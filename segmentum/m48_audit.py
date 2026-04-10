from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .agent import SegmentAgent
from .m47_audit import build_m47_acceptance_report
from .m47_reacceptance import build_m47_reacceptance_report
from .runtime import SegmentRuntime


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

FORMAL_CONCLUSION_NOT_ISSUED = "NOT_ISSUED"

GATE_FLAG_ROUNDTRIP = "memory_flag_roundtrip"
GATE_SUPPRESSION = "memory_suppression_contract"
GATE_ABLATION = "ablation_contrast"
GATE_BIAS_EXPOSURE = "last_memory_context_bias_fields"
GATE_M47_DEMOTION = "m47_runtime_snapshot_demoted"
GATE_LAYER_MODEL = "three_layer_acceptance_documented"

GATE_ORDER = (
    GATE_FLAG_ROUNDTRIP,
    GATE_SUPPRESSION,
    GATE_ABLATION,
    GATE_BIAS_EXPOSURE,
    GATE_M47_DEMOTION,
    GATE_LAYER_MODEL,
)

GATE_CODES = {
    GATE_FLAG_ROUNDTRIP: "G1",
    GATE_SUPPRESSION: "G2",
    GATE_ABLATION: "G3",
    GATE_BIAS_EXPOSURE: "G4",
    GATE_M47_DEMOTION: "G5",
    GATE_LAYER_MODEL: "G6",
}

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

M48_ABLATION_EVIDENCE_PATH = ARTIFACTS_DIR / "m48_ablation_contrast.json"
M48_REPORT_PATH = REPORTS_DIR / "m48_acceptance_report.json"
M48_SUMMARY_PATH = REPORTS_DIR / "m48_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_output_paths(
    *,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, Path]:
    resolved_output_root = Path(output_root).resolve() if output_root is not None else None
    resolved_artifacts_dir = (
        Path(artifacts_dir).resolve()
        if artifacts_dir is not None
        else (resolved_output_root / "artifacts" if resolved_output_root is not None else ARTIFACTS_DIR)
    )
    resolved_reports_dir = (
        Path(reports_dir).resolve()
        if reports_dir is not None
        else (resolved_output_root / "reports" if resolved_output_root is not None else REPORTS_DIR)
    )
    return {
        "ablation_evidence": resolved_artifacts_dir / M48_ABLATION_EVIDENCE_PATH.name,
        "report": resolved_reports_dir / M48_REPORT_PATH.name,
        "summary": resolved_reports_dir / M48_SUMMARY_PATH.name,
    }


def _record(
    *,
    gate: str,
    observed: dict[str, object],
    criteria_checks: dict[str, bool],
    notes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "gate": gate,
        "status": STATUS_PASS if criteria_checks and all(criteria_checks.values()) else STATUS_FAIL,
        "observed": observed,
        "criteria_checks": criteria_checks,
        "notes": list(notes or []),
    }


def _gate_summary(gate: str, records: list[dict[str, object]]) -> dict[str, object]:
    statuses = [str(record["status"]) for record in records]
    status = STATUS_FAIL if any(item == STATUS_FAIL for item in statuses) else STATUS_PASS
    return {
        "gate": gate,
        "status": status,
        "scenario_ids": [gate],
        "counts": {
            "total": len(records),
            "passed": sum(1 for item in statuses if item == STATUS_PASS),
            "failed": sum(1 for item in statuses if item == STATUS_FAIL),
        },
        "notes": [note for record in records for note in record.get("notes", []) if isinstance(note, str) and note],
    }


def _action_entropy(actions: list[str]) -> float:
    total = len(actions)
    if total == 0:
        return 0.0
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)


def _avoidance_ratio(actions: list[str]) -> float:
    if not actions:
        return 0.0
    avoidance_actions = {"hide", "rest"}
    return sum(1 for action in actions if action in avoidance_actions) / len(actions)


def _action_distribution(actions: list[str]) -> dict[str, float]:
    total = len(actions) or 1
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    return {
        key: round(value / total, 6)
        for key, value in sorted(counts.items())
    }


def _direct_query_inputs(runtime: SegmentRuntime) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, object]]:
    observed = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_decision_observation).items()
    }
    baseline_prediction = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_memory_context.get("prediction_before_memory", {})).items()
    }
    errors = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_memory_context.get("errors", {})).items()
    }
    current_state_snapshot = {
        "observation": dict(observed),
        "prediction": dict(baseline_prediction),
        "errors": dict(errors),
        "body_state": runtime.agent._current_body_state(),
    }
    return observed, baseline_prediction, errors, current_state_snapshot


def _rollout_trace(*, memory_enabled: bool, seed: int = 42, cycles: int = 20) -> dict[str, object]:
    runtime = SegmentRuntime.load_or_create(seed=seed, reset=True, memory_enabled=memory_enabled)
    rows: list[dict[str, object]] = []
    for _ in range(cycles):
        runtime.step(verbose=False)
        diagnostics = runtime.agent.last_decision_diagnostics
        context = dict(runtime.agent.last_memory_context)
        aggregate = dict(context.get("aggregate", {}))
        state_delta = {
            str(key): float(value)
            for key, value in dict(context.get("state_delta", {})).items()
        }
        rows.append(
            {
                "cycle": int(runtime.agent.cycle),
                "choice": str(diagnostics.chosen.choice if diagnostics is not None else runtime.agent.last_decision_choice),
                "memory_enabled": bool(context.get("memory_enabled", runtime.agent.memory_enabled)),
                "memory_hit": bool(context.get("memory_hit", False)),
                "memory_bias": float(context.get("memory_bias", 0.0)),
                "pattern_bias": float(context.get("pattern_bias", 0.0)),
                "chronic_threat_bias": float(aggregate.get("chronic_threat_bias", 0.0)),
                "protected_anchor_bias": float(aggregate.get("protected_anchor_bias", 0.0)),
                "state_delta": state_delta,
                "max_abs_state_delta": max((abs(value) for value in state_delta.values()), default=0.0),
                "retrieved_episode_ids": list(context.get("retrieved_episode_ids", [])),
            }
        )
    actions = [str(row["choice"]) for row in rows]
    return {
        "memory_enabled": memory_enabled,
        "seed": seed,
        "cycles": cycles,
        "decision_sequence": actions,
        "decision_entropy": _action_entropy(actions),
        "avoidance_ratio": _avoidance_ratio(actions),
        "action_distribution": _action_distribution(actions),
        "trace": rows,
        "final_context": dict(runtime.agent.last_memory_context),
        "episode_count": len(runtime.agent.long_term_memory.episodes),
        "export_snapshot_memory_enabled": bool(runtime.export_snapshot()["agent"]["memory_enabled"]),
    }


def build_m48_ablation_evidence(*, seed: int = 42, cycles: int = 20) -> dict[str, object]:
    enabled = _rollout_trace(memory_enabled=True, seed=seed, cycles=cycles)
    disabled = _rollout_trace(memory_enabled=False, seed=seed, cycles=cycles)
    control = _rollout_trace(memory_enabled=True, seed=seed, cycles=cycles)

    seq_on = list(enabled["decision_sequence"])
    seq_off = list(disabled["decision_sequence"])
    seq_control = list(control["decision_sequence"])
    differing_cycles = [
        index + 1
        for index, (left, right) in enumerate(zip(seq_on, seq_off))
        if left != right
    ]
    threat_rows = [
        index
        for index, row in enumerate(enabled["trace"])
        if float(row["chronic_threat_bias"]) > 0.1
    ]
    enabled_threat_actions = [seq_on[index] for index in threat_rows]
    disabled_threat_actions = [seq_off[index] for index in threat_rows]
    enabled_nonzero_bias_cycles = [
        int(row["cycle"])
        for row in enabled["trace"]
        if abs(float(row["memory_bias"])) > 1e-9 or abs(float(row["pattern_bias"])) > 1e-9
    ]
    disabled_nonzero_state_delta_cycles = [
        int(row["cycle"])
        for row in disabled["trace"]
        if float(row["max_abs_state_delta"]) > 1e-9
    ]
    return {
        "seed": seed,
        "cycles": cycles,
        "enabled": enabled,
        "disabled": disabled,
        "negative_control": control,
        "comparison": {
            "differing_cycles": differing_cycles,
            "decision_sequences_identical_for_negative_control": seq_on == seq_control,
            "entropy_delta": round(float(enabled["decision_entropy"]) - float(disabled["decision_entropy"]), 6),
            "enabled_threat_cycle_count": len(threat_rows),
            "enabled_avoidance_ratio_under_threat": _avoidance_ratio(enabled_threat_actions),
            "disabled_avoidance_ratio_under_threat": _avoidance_ratio(disabled_threat_actions),
            "enabled_nonzero_bias_cycles": enabled_nonzero_bias_cycles,
            "disabled_nonzero_state_delta_cycles": disabled_nonzero_state_delta_cycles,
        },
    }


def _gate_flag_roundtrip(*, seed: int) -> dict[str, object]:
    agent = SegmentAgent(memory_enabled=False)
    agent_payload = agent.to_dict()
    restored_agent = SegmentAgent.from_dict(agent_payload)

    with TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"

        fresh_runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            seed=seed,
            reset=True,
            memory_enabled=False,
        )
        fresh_runtime.save_snapshot()
        restored_runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            seed=seed + 1,
            memory_enabled=False,
        )

    with TemporaryDirectory() as tmp_dir:
        corrupt_path = Path(tmp_dir) / "segment_state.json"
        corrupt_path.write_text("{not-json", encoding="utf-8")
        recovered_runtime = SegmentRuntime.load_or_create(
            state_path=corrupt_path,
            seed=seed + 2,
            memory_enabled=False,
        )

    observed = {
        "constructor_memory_enabled": agent.memory_enabled,
        "payload_memory_enabled": bool(agent_payload["memory_enabled"]),
        "restored_agent_memory_enabled": restored_agent.memory_enabled,
        "fresh_runtime_memory_enabled": fresh_runtime.agent.memory_enabled,
        "export_snapshot_memory_enabled": bool(fresh_runtime.export_snapshot()["agent"]["memory_enabled"]),
        "restored_runtime_memory_enabled": restored_runtime.agent.memory_enabled,
        "recovered_runtime_memory_enabled": recovered_runtime.agent.memory_enabled,
        "recovered_state_load_status": str(recovered_runtime.state_load_status),
    }
    criteria_checks = {
        "constructor_flag_exists": observed["constructor_memory_enabled"] is False,
        "agent_roundtrip_restores_flag": observed["payload_memory_enabled"] is False
        and observed["restored_agent_memory_enabled"] is False,
        "fresh_runtime_respects_flag": observed["fresh_runtime_memory_enabled"] is False,
        "snapshot_exports_flag": observed["export_snapshot_memory_enabled"] is False,
        "restored_runtime_respects_explicit_override": observed["restored_runtime_memory_enabled"] is False,
        "recovered_runtime_respects_explicit_override": observed["recovered_runtime_memory_enabled"] is False,
    }
    return _record(
        gate=GATE_FLAG_ROUNDTRIP,
        observed=observed,
        criteria_checks=criteria_checks,
    )


def _gate_suppression_contract(*, seed: int) -> dict[str, object]:
    runtime = SegmentRuntime.load_or_create(seed=seed, reset=True, memory_enabled=False)
    runtime.run(cycles=8, verbose=False)
    observed, baseline_prediction, errors, current_state_snapshot = _direct_query_inputs(runtime)
    helper_retrieval = runtime.agent._retrieve_decision_memories(
        observed=observed,
        baseline_prediction=baseline_prediction,
        baseline_errors=errors,
        current_state_snapshot=current_state_snapshot,
        k=3,
    )
    helper_context = runtime.agent._build_memory_context(
        observed=observed,
        baseline_prediction=baseline_prediction,
        errors=errors,
        similar_memories=[{"episode_id": "synthetic-memory"}],
    )
    aggregate = dict(helper_context.get("aggregate", {}))
    state_delta = {
        str(key): float(value)
        for key, value in dict(helper_context.get("state_delta", {})).items()
    }
    observed_payload = {
        "helper_retrieved_count": len(helper_retrieval),
        "helper_state_delta": state_delta,
        "helper_aggregate": aggregate,
        "helper_memory_hit": bool(helper_context.get("memory_hit", False)),
        "runtime_last_memory_context": dict(runtime.agent.last_memory_context),
        "episode_count": len(runtime.agent.long_term_memory.episodes),
    }
    criteria_checks = {
        "helper_retrieval_returns_empty": len(helper_retrieval) == 0,
        "helper_context_memory_hit_false": observed_payload["helper_memory_hit"] is False,
        "helper_state_delta_zeroed": bool(state_delta) and all(abs(value) <= 1e-9 for value in state_delta.values()),
        "helper_biases_zeroed": float(aggregate.get("chronic_threat_bias", 0.0)) == 0.0
        and float(aggregate.get("protected_anchor_bias", 0.0)) == 0.0,
        "decision_bias_fields_zeroed": float(runtime.agent.last_memory_context.get("memory_bias", 0.0)) == 0.0
        and float(runtime.agent.last_memory_context.get("pattern_bias", 0.0)) == 0.0,
        "episode_recording_continues": observed_payload["episode_count"] > 0,
    }
    return _record(
        gate=GATE_SUPPRESSION,
        observed=observed_payload,
        criteria_checks=criteria_checks,
    )


def _gate_ablation_contrast(ablation_evidence: dict[str, object]) -> dict[str, object]:
    enabled = dict(ablation_evidence["enabled"])
    disabled = dict(ablation_evidence["disabled"])
    control = dict(ablation_evidence["negative_control"])
    comparison = dict(ablation_evidence["comparison"])

    criteria_checks = {
        "divergence_present": len(list(comparison["differing_cycles"])) > 0,
        "entropy_differs": not math.isclose(
            float(enabled["decision_entropy"]),
            float(disabled["decision_entropy"]),
            abs_tol=1e-6,
        ),
        "threat_cycles_present": int(comparison["enabled_threat_cycle_count"]) > 0,
        "avoidance_increases_under_threat": float(comparison["enabled_avoidance_ratio_under_threat"])
        > float(comparison["disabled_avoidance_ratio_under_threat"]),
        "enabled_has_nonzero_state_delta": any(
            float(row["max_abs_state_delta"]) > 0.05 for row in enabled["trace"]
        ),
        "disabled_state_delta_is_zero_every_cycle": all(
            float(row["max_abs_state_delta"]) <= 1e-9 for row in disabled["trace"]
        ),
        "negative_control_is_identical": bool(comparison["decision_sequences_identical_for_negative_control"]),
    }
    return _record(
        gate=GATE_ABLATION,
        observed={
            "enabled_entropy": enabled["decision_entropy"],
            "disabled_entropy": disabled["decision_entropy"],
            "enabled_action_distribution": enabled["action_distribution"],
            "disabled_action_distribution": disabled["action_distribution"],
            "comparison": comparison,
            "negative_control_action_distribution": control["action_distribution"],
        },
        criteria_checks=criteria_checks,
        notes=[
            "Per-cycle rollout evidence is stored in the ablation artifact, not only in last_memory_context.",
        ],
    )


def _gate_bias_exposure(ablation_evidence: dict[str, object]) -> dict[str, object]:
    enabled_rows = list(ablation_evidence["enabled"]["trace"])
    disabled_rows = list(ablation_evidence["disabled"]["trace"])
    criteria_checks = {
        "enabled_rows_expose_float_biases": all(
            isinstance(row.get("memory_bias"), float) and isinstance(row.get("pattern_bias"), float)
            for row in enabled_rows
        ),
        "disabled_rows_expose_float_biases": all(
            isinstance(row.get("memory_bias"), float) and isinstance(row.get("pattern_bias"), float)
            for row in disabled_rows
        ),
        "enabled_run_has_nonzero_bias_cycle": any(
            abs(float(row["memory_bias"])) > 1e-9 or abs(float(row["pattern_bias"])) > 1e-9
            for row in enabled_rows
        ),
        "disabled_run_zeroes_biases_every_cycle": all(
            abs(float(row["memory_bias"])) <= 1e-9 and abs(float(row["pattern_bias"])) <= 1e-9
            for row in disabled_rows
        ),
    }
    return _record(
        gate=GATE_BIAS_EXPOSURE,
        observed={
            "enabled_nonzero_bias_cycles": list(ablation_evidence["comparison"]["enabled_nonzero_bias_cycles"]),
            "disabled_bias_preview": disabled_rows[:3],
        },
        criteria_checks=criteria_checks,
    )


def _gate_m47_demotion() -> dict[str, object]:
    official_report, evidence_report, _, _ = build_m47_acceptance_report()
    reacceptance_report = build_m47_reacceptance_report()
    official_snapshot = dict(evidence_report["runtime_snapshot"])
    reacceptance_snapshot = dict(reacceptance_report["runtime_snapshot"])
    behavioral_gate = dict(reacceptance_report["gate_summaries"]["behavioral_scenario_A_threat_learning"])
    criteria_checks = {
        "official_snapshot_marked_diagnostic_only": bool(official_snapshot.get("diagnostic_only")) is True,
        "reacceptance_snapshot_marked_diagnostic_only": bool(reacceptance_snapshot.get("diagnostic_only")) is True,
        "demotion_reason_mentions_m48": "M4.8" in str(official_snapshot.get("demotion_reason", "")),
        "behavioral_gate_is_explicitly_demoted": bool(behavioral_gate.get("behavioral_claims_demoted")) is True
        and str(behavioral_gate.get("evidence_role")) == "diagnostic_only",
    }
    return _record(
        gate=GATE_M47_DEMOTION,
        observed={
            "official_runtime_snapshot": {
                "diagnostic_only": official_snapshot.get("diagnostic_only"),
                "demotion_reason": official_snapshot.get("demotion_reason"),
            },
            "reacceptance_runtime_snapshot": {
                "diagnostic_only": reacceptance_snapshot.get("diagnostic_only"),
                "demotion_reason": reacceptance_snapshot.get("demotion_reason"),
            },
            "behavioral_gate_summary": behavioral_gate,
        },
        criteria_checks=criteria_checks,
    )


def _gate_layer_model_documented() -> dict[str, object]:
    content = (REPORTS_DIR / "m4_milestone_boundaries.md").read_text(encoding="utf-8")
    criteria_checks = {
        "layers_a_b_c_defined": all(
            phrase in content
            for phrase in (
                "Structural self-consistency",
                "Behavioral causation",
                "Phenomenological alignment",
            )
        ),
        "m45_to_m47_layer_a_only_documented": "M4.5–M4.7 currently satisfy only layer (a)" in content,
        "m48_row_annotated": "| M4.8 | Yes | Yes | — |" in content,
    }
    return _record(
        gate=GATE_LAYER_MODEL,
        observed={
            "document_path": str(REPORTS_DIR / "m4_milestone_boundaries.md"),
        },
        criteria_checks=criteria_checks,
    )


def build_m48_acceptance_report(*, seed: int = 42, cycles: int = 20) -> tuple[dict[str, object], dict[str, object]]:
    ablation_evidence = build_m48_ablation_evidence(seed=seed, cycles=cycles)
    records = [
        _gate_flag_roundtrip(seed=seed),
        _gate_suppression_contract(seed=seed),
        _gate_ablation_contrast(ablation_evidence),
        _gate_bias_exposure(ablation_evidence),
        _gate_m47_demotion(),
        _gate_layer_model_documented(),
    ]
    gate_summaries = {
        gate: _gate_summary(gate, [record for record in records if record["gate"] == gate])
        for gate in GATE_ORDER
    }
    all_passed = all(summary["status"] == STATUS_PASS for summary in gate_summaries.values())
    report = {
        "milestone_id": "M4.8",
        "mode": "official_runtime_acceptance",
        "artifact_lineage": "official_runtime_evidence",
        "primary_evidence_chain": True,
        "generated_at": _now_iso(),
        "git_head": _git_head(),
        "status": "PASS" if all_passed else "INCOMPLETE",
        "acceptance_state": "acceptance_issued" if all_passed else "acceptance_not_issued",
        "recommendation": "ACCEPT" if all_passed else "REPAIR",
        "formal_acceptance_conclusion": "ACCEPT" if all_passed else FORMAL_CONCLUSION_NOT_ISSUED,
        "gate_summaries": gate_summaries,
        "evidence_records": records,
        "failed_gates": [gate for gate, summary in gate_summaries.items() if summary["status"] != STATUS_PASS],
        "notes": [
            "M4.8 proves behavioral causation via same-seed memory ablation contrast.",
            "Per-cycle rollout evidence is required; last_memory_context alone is insufficient.",
        ],
    }
    return report, ablation_evidence


def _write_summary(report: dict[str, object], *, summary_path: Path) -> None:
    lines = [
        "# M4.8 Official Acceptance Summary",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Status: `{report['status']}`",
        f"Formal Acceptance Conclusion: `{report['formal_acceptance_conclusion']}`",
        "",
        "## Gate Status",
        "",
    ]
    for gate in GATE_ORDER:
        lines.append(f"- {GATE_CODES[gate]} `{gate}`: `{report['gate_summaries'][gate]['status']}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- M4.8 requires direct helper-contract checks and per-cycle ablation evidence.",
            "- M4.7 runtime-behavior evidence is treated as diagnostic_only until M4.8 passes.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_m48_acceptance_artifacts(
    *,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    round_started_at: str | None = None,
    seed: int = 42,
    cycles: int = 20,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    report, ablation_evidence = build_m48_acceptance_report(seed=seed, cycles=cycles)
    report["round_started_at"] = round_started_at or _now_iso()
    report["artifacts"] = {
        "ablation_evidence": str(output_paths["ablation_evidence"]),
        "summary": str(output_paths["summary"]),
    }
    report["tests"] = {
        "milestone": [
            "tests/test_m48_ablation_contrast.py",
            "tests/test_m48_acceptance.py",
        ],
    }
    report["freshness"] = {
        "artifact_round_started_at": report["round_started_at"],
        "generated_in_this_run": True,
    }

    output_paths["ablation_evidence"].write_text(
        json.dumps(ablation_evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(report, summary_path=output_paths["summary"])
    return {key: str(path) for key, path in output_paths.items()}
