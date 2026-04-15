from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .agent import SegmentAgent
from .environment import Observation
from .memory import LongTermMemory


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M232_SPEC_PATH = REPORTS_DIR / "m232_milestone_spec.md"
M232_PREPARATION_PATH = REPORTS_DIR / "m232_strict_audit_preparation.md"
M232_TRACE_PATH = ARTIFACTS_DIR / "m232_threat_memory_trace.jsonl"
M232_ABLATION_PATH = ARTIFACTS_DIR / "m232_threat_memory_ablation.json"
M232_STRESS_PATH = ARTIFACTS_DIR / "m232_threat_memory_stress.json"
M232_REPORT_PATH = REPORTS_DIR / "m232_acceptance_report.json"
M232_SUMMARY_PATH = REPORTS_DIR / "m232_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (232, 464)
M232_TESTS: tuple[str, ...] = (
    "tests/test_restart_memory_protection.py",
    "tests/test_m28_attention.py",
    "tests/test_m2_targeted_repair.py",
    "tests/test_m232_acceptance.py",
    "tests/test_m232_audit_preparation.py",
)
M232_REGRESSIONS: tuple[str, ...] = (
    "tests/test_memory.py",
    "tests/test_threat_profile_learning.py",
)
M232_GATES: tuple[str, ...] = (
    "schema",
    "protection",
    "causality",
    "attention_prediction_influence",
    "regression",
    "artifact_freshness",
)

THREAT_OBSERVATION = {
    "food": 0.12,
    "danger": 0.82,
    "novelty": 0.22,
    "shelter": 0.10,
    "temperature": 0.46,
    "social": 0.18,
}
THREAT_PREDICTION = {
    "food": 0.60,
    "danger": 0.20,
    "novelty": 0.40,
    "shelter": 0.40,
    "temperature": 0.50,
    "social": 0.30,
}
THREAT_OUTCOME = {
    "energy_delta": -0.10,
    "stress_delta": 0.28,
    "fatigue_delta": 0.18,
    "temperature_delta": 0.02,
    "free_energy_drop": -0.45,
}
THREAT_BODY_STATE = {
    "energy": 0.50,
    "stress": 0.40,
    "fatigue": 0.25,
    "temperature": 0.46,
}
ATTENTION_CHALLENGE_OBSERVATION = {
    "danger": 0.32,
    "novelty": 0.55,
    "food": 0.52,
}
ATTENTION_CHALLENGE_PREDICTION = {
    "danger": 0.28,
    "novelty": 0.30,
    "food": 0.50,
}
ATTENTION_CHALLENGE_ERRORS = {
    "danger": 0.04,
    "novelty": 0.25,
    "food": 0.02,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso8601(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _artifact_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(timespec="seconds"),
    }


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty_paths() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _suite_execution_record(*, label: str, paths: Iterable[str], execute: bool) -> dict[str, object]:
    normalized_paths = [str(path) for path in paths]
    if not execute:
        return {
            "label": label,
            "paths": normalized_paths,
            "executed": False,
            "passed": False,
            "returncode": None,
            "command": [],
            "stdout": "",
            "stderr": "",
            "execution_source": "skipped",
            "started_at": None,
            "completed_at": None,
        }

    command = [sys.executable, "-m", "pytest", *normalized_paths, "-q"]
    started_at = _now_iso()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    completed_at = _now_iso()
    return {
        "label": label,
        "paths": normalized_paths,
        "executed": True,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "execution_source": "subprocess",
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _is_authentic_execution_record(record: dict[str, object], *, expected_paths: Iterable[str]) -> bool:
    expected = [str(path) for path in expected_paths]
    command = record.get("command", [])
    started_at = _parse_iso8601(record.get("started_at"))
    completed_at = _parse_iso8601(record.get("completed_at"))
    return bool(
        isinstance(record, dict)
        and bool(record.get("executed"))
        and record.get("execution_source") == "subprocess"
        and isinstance(command, list)
        and len(command) >= 4
        and command[1:3] == ["-m", "pytest"]
        and list(record.get("paths", [])) == expected
        and started_at is not None
        and completed_at is not None
        and completed_at >= started_at
    )


def _freshness_gate(
    *,
    artifacts: dict[str, str],
    audit_started_at: str,
    generated_at: str,
    milestone_execution: dict[str, object],
    regression_execution: dict[str, object],
    strict: bool,
) -> tuple[bool, dict[str, object]]:
    audit_started = _parse_iso8601(audit_started_at)
    generated = _parse_iso8601(generated_at)
    artifact_records = {
        name: _artifact_record(Path(path))
        for name, path in artifacts.items()
        if Path(path).exists()
    }
    evidence_times = [
        _parse_iso8601(record.get("modified_at"))
        for name, record in artifact_records.items()
        if name in {"canonical_trace", "ablation", "stress"}
    ]
    evidence_times_ok = bool(audit_started and generated and evidence_times) and all(
        modified is not None and audit_started <= modified <= generated
        for modified in evidence_times
    )
    report_times = [
        _parse_iso8601(record.get("modified_at"))
        for name, record in artifact_records.items()
        if name in {"report", "summary"}
    ]
    report_times_ok = bool(audit_started and report_times) and all(
        modified is not None and audit_started <= modified
        for modified in report_times
    )
    milestone_authentic = _is_authentic_execution_record(
        milestone_execution,
        expected_paths=M232_TESTS,
    )
    regression_authentic = _is_authentic_execution_record(
        regression_execution,
        expected_paths=M232_REGRESSIONS,
    )
    suite_times = [
        _parse_iso8601(milestone_execution.get("started_at")),
        _parse_iso8601(milestone_execution.get("completed_at")),
        _parse_iso8601(regression_execution.get("started_at")),
        _parse_iso8601(regression_execution.get("completed_at")),
    ]
    suite_times_ok = bool(audit_started and generated) and all(
        timestamp is not None and audit_started <= timestamp <= generated
        for timestamp in suite_times
    )
    dirty_paths = _git_dirty_paths()
    freshness_ok = evidence_times_ok and report_times_ok and (
        not strict or (milestone_authentic and regression_authentic and suite_times_ok)
    )
    return freshness_ok, {
        "strict": strict,
        "audit_started_at": audit_started_at,
        "generated_at": generated_at,
        "artifact_records": artifact_records,
        "evidence_times_within_round": evidence_times_ok,
        "report_times_within_round": report_times_ok,
        "milestone_execution_authentic": milestone_authentic,
        "regression_execution_authentic": regression_authentic,
        "suite_times_within_round": suite_times_ok,
        "current_round": freshness_ok,
        "git": {
            "head": _git_commit(),
            "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
        },
    }


def preparation_manifest() -> dict[str, object]:
    return {
        "milestone_id": "M2.32",
        "title": "Threat Trace Protection And Trauma-Like Encoding",
        "status": "PREPARATION_READY",
        "assumption_source": str(M232_SPEC_PATH),
        "seed_set": list(SEED_SET),
        "artifacts": {
            "specification": str(M232_SPEC_PATH),
            "preparation": str(M232_PREPARATION_PATH),
            "canonical_trace": str(M232_TRACE_PATH),
            "ablation": str(M232_ABLATION_PATH),
            "stress": str(M232_STRESS_PATH),
            "report": str(M232_REPORT_PATH),
            "summary": str(M232_SUMMARY_PATH),
        },
        "tests": {
            "milestone": list(M232_TESTS),
            "regressions": list(M232_REGRESSIONS),
        },
        "gates": list(M232_GATES),
    }


def _threat_errors() -> dict[str, float]:
    return {
        key: THREAT_OBSERVATION[key] - THREAT_PREDICTION[key]
        for key in THREAT_OBSERVATION
    }


def _threat_observation_model() -> Observation:
    return Observation(**THREAT_OBSERVATION)


def _seed_threat_memory(
    agent: SegmentAgent,
    *,
    count: int = 5,
    protect_anchor: bool,
) -> dict[str, object]:
    episode_ids: list[str] = []
    for cycle in range(1, count + 1):
        payload = agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=THREAT_OBSERVATION,
            prediction=THREAT_PREDICTION,
            errors=_threat_errors(),
            action="forage",
            outcome=THREAT_OUTCOME,
            body_state=THREAT_BODY_STATE,
        )
        episode_ids.append(str(payload.get("episode_id", "")))

    protected_episode_id = episode_ids[-1]
    protected_count = 0
    if protect_anchor:
        protected_count = agent.long_term_memory.protect_episode_ids(
            [protected_episode_id],
            reason="chronic_threat_trace",
            continuity_tag="structural_trace",
        )

    return {
        "episode_ids": episode_ids,
        "protected_episode_id": protected_episode_id,
        "protected_count": protected_count,
    }


def _current_priors(agent: SegmentAgent) -> dict[str, float]:
    return agent.strategic_layer.priors(
        agent.energy,
        agent.stress,
        agent.fatigue,
        agent.temperature,
        agent.dopamine,
        agent.drive_system,
        personality_modulation=agent._personality_strategic_modulation,
    )


def _build_runtime_attention_snapshot(
    agent: SegmentAgent,
    *,
    memory_context: dict[str, object],
) -> dict[str, object]:
    priors = _current_priors(agent)
    baseline_prediction = agent.world_model.predict(priors)
    full_trace = agent.attention_bottleneck.allocate(
        observation=ATTENTION_CHALLENGE_OBSERVATION,
        prediction=ATTENTION_CHALLENGE_PREDICTION,
        errors=ATTENTION_CHALLENGE_ERRORS,
        narrative_priors=agent.self_model.narrative_priors.to_dict(),
        tick=agent.cycle,
        memory_context=memory_context,
    )
    ablated_context = {
        **memory_context,
        "sensitive_channels": [],
        "attention_biases": {},
        "aggregate": {
            **dict(memory_context.get("aggregate", {})),
            "chronic_threat_bias": 0.0,
            "protected_anchor_bias": 0.0,
        },
    }
    ablated_trace = agent.attention_bottleneck.allocate(
        observation=ATTENTION_CHALLENGE_OBSERVATION,
        prediction=ATTENTION_CHALLENGE_PREDICTION,
        errors=ATTENTION_CHALLENGE_ERRORS,
        narrative_priors=agent.self_model.narrative_priors.to_dict(),
        tick=agent.cycle,
        memory_context=ablated_context,
    )
    full_prediction = agent.world_model.predict(priors, memory_context=memory_context)
    ablated_prediction = agent.world_model.predict(
        priors,
        memory_context={
            **memory_context,
            "state_projection": {},
            "state_delta": {},
            "prediction_blend": 0.0,
            "delta_gain": 0.0,
        },
    )
    return {
        "baseline_prediction": baseline_prediction,
        "full_attention_selected": list(full_trace.allocation.selected_channels),
        "ablated_attention_selected": list(ablated_trace.allocation.selected_channels),
        "full_attention_scores": dict(full_trace.salience_scores),
        "ablated_attention_scores": dict(ablated_trace.salience_scores),
        "full_prediction": full_prediction,
        "ablated_prediction": ablated_prediction,
        "full_prediction_delta": {
            key: full_prediction.get(key, 0.0) - baseline_prediction.get(key, 0.0)
            for key in baseline_prediction
        },
        "ablated_prediction_delta": {
            key: ablated_prediction.get(key, 0.0) - baseline_prediction.get(key, 0.0)
            for key in baseline_prediction
        },
    }


def _run_decision_probe(*, seed: int, protect_anchor: bool) -> dict[str, object]:
    agent = SegmentAgent(rng=random.Random(seed))
    agent.attention_bottleneck.enabled = True
    agent.attention_bottleneck.capacity = 1
    seed_info = _seed_threat_memory(agent, count=5, protect_anchor=protect_anchor)

    agent.cycle = 40
    agent.decision_cycle(_threat_observation_model())
    agent.cycle = 41
    result = agent.decision_cycle(_threat_observation_model())
    diagnostics = result["diagnostics"]
    ranking = [option.choice for option in diagnostics.ranked_options[:3]]
    scores = {
        option.choice: float(option.policy_score)
        for option in diagnostics.ranked_options[:3]
    }
    attention_probe = _build_runtime_attention_snapshot(
        agent,
        memory_context=dict(agent.last_memory_context),
    )
    return {
        "seed": seed,
        "protect_anchor": protect_anchor,
        "agent": agent,
        "result": result,
        "diagnostics": diagnostics,
        "seed_info": seed_info,
        "ranking": ranking,
        "scores": scores,
        "attention_probe": attention_probe,
    }


def _run_restart_stress_probe(*, seed: int) -> dict[str, object]:
    memory = LongTermMemory()
    memory.max_active_age = 3
    episode_ids: list[str] = []
    for cycle in range(1, 6):
        payload = memory.store_episode(
            cycle=cycle,
            observation=THREAT_OBSERVATION,
            prediction=THREAT_PREDICTION,
            errors=_threat_errors(),
            action="forage",
            outcome=THREAT_OUTCOME,
            body_state=THREAT_BODY_STATE,
        )
        episode_ids.append(str(payload.get("episode_id", "")))
    protected_episode_id = episode_ids[-1]
    protected_count = memory.protect_episode_ids(
        [protected_episode_id],
        reason="chronic_threat_trace",
        continuity_tag="structural_trace",
    )
    memory.activate_restart_continuity_window(current_cycle=6, duration=64)
    removed = memory.compress_episodes(current_cycle=12)
    restored = LongTermMemory.from_dict(memory.to_dict())
    restart_ids = {
        str(item.get("episode_id", ""))
        for item in restored.restart_anchor_payload()
    }
    retrieved = restored.retrieve_similar_memories(
        {
            "observation": THREAT_OBSERVATION,
            "prediction": THREAT_PREDICTION,
            "errors": _threat_errors(),
            "body_state": THREAT_BODY_STATE,
        },
        k=3,
    )
    return {
        "seed": seed,
        "protected_count": protected_count,
        "protected_episode_id": protected_episode_id,
        "compress_removed": removed,
        "restart_anchor_ids": sorted(restart_ids),
        "retrieved_episode_ids": [
            str(item.get("episode_id", ""))
            for item in retrieved
            if item.get("episode_id")
        ],
        "retrieved_similarities": {
            str(item.get("episode_id", "")): float(item.get("similarity", 0.0))
            for item in retrieved
            if item.get("episode_id")
        },
    }


def build_m232_runtime_evidence() -> dict[str, object]:
    full = _run_decision_probe(seed=SEED_SET[0], protect_anchor=True)
    unprotected = _run_decision_probe(seed=SEED_SET[0], protect_anchor=False)
    stress = _run_restart_stress_probe(seed=SEED_SET[1])

    full_diag = full["diagnostics"]
    unprotected_diag = unprotected["diagnostics"]
    full_probe = full["attention_probe"]
    full_danger_delta = float(full_diag.prediction_delta.get("danger", 0.0))
    no_trace_danger_delta = float(unprotected_diag.prediction_delta.get("danger", 0.0))
    no_shaping_danger_delta = float(full_probe["ablated_prediction_delta"].get("danger", 0.0))

    protected_episode_id = str(full["seed_info"]["protected_episode_id"])
    full_context = dict(full["agent"].last_memory_context)
    aggregate = dict(full_context.get("aggregate", {}))

    trace_records = [
        {
            "seed": full["seed"],
            "event": "protected_anchor_created",
            "episode_id": protected_episode_id,
            "retrieved_episode_ids": list(full_diag.retrieved_episode_ids),
            "continuity_tags": list(
                next(
                    (
                        payload.get("continuity_tags", [])
                        for payload in full["agent"].long_term_memory.episodes
                        if str(payload.get("episode_id", "")) == protected_episode_id
                    ),
                    [],
                )
            ),
            "restart_protected": protected_episode_id in {
                str(item.get("episode_id", ""))
                for item in full["agent"].long_term_memory.restart_anchor_payload()
            },
            "memory_protection_reasons": list(
                next(
                    (
                        payload.get("memory_protection_reasons", [])
                        for payload in full["agent"].long_term_memory.episodes
                        if str(payload.get("episode_id", "")) == protected_episode_id
                    ),
                    [],
                )
            ),
            "memory_context_summary": str(full_diag.memory_context_summary),
        },
        {
            "seed": full["seed"],
            "event": "attention_and_prediction_shift",
            "attention_selected_channels": list(full_diag.attention_selected_channels),
            "ranking": list(full["ranking"]),
            "policy_scores": dict(full["scores"]),
            "prediction_before_memory": dict(full_diag.prediction_before_memory),
            "prediction_after_memory": dict(full_diag.prediction_after_memory),
            "prediction_delta": dict(full_diag.prediction_delta),
            "aggregate": {
                "chronic_threat_bias": float(aggregate.get("chronic_threat_bias", 0.0)),
                "protected_anchor_bias": float(aggregate.get("protected_anchor_bias", 0.0)),
            },
            "sensitive_channels": list(full_context.get("sensitive_channels", [])),
            "protected_anchor_retrieved": protected_episode_id in set(full_diag.retrieved_episode_ids),
        },
    ]

    ablation = {
        "full_mechanism": {
            "protected_anchor_survives_restart": stress["protected_episode_id"] in set(
                stress["restart_anchor_ids"]
            ),
            "danger_prediction_delta": full_danger_delta,
            "danger_attention_promoted": "danger" in set(full_diag.attention_selected_channels),
            "top_ranked_actions": list(full["ranking"]),
        },
        "without_structural_trace_protection": {
            "protected_anchor_survives_restart": False,
            "danger_prediction_delta": no_trace_danger_delta,
            "danger_attention_promoted": "danger" in set(unprotected_diag.attention_selected_channels),
            "top_ranked_actions": list(unprotected["ranking"]),
        },
        "without_attention_bias": {
            "danger_attention_promoted": "danger" in set(full_probe["ablated_attention_selected"]),
            "attention_selected_channels": list(full_probe["ablated_attention_selected"]),
        },
        "without_prediction_shaping": {
            "danger_prediction_delta": no_shaping_danger_delta,
        },
        "degradation_checks": {
            "protection_degrades_without_anchor_mechanism": False,
            "prediction_shift_degrades_without_threat_trace": (
                full_danger_delta > (no_trace_danger_delta + 1e-9)
                or full_danger_delta > (no_shaping_danger_delta + 1e-9)
            ),
            "attention_shift_degrades_without_sensitive_pattern_bias": "danger" in set(
                full_probe["full_attention_selected"]
            ) and "danger" not in set(full_probe["ablated_attention_selected"]),
            "action_ranking_changes_under_full_mechanism": list(full["ranking"]) != list(unprotected["ranking"]),
        },
    }
    ablation["degradation_checks"]["protection_degrades_without_anchor_mechanism"] = (
        full["seed_info"]["protected_count"] == 1
        and unprotected["seed_info"]["protected_count"] == 0
        and protected_episode_id in {
            str(item.get("episode_id", ""))
            for item in full["agent"].long_term_memory.restart_anchor_payload()
        }
    )

    stress_payload = {
        "stress_checks": {
            "protect_episode_ids_handles_non_identity_anchor": stress["protected_count"] == 1,
            "structural_trace_anchor_remains_restart_protected": stress["protected_episode_id"]
            in set(stress["restart_anchor_ids"]),
            "memory_sensitive_pattern_promotes_threat_channel": "danger"
            in set(full_probe["full_attention_selected"]),
            "prediction_surface_remains_shifted_under_replay": float(
                full_diag.prediction_delta.get("danger", 0.0)
            )
            > 0.0,
            "roundtrip_retrieval_preserves_anchor": stress["protected_episode_id"]
            in set(stress["retrieved_episode_ids"]),
        },
        "details": stress,
    }

    gates = {
        "schema": {
            "passed": (
                bool(trace_records[0].get("episode_id"))
                and bool(trace_records[0].get("continuity_tags"))
                and bool(trace_records[0].get("restart_protected"))
            ),
            "evidence": ["episode_id", "continuity_tags", "restart_protected"],
        },
        "protection": {
            "passed": bool(stress_payload["stress_checks"]["protect_episode_ids_handles_non_identity_anchor"])
            and bool(stress_payload["stress_checks"]["structural_trace_anchor_remains_restart_protected"]),
            "evidence": ["structural_trace", "chronic_threat_trace"],
        },
        "causality": {
            "passed": bool(ablation["degradation_checks"]["prediction_shift_degrades_without_threat_trace"])
            and bool(stress_payload["stress_checks"]["roundtrip_retrieval_preserves_anchor"]),
            "evidence": [
                "danger_prediction_delta",
                "protected_anchor_survives_restart",
                "action_ranking_changes_under_full_mechanism",
            ],
        },
        "attention_prediction_influence": {
            "passed": bool(ablation["degradation_checks"]["attention_shift_degrades_without_sensitive_pattern_bias"])
            and float(full_diag.prediction_delta.get("danger", 0.0))
            > float(full_probe["ablated_prediction_delta"].get("danger", 0.0)),
            "evidence": ["attention_selected_channels", "prediction_delta"],
        },
    }

    return {
        "trace_records": trace_records,
        "ablation": ablation,
        "stress": stress_payload,
        "gates": gates,
    }


def _write_trace(path: Path, trace_records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True, sort_keys=True) for record in trace_records) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _summary_markdown(report: dict[str, object]) -> str:
    gates = report["gates"]
    lines = [
        "# M2.32 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Protection gate: {'PASS' if gates['protection']['passed'] else 'FAIL'}",
        f"- Causality gate: {'PASS' if gates['causality']['passed'] else 'FAIL'}",
        f"- Attention/prediction gate: {'PASS' if gates['attention_prediction_influence']['passed'] else 'FAIL'}",
        f"- Regression gate: {'PASS' if gates['regression']['passed'] else 'FAIL'}",
        f"- Freshness gate: {'PASS' if gates['artifact_freshness']['passed'] else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


def write_m232_acceptance_artifacts(
    *,
    strict: bool = True,
    execute_test_suites: bool = True,
    milestone_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
) -> dict[str, str]:
    if strict:
        for injected, label in (
            (milestone_execution, "milestone"),
            (regression_execution, "regression"),
        ):
            if injected is not None and injected.get("execution_source") != "subprocess":
                raise ValueError(
                    f"strict M2.32 audit refuses injected execution records for {label} tests"
                )

    audit_started_at = _now_iso()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    runtime_evidence = build_m232_runtime_evidence()
    _write_trace(M232_TRACE_PATH, runtime_evidence["trace_records"])
    _write_json(M232_ABLATION_PATH, runtime_evidence["ablation"])
    _write_json(M232_STRESS_PATH, runtime_evidence["stress"])

    milestone_execution = milestone_execution or _suite_execution_record(
        label="m232-milestone",
        paths=M232_TESTS,
        execute=execute_test_suites,
    )
    regression_execution = regression_execution or _suite_execution_record(
        label="m232-regression",
        paths=M232_REGRESSIONS,
        execute=execute_test_suites,
    )

    artifacts = {
        "specification": str(M232_SPEC_PATH),
        "preparation": str(M232_PREPARATION_PATH),
        "canonical_trace": str(M232_TRACE_PATH),
        "ablation": str(M232_ABLATION_PATH),
        "stress": str(M232_STRESS_PATH),
        "report": str(M232_REPORT_PATH),
        "summary": str(M232_SUMMARY_PATH),
    }

    def _assemble_report(
        *,
        generated_at_value: str,
        freshness_ok_value: bool,
        freshness_payload_value: dict[str, object],
    ) -> dict[str, object]:
        gates = {
            **runtime_evidence["gates"],
            "regression": {
                "passed": bool(regression_execution.get("executed")) and bool(regression_execution.get("passed")),
                "execution": regression_execution,
            },
            "artifact_freshness": {
                "passed": freshness_ok_value,
                "details": freshness_payload_value,
            },
        }
        milestone_ok = bool(milestone_execution.get("executed")) and bool(milestone_execution.get("passed"))
        for gate_name in ("schema", "protection", "causality", "attention_prediction_influence"):
            gates[gate_name]["passed"] = bool(gates[gate_name]["passed"]) and milestone_ok
        all_passed = all(bool(payload["passed"]) for payload in gates.values())
        return {
            "milestone_id": "M2.32",
            "title": "Threat Trace Protection And Trauma-Like Encoding",
            "strict": strict,
            "status": "PASS" if all_passed else "FAIL",
            "recommendation": "ACCEPT" if all_passed else "BLOCK",
            "generated_at": generated_at_value,
            "seed_set": list(SEED_SET),
            "artifacts": artifacts,
            "tests": {
                "milestone": milestone_execution,
                "regressions": regression_execution,
            },
            "gates": gates,
            "findings": [],
            "residual_risks": [] if all_passed else ["runtime evidence or freshness gate missing"],
            "freshness": freshness_payload_value,
        }

    provisional_generated_at = _now_iso()
    provisional_report = _assemble_report(
        generated_at_value=provisional_generated_at,
        freshness_ok_value=False,
        freshness_payload_value={
            "strict": strict,
            "audit_started_at": audit_started_at,
            "generated_at": provisional_generated_at,
            "artifact_records": {},
            "evidence_times_within_round": False,
            "report_times_within_round": False,
            "milestone_execution_authentic": False,
            "regression_execution_authentic": False,
            "suite_times_within_round": False,
            "current_round": False,
            "git": {
                "head": _git_commit(),
                "dirty": bool(_git_dirty_paths()),
                "dirty_paths": _git_dirty_paths(),
            },
        },
    )
    M232_REPORT_PATH.write_text(
        json.dumps(provisional_report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    M232_SUMMARY_PATH.write_text(_summary_markdown(provisional_report), encoding="utf-8")

    final_generated_at = _now_iso()
    freshness_ok, freshness_payload = _freshness_gate(
        artifacts=artifacts,
        audit_started_at=audit_started_at,
        generated_at=final_generated_at,
        milestone_execution=milestone_execution,
        regression_execution=regression_execution,
        strict=strict,
    )
    final_report = _assemble_report(
        generated_at_value=final_generated_at,
        freshness_ok_value=freshness_ok,
        freshness_payload_value=freshness_payload,
    )
    M232_REPORT_PATH.write_text(json.dumps(final_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M232_SUMMARY_PATH.write_text(_summary_markdown(final_report), encoding="utf-8")
    return artifacts
