from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m45_acceptance_data import REGRESSION_TARGETS, build_m45_acceptance_payload


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M45_CANONICAL_TRACE_PATH = ARTIFACTS_DIR / "m45_canonical_trace.json"
M45_ABLATION_PATH = ARTIFACTS_DIR / "m45_ablation.json"
M45_FAILURE_INJECTION_PATH = ARTIFACTS_DIR / "m45_failure_injection.json"
M45_REPORT_PATH = REPORTS_DIR / "m45_acceptance_report.json"
M45_SUMMARY_PATH = REPORTS_DIR / "m45_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
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
        "canonical_trace": resolved_artifacts_dir / M45_CANONICAL_TRACE_PATH.name,
        "ablation": resolved_artifacts_dir / M45_ABLATION_PATH.name,
        "failure_injection": resolved_artifacts_dir / M45_FAILURE_INJECTION_PATH.name,
        "report": resolved_reports_dir / M45_REPORT_PATH.name,
        "summary": resolved_reports_dir / M45_SUMMARY_PATH.name,
    }


def _top_blockers(findings: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    severity_rank = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
    return sorted(
        findings,
        key=lambda item: (
            severity_rank.get(str(item.get("severity", "S3")), 99),
            str(item.get("label", "")),
        ),
    )[:limit]


def _probe(payload: dict[str, Any], channel: str, probe_id: str) -> dict[str, Any]:
    probes = payload.get(f"{channel}_probes", {})
    if not isinstance(probes, dict):
        return {}
    probe = probes.get(probe_id, {})
    if not isinstance(probe, dict):
        return {}
    observed = probe.get("observed", {})
    return dict(observed) if isinstance(observed, dict) else {}


def _all_gate_evidence_present(gates: dict[str, dict[str, Any]]) -> bool:
    return all(isinstance(gate.get("evidence"), dict) and bool(gate["evidence"]) for gate in gates.values())


def _store_level_rank(level: Any) -> int:
    return {"short": 0, "mid": 1, "long": 2}.get(str(level), -1)


def _confidence_pairs_are_structured(pairs: Any) -> bool:
    if not isinstance(pairs, list) or len(pairs) != 4:
        return False
    required = {"label", "entry_id", "source_type", "source_confidence", "reality_confidence"}
    source_types: set[str] = set()
    values: set[tuple[Any, Any]] = set()
    for item in pairs:
        if not isinstance(item, dict) or not required.issubset(item):
            return False
        source_types.add(str(item["source_type"]))
        values.add((item["source_confidence"], item["reality_confidence"]))
    return source_types == {"experience", "hearsay", "inference", "reconstruction"} and len(values) == 4


def _drift_case_matches(case: Any, *, label: str, source_changed: bool, reality_changed: bool) -> bool:
    required = {"label", "entry_id", "source_type", "before", "after", "source_changed", "reality_changed"}
    if not isinstance(case, dict) or not required.issubset(case):
        return False
    before = case.get("before", {})
    after = case.get("after", {})
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    return bool(
        case.get("label") == label
        and case.get("source_changed") is source_changed
        and case.get("reality_changed") is reality_changed
        and before.get("source_confidence") is not None
        and before.get("reality_confidence") is not None
        and after.get("source_confidence") is not None
        and after.get("reality_confidence") is not None
    )


def _confidence_drift_cases_show_independence(cases: Any) -> bool:
    if not isinstance(cases, list):
        return False
    case_by_label = {
        item.get("label"): item
        for item in cases
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    }
    return bool(
        _drift_case_matches(case_by_label.get("source_only_drift"), label="source_only_drift", source_changed=True, reality_changed=False)
        and _drift_case_matches(case_by_label.get("reality_only_drift"), label="reality_only_drift", source_changed=False, reality_changed=True)
    )


def _regression_summary_passes(summary: dict[str, Any]) -> bool:
    return bool(
        summary
        and summary.get("executed") is True
        and summary.get("returncode") == 0
        and summary.get("passed") is True
        and summary.get("files") == REGRESSION_TARGETS
    )


def _run_regression_summary() -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", *REGRESSION_TARGETS, "-q"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_seconds = round(time.perf_counter() - started, 3)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = "\n".join(item for item in [stdout.strip(), stderr.strip()] if item).strip()
    summary_line = ""
    if combined:
        summary_line = combined.splitlines()[-1]
    passed_match = re.search(r"(\d+)\s+passed", combined)
    return {
        "executed": True,
        "command": command,
        "files": list(REGRESSION_TARGETS),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "passed_count": int(passed_match.group(1)) if passed_match else 0,
        "duration_seconds": duration_seconds,
        "summary_line": summary_line,
        "stdout_tail": combined.splitlines()[-5:],
    }


def _transition_passes(record: dict[str, Any], *, old_level: str, new_level: str) -> bool:
    required = {
        "reason",
        "effective_cycle",
        "elapsed_since_baseline",
        "old_level",
        "new_level",
        "trace_before",
        "trace_after",
        "accessibility_before",
        "accessibility_after",
        "trace_rate_before",
        "trace_rate_after",
        "access_rate_before",
        "access_rate_after",
    }
    if not required.issubset(record):
        return False
    return bool(
        record.get("old_level") == old_level
        and record.get("new_level") == new_level
        and float(record.get("elapsed_since_baseline", 0.0)) > 0.0
        and float(record.get("trace_after", 0.0)) > float(record.get("trace_before", 0.0))
        and float(record.get("accessibility_after", 0.0)) > float(record.get("accessibility_before", 0.0))
    )


def _catalog_integrity(payload: dict[str, Any]) -> bool:
    catalog = payload.get("probe_catalog", {})
    boundary = payload.get("boundary_probes", {})
    integration = payload.get("integration_probes", {})
    if not isinstance(catalog, dict) or not isinstance(boundary, dict) or not isinstance(integration, dict):
        return False
    boundary_ids = {item.get("id") for item in catalog.get("boundary", []) if isinstance(item, dict)}
    integration_ids = {item.get("id") for item in catalog.get("integration", []) if isinstance(item, dict)}
    return bool(
        boundary_ids == set(boundary)
        and integration_ids == set(integration)
        and list(catalog.get("regression_targets", [])) == list(REGRESSION_TARGETS)
    )


def _evaluate_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    boundary_data_model = _probe(payload, "boundary", "data_model_boundary")
    integration_data_model = _probe(payload, "integration", "data_model_integration")
    boundary_salience = _probe(payload, "boundary", "salience_boundary")
    integration_salience = _probe(payload, "integration", "salience_integration")
    boundary_encoding = _probe(payload, "boundary", "encoding_boundary")
    integration_encoding = _probe(payload, "integration", "encoding_integration")
    boundary_decay = _probe(payload, "boundary", "decay_boundary")
    integration_decay = _probe(payload, "integration", "decay_integration")
    boundary_legacy = _probe(payload, "boundary", "legacy_bridge_boundary")
    integration_legacy = _probe(payload, "integration", "legacy_bridge_integration")
    boundary_transitions = _probe(payload, "boundary", "store_transitions_boundary")
    integration_transitions = _probe(payload, "integration", "store_transitions_integration")
    regression_summary = dict(payload.get("regression_summary", {}))
    ablation = dict(payload.get("ablation", {}))
    failure_injection = dict(payload.get("failure_injection", {}))

    source_defaults_match = (
        tuple(boundary_encoding.get("source_defaults", {}).get("experience", ())) == (0.9, 0.85)
        and tuple(boundary_encoding.get("source_defaults", {}).get("hearsay", ())) == (0.7, 0.5)
        and tuple(boundary_encoding.get("source_defaults", {}).get("inference", ())) == (0.9, 0.35)
        and tuple(boundary_encoding.get("source_defaults", {}).get("reconstruction", ())) == (0.4, 0.5)
    )
    trace_curves = dict(boundary_decay.get("trace_curves", {}))
    accessibility_curves = dict(boundary_decay.get("accessibility_curves", {}))
    curve_relationships_hold = bool(
        trace_curves and accessibility_curves
        and all(
            trace_curves["short"][index] < trace_curves["mid"][index] < trace_curves["long"][index]
            and accessibility_curves["short"][index] < accessibility_curves["mid"][index] < accessibility_curves["long"][index]
            and trace_curves["short"][index] > accessibility_curves["short"][index]
            and trace_curves["mid"][index] > accessibility_curves["mid"][index]
            and trace_curves["long"][index] > accessibility_curves["long"][index]
            and trace_curves["procedural_long"][index] > trace_curves["long"][index]
            for index in range(len(boundary_decay.get("timepoints", [])))
        )
    )
    forgetting_paths = dict(boundary_decay.get("forgetting_paths", {}))
    cleanup_report = dict(boundary_decay.get("cleanup_report", {}))
    confidence_pairs = integration_encoding.get("confidence_pairs", [])
    confidence_drift_cases = integration_encoding.get("confidence_drift_cases", [])

    preliminary_gates = {
        "data_model_integrity": {
            "passed": bool(
                boundary_data_model.get("round_trip_equal")
                and boundary_data_model.get("content_hash_stable")
                and boundary_data_model.get("version_incremented")
                and boundary_data_model.get("procedural_guard")
                and boundary_data_model.get("anchor_guard")
                and integration_data_model.get("round_trip_equal")
                and integration_data_model.get("semantic_lineage_type")
                and integration_data_model.get("inferred_lineage_type")
                and dict(integration_data_model.get("protected_anchor_strengths", {})).get("action") in {"strong", "locked"}
            ),
            "blocking": True,
            "evidence": {"boundary": boundary_data_model, "integration": integration_data_model},
        },
        "salience_auditability": {
            "passed": bool(
                float(boundary_salience.get("max_diff", 1.0)) < 1e-9
                and all(
                    key in dict(boundary_salience.get("audit_inputs", {}))
                    for key in ("relevance_goal", "relevance_threat", "relevance_self", "relevance_social", "relevance_reward")
                )
                and integration_salience.get("has_signal_breakdown")
                and bool(integration_salience.get("self_evidence"))
                and bool(dict(integration_salience.get("relevance_audit", {})).get("formula"))
            ),
            "blocking": True,
            "evidence": {"boundary": boundary_salience, "integration": integration_salience},
        },
        "encoding_pipeline": {
            "passed": bool(
                sorted(boundary_encoding.get("memory_classes", [])) == ["episodic", "inferred", "procedural", "semantic"]
                and source_defaults_match
                and boundary_encoding.get("high_arousal_store_level") == "long"
                and boundary_encoding.get("high_salience_store_level") == "long"
                and float(boundary_encoding.get("identity_self_relevance", 0.0)) > float(boundary_encoding.get("noise_self_relevance", 1.0))
                and float(boundary_encoding.get("first_person_self_relevance", 1.0)) < 0.2
                and bool(boundary_encoding.get("procedural_steps"))
                and boundary_encoding.get("semantic_lineage_type")
                and boundary_encoding.get("inferred_lineage_type")
                and float(dict(boundary_encoding.get("reward_probe", {})).get("relevance_reward", 1.0)) <= 0.05
                and bool(dict(boundary_encoding.get("reward_probe", {})).get("threat_evidence"))
                and _store_level_rank(integration_encoding.get("identity_store_level")) > _store_level_rank(integration_encoding.get("noise_store_level"))
                and bool(integration_encoding.get("identity_retention_reasons"))
                and _confidence_pairs_are_structured(confidence_pairs)
                and _confidence_drift_cases_show_independence(confidence_drift_cases)
            ),
            "blocking": True,
            "evidence": {"boundary": boundary_encoding, "integration": integration_encoding},
        },
        "dual_decay_correctness": {
            "passed": bool(
                curve_relationships_hold
                and list(forgetting_paths.get("deleted_short_residue", []))
                and list(forgetting_paths.get("dormant_marked", []))
                and list(forgetting_paths.get("abstracted_entries", []))
                and list(forgetting_paths.get("source_confidence_drifted", []))
                and list(forgetting_paths.get("reality_confidence_drifted", []))
                and "deleted_short_residue" in cleanup_report
                and integration_decay.get("processed_entries", 0) >= 2
                and (
                    list(integration_decay.get("abstracted_entries", []))
                    or list(integration_decay.get("source_confidence_drifted", []))
                    or list(integration_decay.get("reality_confidence_drifted", []))
                )
            ),
            "blocking": True,
            "evidence": {"boundary": boundary_decay, "integration": integration_decay},
        },
        "legacy_bridge": {
            "passed": bool(
                boundary_legacy.get("timestamp_preserved")
                and boundary_legacy.get("action_reflected") == "hide_revised"
                and boundary_legacy.get("outcome_reflected") == "mutated_outcome"
                and float(boundary_legacy.get("prediction_error_reflected", 0.0)) == 0.99
                and float(boundary_legacy.get("total_surprise_reflected", 0.0)) == 0.77
                and int(boundary_legacy.get("support_count_reflected", 0)) == 5
                and dict(boundary_legacy.get("unknown_fields_preserved", {})).get("custom_flag") == "keep-me"
                and int(integration_legacy.get("store_entries_after_store", 0)) >= 1
                and int(integration_legacy.get("merge_support_delta", 0)) == 1
                and int(integration_legacy.get("restored_store_entries", 0)) >= 1
                and int(integration_legacy.get("replay_batch_size", 0)) >= 1
                and _regression_summary_passes(regression_summary)
            ),
            "blocking": True,
            "evidence": {
                "boundary": boundary_legacy,
                "integration": integration_legacy,
                "regression_summary": regression_summary,
            },
        },
        "store_level_transitions": {
            "passed": bool(
                _store_level_rank(boundary_transitions.get("identity_store_level")) >= _store_level_rank("mid")
                and boundary_transitions.get("long_store_level") == "long"
                and boundary_transitions.get("noise_store_level") == "short"
                and _transition_passes(dict(boundary_transitions.get("identity_transition", {})), old_level="short", new_level="mid")
                and _transition_passes(dict(boundary_transitions.get("long_transition", {})), old_level="mid", new_level="long")
                and _store_level_rank(integration_transitions.get("identity_store_level")) > _store_level_rank(integration_transitions.get("noise_store_level"))
                and _store_level_rank(integration_transitions.get("identity_linked_store_level")) > _store_level_rank(integration_transitions.get("identity_null_store_level"))
                and float(integration_transitions.get("identity_score_delta", 0.0)) > 0.0
                and integration_transitions.get("identity_null_store_level") == "short"
                and float(integration_transitions.get("neutral_promotion_rate", 1.0)) < 0.05
            ),
            "blocking": True,
            "evidence": {"boundary": boundary_transitions, "integration": integration_transitions},
        },
    }

    unsupported_claims: list[str] = []
    if not _catalog_integrity(payload):
        unsupported_claims.append("probe_catalog_mismatch")
    if not regression_summary:
        unsupported_claims.append("missing_regression_summary")

    gates = dict(preliminary_gates)
    gates["report_honesty"] = {
        "passed": bool(
            _all_gate_evidence_present(preliminary_gates)
            and _catalog_integrity(payload)
            and _regression_summary_passes(regression_summary)
            and bool(failure_injection.get("cases"))
            and float(ablation.get("identity_relevance_drop", 0.0)) > 0.0
            and not unsupported_claims
        ),
        "blocking": True,
        "evidence": {
            "all_gates_have_evidence": _all_gate_evidence_present(preliminary_gates),
            "probe_catalog_complete": _catalog_integrity(payload),
            "regression_summary": regression_summary,
            "failure_injection_cases": list(failure_injection.get("cases", [])),
            "ablation_signal_drop": ablation.get("identity_relevance_drop"),
            "unsupported_claims": unsupported_claims,
        },
    }

    findings: list[dict[str, Any]] = []
    for gate_name, gate in gates.items():
        if gate["passed"]:
            continue
        findings.append(
            {
                "severity": "S1",
                "label": gate_name,
                "detail": f"M4.5 gate {gate_name} did not meet its evidence requirements.",
            }
        )
    status = "PASS" if all(gate["passed"] for gate in gates.values() if gate["blocking"]) else "FAIL"
    acceptance_state = "acceptance_pass" if status == "PASS" else "acceptance_fail"
    recommendation = "ACCEPT" if status == "PASS" else "BLOCK"
    return {
        "status": status,
        "acceptance_state": acceptance_state,
        "gates": gates,
        "failed_gates": [name for name, gate in gates.items() if not gate["passed"]],
        "findings": findings,
        "top_blockers": _top_blockers(findings),
        "headline_metrics": {
            "identity_relevance_gap": ablation.get("identity_relevance_drop"),
            "identity_store_level": integration_encoding.get("identity_store_level"),
            "noise_store_level": integration_encoding.get("noise_store_level"),
            "regression_passed": regression_summary.get("passed"),
        },
        "recommendation": recommendation,
    }


def _residual_risks_for_failed_gates(failed_gates: list[str]) -> list[dict[str, str]]:
    if not failed_gates:
        return []
    return [
        {
            "未完成项": f"{gate_name} gate 尚未通过",
            "当前替代方案": "官方验收产物保留 FAIL，并仅陈述当前真实 gate 结果。",
            "风险残留点": f"M4.5 仍被 {gate_name} 阻塞，不能声明 acceptance_pass。",
            "后续建议": f"修复 {gate_name} 对应实现或证据链后重新生成官方产物并复核。",
        }
        for gate_name in failed_gates
    ]


def _write_summary(report: dict[str, Any], *, summary_path: Path) -> None:
    lines = [
        "# M4.5 Acceptance Summary",
        "",
        f"Status: `{report['status']}`",
        f"Recommendation: `{report['recommendation']}`",
        "",
        "## Gate Status",
        "",
    ]
    for gate_name, gate in report["gates"].items():
        lines.append(f"- `{gate_name}`: `{'PASS' if gate['passed'] else 'FAIL'}`")
    residual_risks = report.get("residual_risks", [])
    if report.get("status") == "FAIL" and isinstance(residual_risks, list) and residual_risks:
        lines.extend(["", "## 未完成项与风险说明", ""])
        for item in residual_risks:
            if not isinstance(item, dict):
                continue
            lines.append(f"- 未完成项：{item.get('未完成项', '')}")
            lines.append(f"  当前替代方案：{item.get('当前替代方案', '')}")
            lines.append(f"  风险残留点：{item.get('风险残留点', '')}")
            lines.append(f"  后续建议：{item.get('后续建议', '')}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_m45_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    regression_summary = _run_regression_summary()
    payload = build_m45_acceptance_payload(regression_summary=regression_summary)
    evaluation = _evaluate_acceptance(payload)
    residual_risks = _residual_risks_for_failed_gates(list(evaluation["failed_gates"]))
    report = {
        "milestone_id": "M4.5",
        "generated_at": _now_iso(),
        "round_started_at": round_started_at or _now_iso(),
        "git_head": _git_head(),
        "status": evaluation["status"],
        "acceptance_state": evaluation["acceptance_state"],
        "seed_set": [45],
        "artifacts": {
            "canonical_trace": str(output_paths["canonical_trace"]),
            "ablation": str(output_paths["ablation"]),
            "failure_injection": str(output_paths["failure_injection"]),
            "summary": str(output_paths["summary"]),
        },
        "tests": {
            "milestone": ["tests/test_m45_memory_core.py", "tests/test_m45_acceptance.py"],
            "regression": list(REGRESSION_TARGETS),
        },
        "gates": evaluation["gates"],
        "failed_gates": evaluation["failed_gates"],
        "findings": evaluation["findings"],
        "top_blockers": evaluation["top_blockers"],
        "headline_metrics": evaluation["headline_metrics"],
        "residual_risks": residual_risks,
        "freshness": {
            "artifact_round_started_at": round_started_at or _now_iso(),
            "generated_in_this_run": True,
        },
        "recommendation": evaluation["recommendation"],
    }
    output_paths["canonical_trace"].write_text(
        json.dumps(
            {
                "probe_catalog": payload["probe_catalog"],
                "boundary_probes": payload["boundary_probes"],
                "integration_probes": payload["integration_probes"],
                "regression_summary": payload["regression_summary"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_paths["ablation"].write_text(
        json.dumps(payload["ablation"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["failure_injection"].write_text(
        json.dumps(payload["failure_injection"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(report, summary_path=output_paths["summary"])
    return {key: str(path) for key, path in output_paths.items()}


publish_m45_acceptance_artifacts = write_m45_acceptance_artifacts
