from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m46_acceptance_data import REGRESSION_TARGETS, build_m46_acceptance_payload
from .m46_reacceptance import (
    FORMAL_CONCLUSION_NOT_ISSUED,
    GATE_CODES,
    GATE_HONESTY,
    GATE_ORDER,
    STATUS_FAIL,
    STATUS_NOT_RUN,
    build_m46_reacceptance_report,
    _build_honesty_record,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

OFFICIAL_M46_ACCEPTANCE_NOTICE = "Official M4.6 runtime acceptance artifact built from real evidence records."
LEGACY_M46_ACCEPTANCE_NOTICE = (
    "Legacy self-attested M4.6 acceptance artifact. Historical only; not the primary evidence chain."
)

M46_CANONICAL_TRACE_PATH = ARTIFACTS_DIR / "m46_canonical_trace.json"
M46_ABLATION_PATH = ARTIFACTS_DIR / "m46_ablation.json"
M46_FAILURE_INJECTION_PATH = ARTIFACTS_DIR / "m46_failure_injection.json"
M46_REPORT_PATH = REPORTS_DIR / "m46_acceptance_report.json"
M46_SUMMARY_PATH = REPORTS_DIR / "m46_acceptance_summary.md"


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
        "canonical_trace": resolved_artifacts_dir / M46_CANONICAL_TRACE_PATH.name,
        "ablation": resolved_artifacts_dir / M46_ABLATION_PATH.name,
        "failure_injection": resolved_artifacts_dir / M46_FAILURE_INJECTION_PATH.name,
        "report": resolved_reports_dir / M46_REPORT_PATH.name,
        "summary": resolved_reports_dir / M46_SUMMARY_PATH.name,
    }


def _records_by_gate(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        gate_name: [record for record in records if record.get("gate") == gate_name]
        for gate_name in GATE_ORDER
    }


def _failure_injection_case(
    *,
    case_id: str,
    tamper: str,
    raw_records: list[dict[str, Any]],
    include_regressions: bool,
) -> dict[str, Any]:
    honesty_record = _build_honesty_record(deepcopy(raw_records), include_regressions=include_regressions)
    observed = dict(honesty_record.get("observed", {}))
    return {
        "case": case_id,
        "tamper": tamper,
        "honesty_status": honesty_record["status"],
        "failed_closed": honesty_record["status"] == STATUS_FAIL,
        "missing_integration_scenarios": list(observed.get("missing_integration_scenarios", [])),
        "mismatched_status_records": list(observed.get("mismatched_status_records", [])),
        "mismatched_source_api_call_id_records": list(observed.get("mismatched_source_api_call_id_records", [])),
        "duplicate_source_api_call_ids": list(observed.get("duplicate_source_api_call_ids", [])),
        "external_check_failures": dict(observed.get("external_check_failures", {})),
    }


def _run_failure_injection_suite(
    *,
    report: dict[str, Any],
    include_regressions: bool,
) -> dict[str, Any]:
    baseline_records = [
        deepcopy(record)
        for record in list(report["evidence_records"])
        if record.get("gate") != GATE_HONESTY
    ]

    cases: list[dict[str, Any]] = []

    fake_regression = deepcopy(baseline_records)
    regression_record = next(record for record in fake_regression if record["scenario_id"] == "legacy_regression_prereq")
    regression_record["status"] = "PASS"
    regression_record["observed"] = {
        "executed": True,
        "files": list(REGRESSION_TARGETS),
        "returncode": 0,
        "passed": True,
        "summary_line": "synthetic pass",
        "stdout_tail": ["synthetic pass"],
    }
    cases.append(
        _failure_injection_case(
            case_id="fake_regression_pass",
            tamper="Replace the regression prerequisite with a synthetic PASS summary.",
            raw_records=fake_regression,
            include_regressions=include_regressions,
        )
    )

    missing_integration = [
        deepcopy(record)
        for record in baseline_records
        if record["scenario_id"] != "consolidation_validation_linkage"
    ]
    cases.append(
        _failure_injection_case(
            case_id="missing_integration_record",
            tamper="Delete the required consolidation-validation integration scenario.",
            raw_records=missing_integration,
            include_regressions=include_regressions,
        )
    )

    tampered_candidates = deepcopy(baseline_records)
    retrieval_record = next(record for record in tampered_candidates if record["scenario_id"] == "retrieval_tag_primary")
    retrieval_record["observed"]["candidate_ids"] = ["bogus-id"]
    cases.append(
        _failure_injection_case(
            case_id="tampered_candidate_ids",
            tamper="Corrupt candidate_ids so they no longer match the recorded retrieval candidates.",
            raw_records=tampered_candidates,
            include_regressions=include_regressions,
        )
    )

    tampered_validated_linkage = deepcopy(baseline_records)
    linkage_record = next(record for record in tampered_validated_linkage if record["scenario_id"] == "consolidation_validation_linkage")
    linkage_record["observed"]["report"]["validated_inference_ids"] = ["bogus-validated-id"]
    cases.append(
        _failure_injection_case(
            case_id="tampered_validated_linkage",
            tamper="Corrupt validated_inference_ids so they no longer match the LONG entries recorded in evidence.",
            raw_records=tampered_validated_linkage,
            include_regressions=include_regressions,
        )
    )

    empty_observed = deepcopy(baseline_records)
    empty_record = next(record for record in empty_observed if record["scenario_id"] == "retrieval_context_rich")
    empty_record["observed"] = {}
    cases.append(
        _failure_injection_case(
            case_id="empty_observed_payload",
            tamper="Erase a scenario observed payload entirely.",
            raw_records=empty_observed,
            include_regressions=include_regressions,
        )
    )

    duplicate_call_ids = deepcopy(baseline_records)
    duplicate_call_ids[1]["source_api_call_id"] = duplicate_call_ids[0]["source_api_call_id"]
    cases.append(
        _failure_injection_case(
            case_id="duplicate_source_api_call_id",
            tamper="Duplicate source_api_call_id across two scenarios.",
            raw_records=duplicate_call_ids,
            include_regressions=include_regressions,
        )
    )

    return {
        "suite": "acceptance_method_negative_controls",
        "baseline_honesty_status": report["gate_summaries"][GATE_HONESTY]["status"],
        "case_count": len(cases),
        "all_cases_failed_closed": all(case["failed_closed"] for case in cases),
        "cases": cases,
    }


def _build_ablation_summary(failure_injection: dict[str, Any]) -> dict[str, Any]:
    cases = list(failure_injection.get("cases", []))
    return {
        "comparison": "full_evidence_vs_tampered_evidence",
        "tampered_case_count": len(cases),
        "tampered_cases_failed_closed": sum(1 for case in cases if case.get("failed_closed") is True),
        "all_tampered_cases_failed_closed": failure_injection.get("all_cases_failed_closed") is True,
        "case_ids": [str(case.get("case")) for case in cases],
    }


def _gate_payload(
    *,
    gate_name: str,
    gate_summary: dict[str, Any],
    gate_records: list[dict[str, Any]],
    failure_injection: dict[str, Any] | None = None,
    ablation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(gate_summary["status"])
    passed = status == "PASS"
    evidence: dict[str, Any] = {
        "summary": gate_summary,
        "records": gate_records,
    }
    if gate_name == GATE_HONESTY:
        evidence["failure_injection"] = failure_injection or {}
        evidence["ablation"] = ablation or {}
        passed = (
            status == "PASS"
            and bool(failure_injection)
            and failure_injection.get("all_cases_failed_closed") is True
            and bool(ablation)
            and ablation.get("all_tampered_cases_failed_closed") is True
        )
    return {
        "status": status,
        "passed": passed,
        "blocking": True,
        "evidence": evidence,
    }


def _official_status_and_conclusion(gates: dict[str, dict[str, Any]]) -> tuple[str, str, str, str]:
    failed_gates = [
        name
        for name, gate in gates.items()
        if gate["status"] == STATUS_FAIL or (name == GATE_HONESTY and gate["passed"] is False)
    ]
    not_run_gates = [name for name, gate in gates.items() if gate["status"] == STATUS_NOT_RUN]
    if failed_gates:
        return "FAIL", "acceptance_fail", "BLOCK", "BLOCK"
    if not_run_gates:
        return "INCOMPLETE", "acceptance_not_issued", "DEFER", FORMAL_CONCLUSION_NOT_ISSUED
    return "PASS", "acceptance_pass", "ACCEPT", "ACCEPT"


def _headline_metrics(report: dict[str, Any], failure_injection: dict[str, Any]) -> dict[str, Any]:
    gate_summaries = dict(report["gate_summaries"])
    return {
        "retrieval_scenarios": gate_summaries["retrieval_multi_cue"]["counts"]["total"],
        "reconsolidation_scenarios": gate_summaries["reconsolidation"]["counts"]["total"],
        "validated_negative_controls": sum(1 for case in failure_injection.get("cases", []) if case.get("failed_closed") is True),
        "regression_status": gate_summaries["legacy_integration"]["status"],
    }


def _findings_for_gates(gates: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for gate_name, gate in gates.items():
        if gate["status"] == STATUS_FAIL or (gate_name == GATE_HONESTY and gate["passed"] is False):
            findings.append(
                {
                    "severity": "S1",
                    "label": gate_name,
                    "detail": f"M4.6 gate {gate_name} did not meet the official runtime evidence requirements.",
                }
            )
        elif gate["status"] == STATUS_NOT_RUN:
            findings.append(
                {
                    "severity": "S2",
                    "label": gate_name,
                    "detail": f"M4.6 gate {gate_name} remains not run, so formal acceptance cannot be issued.",
                }
            )
    return findings


def build_m46_acceptance_report(*, include_regressions: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence_report = build_m46_reacceptance_report(include_regressions=include_regressions)
    failure_injection = _run_failure_injection_suite(
        report=evidence_report,
        include_regressions=include_regressions,
    )
    ablation = _build_ablation_summary(failure_injection)
    records_by_gate = _records_by_gate(list(evidence_report["evidence_records"]))
    gates = {
        gate_name: _gate_payload(
            gate_name=gate_name,
            gate_summary=dict(evidence_report["gate_summaries"][gate_name]),
            gate_records=records_by_gate[gate_name],
            failure_injection=failure_injection if gate_name == GATE_HONESTY else None,
            ablation=ablation if gate_name == GATE_HONESTY else None,
        )
        for gate_name in GATE_ORDER
    }
    status, acceptance_state, recommendation, formal_conclusion = _official_status_and_conclusion(gates)
    failed_gates = [
        name
        for name, gate in gates.items()
        if gate["status"] == STATUS_FAIL or (name == GATE_HONESTY and gate["passed"] is False)
    ]
    not_run_gates = [name for name, gate in gates.items() if gate["status"] == STATUS_NOT_RUN]
    report = {
        "milestone_id": "M4.6",
        "mode": "official_runtime_acceptance",
        "artifact_lineage": "official_runtime_evidence",
        "primary_evidence_chain": True,
        "generated_at": _now_iso(),
        "git_head": _git_head(),
        "status": status,
        "acceptance_state": acceptance_state,
        "recommendation": recommendation,
        "formal_acceptance_conclusion": formal_conclusion,
        "seed_set": [46],
        "gate_summaries": evidence_report["gate_summaries"],
        "evidence_records": evidence_report["evidence_records"],
        "gates": gates,
        "failed_gates": failed_gates,
        "not_run_gates": not_run_gates,
        "findings": _findings_for_gates(gates),
        "headline_metrics": _headline_metrics(evidence_report, failure_injection),
        "regression_policy": evidence_report["regression_policy"],
        "notes": [
            OFFICIAL_M46_ACCEPTANCE_NOTICE,
            "This report derives M4.6 verdicts from real runtime evidence records rather than synthetic acceptance payloads.",
            "Legacy self-attested artifacts remain historical compatibility outputs only.",
        ],
    }
    return report, evidence_report, failure_injection, ablation


def _write_summary(report: dict[str, Any], *, summary_path: Path) -> None:
    lines = [
        "# M4.6 Acceptance Summary",
        "",
        OFFICIAL_M46_ACCEPTANCE_NOTICE,
        "",
        f"Mode: `{report['mode']}`",
        f"Status: `{report['status']}`",
        f"Formal Acceptance Conclusion: `{report['formal_acceptance_conclusion']}`",
        f"Recommendation: `{report['recommendation']}`",
        "",
        "## Gate Status",
        "",
    ]
    for gate_name in GATE_ORDER:
        gate = report["gates"][gate_name]
        lines.append(
            f"- {GATE_CODES[gate_name]} `{gate_name}`: `{gate['status']}` "
            f"(passed={'true' if gate['passed'] else 'false'})"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is the primary M4.6 evidence chain.",
            "- If `legacy_integration` is `NOT_RUN`, formal acceptance remains `NOT_ISSUED`.",
            "- Legacy self-attested outputs are historical only and do not control PASS/BLOCK.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_m46_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    include_regressions: bool = False,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    report, evidence_report, failure_injection, ablation = build_m46_acceptance_report(
        include_regressions=include_regressions
    )
    report["round_started_at"] = round_started_at or _now_iso()
    report["artifacts"] = {
        "canonical_trace": str(output_paths["canonical_trace"]),
        "ablation": str(output_paths["ablation"]),
        "failure_injection": str(output_paths["failure_injection"]),
        "summary": str(output_paths["summary"]),
    }
    report["tests"] = {
        "milestone": ["tests/test_m46_memory_core.py", "tests/test_m46_acceptance.py", "tests/test_m46_reacceptance.py"],
        "regression": list(REGRESSION_TARGETS),
    }
    report["freshness"] = {
        "artifact_round_started_at": report["round_started_at"],
        "generated_in_this_run": True,
    }
    output_paths["canonical_trace"].write_text(
        json.dumps(
            {
                "mode": evidence_report["mode"],
                "formal_acceptance_conclusion": evidence_report["formal_acceptance_conclusion"],
                "gate_summaries": evidence_report["gate_summaries"],
                "evidence_records": evidence_report["evidence_records"],
                "regression_policy": evidence_report["regression_policy"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_paths["ablation"].write_text(json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["failure_injection"].write_text(
        json.dumps(failure_injection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(report, summary_path=output_paths["summary"])
    return {key: str(path) for key, path in output_paths.items()}


def _probe(payload: dict[str, Any], channel: str, probe_id: str) -> dict[str, Any]:
    probes = payload.get(f"{channel}_probes", {})
    probe = probes.get(probe_id, {}) if isinstance(probes, dict) else {}
    observed = probe.get("observed", {}) if isinstance(probe, dict) else {}
    return dict(observed) if isinstance(observed, dict) else {}


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


def _all_gate_evidence_present(gates: dict[str, dict[str, Any]]) -> bool:
    return all(isinstance(gate.get("evidence"), dict) and bool(gate["evidence"]) for gate in gates.values())


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
    combined = "\n".join(item for item in [completed.stdout.strip(), completed.stderr.strip()] if item).strip()
    summary_line = combined.splitlines()[-1] if combined else ""
    return {
        "executed": True,
        "command": command,
        "files": list(REGRESSION_TARGETS),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": duration_seconds,
        "summary_line": summary_line,
        "stdout_tail": combined.splitlines()[-5:],
    }


def _evaluate_legacy_acceptance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    retrieval = _probe(payload, "boundary", "retrieval_boundary")
    competition = _probe(payload, "boundary", "competition_boundary")
    reconstruction = _probe(payload, "boundary", "reconstruction_boundary")
    reconsolidation = _probe(payload, "boundary", "reconsolidation_boundary")
    consolidation = _probe(payload, "boundary", "consolidation_boundary")
    inference = _probe(payload, "boundary", "inference_boundary")
    legacy_bridge = _probe(payload, "boundary", "legacy_bridge_boundary")
    regression_summary = dict(payload.get("regression_summary", {}))
    ablation = dict(payload.get("ablation", {}))
    failure_injection = dict(payload.get("failure_injection", {}))

    preliminary_gates = {
        "retrieval_multi_cue": {
            "passed": bool(
                retrieval.get("scenario_count", 0) >= 5
                and retrieval.get("tag_top_id") == "tag-primary"
                and retrieval.get("context_top_id") == "context-rich"
                and retrieval.get("mood_top_id") == "negative-mood"
                and retrieval.get("low_access_not_top") is True
                and retrieval.get("dormant_present") is False
                and retrieval.get("recall_primary_id") == "tag-primary"
                and retrieval.get("recall_aux_ids")
                and retrieval.get("recall_is_reconstructed") is True
                and retrieval.get("procedural_outline")
            ),
            "blocking": True,
            "evidence": {"boundary": retrieval},
        },
        "candidate_competition": {
            "passed": bool(
                competition.get("dominant_confidence") == "high"
                and competition.get("dominant_interference_risk") is False
                and competition.get("close_confidence") == "low"
                and competition.get("close_interference_risk") is True
                and competition.get("close_competitor_ids")
                and competition.get("close_interpretations")
            ),
            "blocking": True,
            "evidence": {"boundary": competition},
        },
        "reconstruction_mechanism": {
            "passed": bool(
                reconstruction.get("trigger_a") == "abstract_short_content"
                and reconstruction.get("trigger_b") == "semantic_abstractness"
                and reconstruction.get("trigger_c") == "low_reality_after_retrieval"
                and reconstruction.get("locked_preserved") is True
                and reconstruction.get("weak_filled") is True
                and reconstruction.get("procedural_steps_preserved") is True
                and "night_shift" in reconstruction.get("procedural_contexts", [])
                and reconstruction.get("source_type") == "reconstruction"
                and reconstruction.get("version_changed") is True
            ),
            "blocking": True,
            "evidence": {"boundary": reconstruction},
        },
        "reconsolidation": {
            "passed": bool(
                reconsolidation.get("update_types", {}).get("reinforce") == "reinforcement_only"
                and reconsolidation.get("update_types", {}).get("rebind") == "contextual_rebinding"
                and reconsolidation.get("update_types", {}).get("reconstruct") == "structural_reconstruction"
                and reconsolidation.get("update_types", {}).get("conflict") == "conflict_marking"
                and "mood_context" in reconsolidation.get("rebind_fields", [])
                and reconsolidation.get("reconstruct_fields")
                and "factual" in reconsolidation.get("conflict_flags", [])
            ),
            "blocking": True,
            "evidence": {"boundary": reconsolidation},
        },
        "offline_consolidation_pipeline": {
            "passed": bool(
                consolidation.get("upgrade_promoted_ids")
                and consolidation.get("extracted_pattern_ids")
                and consolidation.get("replay_created_ids")
                and "cleanup-short" in consolidation.get("cleanup_deleted_ids", [])
                and consolidation.get("semantic_created") is True
                and consolidation.get("inferred_created") is True
                and isinstance(consolidation.get("report"), dict)
            ),
            "blocking": True,
            "evidence": {"boundary": consolidation},
        },
        "inference_validation_gate": {
            "passed": bool(
                inference.get("validated", {}).get("validation_status") == "validated"
                and inference.get("validated", {}).get("passed") is True
                and inference.get("unvalidated", {}).get("validation_status") in {"unvalidated", "contradicted"}
                and inference.get("unvalidated", {}).get("passed") is False
                and inference.get("unvalidated_aux_blocked") is True
            ),
            "blocking": True,
            "evidence": {"boundary": inference},
        },
        "legacy_integration": {
            "passed": bool(
                legacy_bridge.get("store_cycle_callable") is True
                and legacy_bridge.get("replay_batch_size", 0) >= 1
                and legacy_bridge.get("entries_match_after_bridge") is True
                and _regression_summary_passes(regression_summary)
            ),
            "blocking": True,
            "evidence": {"boundary": legacy_bridge, "regression_summary": regression_summary},
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
            and failure_injection.get("cases")
            and float(ablation.get("top_score_gap", 0.0)) >= 0.0
            and not unsupported_claims
        ),
        "blocking": True,
        "evidence": {
            "all_gates_have_evidence": _all_gate_evidence_present(preliminary_gates),
            "probe_catalog_complete": _catalog_integrity(payload),
            "regression_summary": regression_summary,
            "failure_injection_cases": list(failure_injection.get("cases", [])),
            "ablation": ablation,
            "unsupported_claims": unsupported_claims,
        },
    }

    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    status = "PASS" if not failed_gates else "FAIL"
    return {
        "status": status,
        "acceptance_state": "acceptance_pass" if status == "PASS" else "acceptance_fail",
        "gates": gates,
        "failed_gates": failed_gates,
        "headline_metrics": {
            "retrieval_scenarios": retrieval.get("scenario_count"),
            "competition_low_confidence": competition.get("close_confidence"),
            "regression_passed": regression_summary.get("passed"),
        },
        "recommendation": "ACCEPT" if status == "PASS" else "BLOCK",
        "findings": [
            {
                "severity": "S1",
                "label": gate_name,
                "detail": f"M4.6 gate {gate_name} did not meet its legacy evidence requirements.",
            }
            for gate_name in failed_gates
        ],
    }


def _residual_risks_for_failed_gates(failed_gates: list[str]) -> list[dict[str, str]]:
    return [
        {
            "unfinished_item": f"{gate_name} gate has not passed",
            "current_fallback": "Use the official runtime evidence chain for current audit truth.",
            "residual_risk": f"M4.6 remains blocked by {gate_name}; this legacy artifact must stay historical-only.",
            "next_step": f"Regenerate the official runtime acceptance artifacts after evidence or implementation changes.",
        }
        for gate_name in failed_gates
    ]


def _write_legacy_summary(report: dict[str, Any], *, summary_path: Path) -> None:
    lines = [
        "# Legacy M4.6 Acceptance Summary",
        "",
        LEGACY_M46_ACCEPTANCE_NOTICE,
        "Use the official M4.6 acceptance artifacts for the current primary evidence chain.",
        "",
        f"Status: `{report['status']}`",
        f"Recommendation: `{report['recommendation']}`",
        "",
        "## Gate Status",
        "",
    ]
    for gate_name, gate in report["gates"].items():
        lines.append(f"- `{gate_name}`: `{'PASS' if gate['passed'] else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This legacy summary is a historical self-attested acceptance view.",
            "- It is not the primary M4.6 evidence chain and does not supersede runtime evidence artifacts.",
        ]
    )
    residual_risks = report.get("residual_risks", [])
    if report.get("status") == "FAIL" and residual_risks:
        lines.extend(["", "## Residual Risks", ""])
        for item in residual_risks:
            lines.append(f"- Unfinished item: {item.get('unfinished_item', '')}")
            lines.append(f"  Current fallback: {item.get('current_fallback', '')}")
            lines.append(f"  Residual risk: {item.get('residual_risk', '')}")
            lines.append(f"  Next step: {item.get('next_step', '')}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_m46_legacy_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    regression_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    effective_regression_summary = dict(regression_summary) if regression_summary is not None else _run_regression_summary()
    payload = build_m46_acceptance_payload(regression_summary=effective_regression_summary)
    evaluation = _evaluate_legacy_acceptance_payload(payload)
    residual_risks = _residual_risks_for_failed_gates(list(evaluation["failed_gates"]))
    report = {
        "milestone_id": "M4.6",
        "artifact_lineage": "legacy_self_attested_acceptance",
        "primary_evidence_chain": False,
        "legacy_notice": LEGACY_M46_ACCEPTANCE_NOTICE,
        "generated_at": _now_iso(),
        "round_started_at": round_started_at or _now_iso(),
        "git_head": _git_head(),
        "status": evaluation["status"],
        "acceptance_state": evaluation["acceptance_state"],
        "seed_set": [46],
        "artifacts": {
            "canonical_trace": str(output_paths["canonical_trace"]),
            "ablation": str(output_paths["ablation"]),
            "failure_injection": str(output_paths["failure_injection"]),
            "summary": str(output_paths["summary"]),
        },
        "tests": {
            "milestone": ["tests/test_m46_memory_core.py", "tests/test_m46_acceptance.py"],
            "regression": list(REGRESSION_TARGETS),
        },
        "gates": evaluation["gates"],
        "failed_gates": evaluation["failed_gates"],
        "findings": evaluation["findings"],
        "headline_metrics": evaluation["headline_metrics"],
        "residual_risks": residual_risks,
        "freshness": {
            "artifact_round_started_at": round_started_at or _now_iso(),
            "generated_in_this_run": True,
        },
        "recommendation": evaluation["recommendation"],
        "notes": [
            LEGACY_M46_ACCEPTANCE_NOTICE,
            "This artifact is kept for historical compatibility and should not be treated as the current primary evidence chain.",
        ],
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
    output_paths["ablation"].write_text(json.dumps(payload["ablation"], indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["failure_injection"].write_text(
        json.dumps(payload["failure_injection"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_legacy_summary(report, summary_path=output_paths["summary"])
    return {key: str(path) for key, path in output_paths.items()}


_evaluate_acceptance = _evaluate_legacy_acceptance_payload

# Historical compatibility alias retained for old scripts.
publish_m46_acceptance_artifacts = write_m46_legacy_acceptance_artifacts
