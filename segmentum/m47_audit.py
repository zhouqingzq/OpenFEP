from __future__ import annotations

from copy import deepcopy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m47_reacceptance import (
    FORMAL_CONCLUSION_NOT_ISSUED,
    GATE_CODES,
    GATE_HONESTY,
    GATE_ORDER,
    GATE_REGRESSION,
    REGRESSION_TARGETS,
    STATUS_FAIL,
    STATUS_NOT_RUN,
    _all_gate_records,
    _build_honesty_record,
    _gate_summary,
    blocking_failed_gate_names,
    blocking_not_run_gate_names,
    build_m47_evidence_records,
    iter_blocking_gate_summaries,
    rollup_evidence_rebuild_status,
)
from .m47_runtime import M47_RUNTIME_SNAPSHOT_PATH, build_m47_runtime_snapshot


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

OFFICIAL_M47_ACCEPTANCE_NOTICE = "Official M4.7 runtime acceptance artifact built from shared workload evidence."
LEGACY_M47_ACCEPTANCE_NOTICE = "Legacy-style M4.7 acceptance payload. Historical compatibility only."

M47_CANONICAL_TRACE_PATH = ARTIFACTS_DIR / "m47_canonical_trace.json"
M47_ABLATION_PATH = ARTIFACTS_DIR / "m47_ablation.json"
M47_FAILURE_INJECTION_PATH = ARTIFACTS_DIR / "m47_failure_injection.json"
M47_REPORT_PATH = REPORTS_DIR / "m47_acceptance_report.json"
M47_SUMMARY_PATH = REPORTS_DIR / "m47_acceptance_summary.md"


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
        "runtime_snapshot": resolved_artifacts_dir / M47_RUNTIME_SNAPSHOT_PATH.name,
        "canonical_trace": resolved_artifacts_dir / M47_CANONICAL_TRACE_PATH.name,
        "ablation": resolved_artifacts_dir / M47_ABLATION_PATH.name,
        "failure_injection": resolved_artifacts_dir / M47_FAILURE_INJECTION_PATH.name,
        "report": resolved_reports_dir / M47_REPORT_PATH.name,
        "summary": resolved_reports_dir / M47_SUMMARY_PATH.name,
    }


def _records_by_gate(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {gate_name: [record for record in records if record.get("gate") == gate_name] for gate_name in GATE_ORDER}


def _failure_injection_case(
    *,
    case_id: str,
    tamper: str,
    raw_records: list[dict[str, Any]],
    include_regressions: bool,
) -> dict[str, Any]:
    honesty_record = _build_honesty_record(
        deepcopy(raw_records),
        include_regressions=include_regressions,
        diagnostic_only=False,
    )
    observed = dict(honesty_record.get("observed", {}))
    return {
        "case": case_id,
        "tamper": tamper,
        "honesty_status": honesty_record["status"],
        "failed_closed": honesty_record["status"] == STATUS_FAIL,
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

    tampered_effect_size = deepcopy(baseline_records)
    scenario_a = next(record for record in tampered_effect_size if record["gate"] == "behavioral_scenario_A_threat_learning")
    scenario_a["observed"]["cohens_d"] = 9.99
    cases.append(
        _failure_injection_case(
            case_id="tampered_effect_size",
            tamper="Overwrite G4 Cohen's d while leaving raw salience samples unchanged.",
            raw_records=tampered_effect_size,
            include_regressions=include_regressions,
        )
    )

    missing_evidence = [deepcopy(record) for record in baseline_records if record["gate"] != "identity_continuity_retention"]
    cases.append(
        _failure_injection_case(
            case_id="missing_gate_record",
            tamper="Delete the identity continuity gate record entirely.",
            raw_records=missing_evidence,
            include_regressions=include_regressions,
        )
    )

    fake_random_noise = deepcopy(baseline_records)
    scenario_gy = next(record for record in fake_random_noise if record["gate"] == "behavioral_scenario_E_natural_misattribution")
    scenario_gy["observed"]["random_noise_injected"] = True
    scenario_gy["observed"]["reconstruction_trace"]["borrowed_source_ids"] = []
    cases.append(
        _failure_injection_case(
            case_id="fake_random_noise_misattribution",
            tamper="Claim the Gy error is noise-driven and erase its borrowed-source trace.",
            raw_records=fake_random_noise,
            include_regressions=include_regressions,
        )
    )

    fake_regression = deepcopy(baseline_records)
    regression_record = next(record for record in fake_regression if record["scenario_id"] == "m41_to_m46_regression_prereq")
    regression_record["status"] = "PASS"
    regression_record["observed"] = {
        "executed": True,
        "expected_targets": list(REGRESSION_TARGETS),
        "reason": "synthetic pass",
        "passed": True,
    }
    cases.append(
        _failure_injection_case(
            case_id="fake_regression_pass",
            tamper="Replace the regression prerequisite with a synthetic PASS summary.",
            raw_records=fake_regression,
            include_regressions=include_regressions,
        )
    )

    duplicate_ids = deepcopy(baseline_records)
    duplicate_ids[1]["source_api_call_id"] = duplicate_ids[0]["source_api_call_id"]
    cases.append(
        _failure_injection_case(
            case_id="duplicate_source_api_call_id",
            tamper="Duplicate source_api_call_id across two records.",
            raw_records=duplicate_ids,
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
    return {
        "code": GATE_CODES[gate_name],
        "status": status,
        "passed": status == "PASS",
        "scenario_ids": list(gate_summary["scenario_ids"]),
        "evidence": {
            "records": gate_records,
            "failure_injection": failure_injection,
            "ablation": ablation,
        },
    }


def _official_status_and_conclusion(
    gates: dict[str, dict[str, Any]],
    gate_summaries: dict[str, dict[str, Any]],
    snapshot: dict[str, Any],
) -> tuple[str, str, str, str]:
    if blocking_failed_gate_names(gates, gate_summaries, snapshot):
        return "FAIL", "REJECT", "REJECT", "REJECT"
    if blocking_not_run_gate_names(gates, gate_summaries, snapshot):
        return "INCOMPLETE", "DEFER", "DEFER", FORMAL_CONCLUSION_NOT_ISSUED
    return "PASS", "ACCEPT", "ACCEPT", "ACCEPT"


def _headline_metrics(evidence_report: dict[str, Any], failure_injection: dict[str, Any]) -> dict[str, Any]:
    gate_summaries = evidence_report["gate_summaries"]
    return {
        "state_vector_scenarios": gate_summaries["state_vector_dynamics"]["counts"]["total"],
        "long_horizon_cycles": sum(item["observed"]["cycle_count"] for item in evidence_report["evidence_records"] if item["gate"] == "behavioral_scenario_C_consolidation"),
        "validated_negative_controls": sum(1 for case in failure_injection.get("cases", []) if case.get("failed_closed") is True),
        "regression_status": gate_summaries[GATE_REGRESSION]["status"],
    }


def _findings_for_gates(
    gates: dict[str, dict[str, Any]],
    gate_summaries: dict[str, dict[str, Any]],
    snapshot: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for gate_name, _ in iter_blocking_gate_summaries(gate_summaries, snapshot):
        gate = gates[gate_name]
        if gate["status"] == STATUS_FAIL or (gate_name == GATE_HONESTY and gate["passed"] is False):
            findings.append(
                {
                    "severity": "S1",
                    "label": gate_name,
                    "detail": f"M4.7 gate {gate_name} did not meet the official runtime evidence requirements.",
                }
            )
        elif gate["status"] == STATUS_NOT_RUN:
            findings.append(
                {
                    "severity": "S2",
                    "label": gate_name,
                    "detail": f"M4.7 gate {gate_name} remains not run, so formal acceptance cannot be issued.",
                }
            )
    return findings


def build_m47_acceptance_report(
    *,
    include_regressions: bool = False,
    runtime_snapshot: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot = deepcopy(runtime_snapshot) if runtime_snapshot is not None else build_m47_runtime_snapshot()
    # M4.8 demotion: runtime snapshot is diagnostic-only, not acceptance evidence.
    # Behavioral claims require M4.8 ablation contrast proof.
    snapshot["diagnostic_only"] = True
    snapshot["demotion_reason"] = "M4.7 behavioral claims depend on M4.8 ablation evidence; this snapshot satisfies only structural self-consistency (layer a)."
    evidence_records = build_m47_evidence_records(
        include_regressions=include_regressions,
        runtime_snapshot=snapshot,
    )
    gate_summaries = {
        gate: _gate_summary(gate, _all_gate_records(evidence_records, gate))
        for gate in GATE_ORDER
        if gate != GATE_HONESTY
    }
    honesty_record = _build_honesty_record(
        evidence_records,
        include_regressions=include_regressions,
        diagnostic_only=bool(snapshot.get("diagnostic_only")),
    )
    evidence_records.append(honesty_record)
    gate_summaries[GATE_HONESTY] = _gate_summary(GATE_HONESTY, [honesty_record])
    evidence_rebuild_status = rollup_evidence_rebuild_status(gate_summaries, snapshot)
    evidence_report = {
        "milestone_id": "M4.7",
        "mode": "independent_evidence_rebuild",
        "generated_at": _now_iso(),
        "formal_acceptance_conclusion": FORMAL_CONCLUSION_NOT_ISSUED,
        "evidence_rebuild_status": evidence_rebuild_status,
        "runtime_snapshot": snapshot,
        "regression_policy": {
            "include_regressions": include_regressions,
            "regression_targets": list(REGRESSION_TARGETS),
            "live_only": True,
        },
        "gate_summaries": gate_summaries,
        "evidence_records": evidence_records,
        "anti_degeneration_addendum": [
            {
                "risk": "shared_runtime_snapshot",
                "current_mitigation": "Official acceptance and reacceptance both read the same shared runtime snapshot.",
                "residual_risk": "The shared snapshot still depends on a curated corpus.",
            }
        ],
        "notes": [
            "Official acceptance evidence is built directly from the shared runtime snapshot.",
            "Reacceptance and official acceptance grade the same raw evidence chain.",
        ],
    }
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
    status, acceptance_state, recommendation, formal_conclusion = _official_status_and_conclusion(
        gates, evidence_report["gate_summaries"], snapshot
    )
    failed_gates = blocking_failed_gate_names(gates, evidence_report["gate_summaries"], snapshot)
    not_run_gates = blocking_not_run_gate_names(gates, evidence_report["gate_summaries"], snapshot)
    report = {
        "milestone_id": "M4.7",
        "mode": "official_runtime_acceptance",
        "artifact_lineage": "official_runtime_evidence",
        "primary_evidence_chain": True,
        "generated_at": _now_iso(),
        "git_head": _git_head(),
        "status": status,
        "acceptance_state": acceptance_state,
        "recommendation": recommendation,
        "formal_acceptance_conclusion": formal_conclusion,
        "seed_set": [47],
        "gate_summaries": evidence_report["gate_summaries"],
        "evidence_records": evidence_report["evidence_records"],
        "gates": gates,
        "failed_gates": failed_gates,
        "not_run_gates": not_run_gates,
        "findings": _findings_for_gates(gates, evidence_report["gate_summaries"], snapshot),
        "headline_metrics": _headline_metrics(evidence_report, failure_injection),
        "regression_policy": evidence_report["regression_policy"],
        "anti_degeneration_addendum": evidence_report["anti_degeneration_addendum"],
        "notes": [
            OFFICIAL_M47_ACCEPTANCE_NOTICE,
            "This report derives M4.7 verdicts from shared runtime evidence rather than helper-generated scenario payloads.",
            "Regression remains a blocking prerequisite for formal issuance.",
        ],
    }
    return report, evidence_report, failure_injection, ablation


def _write_summary(report: dict[str, Any], *, summary_path: Path) -> None:
    lines = [
        "# M4.7 Official Acceptance Summary",
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
            "- This is the primary M4.7 evidence chain.",
            "- If `regression` is `NOT_RUN`, formal acceptance remains `NOT_ISSUED`.",
            "- Negative controls must all fail closed for the honesty gate to count as passed.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_m47_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    include_regressions: bool = False,
    runtime_snapshot: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = deepcopy(runtime_snapshot) if runtime_snapshot is not None else build_m47_runtime_snapshot()
    report, evidence_report, failure_injection, ablation = build_m47_acceptance_report(
        include_regressions=include_regressions,
        runtime_snapshot=snapshot,
    )
    report["round_started_at"] = round_started_at or _now_iso()
    report["artifacts"] = {
        "runtime_snapshot": str(output_paths["runtime_snapshot"]),
        "canonical_trace": str(output_paths["canonical_trace"]),
        "ablation": str(output_paths["ablation"]),
        "failure_injection": str(output_paths["failure_injection"]),
        "summary": str(output_paths["summary"]),
    }
    report["tests"] = {
        "milestone": [
            "tests/test_m47_memory_core.py",
            "tests/test_m47_reacceptance.py",
            "tests/test_m47_acceptance.py",
        ],
        "regression": list(REGRESSION_TARGETS),
    }
    report["freshness"] = {
        "artifact_round_started_at": report["round_started_at"],
        "generated_in_this_run": True,
    }
    output_paths["runtime_snapshot"].write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["canonical_trace"].write_text(
        json.dumps(
            {
                "mode": evidence_report["mode"],
                "formal_acceptance_conclusion": evidence_report["formal_acceptance_conclusion"],
                "runtime_snapshot": evidence_report["runtime_snapshot"],
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
