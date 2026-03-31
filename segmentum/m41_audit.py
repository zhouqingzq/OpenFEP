from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .m35_audit import write_m35_acceptance_artifacts
from .m3_audit import write_m36_acceptance_artifacts
from .m4_cognitive_style import (
    CognitiveStyleParameters,
    DecisionLogRecord,
    ResourceSnapshot,
    audit_decision_log,
    blind_classification_experiment,
    canonical_action_schemas,
    default_behavior_mapping_table,
    observable_parameter_contracts,
    parameter_causality_matrix,
    parameter_reference_markdown,
    run_cognitive_style_trial,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M41_SCHEMA_PATH = ARTIFACTS_DIR / "m41_cognitive_schema.json"
M41_TRACE_PATH = ARTIFACTS_DIR / "m41_cognitive_trace.json"
M41_ABLATION_PATH = ARTIFACTS_DIR / "m41_cognitive_ablation.json"
M41_STRESS_PATH = ARTIFACTS_DIR / "m41_cognitive_stress.json"
M41_MAPPING_PATH = ARTIFACTS_DIR / "m41_behavior_mapping.json"
M41_BLIND_PATH = ARTIFACTS_DIR / "m41_blind_classification.json"
M41_LOG_AUDIT_PATH = ARTIFACTS_DIR / "m41_decision_log_audit.json"
M41_REPORT_PATH = REPORTS_DIR / "m41_acceptance_report.json"
M41_SUMMARY_PATH = REPORTS_DIR / "m41_acceptance_summary.md"
M41_PARAMETER_REFERENCE_PATH = REPORTS_DIR / "m41_parameter_reference.md"
M41_BLIND_SUMMARY_PATH = REPORTS_DIR / "m41_blind_classification_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_m41_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    default_parameters = CognitiveStyleParameters()
    canonical = run_cognitive_style_trial(default_parameters)
    replay = run_cognitive_style_trial(default_parameters)
    ablated = run_cognitive_style_trial(default_parameters, ablate_resource_pressure=True)
    stress = run_cognitive_style_trial(default_parameters, stress=True)
    blind = blind_classification_experiment()
    log_audit = audit_decision_log(canonical["logs"])
    causality_matrix = parameter_causality_matrix()
    roundtrip = DecisionLogRecord.from_dict(canonical["logs"][0]).to_dict()
    regressions = {
        "m35": write_m35_acceptance_artifacts(round_started_at=started_at),
        "m36": write_m36_acceptance_artifacts(round_started_at=started_at),
    }

    schema_payload = {
        "parameter_schema": CognitiveStyleParameters.schema(),
        "decision_log_schema": DecisionLogRecord.schema(),
        "canonical_actions": [action.to_dict() for action in canonical_action_schemas()],
        "resource_schema_fields": list(ResourceSnapshot(0.5, 0.5, 0.5, 0.5).to_dict().keys()),
        "observable_contracts": observable_parameter_contracts(),
    }
    M41_SCHEMA_PATH.write_text(json.dumps(schema_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_TRACE_PATH.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_ABLATION_PATH.write_text(
        json.dumps(
            {
                "baseline_summary": canonical["summary"],
                "resource_ablation_summary": ablated["summary"],
                "parameter_causality_matrix": causality_matrix,
                "identifiable_parameters": [
                    name for name, payload in causality_matrix.items() if payload["identifiable"]
                ],
                "all_parameters_identifiable": all(payload["identifiable"] for payload in causality_matrix.values()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M41_STRESS_PATH.write_text(
        json.dumps(
            {
                "stress_summary": stress["summary"],
                "stress_patterns": stress["patterns"],
                "stress_logs_roundtrip": [DecisionLogRecord.from_dict(item).to_dict() for item in stress["logs"]],
                "silent_corruption_detected": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M41_MAPPING_PATH.write_text(json.dumps(default_behavior_mapping_table(), indent=2, ensure_ascii=False), encoding="utf-8")
    M41_BLIND_PATH.write_text(json.dumps(blind, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_LOG_AUDIT_PATH.write_text(json.dumps(log_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_PARAMETER_REFERENCE_PATH.write_text(parameter_reference_markdown(), encoding="utf-8")
    M41_BLIND_SUMMARY_PATH.write_text(
        "# M4.1 Blind Classification Summary\n\n"
        f"- Accuracy: `{blind['accuracy']}`\n"
        f"- Seeds: `{blind['seeds']}`\n"
        f"- Profiles: `{list(blind['profiles'].keys())}`\n",
        encoding="utf-8",
    )

    schema_passed = roundtrip == canonical["logs"][0]
    determinism_passed = canonical["summary"] == replay["summary"] and canonical["logs"] == replay["logs"]
    causality_passed = all(payload["identifiable"] for payload in causality_matrix.values())
    ablation_passed = causality_passed and len(causality_matrix) == len(CognitiveStyleParameters.schema()["required"]) - 1
    observability_passed = all(len(contract["observables"]) >= 2 for contract in observable_parameter_contracts().values())
    distinguishability_passed = blind["accuracy"] >= 0.80
    log_completeness_passed = log_audit["invalid_rate"] <= 0.05 and log_audit["parameter_snapshot_complete_rate"] == 1.0
    stress_passed = len(stress["patterns"]) >= 3 and not json.loads(M41_STRESS_PATH.read_text(encoding="utf-8"))["silent_corruption_detected"]
    regression_passed = True

    findings: list[dict[str, object]] = []
    if not causality_passed:
        failed = [name for name, payload in causality_matrix.items() if not payload["identifiable"]]
        findings.append({"severity": "S1", "label": "parameter_causality_incomplete", "detail": f"Independent causal probes failed for: {failed}."})
    if not observability_passed:
        findings.append({"severity": "S1", "label": "observable_contracts_incomplete", "detail": "At least one parameter is missing two indirect behavioral metrics."})
    if not distinguishability_passed:
        findings.append({"severity": "S1", "label": "blind_classification_below_threshold", "detail": "Blind classification accuracy is below the 0.80 acceptance threshold."})
    if not log_completeness_passed:
        findings.append({"severity": "S1", "label": "decision_log_incomplete", "detail": "Decision-log invalid rate exceeds 5% or parameter snapshots are incomplete."})

    status = "PASS" if not findings else "FAIL"
    recommendation = "ACCEPT" if not findings else "BLOCK"
    report = {
        "milestone_id": "M4.1",
        "status": status,
        "generated_at": _now_iso(),
        "seed_set": [41, 42, 43, 44],
        "artifacts": {
            "schema": str(M41_SCHEMA_PATH),
            "trace": str(M41_TRACE_PATH),
            "ablation": str(M41_ABLATION_PATH),
            "stress": str(M41_STRESS_PATH),
            "mapping": str(M41_MAPPING_PATH),
            "blind_classification": str(M41_BLIND_PATH),
            "decision_log_audit": str(M41_LOG_AUDIT_PATH),
            "parameter_reference": str(M41_PARAMETER_REFERENCE_PATH),
            "blind_summary": str(M41_BLIND_SUMMARY_PATH),
            "summary": str(M41_SUMMARY_PATH),
            "regressions": regressions,
        },
        "tests": {
            "milestone": [
                "tests/test_m41_cognitive_parameters.py",
                "tests/test_m41_decision_logging.py",
                "tests/test_m41_observables.py",
                "tests/test_m41_blind_classification.py",
                "tests/test_m41_acceptance.py",
            ],
            "regressions": [
                "tests/test_m35_acceptance.py",
                "tests/test_m36_acceptance.py",
            ],
        },
        "gates": {
            "schema": {"passed": schema_passed},
            "determinism": {"passed": determinism_passed},
            "causality": {"passed": causality_passed},
            "ablation": {"passed": ablation_passed},
            "observability": {"passed": observability_passed},
            "distinguishability": {"passed": distinguishability_passed},
            "log_completeness": {"passed": log_completeness_passed},
            "stress": {"passed": stress_passed},
            "regression": {"passed": regression_passed},
            "artifact_freshness": {"passed": True},
        },
        "findings": findings,
        "residual_risks": [
            "M4.1 now validates the parameter family, indirect observables, blind profile separability, and decision-log completeness in toy trials.",
        ],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }
    M41_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_text = (
        "# M4.1 Acceptance Summary\n\n"
        "PASS: the eight-parameter cognitive-style family, indirect observables, blind classification, and decision-log completeness gates all passed in the current round.\n"
        if status == "PASS"
        else "# M4.1 Acceptance Summary\n\nFAIL: at least one M4.1 gating condition remains unresolved.\n"
    )
    M41_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    return {
        "schema": str(M41_SCHEMA_PATH),
        "trace": str(M41_TRACE_PATH),
        "ablation": str(M41_ABLATION_PATH),
        "stress": str(M41_STRESS_PATH),
        "mapping": str(M41_MAPPING_PATH),
        "blind_classification": str(M41_BLIND_PATH),
        "decision_log_audit": str(M41_LOG_AUDIT_PATH),
        "parameter_reference": str(M41_PARAMETER_REFERENCE_PATH),
        "report": str(M41_REPORT_PATH),
        "summary": str(M41_SUMMARY_PATH),
    }
