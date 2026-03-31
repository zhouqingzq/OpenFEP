from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m35_audit import write_m35_acceptance_artifacts
from .m3_audit import write_m36_acceptance_artifacts
from .m4_cognitive_style import (
    CognitiveStyleParameters,
    DecisionLogRecord,
    ResourceSnapshot,
    audit_decision_log,
    blind_classification_experiment,
    canonical_action_schemas,
    compute_trial_variation,
    default_behavior_mapping_table,
    metrics_have_executable_registry,
    observable_parameter_contracts,
    observable_metrics_registry,
    parameter_intervention_sensitivity_matrix,
    parameter_reference_markdown,
    run_cognitive_style_trial,
    validate_acceptance_report,
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


def _all_metrics_sufficient(metric_payload: dict[str, dict[str, object]]) -> bool:
    return all(not bool(payload.get("insufficient_data")) for payload in metric_payload.values())


def _metric_values(metric_payload: dict[str, dict[str, object]]) -> dict[str, float]:
    return {
        name: float(payload["value"])
        for name, payload in metric_payload.items()
        if payload.get("value") is not None and not payload.get("insufficient_data")
    }


def _load_json_dict(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _text_contains(path_str: str, required_snippets: list[str]) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {"present": False, "matched": [], "size_bytes": 0}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"present": False, "matched": [], "size_bytes": 0}
    matched = [snippet for snippet in required_snippets if snippet in text]
    return {"present": True, "matched": matched, "size_bytes": len(text.encode("utf-8"))}


def _evaluate_regression_dependency(name: str, artifacts: dict[str, str]) -> dict[str, Any]:
    report = _load_json_dict(artifacts.get("report", ""))
    summary = _text_contains(artifacts.get("summary", ""), [name.upper(), "Summary", "PASS", "FAIL"])
    trace = _load_json_dict(artifacts.get("trace", ""))
    dependency = {
        "artifact_keys": sorted(artifacts.keys()),
        "report_present": report is not None,
        "summary_present": summary["present"],
        "trace_present": trace is not None,
        "summary_markers": summary["matched"],
        "report_status": report.get("status") if report else None,
        "report_milestone_id": report.get("milestone_id") if report else None,
        "report_has_gates": bool(report and isinstance(report.get("gates"), dict) and report["gates"]),
        "trace_keys": sorted(trace.keys())[:10] if trace else [],
    }
    dependency["passed"] = bool(
        dependency["report_present"]
        and dependency["summary_present"]
        and dependency["trace_present"]
        and dependency["report_status"] == "PASS"
        and dependency["report_has_gates"]
        and dependency["trace_keys"]
    )
    return dependency


def _evaluate_self_artifact_evidence(
    *,
    schema_path: Path,
    trace_path: Path,
    ablation_path: Path,
    blind_path: Path,
    blind_summary_path: Path,
    parameter_reference_path: Path,
) -> dict[str, Any]:
    schema = _load_json_dict(str(schema_path))
    trace = _load_json_dict(str(trace_path))
    ablation = _load_json_dict(str(ablation_path))
    blind = _load_json_dict(str(blind_path))
    blind_summary = _text_contains(
        str(blind_summary_path),
        ["M4.1", "toy_internal_distinguishability", "Train/eval split", "Generator family"],
    )
    reference = _text_contains(
        str(parameter_reference_path),
        ["uncertainty_sensitivity", "resource_pressure_sensitivity", "virtual_prediction_error_gain"],
    )
    evidence = {
        "schema": {
            "present": schema is not None,
            "has_parameter_schema": bool(schema and "parameter_schema" in schema),
            "has_observable_registry": bool(schema and schema.get("observable_registry")),
        },
        "trace": {
            "present": trace is not None,
            "log_count": len(trace.get("logs", [])) if trace else 0,
            "pattern_count": len(trace.get("patterns", [])) if trace else 0,
            "summary_keys": sorted(trace.get("summary", {}).keys()) if trace else [],
        },
        "ablation": {
            "present": ablation is not None,
            "has_intervention_probe": bool(ablation and ablation.get("parameter_intervention_sensitivity")),
            "has_trial_variation": bool(ablation and ablation.get("trial_variation_vs_ablation")),
        },
        "blind": {
            "present": blind is not None,
            "analysis_type": blind.get("analysis_type") if blind else None,
            "sample_count": blind.get("sample_count") if blind else 0,
            "has_train_eval_split": bool(blind and blind.get("train_eval_split")),
        },
        "blind_summary": blind_summary,
        "parameter_reference": reference,
    }
    evidence["passed"] = bool(
        evidence["schema"]["has_parameter_schema"]
        and evidence["schema"]["has_observable_registry"]
        and evidence["trace"]["log_count"] > 0
        and evidence["trace"]["pattern_count"] > 0
        and evidence["ablation"]["has_intervention_probe"]
        and evidence["ablation"]["has_trial_variation"]
        and evidence["blind"]["analysis_type"] == "toy_internal_distinguishability"
        and evidence["blind"]["sample_count"] > 0
        and evidence["blind"]["has_train_eval_split"]
        and len(evidence["blind_summary"]["matched"]) >= 3
        and len(evidence["parameter_reference"]["matched"]) == 3
    )
    return evidence


def write_m41_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()

    default_parameters = CognitiveStyleParameters()
    canonical = run_cognitive_style_trial(default_parameters, seed=41)
    replay = run_cognitive_style_trial(default_parameters, seed=41)
    variant = run_cognitive_style_trial(default_parameters, seed=42)
    ablated = run_cognitive_style_trial(default_parameters, seed=41, ablate_resource_pressure=True)
    stress = run_cognitive_style_trial(default_parameters, seed=41, stress=True)
    blind = blind_classification_experiment()
    log_audit = audit_decision_log(canonical["logs"])
    intervention_matrix = parameter_intervention_sensitivity_matrix()
    roundtrip = DecisionLogRecord.from_dict(canonical["logs"][0]).to_dict()
    variation = compute_trial_variation(canonical, variant)
    ablation_variation = compute_trial_variation(canonical, ablated)
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
        "observable_registry": observable_metrics_registry(),
    }
    M41_SCHEMA_PATH.write_text(json.dumps(schema_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_TRACE_PATH.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    M41_ABLATION_PATH.write_text(
        json.dumps(
            {
                "baseline_summary": canonical["summary"],
                "resource_ablation_summary": ablated["summary"],
                "trial_variation_vs_ablation": ablation_variation,
                "parameter_intervention_sensitivity": intervention_matrix,
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
                "stress_metric_values": stress["observable_metric_values"],
                "stress_logs_roundtrip": [DecisionLogRecord.from_dict(item).to_dict() for item in stress["logs"]],
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
        f"- Analysis type: `{blind['analysis_type']}`\n"
        f"- Accuracy: `{blind['accuracy']}`\n"
        f"- Baseline accuracy: `{blind['baseline_accuracy']}`\n"
        f"- Train/eval split: `{blind['train_eval_split']}`\n"
        f"- Generator family: `{blind['generator_family']}`\n"
        f"- External validation: `{blind['external_validation']}`\n"
        f"- Feature set: `{blind['feature_set']}`\n",
        encoding="utf-8",
    )

    contracts = observable_parameter_contracts()
    all_metrics_have_evaluator = metrics_have_executable_registry()
    canonical_metrics_sufficient = _all_metrics_sufficient(canonical["observable_metrics"])
    min_recall = min(payload["recall"] for payload in blind["per_class"].values())

    schema_integrity_passed = roundtrip == canonical["logs"][0] and set(CognitiveStyleParameters.schema()["required"]) == set(
        canonical["logs"][0]["parameter_snapshot"].keys()
    )
    trial_variation_passed = variation["varies"]
    observability_passed = (
        all_metrics_have_evaluator
        and all(len(contract["observables"]) >= 2 for contract in contracts.values())
        and canonical_metrics_sufficient
    )
    intervention_sensitivity_passed = all(payload["identifiable"] for payload in intervention_matrix.values())
    blind_distinguishability_passed = (
        blind["accuracy"] >= 0.80
        and blind["accuracy"] > blind["baseline_accuracy"] + 0.25
        and min_recall >= 0.75
    )
    log_completeness_passed = log_audit["invalid_rate"] <= 0.05 and log_audit["parameter_snapshot_complete_rate"] == 1.0
    stress_behavior_passed = (
        "resource_conservation" in {pattern["label"] for pattern in stress["patterns"]}
        and stress["observable_metric_values"].get("high_pressure_low_cost_ratio", 0.0) >= 0.55
    )
    self_artifact_evidence = _evaluate_self_artifact_evidence(
        schema_path=M41_SCHEMA_PATH,
        trace_path=M41_TRACE_PATH,
        ablation_path=M41_ABLATION_PATH,
        blind_path=M41_BLIND_PATH,
        blind_summary_path=M41_BLIND_SUMMARY_PATH,
        parameter_reference_path=M41_PARAMETER_REFERENCE_PATH,
    )
    regression_dependencies = {
        "m35": _evaluate_regression_dependency("m35", regressions["m35"]),
        "m36": _evaluate_regression_dependency("m36", regressions["m36"]),
    }
    regression_passed = bool(
        self_artifact_evidence["passed"]
        and regression_dependencies
        and all(payload["passed"] for payload in regression_dependencies.values())
    )

    blocker_mapping = {
        "blind_classification_is_threshold_leakage": "blind_distinguishability",
        "intervention_probe_is_not_causal_inference": "intervention_sensitivity",
        "shared_sequence_hides_seed_variation": "trial_variation",
        "observable_formulas_must_be_executable": "observability",
        "acceptance_must_fail_without_evidence": "regression",
        "tests_must_check_negative_cases": "regression",
    }

    gates = {
        "schema_integrity": {
            "passed": schema_integrity_passed,
            "blocking": True,
            "evidence": {
                "roundtrip_equal": roundtrip == canonical["logs"][0],
                "parameter_snapshot_keys": sorted(canonical["logs"][0]["parameter_snapshot"].keys()),
            },
        },
        "trial_variation": {
            "passed": trial_variation_passed,
            "blocking": True,
            "evidence": variation,
        },
        "observability": {
            "passed": observability_passed,
            "blocking": True,
            "evidence": {
                "registry_is_executable": all_metrics_have_evaluator,
                "all_parameters_have_two_metrics": all(len(contract["observables"]) >= 2 for contract in contracts.values()),
                "canonical_metrics_sufficient": canonical_metrics_sufficient,
                "metric_values": _metric_values(canonical["observable_metrics"]),
            },
        },
        "intervention_sensitivity": {
            "passed": intervention_sensitivity_passed,
            "blocking": True,
            "evidence": {
                "analysis_type": "intervention_sensitivity",
                "all_identifiable": intervention_sensitivity_passed,
                "probes": intervention_matrix,
            },
        },
        "blind_distinguishability": {
            "passed": blind_distinguishability_passed,
            "blocking": True,
            "evidence": {
                "analysis_type": blind["analysis_type"],
                "benchmark_scope": blind["benchmark_scope"],
                "accuracy": blind["accuracy"],
                "baseline_accuracy": blind["baseline_accuracy"],
                "per_class": blind["per_class"],
                "feature_set": blind["feature_set"],
                "sample_count": blind["sample_count"],
                "train_eval_split": blind["train_eval_split"],
                "generator_family": blind["generator_family"],
                "external_validation": blind["external_validation"],
                "validation_limits": blind["validation_limits"],
            },
        },
        "log_completeness": {
            "passed": log_completeness_passed,
            "blocking": True,
            "evidence": log_audit,
        },
        "stress_behavior": {
            "passed": stress_behavior_passed,
            "blocking": True,
            "evidence": {
                "patterns": stress["patterns"],
                "metric_values": stress["observable_metric_values"],
            },
        },
        "regression": {
            "passed": regression_passed,
            "blocking": True,
            "evidence": {
                "self_artifacts": self_artifact_evidence,
                "dependencies": regression_dependencies,
            },
        },
    }

    findings: list[dict[str, object]] = []
    for gate_name, payload in gates.items():
        if not payload["passed"]:
            findings.append(
                {
                    "severity": "S1",
                    "label": f"{gate_name}_failed",
                    "detail": f"M4.1 gate `{gate_name}` did not meet its evidence threshold.",
                }
            )

    status = "PASS" if all(bool(payload["passed"]) for payload in gates.values()) else "FAIL"
    recommendation = "ACCEPT" if status == "PASS" else "BLOCK"
    report = {
        "milestone_id": "M4.1",
        "status": status,
        "generated_at": _now_iso(),
        "analysis_scope": "toy cognitive-style benchmark with falsifiable gates",
        "seed_set": {"canonical": 41, "variation": 42, "intervention": [41, 42, 43], "blind_eval": blind["train_eval_split"]["eval_seeds"]},
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
        "gates": gates,
        "blocker_mapping": blocker_mapping,
        "findings": findings,
        "residual_risks": [
            "M4.1 intervention probes are sensitivity checks under controlled parameter toggles, not scientific causal inference.",
            "Blind distinguishability remains a train/eval seed split within the same generator family.",
            "The blind distinguishability benchmark is toy/internal and not an external blind validation.",
        ],
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
        "recommendation": recommendation,
    }
    report_validation = validate_acceptance_report(report)
    report["report_validation"] = report_validation
    if not report_validation["valid"]:
        report["status"] = "FAIL"
        report["recommendation"] = "BLOCK"
        report["findings"].append(
            {
                "severity": "S1",
                "label": "report_validation_failed",
                "detail": f"Acceptance report schema validation failed: {report_validation['errors']}",
            }
        )

    M41_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_text = (
        "# M4.1 Acceptance Summary\n\n"
        f"Status: `{report['status']}`\n\n"
        "- M4.1 is treated as a toy cognitive-style benchmark, not a causal inference system.\n"
        "- Intervention evidence is a sensitivity probe based on parameter interventions, not formal causal identification.\n"
        "- Blind distinguishability uses a train/eval seed split inside the same generator family and is not external blind validation.\n"
        f"- Trial variation gate: `{gates['trial_variation']['passed']}`\n"
        f"- Intervention sensitivity gate: `{gates['intervention_sensitivity']['passed']}`\n"
        f"- Blind distinguishability gate: `{gates['blind_distinguishability']['passed']}`\n"
        f"- Recommendation: `{report['recommendation']}`\n"
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
