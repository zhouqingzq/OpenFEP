from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m4_cognitive_style import (
    CognitiveStyleParameters,
    audit_decision_log,
    audit_observable_contracts,
    compute_observable_metrics,
    observable_parameter_contracts,
    parameter_intervention_sensitivity_matrix,
    run_cognitive_style_trial,
    validate_acceptance_report,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M41_REPORT_PATH = REPORTS_DIR / "m41_acceptance_report.json"
M41_SUMMARY_PATH = REPORTS_DIR / "m41_acceptance_summary.md"
M41_SCOPE_PATH = REPORTS_DIR / "m41_scope_redefinition.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gate(passed: bool, evidence: dict[str, Any], *, blocking: bool = True, validation_type: str | None = None) -> dict[str, Any]:
    payload = {"passed": bool(passed), "blocking": blocking, "evidence": evidence}
    if validation_type is not None:
        payload["validation_type"] = validation_type
    return payload


def _report_status_for_gates(gates: dict[str, dict[str, Any]]) -> str:
    return "FAIL" if any(
        payload.get("blocking") and payload.get("passed") is False
        for payload in gates.values()
    ) else "PASS"


# ---------------------------------------------------------------------------
# G1: Schema completeness — 8-param roundtrip + DecisionLogRecord fields
# ---------------------------------------------------------------------------

def _evaluate_g1_schema() -> dict[str, Any]:
    params = CognitiveStyleParameters(exploration_bias=0.81, resource_pressure_sensitivity=0.93)
    roundtripped = CognitiveStyleParameters.from_dict(params.to_dict())
    original = params.to_dict()
    restored = roundtripped.to_dict()
    param_names = [k for k in original if k != "schema_version"]
    max_loss = max(abs(original[k] - restored[k]) for k in param_names)
    roundtrip_ok = max_loss < 1e-6

    # Run a short trial and audit its log for schema completeness
    trial = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41)
    audit = audit_decision_log(trial["logs"])
    snapshot_rate = audit["parameter_snapshot_complete_rate"]

    return {
        "roundtrip_precision_loss": max_loss,
        "roundtrip_ok": roundtrip_ok,
        "parameter_count": len(param_names),
        "invalid_rate": audit["invalid_rate"],
        "parameter_snapshot_complete_rate": snapshot_rate,
        "passed": roundtrip_ok and audit["invalid_rate"] <= 0.05 and snapshot_rate == 1.0,
    }


# ---------------------------------------------------------------------------
# G2: Trial variability — seed determinism + parameter sensitivity
# ---------------------------------------------------------------------------

def _evaluate_g2_variability() -> dict[str, Any]:
    p = CognitiveStyleParameters()
    r1 = run_cognitive_style_trial(p, seed=1)
    r2 = run_cognitive_style_trial(p, seed=2)
    r3 = run_cognitive_style_trial(p, seed=1)

    actions_1 = r1["summary"]["selected_actions"]
    actions_2 = r2["summary"]["selected_actions"]
    actions_3 = r3["summary"]["selected_actions"]

    diff_seed_differ = actions_1 != actions_2
    same_seed_identical = actions_1 == actions_3

    p_alt = CognitiveStyleParameters(exploration_bias=0.05)
    r4 = run_cognitive_style_trial(p_alt, seed=1)
    actions_4 = r4["summary"]["selected_actions"]
    diff_param_differ = actions_1 != actions_4

    return {
        "different_seed_produces_different_actions": diff_seed_differ,
        "same_seed_produces_identical_actions": same_seed_identical,
        "different_param_produces_different_actions": diff_param_differ,
        "passed": diff_seed_differ and same_seed_identical and diff_param_differ,
    }


# ---------------------------------------------------------------------------
# G3: Observability — each param >= 2 observable metrics, evaluators work
# ---------------------------------------------------------------------------

def _evaluate_g3_observability() -> dict[str, Any]:
    contracts = observable_parameter_contracts()
    all_have_two = all(len(c["observables"]) >= 2 for c in contracts.values())
    contract_audit = audit_observable_contracts()
    informative_per_parameter = {
        parameter_name: payload["informative_observables"]
        for parameter_name, payload in contract_audit["per_parameter"].items()
    }

    # Check evaluators execute without error on a 100-tick trial
    trial = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41)
    metrics = compute_observable_metrics(trial["logs"])
    with_data = [k for k, v in metrics.items() if not v.get("insufficient_data", False)]

    # Check sparse data triggers insufficient_data
    sparse_metrics = compute_observable_metrics(trial["logs"][:2])
    insufficient_count = sum(1 for v in sparse_metrics.values() if v.get("insufficient_data", False))
    sparse_ratio = insufficient_count / max(len(sparse_metrics), 1)

    return {
        "parameter_count": len(contracts),
        "all_params_have_two_observables": all_have_two,
        "registry_executable": contract_audit["registry_executable"],
        "metrics_with_data": len(with_data),
        "total_metrics": len(metrics),
        "sparse_insufficient_ratio": round(sparse_ratio, 4),
        "sparse_threshold_works": sparse_ratio >= 0.50,
        "direction_mismatch_count": contract_audit["direction_mismatch_count"],
        "uninformative_metric_count": contract_audit["uninformative_metric_count"],
        "informative_observables_per_parameter": informative_per_parameter,
        "passed": (
            all_have_two
            and contract_audit["registry_executable"]
            and len(with_data) == len(metrics)
            and sparse_ratio >= 0.50
            and contract_audit["direction_mismatch_count"] == 0
            and all(count >= 2 for count in informative_per_parameter.values())
        ),
    }


# ---------------------------------------------------------------------------
# G4: Intervention sensitivity — parameter -> observable causal direction
# ---------------------------------------------------------------------------

def _evaluate_g4_sensitivity() -> dict[str, Any]:
    matrix = parameter_intervention_sensitivity_matrix()
    identifiable_count = sum(1 for v in matrix.values() if v["identifiable"])
    total = len(matrix)
    all_identifiable = identifiable_count == total

    # Collect per-parameter evidence
    per_param = {
        name: {
            "metric": v["target_metric"],
            "delta": v["delta"],
            "identifiable": v["identifiable"],
            "expectation": v["expectation"],
        }
        for name, v in matrix.items()
    }

    return {
        "identifiable_count": identifiable_count,
        "total_parameters": total,
        "all_identifiable": all_identifiable,
        "per_parameter": per_param,
        "passed": all_identifiable,
    }


# ---------------------------------------------------------------------------
# G5: Log completeness — invalid rate <= 0.05, all required fields present
# ---------------------------------------------------------------------------

def _evaluate_g5_log_completeness() -> dict[str, Any]:
    trial = run_cognitive_style_trial(CognitiveStyleParameters(), seed=41)
    audit = audit_decision_log(trial["logs"])
    return {
        "total_records": audit["total_records"],
        "valid_records": audit["valid_records"],
        "invalid_rate": audit["invalid_rate"],
        "parameter_snapshot_complete_rate": audit["parameter_snapshot_complete_rate"],
        "invalid_value_counts": audit["invalid_value_counts"],
        "semantic_invalid_counts": audit["semantic_invalid_counts"],
        "passed": audit["invalid_rate"] <= 0.05,
    }


# ---------------------------------------------------------------------------
# G6: Stress behavior — resource conservation under pressure
# ---------------------------------------------------------------------------

def _evaluate_g6_stress() -> dict[str, Any]:
    params = CognitiveStyleParameters(resource_pressure_sensitivity=0.95)
    stress_trial = run_cognitive_style_trial(params, seed=41, stress=True)
    actions = stress_trial["summary"]["selected_actions"]
    low_cost_actions = {"rest", "conserve", "recover", "scan"}
    low_cost_count = sum(1 for a in actions if a in low_cost_actions)
    ratio = round(low_cost_count / max(len(actions), 1), 4)

    return {
        "total_actions": len(actions),
        "low_cost_actions": low_cost_count,
        "high_pressure_low_cost_ratio": ratio,
        "passed": ratio >= 0.55,
    }


# ---------------------------------------------------------------------------
# R1: Report structure self-consistency
# ---------------------------------------------------------------------------

def _compute_r1_structure_evidence(
    *,
    gates: dict[str, dict[str, Any]],
    failed_gates: list[str],
    status: str,
) -> dict[str, Any]:
    expected_failed = sorted(name for name, payload in gates.items() if not payload.get("passed"))
    expected_status = _report_status_for_gates(gates)
    return {
        "gate_count": len(gates),
        "all_gates_have_evidence": all(
            isinstance(payload.get("evidence"), dict) and bool(payload.get("evidence"))
            for payload in gates.values()
        ),
        "all_gates_have_passed_flag": all("passed" in payload for payload in gates.values()),
        "failed_gates_match_recomputed": sorted(failed_gates) == expected_failed,
        "status_matches_blocking_gates": status == expected_status,
        "recomputed_failed_gates": expected_failed,
        "recomputed_status": expected_status,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_m41_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()

    # Evaluate G1-G6 interface gates
    g1 = _evaluate_g1_schema()
    g2 = _evaluate_g2_variability()
    g3 = _evaluate_g3_observability()
    g4 = _evaluate_g4_sensitivity()
    g5 = _evaluate_g5_log_completeness()
    g6 = _evaluate_g6_stress()

    gates = {
        "g1_schema_completeness": _gate(
            g1["passed"], g1, validation_type="interface_contract",
        ),
        "g2_trial_variability": _gate(
            g2["passed"], g2, validation_type="interface_contract",
        ),
        "g3_observability": _gate(
            g3["passed"], g3, validation_type="interface_contract",
        ),
        "g4_intervention_sensitivity": _gate(
            g4["passed"], g4, validation_type="interface_contract",
        ),
        "g5_log_completeness": _gate(
            g5["passed"], g5, validation_type="interface_contract",
        ),
        "g6_stress_behavior": _gate(
            g6["passed"], g6, validation_type="interface_contract",
        ),
    }

    # R1: report structure self-consistency (computed over G1-G6 first)
    provisional_failed = sorted(n for n, p in gates.items() if not p["passed"])
    provisional_status = _report_status_for_gates(gates)
    r1_evidence = _compute_r1_structure_evidence(
        gates=gates, failed_gates=provisional_failed, status=provisional_status,
    )
    gates["r1_report_structure"] = _gate(
        all(r1_evidence[k] for k in (
            "all_gates_have_evidence",
            "all_gates_have_passed_flag",
            "failed_gates_match_recomputed",
            "status_matches_blocking_gates",
        )),
        r1_evidence,
        validation_type="report_integrity",
    )

    # Recompute after adding R1
    final_failed = sorted(n for n, p in gates.items() if not p["passed"])
    final_status = _report_status_for_gates(gates)
    r1_evidence_final = _compute_r1_structure_evidence(
        gates=gates, failed_gates=final_failed, status=final_status,
    )
    gates["r1_report_structure"] = _gate(
        all(r1_evidence_final[k] for k in (
            "all_gates_have_evidence",
            "all_gates_have_passed_flag",
            "failed_gates_match_recomputed",
            "status_matches_blocking_gates",
        )),
        r1_evidence_final,
        validation_type="report_integrity",
    )

    failed_gates = sorted(n for n, p in gates.items() if not p["passed"])
    status = _report_status_for_gates(gates)

    findings = [
        {
            "severity": "S1",
            "label": f"{gate_name}_failed",
            "detail": f"M4.1 gate `{gate_name}` did not meet the acceptance threshold.",
        }
        for gate_name in failed_gates
    ]

    scope = {
        "milestone_goal": (
            "Translate 'prior preference structure under finite energy constraints' "
            "into a unified parameter interface, observable interface, and logging interface "
            "that provides a common language for downstream benchmark tasks and open-world validation."
        ),
        "gates_in_scope": [
            "g1_schema_completeness",
            "g2_trial_variability",
            "g3_observability",
            "g4_intervention_sensitivity",
            "g5_log_completeness",
            "g6_stress_behavior",
        ],
        "deferred_to_later_milestones": [
            "cross-generator blind classification (G7)",
            "parameter falsification (G8)",
            "cross-generator parameter recovery (G9)",
            "inference engine audit (G10)",
            "baseline non-hardcoded audit (G11)",
        ],
        "rationale": (
            "G1-G6 verify the interface contracts (parameter roundtrip, observability, "
            "intervention sensitivity, log completeness, stress behavior). "
            "G7-G11 verify whether latent parameters correspond to real cognitive structure, "
            "which requires external human data and belongs to subsequent validation milestones."
        ),
    }

    report = {
        "milestone_id": "M4.1",
        "status": status,
        "generated_at": _now_iso(),
        "scope": scope,
        "gates": gates,
        "failed_gates": failed_gates,
        "findings": findings,
        "recommendation": "ACCEPT" if status == "PASS" else "BLOCK",
        "freshness": {"generated_this_round": True, "round_started_at": started_at},
    }

    report_validation = validate_acceptance_report(report)
    report["report_validation"] = report_validation
    if not report_validation["valid"]:
        report["status"] = "FAIL"
        report["recommendation"] = "BLOCK"
        report["failed_gates"] = sorted(set(report["failed_gates"]) | {"r1_report_structure"})
        report["findings"].append({
            "severity": "S1",
            "label": "report_validation_failed",
            "detail": f"Acceptance report validation failed: {report_validation['errors']}",
        })

    M41_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# M4.1 Acceptance Summary",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "M4.1 validates the unified interface layer: parameter interface (`CognitiveStyleParameters`),",
        "observable interface (parameter-to-metric contracts), and logging interface (`DecisionLogRecord`).",
        "",
        "## Gate Results",
        "",
    ]
    for name, payload in gates.items():
        mark = "PASS" if payload["passed"] else "FAIL"
        summary_lines.append(f"- `{name}`: **{mark}**")
    if failed_gates:
        summary_lines += ["", "## Failed Gates", ""]
        for g in failed_gates:
            summary_lines.append(f"- `{g}`")
    summary_lines.append("")
    M41_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    scope_doc = (
        "# M4.1 Scope Definition\n\n"
        "## Goal\n\n"
        "Translate 'prior preference structure under finite energy constraints' into a unified\n"
        "parameter interface, observable interface, and logging interface that provides a common\n"
        "language for downstream benchmark tasks and open-world validation.\n\n"
        "## In Scope (G1-G6)\n\n"
        "- G1: Schema completeness — 8-parameter roundtrip, DecisionLogRecord field audit\n"
        "- G2: Trial variability — seed determinism, parameter sensitivity\n"
        "- G3: Observability — each parameter maps to >= 2 computable observable metrics\n"
        "- G4: Intervention sensitivity — parameter changes cause expected metric changes\n"
        "- G5: Log completeness — invalid record rate <= 0.05, all required fields present\n"
        "- G6: Stress behavior — resource conservation under energy pressure\n\n"
        "## Deferred to Later Milestones\n\n"
        "- Cross-generator blind classification (requires independent external generator)\n"
        "- Parameter falsification (requires robust control metrics)\n"
        "- Cross-generator parameter recovery (requires external human data)\n"
        "- Inference engine data-driven audit\n"
        "- Baseline non-hardcoded audit\n\n"
        "These items verify whether latent parameters correspond to real cognitive structure.\n"
        "They require external human data and independent annotation, not interface-layer testing.\n"
    )
    M41_SCOPE_PATH.write_text(scope_doc, encoding="utf-8")

    return {
        "report": str(M41_REPORT_PATH),
        "summary": str(M41_SUMMARY_PATH),
        "scope": str(M41_SCOPE_PATH),
    }
