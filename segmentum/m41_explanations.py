from __future__ import annotations

from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, DecisionLogRecord, PARAMETER_REFERENCE


def _round(value: float) -> float:
    return round(float(value), 6)


def _normalized_record(record: DecisionLogRecord | dict[str, Any]) -> DecisionLogRecord:
    return record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record)


def explain_decision_record(
    record: DecisionLogRecord | dict[str, Any],
    *,
    parameters: CognitiveStyleParameters | dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalized_record(record)
    params = parameters if isinstance(parameters, CognitiveStyleParameters) else CognitiveStyleParameters.from_dict(dict(parameters))
    evidence = normalized.observation_evidence
    drivers = [
        {
            "parameter": "uncertainty_sensitivity",
            "contribution": _round(float(params.uncertainty_sensitivity) * float(evidence.get("uncertainty", 0.0))),
            "mechanism": "Ambiguity raises the attractiveness of inspect-like or delayed commitment policies.",
            "evidence": f"uncertainty={_round(float(evidence.get('uncertainty', 0.0)))}",
        },
        {
            "parameter": "error_aversion",
            "contribution": _round(float(params.error_aversion) * float(evidence.get("expected_error", 0.0))),
            "mechanism": "Elevated expected failure cost shifts policy toward conservative or recovery actions.",
            "evidence": f"expected_error={_round(float(evidence.get('expected_error', 0.0)))}",
        },
        {
            "parameter": "resource_pressure_sensitivity",
            "contribution": _round(
                float(params.resource_pressure_sensitivity)
                * (1.0 - ((float(normalized.resource_state.get("energy", 0.0)) + float(normalized.resource_state.get("budget", 0.0)) + float(normalized.resource_state.get("time_remaining", 0.0))) / 3.0))
            ),
            "mechanism": "Low energy, time, or budget increase conservation and recovery pressure.",
            "evidence": f"resource_state={normalized.resource_state}",
        },
        {
            "parameter": "virtual_prediction_error_gain",
            "contribution": _round(
                float(params.virtual_prediction_error_gain)
                * max(0.0, float(normalized.prediction_error_vector.get("virtual_error", 0.0)) - float(normalized.prediction_error_vector.get("direct_error", 0.0)))
            ),
            "mechanism": "Imagined loss channels suppress risky commitment when counterfactual warning exceeds direct evidence.",
            "evidence": f"prediction_error_vector={normalized.prediction_error_vector}",
        },
        {
            "parameter": "confidence_gain",
            "contribution": _round(float(params.confidence_gain) * float(evidence.get("evidence_strength", 0.0))),
            "mechanism": "Stronger evidence increases commitment readiness and subjective confidence.",
            "evidence": f"evidence_strength={_round(float(evidence.get('evidence_strength', 0.0)))}",
        },
    ]
    ranked = sorted(drivers, key=lambda item: (-float(item["contribution"]), item["parameter"]))
    return {
        "tick": normalized.tick,
        "selected_action": normalized.selected_action,
        "top_drivers": ranked[:3],
        "causal_hypothesis": f"The action `{normalized.selected_action}` emerged from the highest combined pressure among the top-ranked mechanisms.",
        "counterfactual_note": f"If `{ranked[0]['parameter']}` were reduced, the selected action would be less favored in this context.",
    }


def build_behavior_explanation_report(
    records: list[DecisionLogRecord | dict[str, Any]],
    *,
    parameters: CognitiveStyleParameters | dict[str, Any],
) -> dict[str, Any]:
    normalized = [_normalized_record(record) for record in records]
    params = parameters if isinstance(parameters, CognitiveStyleParameters) else CognitiveStyleParameters.from_dict(dict(parameters))
    decision_reports = [explain_decision_record(record, parameters=params) for record in normalized]
    driver_totals = {name: 0.0 for name in PARAMETER_REFERENCE}
    for report in decision_reports:
        for driver in report["top_drivers"]:
            driver_totals[driver["parameter"]] += float(driver["contribution"])
    dominant_mechanisms = sorted(driver_totals, key=lambda name: (-driver_totals[name], name))[:4]
    mechanism_graph = {
        "nodes": [
            {"id": name, "label": name, "role": "parameter" if name in PARAMETER_REFERENCE else "decision"}
            for name in [*PARAMETER_REFERENCE.keys(), "selected_action", "confidence", "recovery", "commitment"]
        ],
        "edges": [
            {
                "source": name,
                "target": "selected_action",
                "weight": _round(driver_totals[name] / max(1, len(decision_reports))),
                "hypothesis": PARAMETER_REFERENCE[name]["decision_path"],
            }
            for name in dominant_mechanisms
        ],
    }
    return {
        "analysis_type": "behavior_explanation_report",
        "record_count": len(normalized),
        "dominant_mechanisms": dominant_mechanisms,
        "mechanism_graph": mechanism_graph,
        "decision_explanations": decision_reports,
        "summary": {
            "top_mechanism": dominant_mechanisms[0] if dominant_mechanisms else None,
            "explanation_density": _round(sum(driver_totals.values()) / max(1, len(decision_reports))),
        },
    }
