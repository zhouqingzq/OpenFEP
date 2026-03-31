from __future__ import annotations

from statistics import mean
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE, PROFILE_REGISTRY, compute_observable_metrics, metric_values_from_payload
from .m41_inference import classify_inferred_style, infer_cognitive_style


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _default_payload() -> dict[str, Any]:
    return CognitiveStyleParameters().to_dict()


def infer_no_style_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    parameters = CognitiveStyleParameters()
    classification = classify_inferred_style(parameters)
    return {
        "model_label": "no_style_parameter_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": 0.22,
        "classification": classification,
        "parameter_estimates": {name: {"estimate": getattr(parameters, name), "confidence": 0.22} for name in PARAMETER_REFERENCE},
    }


def infer_risk_resource_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metric_values_from_payload(compute_observable_metrics(records))
    payload = _default_payload()
    payload["error_aversion"] = _clamp01(mean([metrics.get("high_expected_error_rejection_rate", payload["error_aversion"]), metrics.get("post_error_conservative_shift", payload["error_aversion"])]))
    payload["resource_pressure_sensitivity"] = _clamp01(mean([metrics.get("high_pressure_low_cost_ratio", payload["resource_pressure_sensitivity"]), metrics.get("recovery_trigger_rate", payload["resource_pressure_sensitivity"])]))
    payload["virtual_prediction_error_gain"] = _clamp01(metrics.get("conflict_avoidance_shift", payload["virtual_prediction_error_gain"]) * 0.65 + 0.15)
    parameters = CognitiveStyleParameters.from_dict(payload)
    classification = classify_inferred_style(parameters)
    return {
        "model_label": "risk_resource_only_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": 0.38,
        "classification": classification,
        "parameter_estimates": {name: {"estimate": payload[name], "confidence": 0.38 if name in {"error_aversion", "resource_pressure_sensitivity", "virtual_prediction_error_gain"} else 0.18} for name in PARAMETER_REFERENCE},
    }


def infer_confidence_only_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metric_values_from_payload(compute_observable_metrics(records))
    payload = _default_payload()
    payload["uncertainty_sensitivity"] = _clamp01(metrics.get("uncertainty_confidence_drop_rate", payload["uncertainty_sensitivity"]))
    payload["confidence_gain"] = _clamp01(mean([metrics.get("confidence_evidence_slope", payload["confidence_gain"]), metrics.get("high_evidence_commit_rate", payload["confidence_gain"])]))
    parameters = CognitiveStyleParameters.from_dict(payload)
    classification = classify_inferred_style(parameters)
    return {
        "model_label": "confidence_only_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": 0.34,
        "classification": classification,
        "parameter_estimates": {name: {"estimate": payload[name], "confidence": 0.40 if name in {"uncertainty_sensitivity", "confidence_gain"} else 0.16} for name in PARAMETER_REFERENCE},
    }


def infer_simple_rl_heuristic_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metric_values_from_payload(compute_observable_metrics(records))
    payload = _default_payload()
    payload["exploration_bias"] = _clamp01(mean([metrics.get("novel_action_ratio", payload["exploration_bias"]), metrics.get("choice_repeat_suppression", payload["exploration_bias"])]))
    payload["update_rigidity"] = _clamp01(metrics.get("mean_update_inverse", payload["update_rigidity"]))
    payload["attention_selectivity"] = _clamp01(metrics.get("evidence_aligned_choice_rate", payload["attention_selectivity"]) * 0.8 + 0.1)
    parameters = CognitiveStyleParameters.from_dict(payload)
    classification = classify_inferred_style(parameters)
    return {
        "model_label": "simple_rl_heuristic_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": 0.36,
        "classification": classification,
        "parameter_estimates": {name: {"estimate": payload[name], "confidence": 0.40 if name in {"exploration_bias", "update_rigidity", "attention_selectivity"} else 0.18} for name in PARAMETER_REFERENCE},
    }


BASELINE_INFERENCE_REGISTRY = {
    "style_inference_model": infer_cognitive_style,
    "no_style_parameter_model": infer_no_style_baseline,
    "risk_resource_only_model": infer_risk_resource_baseline,
    "confidence_only_model": infer_confidence_only_baseline,
    "simple_rl_heuristic_model": infer_simple_rl_heuristic_baseline,
}
