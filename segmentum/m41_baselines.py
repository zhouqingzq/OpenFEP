from __future__ import annotations

"""Legacy same-framework baseline models for synthetic style inference.

These baselines remain useful for sandbox comparisons inside the repository's
synthetic cognitive-style family. They should not be interpreted as external
benchmark baselines for the narrowed M4.1 acceptance claim.
"""

import hashlib
import random
from statistics import mean, pvariance
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE, PROFILE_REGISTRY, compute_observable_metrics, metric_values_from_payload
from .m41_inference import classify_inferred_style, infer_cognitive_style


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _default_payload() -> dict[str, Any]:
    return CognitiveStyleParameters().to_dict()


def _build_parameter_estimates(payload: dict[str, float], active: set[str], confidence: float) -> dict[str, Any]:
    return {
        name: {
            "estimate": payload[name],
            "confidence": round(confidence if name in active else 0.0, 6),
            "identifiable": name in active,
        }
        for name in PARAMETER_REFERENCE
    }


def _metric_payloads(
    metrics: dict[str, dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    return [dict(metrics.get(metric_name, {})) for metric_name in metric_names]


def _baseline_fit_confidence(
    metrics: dict[str, dict[str, Any]],
    *,
    metric_names: list[str],
) -> float:
    payloads = _metric_payloads(metrics, metric_names)
    if not payloads:
        return 0.0

    supported = [
        payload
        for payload in payloads
        if payload.get("value") is not None and not payload.get("insufficient_data", False)
    ]
    coverage = len(supported) / len(payloads)
    if not supported:
        return 0.0

    adequacy = mean(
        min(1.0, float(payload.get("sample_size", 0)) / max(1.0, float(payload.get("min_samples", 1))))
        for payload in supported
    )
    values = [float(payload["value"]) for payload in supported]
    dispersion = pvariance(values) if len(values) > 1 else 0.0
    consensus = 1.0 - min(1.0, dispersion * 4.0)
    sufficient_data_bonus = 1.0 if len(supported) == len(payloads) else 0.35
    return round(_clamp01(coverage * 0.45 + adequacy * 0.25 + consensus * 0.20 + sufficient_data_bonus * 0.10), 6)


def _random_baseline_seed(records: list[dict[str, Any]] | None = None) -> int:
    if not records:
        return 41
    digest_source = "|".join(
        f"{record.get('seed', 0)}:{record.get('tick', 0)}:{record.get('selected_action', '')}"
        for record in records
    )
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def infer_no_style_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    parameters = CognitiveStyleParameters()
    classification = classify_inferred_style(parameters, profile_registry=PROFILE_REGISTRY)
    return {
        "model_label": "no_style_parameter_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": 0.0,
        "classification": classification,
        "parameter_estimates": _build_parameter_estimates(parameters.to_dict(), set(), 0.0),
    }


def infer_risk_resource_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    observable_metrics = compute_observable_metrics(records)
    metrics = metric_values_from_payload(observable_metrics)
    payload = _default_payload()
    payload["error_aversion"] = _clamp01(mean([metrics.get("high_expected_error_rejection_rate", payload["error_aversion"]), metrics.get("post_error_conservative_shift", payload["error_aversion"])]))
    payload["resource_pressure_sensitivity"] = _clamp01(mean([metrics.get("high_pressure_low_cost_ratio", payload["resource_pressure_sensitivity"]), metrics.get("recovery_trigger_rate", payload["resource_pressure_sensitivity"])]))
    payload["virtual_prediction_error_gain"] = _clamp01(mean([metrics.get("conflict_avoidance_shift", payload["virtual_prediction_error_gain"]), metrics.get("counterfactual_loss_sensitivity", payload["virtual_prediction_error_gain"])]))
    identifiable = {"error_aversion", "resource_pressure_sensitivity", "virtual_prediction_error_gain"}
    confidence = _baseline_fit_confidence(
        observable_metrics,
        metric_names=[
            "high_expected_error_rejection_rate",
            "post_error_conservative_shift",
            "high_pressure_low_cost_ratio",
            "recovery_trigger_rate",
            "conflict_avoidance_shift",
            "counterfactual_loss_sensitivity",
        ],
    )
    parameters = CognitiveStyleParameters.from_dict(payload)
    return {
        "model_label": "risk_resource_only_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": confidence,
        "classification": classify_inferred_style(parameters, profile_registry=PROFILE_REGISTRY),
        "parameter_estimates": _build_parameter_estimates(payload, identifiable, confidence),
    }


def infer_confidence_only_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    observable_metrics = compute_observable_metrics(records)
    metrics = metric_values_from_payload(observable_metrics)
    payload = _default_payload()
    payload["uncertainty_sensitivity"] = _clamp01(metrics.get("uncertainty_confidence_drop_rate", payload["uncertainty_sensitivity"]))
    payload["confidence_gain"] = _clamp01(mean([metrics.get("confidence_evidence_slope", payload["confidence_gain"]), metrics.get("high_evidence_commit_rate", payload["confidence_gain"])]))
    identifiable = {"uncertainty_sensitivity", "confidence_gain"}
    confidence = _baseline_fit_confidence(
        observable_metrics,
        metric_names=[
            "uncertainty_confidence_drop_rate",
            "confidence_evidence_slope",
            "high_evidence_commit_rate",
        ],
    )
    parameters = CognitiveStyleParameters.from_dict(payload)
    return {
        "model_label": "confidence_only_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": confidence,
        "classification": classify_inferred_style(parameters, profile_registry=PROFILE_REGISTRY),
        "parameter_estimates": _build_parameter_estimates(payload, identifiable, confidence),
    }


def infer_simple_rl_heuristic_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    observable_metrics = compute_observable_metrics(records)
    metrics = metric_values_from_payload(observable_metrics)
    payload = _default_payload()
    payload["exploration_bias"] = _clamp01(mean([metrics.get("novel_action_ratio", payload["exploration_bias"]), metrics.get("choice_repeat_suppression", payload["exploration_bias"])]))
    payload["update_rigidity"] = _clamp01(metrics.get("mean_update_inverse", payload["update_rigidity"]))
    payload["attention_selectivity"] = _clamp01(metrics.get("evidence_aligned_choice_rate", payload["attention_selectivity"]))
    identifiable = {"exploration_bias", "update_rigidity", "attention_selectivity"}
    confidence = _baseline_fit_confidence(
        observable_metrics,
        metric_names=[
            "novel_action_ratio",
            "choice_repeat_suppression",
            "mean_update_inverse",
            "evidence_aligned_choice_rate",
        ],
    )
    parameters = CognitiveStyleParameters.from_dict(payload)
    return {
        "model_label": "simple_rl_heuristic_model",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": confidence,
        "classification": classify_inferred_style(parameters, profile_registry=PROFILE_REGISTRY),
        "parameter_estimates": _build_parameter_estimates(payload, identifiable, confidence),
    }


def random_baseline_inference(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sampling_seed = _random_baseline_seed(records)
    rng = random.Random(sampling_seed)
    payload = {
        name: round(rng.uniform(0.0, 1.0), 6)
        for name in PARAMETER_REFERENCE
    }
    parameters = CognitiveStyleParameters.from_dict(payload)
    return {
        "model_label": "random_baseline",
        "inferred_parameters": parameters.to_dict(),
        "fit_confidence": 0.0,
        "classification": classify_inferred_style(parameters, profile_registry=PROFILE_REGISTRY),
        "parameter_estimates": _build_parameter_estimates(parameters.to_dict(), set(), 0.0),
        "sampling_strategy": "controlled_uniform_0_1",
        "sampling_seed": sampling_seed,
    }


BASELINE_INFERENCE_REGISTRY = {
    "style_inference_model": infer_cognitive_style,
    "no_style_parameter_model": infer_no_style_baseline,
    "risk_resource_only_model": infer_risk_resource_baseline,
    "confidence_only_model": infer_confidence_only_baseline,
    "simple_rl_heuristic_model": infer_simple_rl_heuristic_baseline,
    "random_baseline": random_baseline_inference,
}


def baseline_scope_note() -> dict[str, str]:
    return {
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "validation_type": "synthetic_holdout_same_framework",
        "interpretation": "synthetic-family baseline comparison only",
    }
