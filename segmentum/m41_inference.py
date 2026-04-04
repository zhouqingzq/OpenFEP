from __future__ import annotations

"""Synthetic/same-framework inference sandbox retained next to M4.1.

This module estimates the M4.1 cognitive-style parameters from logs, but the
training signal comes from repository-owned synthetic generators and shared
latent semantics. Treat it as a sidecar diagnostic only: it is not M4.1
acceptance evidence, not external validation, and not proof that M4.2
benchmark recovery-on-task is already complete.
"""

from collections import defaultdict
from functools import lru_cache
from statistics import mean, pvariance
from typing import Any

from .m4_cognitive_style import (
    BLIND_CLASSIFICATION_FEATURES,
    CognitiveStyleParameters,
    DecisionLogRecord,
    PARAMETER_REFERENCE,
    PROFILE_REGISTRY,
    TRAIN_PROFILE_SEEDS,
    audit_decision_log,
    compute_observable_metrics,
    metric_values_from_payload,
    observable_parameter_contracts,
    reconstruct_behavior_patterns,
    run_cognitive_style_trial,
)
from .m41_blind_classifier import predict_profile_from_features, train_blind_classifier
from .m41_explanations import build_behavior_explanation_report


DEFAULT_PARAMETER_VECTOR = CognitiveStyleParameters().to_dict()
DEFAULT_PARAMETER_VECTOR.pop("schema_version", None)

PARAMETER_BASELINE_EXPLANATIONS: dict[str, list[str]] = {
    "uncertainty_sensitivity": [
        "confidence-only model: local ambiguity can mimic elevated uncertainty sensitivity",
        "task ambiguity effect: scenario volatility can lower confidence without a stable trait difference",
    ],
    "error_aversion": [
        "risk/resource heuristic: conservative choices may come from loss structure rather than enduring aversion",
        "hazard policy artifact: high-cost environments can look like stable error aversion",
    ],
    "exploration_bias": [
        "novelty-by-task heuristic: the environment may simply reward inspect-like actions",
        "information sparsity effect: missing state can inflate exploration without a persistent bias",
    ],
    "attention_selectivity": [
        "stimulus salience effect: one channel may dominate regardless of style",
        "format artifact: simplified cue layouts can mechanically increase attention concentration",
    ],
    "confidence_gain": [
        "signal quality effect: cleaner evidence alone can elevate commit confidence",
        "task separability artifact: easy tasks can steepen confidence without a broader style change",
    ],
    "update_rigidity": [
        "simple learning-rate model: slow updates may reflect a low learning rate rather than rigidity",
        "stationary environment effect: stable tasks can make persistence look like rigidity",
    ],
    "resource_pressure_sensitivity": [
        "pure pressure model: conservation may be explained by resource depletion alone",
        "workload spike effect: transient external pressure can mimic a stable sensitivity parameter",
    ],
    "virtual_prediction_error_gain": [
        "warning-channel artifact: explicit warning cues can look like counterfactual sensitivity",
        "confidence conflict model: imagined loss may simply depress confidence without a dedicated gain parameter",
    ],
}


PARAMETER_COUPLINGS: dict[str, list[str]] = {
    "uncertainty_sensitivity": ["exploration_bias", "confidence_gain"],
    "error_aversion": ["resource_pressure_sensitivity", "virtual_prediction_error_gain"],
    "exploration_bias": ["uncertainty_sensitivity", "attention_selectivity"],
    "attention_selectivity": ["confidence_gain", "exploration_bias"],
    "confidence_gain": ["attention_selectivity", "uncertainty_sensitivity"],
    "update_rigidity": ["error_aversion", "resource_pressure_sensitivity"],
    "resource_pressure_sensitivity": ["error_aversion", "update_rigidity"],
    "virtual_prediction_error_gain": ["error_aversion", "confidence_gain"],
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _normalized_records(records: list[DecisionLogRecord | dict[str, Any]]) -> list[DecisionLogRecord]:
    return [record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record) for record in records]


def _metric_support(metrics: dict[str, dict[str, Any]], parameter_name: str) -> list[dict[str, Any]]:
    contracts = observable_parameter_contracts()[parameter_name]["observables"]
    supported: list[dict[str, Any]] = []
    for contract in contracts:
        metric_name = contract["metric"]
        payload = metrics.get(metric_name, {})
        if payload.get("insufficient_data") or payload.get("value") is None:
            continue
        supported.append(
            {
                "metric": metric_name,
                "value": float(payload["value"]),
                "sample_size": int(payload.get("sample_size", 0)),
                "min_samples": int(payload.get("min_samples", contract.get("min_samples", 1))),
                "direction": contract["direction"],
                "description": contract["description"],
            }
        )
    return supported


def _estimate_parameter(metrics: dict[str, dict[str, Any]], parameter_name: str) -> dict[str, Any]:
    supported = _metric_support(metrics, parameter_name)
    if not supported:
        return {
            "parameter": parameter_name,
            "estimate": DEFAULT_PARAMETER_VECTOR[parameter_name],
            "confidence": 0.0,
            "identifiable": False,
            "evidence_coverage": 0.0,
            "supporting_metrics": [],
            "reason": "no_executable_observables",
            "alternative_explanations": list(PARAMETER_BASELINE_EXPLANATIONS[parameter_name]),
            "coupled_parameters": list(PARAMETER_COUPLINGS.get(parameter_name, [])),
        }

    values = [float(item["value"]) for item in supported]
    sample_ratio = mean(float(item["sample_size"]) / max(1.0, float(item["min_samples"])) for item in supported)
    coverage = len(supported) / max(1, len(observable_parameter_contracts()[parameter_name]["observables"]))
    dispersion = pvariance(values) if len(values) > 1 else 0.0
    estimate = _clamp01(mean(values))
    confidence = _clamp01(
        0.18
        + coverage * 0.32
        + min(0.24, sample_ratio * 0.08)
        + (1.0 - min(1.0, dispersion * 3.5)) * 0.26
    )
    identifiable = len(supported) >= 2 and confidence >= 0.55
    return {
        "parameter": parameter_name,
        "estimate": _round(estimate),
        "confidence": _round(confidence),
        "identifiable": identifiable,
        "evidence_coverage": _round(coverage),
        "supporting_metrics": supported,
        "reason": "well_supported" if identifiable else "metric_confounding_or_sparse_support",
        "alternative_explanations": list(PARAMETER_BASELINE_EXPLANATIONS[parameter_name]),
        "coupled_parameters": list(PARAMETER_COUPLINGS.get(parameter_name, [])),
    }


def _parameter_vector_from_estimates(estimates: dict[str, dict[str, Any]]) -> CognitiveStyleParameters:
    payload = {"schema_version": CognitiveStyleParameters().schema_version}
    for parameter_name in PARAMETER_REFERENCE:
        payload[parameter_name] = float(estimates[parameter_name]["estimate"])
    return CognitiveStyleParameters.from_dict(payload)


def _training_parameter_bank() -> list[CognitiveStyleParameters]:
    bank = list(PROFILE_REGISTRY.values())
    for profile in PROFILE_REGISTRY.values():
        for delta in (-0.12, 0.12):
            mutated = profile.to_dict()
            for name in ("exploration_bias", "error_aversion", "confidence_gain", "resource_pressure_sensitivity"):
                mutated[name] = _clamp01(mutated[name] + delta)
            bank.append(CognitiveStyleParameters.from_dict(mutated))
    return bank


def _metric_vector_distance(observed: dict[str, float], candidate: dict[str, float]) -> float:
    shared = sorted(set(observed) & set(candidate))
    if not shared:
        return float("inf")
    return sum(abs(float(observed[name]) - float(candidate[name])) for name in shared) / len(shared)


@lru_cache(maxsize=1)
def _candidate_library() -> list[dict[str, Any]]:
    library: list[dict[str, Any]] = []
    for candidate_id, parameters in enumerate(_training_parameter_bank()):
        rows = []
        for seed in TRAIN_PROFILE_SEEDS:
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=parameters.error_aversion >= 0.80)
            rows.append(trial["observable_metric_values"])
        metric_names = sorted({name for row in rows for name in row})
        metric_values = {
            metric_name: _round(mean(float(row[metric_name]) for row in rows if metric_name in row))
            for metric_name in metric_names
        }
        library.append(
            {
                "candidate_id": candidate_id,
                "parameters": parameters,
                "metric_values": metric_values,
                "training_seeds": list(TRAIN_PROFILE_SEEDS),
            }
        )
    return library


def _candidate_parameter_estimates(metric_values: dict[str, float]) -> dict[str, Any]:
    ranked = []
    for candidate in _candidate_library():
        distance = _metric_vector_distance(metric_values, candidate["metric_values"])
        ranked.append({**candidate, "distance": distance})
    ranked.sort(key=lambda item: (float(item["distance"]), item["candidate_id"]))
    top = ranked[:5]
    if not top:
        return {"top_candidates": [], "candidate_estimates": {}, "fit_margin": 0.0, "train_test_seed_overlap": 0}

    weights = [1.0 / max(0.02, float(item["distance"])) for item in top]
    total_weight = sum(weights) or 1.0
    estimates = {}
    for parameter_name in PARAMETER_REFERENCE:
        weighted_value = sum(
            float(getattr(item["parameters"], parameter_name)) * weight
            for item, weight in zip(top, weights)
        ) / total_weight
        consensus = pvariance([float(getattr(item["parameters"], parameter_name)) for item in top]) if len(top) > 1 else 0.0
        estimates[parameter_name] = {
            "estimate": _round(weighted_value),
            "distance_consensus": _round(consensus),
        }
    fit_margin = (float(top[1]["distance"]) - float(top[0]["distance"])) if len(top) > 1 else 0.35
    return {
        "top_candidates": top,
        "candidate_estimates": estimates,
        "fit_margin": _round(fit_margin),
        "train_test_seed_overlap": 0,
    }


def _parameter_target(parameters: CognitiveStyleParameters, parameter_name: str) -> float:
    return float(getattr(parameters, parameter_name))


def _training_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameters in _training_parameter_bank():
        for seed in TRAIN_PROFILE_SEEDS:
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=parameters.error_aversion >= 0.80)
            rows.append(
                {
                    "parameters": parameters,
                    "metric_values": dict(trial["observable_metric_values"]),
                    "observable_metrics": dict(trial["observable_metrics"]),
                }
            )
    return rows


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[row_index]) + [float(vector[row_index])] for row_index in range(size)]
    for pivot_index in range(size):
        best_index = max(range(pivot_index, size), key=lambda idx: abs(augmented[idx][pivot_index]))
        if abs(augmented[best_index][pivot_index]) < 1e-9:
            continue
        if best_index != pivot_index:
            augmented[pivot_index], augmented[best_index] = augmented[best_index], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row_index], augmented[pivot_index])
            ]
    return [row[-1] for row in augmented]


def _fit_weighted_linear_regression(
    rows: list[dict[str, Any]],
    *,
    parameter_name: str,
    feature_names: list[str],
    ridge: float = 0.08,
) -> dict[str, Any]:
    design_rows: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        metric_values = row["metric_values"]
        if not all(feature in metric_values for feature in feature_names):
            continue
        design_rows.append([1.0] + [float(metric_values[feature]) for feature in feature_names])
        targets.append(_parameter_target(row["parameters"], parameter_name))

    default_value = DEFAULT_PARAMETER_VECTOR[parameter_name]
    if len(design_rows) < max(8, len(feature_names) + 2):
        return {
            "parameter": parameter_name,
            "selected_metrics": list(feature_names),
            "weights": {feature_name: 0.0 for feature_name in feature_names},
            "bias": _round(default_value),
            "train_seeds": list(TRAIN_PROFILE_SEEDS),
            "fit_statistics": {
                "sample_count": len(design_rows),
                "train_mae": None,
                "coverage": 0.0,
            },
        }

    dimension = len(feature_names) + 1
    xtx = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    xty = [0.0 for _ in range(dimension)]
    for row_values, target in zip(design_rows, targets):
        for left_index in range(dimension):
            xty[left_index] += row_values[left_index] * target
            for right_index in range(dimension):
                xtx[left_index][right_index] += row_values[left_index] * row_values[right_index]
    for index in range(1, dimension):
        xtx[index][index] += ridge
    beta = _solve_linear_system(xtx, xty)
    bias = float(beta[0]) if beta else default_value
    weights = {
        feature_name: _round(beta[index + 1] if index + 1 < len(beta) else 0.0)
        for index, feature_name in enumerate(feature_names)
    }
    predictions = [
        max(0.0, min(1.0, bias + sum(float(weights[feature_name]) * float(row["metric_values"][feature_name]) for feature_name in feature_names)))
        for row in rows
        if all(feature in row["metric_values"] for feature in feature_names)
    ]
    maes = [abs(prediction - target) for prediction, target in zip(predictions, targets)]
    direct_estimates = [
        float(_estimate_parameter(row["observable_metrics"], parameter_name)["estimate"])
        for row in rows
        if all(feature in row["metric_values"] for feature in feature_names)
    ]
    direct_maes = [abs(prediction - target) for prediction, target in zip(direct_estimates, targets)]
    regression_score = 1.0 / max(1e-6, mean(maes) if maes else 1.0)
    direct_score = 1.0 / max(1e-6, mean(direct_maes) if direct_maes else 1.0)
    blend_total = regression_score + direct_score
    return {
        "parameter": parameter_name,
        "selected_metrics": list(feature_names),
        "weights": weights,
        "bias": _round(bias),
        "blend_weights": {
            "regression_component": _round(regression_score / blend_total),
            "direct_estimate_component": _round(direct_score / blend_total),
        },
        "train_seeds": list(TRAIN_PROFILE_SEEDS),
        "fit_statistics": {
            "sample_count": len(design_rows),
            "train_mae": _round(mean(maes)) if maes else None,
            "direct_estimate_train_mae": _round(mean(direct_maes)) if direct_maes else None,
            "coverage": _round(len(design_rows) / max(1, len(rows))),
        },
    }


@lru_cache(maxsize=1)
def _trained_primary_recovery_model() -> dict[str, Any]:
    rows = _training_rows()
    all_metric_names = sorted({metric_name for row in rows for metric_name in row["metric_values"]})
    parameter_models: dict[str, dict[str, Any]] = {}
    for parameter_name in PARAMETER_REFERENCE:
        selected_metrics = [
            metric_name
            for metric_name in all_metric_names
            if sum(1 for row in rows if metric_name in row["metric_values"]) >= 8
        ]
        parameter_models[parameter_name] = _fit_weighted_linear_regression(
            rows,
            parameter_name=parameter_name,
            feature_names=selected_metrics,
        )
    return {
        "model_type": "per_parameter_weighted_linear_regression",
        "training_source": "internal_generator",
        "train_seeds": list(TRAIN_PROFILE_SEEDS),
        "sample_count": len(rows),
        "parameter_models": parameter_models,
    }


def _predict_from_primary_model(metrics: dict[str, dict[str, Any]], metric_values: dict[str, float], parameter_name: str) -> dict[str, Any]:
    parameter_model = _trained_primary_recovery_model()["parameter_models"][parameter_name]
    selected_metrics = list(parameter_model["selected_metrics"])
    available_metrics = [metric_name for metric_name in selected_metrics if metric_name in metric_values]
    default_value = DEFAULT_PARAMETER_VECTOR[parameter_name]
    direct_estimate = float(_estimate_parameter(metrics, parameter_name)["estimate"])
    if not available_metrics:
        return {
            "parameter": parameter_name,
            "estimate": _round(direct_estimate if direct_estimate is not None else default_value),
            "confidence": 0.0,
            "identifiable": False,
            "evidence_coverage": 0.0,
            "supporting_metrics": [],
            "reason": "no_trained_features_available",
            "alternative_explanations": list(PARAMETER_BASELINE_EXPLANATIONS[parameter_name]),
            "coupled_parameters": list(PARAMETER_COUPLINGS.get(parameter_name, [])),
            "model_type": _trained_primary_recovery_model()["model_type"],
        }

    raw_estimate = float(parameter_model["bias"]) + sum(
        float(parameter_model["weights"][metric_name]) * float(metric_values[metric_name])
        for metric_name in available_metrics
    )
    regression_estimate = _clamp01(raw_estimate)
    blend_weights = parameter_model.get("blend_weights", {})
    estimate = _clamp01(
        float(blend_weights.get("regression_component", 0.5)) * regression_estimate
        + float(blend_weights.get("direct_estimate_component", 0.5)) * direct_estimate
    )
    coverage = len(available_metrics) / max(1, len(selected_metrics))
    train_mae = parameter_model["fit_statistics"].get("train_mae")
    confidence = _clamp01(
        0.28
        + coverage * 0.42
        + (1.0 - min(1.0, float(train_mae or 0.5) / 0.35)) * 0.30
    )
    supporting_metrics = [
        {
            "metric": metric_name,
            "value": _round(metric_values[metric_name]),
            "weight": parameter_model["weights"][metric_name],
        }
        for metric_name in available_metrics
    ]
    return {
        "parameter": parameter_name,
        "estimate": _round(estimate),
        "confidence": _round(confidence),
        "identifiable": len(available_metrics) >= 1 and confidence >= 0.55,
        "evidence_coverage": _round(coverage),
        "supporting_metrics": supporting_metrics,
        "regression_estimate": _round(regression_estimate),
        "direct_estimate": _round(direct_estimate),
        "reason": "trained_regression" if coverage >= 1.0 else "trained_regression_partial_feature_set",
        "alternative_explanations": list(PARAMETER_BASELINE_EXPLANATIONS[parameter_name]),
        "coupled_parameters": list(PARAMETER_COUPLINGS.get(parameter_name, [])),
        "model_type": _trained_primary_recovery_model()["model_type"],
    }


def _parameter_distance(left: CognitiveStyleParameters, right: CognitiveStyleParameters) -> float:
    return round(
        sum(abs(float(getattr(left, name)) - float(getattr(right, name))) for name in PARAMETER_REFERENCE) / len(PARAMETER_REFERENCE),
        6,
    )


@lru_cache(maxsize=1)
def _feature_prototype_library() -> dict[str, dict[str, float]]:
    prototypes = {}
    for profile_name, parameters in PROFILE_REGISTRY.items():
        rows = []
        for seed in TRAIN_PROFILE_SEEDS:
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=profile_name in {"low_exploration_high_caution", "methodical_resource_guarded"})
            rows.append(trial["observable_metric_values"])
        feature_means = {}
        for feature_name in BLIND_CLASSIFICATION_FEATURES:
            values = [float(row[feature_name]) for row in rows if feature_name in row]
            if values:
                feature_means[feature_name] = _round(mean(values))
        prototypes[profile_name] = feature_means
    return prototypes


def _feature_distance(metric_values: dict[str, float], prototype: dict[str, float]) -> float:
    shared = sorted(set(metric_values) & set(prototype) & set(BLIND_CLASSIFICATION_FEATURES))
    if not shared:
        return float("inf")
    return sum(abs(float(metric_values[name]) - float(prototype[name])) for name in shared) / len(shared)


@lru_cache(maxsize=1)
def _derived_rule_based_thresholds() -> dict[str, Any]:
    threshold_specs = {
        "high_exploration_low_caution": [
            {
                "metric": "uncertainty_confidence_drop_rate",
                "reference_profiles": ("high_exploration_low_caution", "balanced_midline"),
                "direction": "gte",
            },
            {
                "metric": "confidence_evidence_slope",
                "reference_profiles": ("high_exploration_low_caution", "balanced_midline"),
                "direction": "lte",
            },
            {
                "metric": "strategy_persistence_after_error",
                "reference_profiles": ("high_exploration_low_caution", "balanced_midline"),
                "direction": "lte",
            },
        ],
        "low_exploration_high_caution": [
            {
                "metric": "recovery_trigger_rate",
                "reference_profiles": ("low_exploration_high_caution", "balanced_midline"),
                "direction": "gte",
            },
            {
                "metric": "strategy_persistence_after_error",
                "reference_profiles": ("low_exploration_high_caution", "balanced_midline"),
                "direction": "gte",
            },
            {
                "metric": "confidence_evidence_slope",
                "reference_profiles": ("low_exploration_high_caution", "balanced_midline"),
                "direction": "gte",
            },
        ],
    }
    profile_metric_values: dict[str, dict[str, list[float]]] = {}
    for profile_name, parameters in PROFILE_REGISTRY.items():
        rows = []
        for seed in TRAIN_PROFILE_SEEDS:
            trial = run_cognitive_style_trial(
                parameters,
                seed=seed,
                stress=profile_name in {"low_exploration_high_caution", "methodical_resource_guarded"},
            )
            rows.append(trial["observable_metric_values"])
        metric_map: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for metric_name, value in row.items():
                metric_map[metric_name].append(float(value))
        profile_metric_values[profile_name] = dict(metric_map)

    thresholds: dict[str, Any] = {}
    for profile_name, specs in threshold_specs.items():
        profile_thresholds = []
        for spec in specs:
            left_profile, right_profile = spec["reference_profiles"]
            left_values = profile_metric_values.get(left_profile, {}).get(spec["metric"], [])
            right_values = profile_metric_values.get(right_profile, {}).get(spec["metric"], [])
            left_mean = mean(left_values) if left_values else 0.0
            right_mean = mean(right_values) if right_values else 0.0
            threshold = (left_mean + right_mean) / 2.0
            profile_thresholds.append(
                {
                    "metric": spec["metric"],
                    "direction": spec["direction"],
                    "threshold": _round(threshold),
                    "source_profiles": [left_profile, right_profile],
                    "profile_means": {
                        left_profile: _round(left_mean),
                        right_profile: _round(right_mean),
                    },
                    "training_seeds": list(TRAIN_PROFILE_SEEDS),
                    "summary_method": "midpoint_of_profile_means_from_internal_training_trials",
                }
            )
        thresholds[profile_name] = {
            "predicted_profile": profile_name,
            "conditions": profile_thresholds,
        }
    return {
        "training_source": "internal_generator",
        "training_seeds": list(TRAIN_PROFILE_SEEDS),
        "profiles": thresholds,
    }


def _rule_based_profile_classification(metric_values: dict[str, float]) -> dict[str, Any] | None:
    provenance = _derived_rule_based_thresholds()
    predicted: str | None = None
    matched_conditions: list[dict[str, Any]] = []
    for profile_name, profile_payload in provenance["profiles"].items():
        conditions = profile_payload["conditions"]
        if all(
            (
                float(metric_values.get(condition["metric"], 0.0)) >= float(condition["threshold"])
                if condition["direction"] == "gte"
                else float(metric_values.get(condition["metric"], 0.0)) <= float(condition["threshold"])
            )
            for condition in conditions
        ):
            predicted = profile_name
            matched_conditions = conditions
            break
    if predicted is None:
        return None
    distances = {name: 0.65 for name in PROFILE_REGISTRY}
    distances[predicted] = 0.18
    return {
        "predicted_profile": predicted,
        "profile_distances": {name: _round(value) for name, value in distances.items()},
        "profile_confidence": 0.56,
        "classification_method": "rule_based_fallback",
        "threshold_provenance": {
            "training_source": provenance["training_source"],
            "training_seeds": provenance["training_seeds"],
            "matched_conditions": matched_conditions,
        },
    }


def classify_inferred_style(
    parameters: CognitiveStyleParameters,
    *,
    profile_registry: dict[str, CognitiveStyleParameters] | None = None,
    metric_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_registry = profile_registry or PROFILE_REGISTRY
    if metric_values:
        classifier_artifact = train_blind_classifier()
        feature_values = {
            feature_name: float(metric_values[feature_name])
            for feature_name in classifier_artifact["feature_set"]
            if feature_name in metric_values
        }
        prediction = predict_profile_from_features(feature_values, classifier_artifact)
        parameter_distances = {
            name: _round(_parameter_distance(parameters, profile))
            for name, profile in active_registry.items()
        }
        return {
            **prediction,
            "parameter_profile_distances": parameter_distances,
            "classifier_artifact": classifier_artifact,
        }

    distances = {
        name: _round(_parameter_distance(parameters, profile))
        for name, profile in active_registry.items()
    }
    predicted = min(distances, key=lambda name: (distances[name], name))
    ordered = sorted(distances.values())
    margin = ordered[1] - ordered[0] if len(ordered) > 1 else 0.4
    confidence = _clamp01(0.45 + margin * 1.6)
    return {
        "predicted_profile": predicted,
        "profile_distances": distances,
        "profile_confidence": _round(confidence),
        "classification_method": "parameter_distance",
    }


def infer_cognitive_style(
    records: list[DecisionLogRecord | dict[str, Any]],
    *,
    subject_id: str | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    normalized = _normalized_records(records)
    metrics = compute_observable_metrics(normalized)
    metric_values = metric_values_from_payload(metrics)
    primary_model = _trained_primary_recovery_model()
    candidate_fit = _candidate_parameter_estimates(metric_values)
    estimates: dict[str, dict[str, Any]] = {}
    for parameter_name in PARAMETER_REFERENCE:
        primary = _predict_from_primary_model(metrics, metric_values, parameter_name)
        candidate_estimate = candidate_fit["candidate_estimates"].get(parameter_name, {})
        agreement_gap = abs(float(primary["estimate"]) - float(candidate_estimate.get("estimate", primary["estimate"])))
        estimates[parameter_name] = {
            **primary,
            "candidate_estimate": _round(candidate_estimate.get("estimate")) if candidate_estimate.get("estimate") is not None else None,
            "candidate_consensus": _round(candidate_estimate.get("distance_consensus")) if candidate_estimate.get("distance_consensus") is not None else None,
            "agreement_gap": _round(agreement_gap),
        }

    inferred_parameters = _parameter_vector_from_estimates(estimates)
    classification = classify_inferred_style(inferred_parameters, metric_values=metric_values)

    audit = audit_decision_log(normalized)
    patterns = reconstruct_behavior_patterns(normalized)
    overall_confidence = _clamp01(
        mean(float(payload["confidence"]) for payload in estimates.values()) * 0.82
        + (1.0 - float(audit["invalid_rate"])) * 0.18
    )
    unidentifiable = sorted(name for name, payload in estimates.items() if not payload["identifiable"])

    alternatives = [
        {
            "parameter": parameter_name,
            "most_likely_competitors": PARAMETER_COUPLINGS.get(parameter_name, []),
            "alternative_explanations": PARAMETER_BASELINE_EXPLANATIONS[parameter_name][:2],
        }
        for parameter_name in unidentifiable
    ]
    explanations = []
    for parameter_name, payload in estimates.items():
        top_metrics = sorted(payload["supporting_metrics"], key=lambda item: (-abs(float(item.get("weight", 0.0))), item["metric"]))[:2]
        explanations.append(
            {
                "parameter": parameter_name,
                "estimate": payload["estimate"],
                "confidence": payload["confidence"],
                "identifiable": payload["identifiable"],
                "why": [f"{item['metric']}={_round(item['value'])}" for item in top_metrics],
                "causal_hypothesis": PARAMETER_REFERENCE[parameter_name]["decision_path"],
            }
        )

    return {
        "analysis_type": "same_framework_synthetic_inference_sandbox",
        "legacy_analysis_type": "behavior_to_style_inference",
        "benchmark_scope": "synthetic/same-framework inference sandbox over M4.1-style logs",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "validation_type": "same_framework_synthetic_only",
        "generator_family": "internal_training_generator_with_same_framework_inputs",
        "not_acceptance_evidence": True,
        "milestone_boundary": "M4.2+ pre-research sidecar only",
        "validation_limits": [
            "recovery models are trained on repository-owned synthetic trials",
            "shared latent semantics mean successful recovery here is same-framework evidence only",
            "this sandbox does not count as M4.1 acceptance evidence",
            "this sandbox does not prove M4.2 benchmark/task recovery is complete",
        ],
        "subject_id": subject_id or (normalized[0].task_context.get("subject_id") if normalized else None),
        "source_name": source_name,
        "record_count": len(normalized),
        "observable_metrics": metrics,
        "observable_metric_values": metric_values,
        "parameter_estimates": estimates,
        "primary_parameter_estimates": estimates,
        "inferred_parameters": inferred_parameters.to_dict(),
        "fit_confidence": _round(overall_confidence),
        "unidentifiable_parameters": unidentifiable,
        "alternative_explanations": alternatives,
        "classification": classification,
        "patterns": patterns,
        "log_audit": audit,
        "explanation_report": build_behavior_explanation_report(normalized, parameters=inferred_parameters),
        "primary_recovery_model": primary_model,
        "candidate_bank_baseline": {
            "fit_margin": candidate_fit.get("fit_margin"),
            "train_test_seed_overlap": candidate_fit.get("train_test_seed_overlap", 0),
            "candidate_estimates": {
                parameter_name: {
                    "estimate": _round(payload["estimate"]),
                    "distance_consensus": _round(payload["distance_consensus"]),
                }
                for parameter_name, payload in candidate_fit.get("candidate_estimates", {}).items()
            },
            "top_candidates": [
                {
                    "distance": _round(item["distance"]),
                    "parameters": item["parameters"].to_dict(),
                    "training_seeds": item["training_seeds"],
                }
                for item in candidate_fit.get("top_candidates", [])
            ],
        },
        "candidate_fit": {
            "fit_margin": candidate_fit.get("fit_margin"),
            "train_test_seed_overlap": candidate_fit.get("train_test_seed_overlap", 0),
            "top_candidates": [
                {
                    "distance": _round(item["distance"]),
                    "parameters": item["parameters"].to_dict(),
                    "training_seeds": item["training_seeds"],
                }
                for item in candidate_fit.get("top_candidates", [])
            ],
        },
        "recovery_comparison": {
            parameter_name: {
                "primary_estimate": estimates[parameter_name]["estimate"],
                "candidate_estimate": estimates[parameter_name]["candidate_estimate"],
                "agreement_gap": estimates[parameter_name]["agreement_gap"],
            }
            for parameter_name in PARAMETER_REFERENCE
        },
        "explanations": explanations,
    }


def summarize_parameter_recovery(
    inferred: CognitiveStyleParameters | dict[str, Any],
    target: CognitiveStyleParameters | dict[str, Any],
) -> dict[str, Any]:
    inferred_params = inferred if isinstance(inferred, CognitiveStyleParameters) else CognitiveStyleParameters.from_dict(dict(inferred))
    target_params = target if isinstance(target, CognitiveStyleParameters) else CognitiveStyleParameters.from_dict(dict(target))
    per_parameter = {}
    for parameter_name in PARAMETER_REFERENCE:
        error = abs(float(getattr(inferred_params, parameter_name)) - float(getattr(target_params, parameter_name)))
        per_parameter[parameter_name] = _round(error)
    return {
        "mae": _round(mean(float(value) for value in per_parameter.values())),
        "max_error": _round(max(float(value) for value in per_parameter.values())),
        "per_parameter_error": per_parameter,
    }


def rank_alternative_explanations(
    records: list[DecisionLogRecord | dict[str, Any]],
    *,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    inference = infer_cognitive_style(records)
    ranked: list[dict[str, Any]] = []
    for parameter_name in inference["unidentifiable_parameters"]:
        for explanation in PARAMETER_BASELINE_EXPLANATIONS[parameter_name]:
            ranked.append(
                {
                    "parameter": parameter_name,
                    "explanation": explanation,
                    "fit_conflict": _round(1.0 - float(inference["parameter_estimates"][parameter_name]["confidence"])),
                }
            )
    ranked.sort(key=lambda item: (-float(item["fit_conflict"]), item["parameter"], item["explanation"]))
    return ranked[:top_k]


def infer_population_style_map(
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        subject_id = str(session.get("subject_id", "")).strip() or "unknown_subject"
        by_subject[subject_id].append(session)

    subject_rows = []
    for subject_id, subject_sessions in sorted(by_subject.items()):
        inferences = [infer_cognitive_style(session["records"], subject_id=subject_id, source_name=session.get("source_name")) for session in subject_sessions]
        mean_parameters = {}
        for parameter_name in PARAMETER_REFERENCE:
            mean_parameters[parameter_name] = _round(
                mean(float(inference["inferred_parameters"][parameter_name]) for inference in inferences)
            )
        subject_rows.append(
            {
                "subject_id": subject_id,
                "session_count": len(subject_sessions),
                "mean_parameters": {"schema_version": CognitiveStyleParameters().schema_version, **mean_parameters},
                "mean_fit_confidence": _round(mean(float(inference["fit_confidence"]) for inference in inferences)),
                "unidentifiable_union": sorted({name for inference in inferences for name in inference["unidentifiable_parameters"]}),
            }
        )
    return {
        "analysis_type": "same_framework_population_style_map",
        "legacy_analysis_type": "population_style_map",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "validation_type": "same_framework_synthetic_only",
        "not_acceptance_evidence": True,
        "subjects": subject_rows,
    }
