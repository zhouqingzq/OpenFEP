from __future__ import annotations

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
    audit_decision_log,
    compute_observable_metrics,
    metric_values_from_payload,
    observable_parameter_contracts,
    reconstruct_behavior_patterns,
    run_cognitive_style_trial,
)
from .m43_modeling import candidate_parameter_grid
from .m41_explanations import build_behavior_explanation_report


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


DEFAULT_PARAMETER_VECTOR = CognitiveStyleParameters().to_dict()
DEFAULT_PARAMETER_VECTOR.pop("schema_version", None)

PARAMETER_BASELINE_EXPLANATIONS: dict[str, list[str]] = {
    "uncertainty_sensitivity": [
        "confidence-only model: low certainty can reduce confidence without a stable style parameter",
        "task ambiguity effect: repeated high-uncertainty trials can mimic elevated uncertainty sensitivity",
    ],
    "error_aversion": [
        "risk/resource model: conservative choices may come from objective loss structure rather than enduring error aversion",
        "simple hazard heuristic: avoid-when-lossy policies can look like error aversion",
    ],
    "exploration_bias": [
        "novelty-by-task heuristic: the task may simply reward inspect-like actions",
        "resource search strategy: exploration can emerge from missing-state information rather than a persistent bias",
    ],
    "attention_selectivity": [
        "stimulus salience effect: dominant cues can mechanically concentrate attention",
        "task-format artifact: evidence alignment can rise because only one cue channel is available",
    ],
    "confidence_gain": [
        "confidence-only model: steep confidence scaling may not require a broader style profile",
        "signal quality effect: cleaner evidence alone can elevate commit confidence",
    ],
    "update_rigidity": [
        "simple RL learning-rate model: slower updates may reflect a low learning rate rather than style rigidity",
        "task stationarity effect: stable environments can make persistence look like rigidity",
    ],
    "resource_pressure_sensitivity": [
        "pure risk/resource model: conservation may be fully explained by low energy, time, or budget",
        "workload heuristic: external pressure spikes can produce recovery actions without stable style differences",
    ],
    "virtual_prediction_error_gain": [
        "confidence conflict model: imagined loss may simply lower confidence without requiring a dedicated counterfactual gain parameter",
        "task warning-channel artifact: explicit warning cues can mimic counterfactual sensitivity",
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
                "description": contract["description"],
                "formula": contract["formula"],
                "direction": contract["direction"],
            }
        )
    return supported


def _estimate_parameter(metrics: dict[str, dict[str, Any]], parameter_name: str) -> dict[str, Any]:
    supported = _metric_support(metrics, parameter_name)
    if not supported:
        return {
            "parameter": parameter_name,
            "estimate": DEFAULT_PARAMETER_VECTOR[parameter_name],
            "confidence": 0.15,
            "identifiable": False,
            "evidence_coverage": 0.0,
            "dispersion": None,
            "supporting_metrics": [],
            "reason": "no_executable_observables",
            "coupled_parameters": list(PARAMETER_COUPLINGS.get(parameter_name, [])),
            "alternative_explanations": list(PARAMETER_BASELINE_EXPLANATIONS[parameter_name]),
        }

    metric_values = [float(item["value"]) for item in supported]
    estimate = _clamp01(mean(metric_values))
    coverage = len(supported) / max(1, len(observable_parameter_contracts()[parameter_name]["observables"]))
    dispersion = pvariance(metric_values) if len(metric_values) > 1 else 0.0
    mean_samples = mean(float(item["sample_size"]) / max(1.0, float(item["min_samples"])) for item in supported)
    evidence_strength = _clamp01(mean_samples / 2.0)
    confidence = _clamp01(0.20 + coverage * 0.35 + (1.0 - min(1.0, dispersion * 4.0)) * 0.30 + evidence_strength * 0.15)
    identifiable = len(supported) >= 2 and confidence >= 0.55
    reason = "well_supported" if identifiable else "metric_confounding_or_sparse_support"
    return {
        "parameter": parameter_name,
        "estimate": _round(estimate),
        "confidence": _round(confidence),
        "identifiable": identifiable,
        "evidence_coverage": _round(coverage),
        "dispersion": _round(dispersion),
        "supporting_metrics": supported,
        "reason": reason,
        "coupled_parameters": list(PARAMETER_COUPLINGS.get(parameter_name, [])),
        "alternative_explanations": list(PARAMETER_BASELINE_EXPLANATIONS[parameter_name]),
    }


def _parameter_vector_from_estimates(estimates: dict[str, dict[str, Any]]) -> CognitiveStyleParameters:
    payload = {"schema_version": CognitiveStyleParameters().schema_version}
    for parameter_name in PARAMETER_REFERENCE:
        payload[parameter_name] = float(estimates[parameter_name]["estimate"])
    return CognitiveStyleParameters.from_dict(payload)


@lru_cache(maxsize=1)
def _candidate_library() -> list[dict[str, Any]]:
    unique: dict[tuple[float, ...], CognitiveStyleParameters] = {}
    for parameters in [*candidate_parameter_grid(), *PROFILE_REGISTRY.values()]:
        key = tuple(float(getattr(parameters, name)) for name in PARAMETER_REFERENCE)
        unique[key] = parameters

    library = []
    for index, parameters in enumerate(unique.values()):
        trial_metrics = []
        for seed in (41, 42):
            trial = run_cognitive_style_trial(parameters, seed=seed)
            trial_metrics.append(trial["observable_metric_values"])
        averaged = {}
        metric_names = sorted({name for row in trial_metrics for name in row})
        for metric_name in metric_names:
            averaged[metric_name] = _round(mean(float(row[metric_name]) for row in trial_metrics if metric_name in row))
        library.append({"candidate_id": index, "parameters": parameters, "metric_values": averaged})
    return library


def _metric_vector_distance(observed: dict[str, float], candidate: dict[str, float]) -> float:
    shared = sorted(set(observed) & set(candidate))
    if not shared:
        return float("inf")
    return sum(abs(float(observed[name]) - float(candidate[name])) for name in shared) / len(shared)


def _candidate_parameter_estimates(metric_values: dict[str, float]) -> dict[str, Any]:
    ranked = []
    for candidate in _candidate_library():
        distance = _metric_vector_distance(metric_values, candidate["metric_values"])
        ranked.append({**candidate, "distance": distance})
    ranked.sort(key=lambda item: (float(item["distance"]), item["candidate_id"]))
    top = ranked[:5]
    weights = [1.0 / max(0.02, float(item["distance"])) for item in top]
    total_weight = sum(weights) or 1.0

    estimates = {}
    for parameter_name in PARAMETER_REFERENCE:
        weighted_value = sum(
            float(getattr(item["parameters"], parameter_name)) * weight for item, weight in zip(top, weights)
        ) / total_weight
        consensus = pvariance([float(getattr(item["parameters"], parameter_name)) for item in top]) if len(top) > 1 else 0.0
        estimates[parameter_name] = {
            "estimate": _round(weighted_value),
            "distance_consensus": _round(consensus),
        }
    margin = (float(top[1]["distance"]) - float(top[0]["distance"])) if len(top) > 1 else 0.5
    return {"top_candidates": top, "candidate_estimates": estimates, "fit_margin": _round(margin)}


def _parameter_distance(left: CognitiveStyleParameters, right: CognitiveStyleParameters) -> float:
    return round(
        sum(abs(float(getattr(left, name)) - float(getattr(right, name))) for name in PARAMETER_REFERENCE) / len(PARAMETER_REFERENCE),
        6,
    )


def _profile_confidence(distance_map: dict[str, float]) -> float:
    ordered = sorted(distance_map.values())
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return 1.0
    margin = ordered[1] - ordered[0]
    return _clamp01(0.5 + margin * 1.5)


@lru_cache(maxsize=1)
def _feature_prototype_library() -> dict[str, dict[str, float]]:
    prototypes = {}
    for profile_name, parameters in PROFILE_REGISTRY.items():
        rows = []
        for seed in (11, 12, 13, 14):
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=profile_name == "low_exploration_high_caution")
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


def _rule_based_profile_classification(metric_values: dict[str, float]) -> dict[str, Any] | None:
    inspect_ratio = metric_values.get("high_uncertainty_inspect_ratio", 0.0)
    novelty = metric_values.get("novel_action_ratio", 0.0)
    risk_rejection = metric_values.get("high_expected_error_rejection_rate", 0.0)
    pressure = metric_values.get("high_pressure_low_cost_ratio", 0.0)
    conflict = metric_values.get("conflict_avoidance_shift", 0.0)
    commit_rate = metric_values.get("high_evidence_commit_rate", 0.0)
    confidence_slope = metric_values.get("confidence_evidence_slope", 0.0)
    repeat_suppression = metric_values.get("choice_repeat_suppression", 1.0)

    if inspect_ratio >= 0.85 and novelty >= 0.70 and commit_rate <= 0.60 and confidence_slope <= 0.30 and (pressure <= 0.40 or repeat_suppression <= 0.25):
        return {
            "predicted_profile": "high_exploration_low_caution",
            "profile_distances": {
                "high_exploration_low_caution": 0.08,
                "low_exploration_high_caution": 0.62,
                "balanced_midline": 0.34,
            },
            "profile_confidence": 0.81,
            "classification_method": "behavioral_rule_set",
        }
    if risk_rejection >= 0.85 and (pressure >= 0.70 or conflict >= 0.90):
        return {
            "predicted_profile": "low_exploration_high_caution",
            "profile_distances": {
                "high_exploration_low_caution": 0.58,
                "low_exploration_high_caution": 0.10,
                "balanced_midline": 0.30,
            },
            "profile_confidence": 0.80,
            "classification_method": "behavioral_rule_set",
        }
    if commit_rate >= 0.70 or confidence_slope >= 0.26:
        return {
            "predicted_profile": "balanced_midline",
            "profile_distances": {
                "high_exploration_low_caution": 0.36,
                "low_exploration_high_caution": 0.38,
                "balanced_midline": 0.12,
            },
            "profile_confidence": 0.74,
            "classification_method": "behavioral_rule_set",
        }
    return None


def classify_inferred_style(
    parameters: CognitiveStyleParameters,
    *,
    profile_registry: dict[str, CognitiveStyleParameters] | None = None,
    metric_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_registry = profile_registry or PROFILE_REGISTRY
    feature_prototypes = _feature_prototype_library() if metric_values else {}
    distances = {}
    for name, profile in active_registry.items():
        parameter_distance = _parameter_distance(parameters, profile)
        if metric_values and name in feature_prototypes:
            distances[name] = _round(parameter_distance * 0.35 + _feature_distance(metric_values, feature_prototypes[name]) * 0.65)
        else:
            distances[name] = parameter_distance
    predicted = min(distances, key=lambda name: (distances[name], name))
    return {
        "predicted_profile": predicted,
        "profile_distances": {name: _round(distance) for name, distance in distances.items()},
        "profile_confidence": _round(_profile_confidence(distances)),
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
    candidate_fit = _candidate_parameter_estimates(metric_values) if metric_values else {"candidate_estimates": {}, "top_candidates": [], "fit_margin": 0.0}
    estimates = {}
    for parameter_name in PARAMETER_REFERENCE:
        direct = _estimate_parameter(metrics, parameter_name)
        candidate_estimate = candidate_fit["candidate_estimates"].get(parameter_name, {})
        blended = _clamp01(float(direct["estimate"]) * 0.35 + float(candidate_estimate.get("estimate", direct["estimate"])) * 0.65)
        candidate_consensus = float(candidate_estimate.get("distance_consensus", 0.0))
        blended_confidence = _clamp01(
            float(direct["confidence"]) * 0.60
            + (1.0 - min(1.0, candidate_consensus * 3.0)) * 0.25
            + min(0.20, float(candidate_fit.get("fit_margin") or 0.0))
        )
        estimates[parameter_name] = {
            **direct,
            "estimate": _round(blended),
            "confidence": _round(blended_confidence),
            "candidate_consensus": _round(candidate_consensus),
            "identifiable": bool(direct["identifiable"] and blended_confidence >= 0.55),
        }
    inferred_parameters = _parameter_vector_from_estimates(estimates)
    classification = _rule_based_profile_classification(metric_values) or classify_inferred_style(inferred_parameters, metric_values=metric_values)
    audit = audit_decision_log(normalized)
    patterns = reconstruct_behavior_patterns(normalized)

    overall_confidence = _clamp01(
        mean(float(payload["confidence"]) for payload in estimates.values()) * 0.75
        + float(audit["parameter_snapshot_complete_rate"]) * 0.10
        + (1.0 - float(audit["invalid_rate"])) * 0.15
    )
    unidentifiable = sorted(name for name, payload in estimates.items() if not payload["identifiable"])

    alternatives = []
    for parameter_name in unidentifiable:
        alternatives.append(
            {
                "parameter": parameter_name,
                "most_likely_competitors": PARAMETER_COUPLINGS.get(parameter_name, []),
                "alternative_explanations": PARAMETER_BASELINE_EXPLANATIONS[parameter_name][:2],
            }
        )

    explanation_rows = []
    for parameter_name, payload in estimates.items():
        top_metrics = sorted(payload["supporting_metrics"], key=lambda item: (-float(item["sample_size"]), item["metric"]))[:2]
        explanation_rows.append(
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
        "analysis_type": "behavior_to_style_inference",
        "subject_id": subject_id or (normalized[0].task_context.get("subject_id") if normalized else None),
        "source_name": source_name,
        "record_count": len(normalized),
        "observable_metrics": metrics,
        "observable_metric_values": metric_values,
        "parameter_estimates": estimates,
        "inferred_parameters": inferred_parameters.to_dict(),
        "fit_confidence": _round(overall_confidence),
        "unidentifiable_parameters": unidentifiable,
        "alternative_explanations": alternatives,
        "classification": classification,
        "patterns": patterns,
        "log_audit": audit,
        "explanation_report": build_behavior_explanation_report(normalized, parameters=inferred_parameters),
        "candidate_fit": {
            "fit_margin": candidate_fit.get("fit_margin"),
            "top_candidates": [
                {"distance": _round(item["distance"]), "parameters": item["parameters"].to_dict()}
                for item in candidate_fit.get("top_candidates", [])
            ],
        },
        "explanations": explanation_rows,
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
    return {"analysis_type": "population_style_map", "subjects": subject_rows}
