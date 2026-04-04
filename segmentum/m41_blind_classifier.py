from __future__ import annotations

"""Legacy same-framework synthetic profile-distinguishability sidecar.

This module is intentionally retained as a sandbox benchmark only. Its outputs
must not be used as M4.1 acceptance evidence, external validation, or proof
that M4.2 benchmark recovery is already complete. Both the profile semantics
and the feature pipeline remain anchored in the repository's synthetic
cognitive-style framework.
"""

import random
from statistics import mean
from typing import Any

from .m4_cognitive_style import BLIND_CLASSIFICATION_FEATURES, PROFILE_REGISTRY, TRAIN_PROFILE_SEEDS, EVAL_PROFILE_SEEDS, run_cognitive_style_trial


def _round(value: float) -> float:
    return round(float(value), 6)


def _metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric_name, payload in metrics.items():
        if isinstance(payload, dict) and not payload.get("insufficient_data", False) and payload.get("value") is not None:
            values[metric_name] = float(payload["value"])
        elif isinstance(payload, (int, float)):
            values[metric_name] = float(payload)
    return values


def feature_vector_from_metrics(metrics: dict[str, Any], feature_names: list[str] | None = None) -> dict[str, float]:
    metric_values = _metric_values(metrics)
    active_features = feature_names or BLIND_CLASSIFICATION_FEATURES
    return {name: float(metric_values[name]) for name in active_features if name in metric_values}


def _feature_rows(seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_name, parameters in PROFILE_REGISTRY.items():
        for seed in seeds:
            trial = run_cognitive_style_trial(
                parameters,
                seed=seed,
                stress=profile_name == "low_exploration_high_caution",
            )
            rows.append(
                {
                    "profile_name": profile_name,
                    "seed": seed,
                    "features": feature_vector_from_metrics(trial["observable_metrics"]),
                }
            )
    return rows


def train_blind_classifier(*, train_seeds: list[int] | None = None) -> dict[str, Any]:
    active_train = list(train_seeds or TRAIN_PROFILE_SEEDS)
    rows = _feature_rows(active_train)
    feature_scales: dict[str, dict[str, float]] = {}
    for feature_name in BLIND_CLASSIFICATION_FEATURES:
        values = [float(row["features"][feature_name]) for row in rows if feature_name in row["features"]]
        center = mean(values) if values else 0.0
        spread = max(1e-6, (max(values) - min(values)) if values else 1.0)
        feature_scales[feature_name] = {"center": _round(center), "scale": _round(spread)}

    class_centroids: dict[str, dict[str, float]] = {}
    for profile_name in PROFILE_REGISTRY:
        profile_rows = [row for row in rows if row["profile_name"] == profile_name]
        centroid: dict[str, float] = {}
        for feature_name in BLIND_CLASSIFICATION_FEATURES:
            values = [float(row["features"][feature_name]) for row in profile_rows if feature_name in row["features"]]
            if values:
                centroid[feature_name] = _round(mean(values))
        class_centroids[profile_name] = centroid

    feature_weights: dict[str, float] = {}
    for feature_name in BLIND_CLASSIFICATION_FEATURES:
        centroid_values = [
            float(class_centroids[profile_name][feature_name])
            for profile_name in PROFILE_REGISTRY
            if feature_name in class_centroids[profile_name]
        ]
        between = (max(centroid_values) - min(centroid_values)) if centroid_values else 0.0
        within_values = []
        for row in rows:
            profile_name = row["profile_name"]
            if feature_name in row["features"] and feature_name in class_centroids[profile_name]:
                within_values.append(abs(float(row["features"][feature_name]) - float(class_centroids[profile_name][feature_name])))
        within = mean(within_values) if within_values else 0.0
        feature_weights[feature_name] = _round(max(0.1, min(4.0, between / max(0.02, within))))

    return {
        "model_type": "nearest_centroid",
        "distance_metric": "mean_absolute_distance_on_scaled_features",
        "feature_set": list(BLIND_CLASSIFICATION_FEATURES),
        "train_seeds": active_train,
        "feature_scales": feature_scales,
        "feature_weights": feature_weights,
        "class_centroids": class_centroids,
        "training_summary": {
            "sample_count": len(rows),
            "profiles": sorted(PROFILE_REGISTRY),
            "samples_per_profile": {
                profile_name: sum(1 for row in rows if row["profile_name"] == profile_name)
                for profile_name in PROFILE_REGISTRY
            },
        },
    }


def predict_profile_from_features(features: dict[str, float], classifier_artifact: dict[str, Any]) -> dict[str, Any]:
    scales = classifier_artifact["feature_scales"]
    centroids = classifier_artifact["class_centroids"]
    active_features = [name for name in classifier_artifact["feature_set"] if name in features and name in scales]
    feature_weights = classifier_artifact.get("feature_weights", {})
    profile_distances: dict[str, float] = {}
    for profile_name, centroid in centroids.items():
        shared = [name for name in active_features if name in centroid]
        if not shared:
            profile_distances[profile_name] = float("inf")
            continue
        total_weight = sum(float(feature_weights.get(name, 1.0)) for name in shared) or 1.0
        distance = sum(
            (
                abs(float(features[name]) - float(centroid[name]))
                / max(1e-6, float(scales[name]["scale"]))
            ) * float(feature_weights.get(name, 1.0))
            for name in shared
        ) / total_weight
        profile_distances[profile_name] = _round(distance)
    predicted = min(profile_distances, key=lambda name: (profile_distances[name], name))
    ordered = sorted(value for value in profile_distances.values() if value != float("inf"))
    margin = (ordered[1] - ordered[0]) if len(ordered) > 1 else 0.35
    confidence = max(0.0, min(1.0, 0.45 + margin * 0.35))
    return {
        "predicted_profile": predicted,
        "profile_distances": profile_distances,
        "profile_confidence": _round(confidence),
        "classification_method": "trained_nearest_centroid",
        "features_used": active_features,
    }


def classify_profile_from_metrics(metrics: dict[str, Any], classifier_artifact: dict[str, Any] | None = None) -> str:
    artifact = classifier_artifact or train_blind_classifier()
    features = feature_vector_from_metrics(metrics, artifact["feature_set"])
    return predict_profile_from_features(features, artifact)["predicted_profile"]


def synthetic_profile_distinguishability_benchmark(
    *,
    train_seeds: list[int] | None = None,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    from .m41_external_generator import run_same_framework_holdout_trial
    from .m41_inference import infer_cognitive_style

    classifier_artifact = train_blind_classifier(train_seeds=train_seeds)
    active_eval = list(eval_seeds or EVAL_PROFILE_SEEDS)
    blind_samples: list[dict[str, Any]] = []
    evaluated_samples: list[dict[str, Any]] = []
    confusion_matrix = {
        profile_name: {other_name: 0 for other_name in PROFILE_REGISTRY}
        for profile_name in PROFILE_REGISTRY
    }

    for profile_name, parameters in PROFILE_REGISTRY.items():
        for seed in active_eval:
            trial = run_same_framework_holdout_trial(
                parameters,
                seed=seed,
                tick_count=50,
                scenario_family="synthetic_holdout_eval",
            )
            inference = infer_cognitive_style(trial["logs"])
            features = {
                name: float(inference["observable_metric_values"][name])
                for name in classifier_artifact["feature_set"]
                if name in inference["observable_metric_values"]
            }
            prediction = dict(inference["classification"])
            predicted_profile = prediction["predicted_profile"]
            confusion_matrix[profile_name][predicted_profile] += 1
            blind_samples.append(
                {
                    "sample_id": f"{profile_name}:{seed}",
                    "seed": seed,
                    "metrics": features,
                }
            )
            evaluated_samples.append(
                {
                    "seed": seed,
                    "true_profile": profile_name,
                    "predicted_profile": predicted_profile,
                    "metrics": features,
                    "classification_method": prediction["classification_method"],
                    "profile_distances": prediction["profile_distances"],
                    "features_used": prediction.get("features_used", list(features)),
                }
            )

    accuracy = _round(
        sum(1 for sample in evaluated_samples if sample["true_profile"] == sample["predicted_profile"]) / max(1, len(evaluated_samples))
    )
    per_class: dict[str, dict[str, float]] = {}
    for profile_name in PROFILE_REGISTRY:
        true_positive = confusion_matrix[profile_name][profile_name]
        predicted_positive = sum(confusion_matrix[other][profile_name] for other in PROFILE_REGISTRY)
        actual_positive = sum(confusion_matrix[profile_name].values())
        per_class[profile_name] = {
            "precision": _round(true_positive / predicted_positive) if predicted_positive else 0.0,
            "recall": _round(true_positive / actual_positive) if actual_positive else 0.0,
        }

    null_scores: list[float] = []
    true_profiles = [sample["true_profile"] for sample in evaluated_samples]
    rng = random.Random(41)
    for _ in range(60):
        permuted = list(true_profiles)
        rng.shuffle(permuted)
        null_scores.append(
            sum(1 for sample, label in zip(evaluated_samples, permuted) if label == sample["true_profile"]) / max(1, len(evaluated_samples))
        )
    null_mean = _round(sum(null_scores) / len(null_scores))
    null_std = _round((sum((score - null_mean) ** 2 for score in null_scores) / len(null_scores)) ** 0.5)

    return {
        "analysis_type": "synthetic_holdout_profile_distinguishability",
        "legacy_analysis_type": "cross_generator_blind_classification",
        "benchmark_scope": "sidecar same-framework profile distinguishability on synthetic holdout",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "generator_family": "same_framework_synthetic_holdout",
        "external_validation": False,
        "validation_type": "synthetic_holdout_same_framework",
        "not_acceptance_evidence": True,
        "milestone_boundary": "M4.2+ sidecar diagnostic only",
        "validation_limits": [
            "train data comes from the internal synthetic generator",
            "evaluation data comes from a synthetic holdout generator with shared latent profile semantics",
            "classifier parameters are derived from the training split only",
            "results should be interpreted only as synthetic profile distinguishability, not external validation",
            "this sidecar does not count as M4.1 acceptance evidence and does not prove M4.2 benchmark recovery on tasks",
        ],
        "profiles": {name: params.to_dict() for name, params in PROFILE_REGISTRY.items()},
        "train_eval_split": {"train_seeds": list(classifier_artifact["train_seeds"]), "eval_seeds": active_eval},
        "feature_set": list(classifier_artifact["feature_set"]),
        "classifier_artifact": classifier_artifact,
        "training_summary": classifier_artifact["training_summary"],
        "blind_samples": blind_samples,
        "samples": evaluated_samples,
        "sample_count": len(evaluated_samples),
        "accuracy": accuracy,
        "chance_baseline": {"mean_accuracy": null_mean, "std_accuracy": null_std, "scores": [_round(score) for score in null_scores]},
        "threshold": _round(null_mean + 2.0 * null_std),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "misclassified_samples": [
            {
                "seed": sample["seed"],
                "true_profile": sample["true_profile"],
                "predicted_profile": sample["predicted_profile"],
            }
            for sample in evaluated_samples
            if sample["true_profile"] != sample["predicted_profile"]
        ],
    }


def blind_classification_experiment(
    *,
    train_seeds: list[int] | None = None,
    eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    return synthetic_profile_distinguishability_benchmark(train_seeds=train_seeds, eval_seeds=eval_seeds)
