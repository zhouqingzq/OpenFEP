from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Callable

from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE, PROFILE_REGISTRY, TRAIN_PROFILE_SEEDS, run_cognitive_style_trial
from .m41_baselines import BASELINE_INFERENCE_REGISTRY
from .m41_external_dataset import load_external_behavior_dataset
from .m41_inference import classify_inferred_style, infer_cognitive_style, summarize_parameter_recovery


InferenceFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def _round(value: float) -> float:
    return round(float(value), 6)


def _parameter_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_params = CognitiveStyleParameters.from_dict(dict(left))
    right_params = CognitiveStyleParameters.from_dict(dict(right))
    return _round(
        sum(abs(float(getattr(left_params, name)) - float(getattr(right_params, name))) for name in PARAMETER_REFERENCE)
        / len(PARAMETER_REFERENCE)
    )


def _profile_prototypes_from_internal_training(*, seeds: list[int] | None = None) -> dict[str, dict[str, Any]]:
    active_seeds = seeds or list(TRAIN_PROFILE_SEEDS)
    prototypes: dict[str, dict[str, Any]] = {}
    for profile_name, parameters in PROFILE_REGISTRY.items():
        samples = []
        for seed in active_seeds:
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=profile_name == "low_exploration_high_caution")
            samples.append(infer_cognitive_style(trial["logs"])["inferred_parameters"])
        averaged = {"schema_version": CognitiveStyleParameters().schema_version}
        for parameter_name in PARAMETER_REFERENCE:
            averaged[parameter_name] = _round(mean(float(sample[parameter_name]) for sample in samples))
        prototypes[profile_name] = averaged
    return prototypes


def _ece(rows: list[dict[str, Any]], *, bins: int = 5) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    error = 0.0
    for bucket in range(bins):
        low = bucket / bins
        high = (bucket + 1) / bins
        members = [row for row in rows if low <= float(row["confidence"]) <= high or (bucket == bins - 1 and float(row["confidence"]) == 1.0)]
        if not members:
            continue
        accuracy = mean(1.0 if row["correct"] else 0.0 for row in members)
        confidence = mean(float(row["confidence"]) for row in members)
        error += (len(members) / total) * abs(accuracy - confidence)
    return _round(error)


def _stability_report(session_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[row["subject_id"]].append(row)
    subject_drifts: dict[str, float] = {}
    for subject_id, subject_rows in grouped.items():
        if len(subject_rows) < 2:
            continue
        distances = []
        for index, left in enumerate(subject_rows):
            for right in subject_rows[index + 1 :]:
                distances.append(_parameter_distance(left["inferred_parameters"], right["inferred_parameters"]))
        subject_drifts[subject_id] = _round(mean(distances)) if distances else 0.0
    return {
        "subject_count": len(grouped),
        "subjects_with_repeat_sessions": len(subject_drifts),
        "within_subject_parameter_distance": subject_drifts,
        "mean_within_subject_distance": _round(mean(subject_drifts.values())) if subject_drifts else 0.0,
    }


def _evaluate_model_on_sessions(
    sessions: list[dict[str, Any]],
    *,
    inference_fn: InferenceFn,
    prototypes: dict[str, dict[str, Any]],
    model_label: str,
) -> dict[str, Any]:
    rows = []
    classification_rows = []
    failures = []
    for session in sessions:
        inference = inference_fn(session["records"])
        inferred_parameters = dict(inference["inferred_parameters"])
        predicted_profile = dict(inference.get("classification", {}))
        if not predicted_profile:
            predicted_profile = classify_inferred_style(
                CognitiveStyleParameters.from_dict(inferred_parameters),
                profile_registry={name: CognitiveStyleParameters.from_dict(payload) for name, payload in prototypes.items()},
            )
        recovery = summarize_parameter_recovery(inferred_parameters, session["ground_truth_parameters"]) if session.get("ground_truth_parameters") else None
        correct = predicted_profile["predicted_profile"] == session.get("profile_label")
        row = {
            "model_label": model_label,
            "subject_id": session["subject_id"],
            "session_id": session["session_id"],
            "source_name": session["source_name"],
            "task_name": session["task_name"],
            "true_profile": session.get("profile_label"),
            "predicted_profile": predicted_profile["predicted_profile"],
            "confidence": float(predicted_profile["profile_confidence"]),
            "correct": correct,
            "fit_confidence": float(inference.get("fit_confidence", predicted_profile["profile_confidence"])),
            "inferred_parameters": inferred_parameters,
            "parameter_recovery": recovery,
            "unidentifiable_parameters": list(inference.get("unidentifiable_parameters", [])),
        }
        rows.append(row)
        classification_rows.append({"correct": correct, "confidence": row["confidence"]})
        if (not correct) or float(row["fit_confidence"]) < 0.55 or row["unidentifiable_parameters"]:
            failures.append(
                {
                    "session_id": session["session_id"],
                    "subject_id": session["subject_id"],
                    "true_profile": session.get("profile_label"),
                    "predicted_profile": predicted_profile["predicted_profile"],
                    "fit_confidence": _round(row["fit_confidence"]),
                    "unidentifiable_parameters": row["unidentifiable_parameters"],
                    "most_likely_confusion": min(predicted_profile["profile_distances"], key=predicted_profile["profile_distances"].get),
                }
            )
    mae_values = [float(row["parameter_recovery"]["mae"]) for row in rows if row["parameter_recovery"]]
    return {
        "model_label": model_label,
        "session_count": len(rows),
        "sessions": rows,
        "classification_accuracy": _round(mean(1.0 if row["correct"] else 0.0 for row in rows)) if rows else 0.0,
        "calibration_error": _ece(classification_rows),
        "parameter_recovery_stability": {
            "mean_mae": _round(mean(mae_values)) if mae_values else None,
            "max_mae": _round(max(mae_values)) if mae_values else None,
        },
        "stability_report": _stability_report(rows),
        "failure_samples": failures,
    }


def run_cross_source_holdout_validation(
    *,
    inference_fn: InferenceFn = infer_cognitive_style,
    model_label: str = "style_inference_model",
) -> dict[str, Any]:
    dataset = load_external_behavior_dataset()
    heldout_sessions = [session for session in dataset["sessions"] if session.get("split", "heldout") == "heldout"]
    prototypes = _profile_prototypes_from_internal_training()
    evaluation = _evaluate_model_on_sessions(
        heldout_sessions,
        inference_fn=inference_fn,
        prototypes=prototypes,
        model_label=model_label,
    )
    return {
        "analysis_type": "cross_source_holdout_validation",
        "training_design": {
            "source": "internal_scenario_library",
            "tasks": "synthetic_m41_scenario_families",
            "profile_train_seeds": list(TRAIN_PROFILE_SEEDS),
            "prototypes": prototypes,
        },
        "heldout_design": {
            "source": sorted({session["source_name"] for session in heldout_sessions}),
            "tasks": sorted({session["task_name"] for session in heldout_sessions}),
            "dataset_root": dataset["dataset_root"],
        },
        "metrics": {
            "classification_accuracy": evaluation["classification_accuracy"],
            "calibration_error": evaluation["calibration_error"],
            "parameter_recovery_stability": evaluation["parameter_recovery_stability"],
            "stability_report": evaluation["stability_report"],
        },
        "failure_samples": evaluation["failure_samples"],
        "sessions": evaluation["sessions"],
    }


def compare_models_on_cross_source_holdout() -> dict[str, Any]:
    comparisons = []
    for model_label, inference_fn in BASELINE_INFERENCE_REGISTRY.items():
        comparisons.append(run_cross_source_holdout_validation(inference_fn=inference_fn, model_label=model_label))
    ranked = sorted(
        comparisons,
        key=lambda item: (
            -float(item["metrics"]["classification_accuracy"]),
            float(item["metrics"]["parameter_recovery_stability"]["mean_mae"] or 1.0),
            -float(item["metrics"]["stability_report"]["subjects_with_repeat_sessions"]),
        ),
    )
    return {"analysis_type": "baseline_comparison", "models": ranked}
