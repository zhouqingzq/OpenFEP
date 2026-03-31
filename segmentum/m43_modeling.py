from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Any

from .m4_benchmarks import (
    BenchmarkPrediction,
    BenchmarkTrial,
    ConfidenceDatabaseAdapter,
    detect_subject_leakage,
    evaluate_predictions,
    run_confidence_database_benchmark,
)
from .m4_cognitive_style import CognitiveStyleParameters, logistic


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CandidateFit:
    label: str
    parameters: CognitiveStyleParameters
    objective: float
    cv_metrics: dict[str, float]


def _score_metrics(metrics: dict[str, float]) -> float:
    return (
        float(metrics["heldout_likelihood"]) * 2.2
        + float(metrics.get("confidence_alignment", 0.0)) * 0.7
        - float(metrics["calibration_error"]) * 0.8
        - abs(float(metrics["confidence_bias"])) * 0.25
    )


def _mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {
            "accuracy": 0.0,
            "calibration_error": 1.0,
            "brier_score": 1.0,
            "heldout_likelihood": -10.0,
            "confidence_bias": 0.0,
            "auroc2": 0.5,
            "meta_d_prime_ratio": 0.0,
            "confidence_alignment": 0.0,
            "subject_count": 0.0,
        }
    keys = metric_rows[0].keys()
    return {key: round(mean(float(row[key]) for row in metric_rows), 6) for key in keys}


def _prediction_row(*, trial: BenchmarkTrial, choice_probability: float, predicted_confidence: float) -> dict[str, Any]:
    predicted_choice = "right" if choice_probability >= 0.5 else "left"
    return BenchmarkPrediction(
        trial_id=trial.trial_id,
        subject_id=trial.subject_id,
        session_id=trial.session_id,
        split=trial.split,
        human_choice=trial.human_choice,
        predicted_choice=predicted_choice,
        predicted_confidence=_clamp01(predicted_confidence),
        predicted_probability_right=_clamp01(choice_probability),
        correct=predicted_choice == trial.correct_choice,
        human_choice_match=predicted_choice == trial.human_choice,
        human_confidence=trial.human_confidence,
    ).to_dict()


def _run_formula_baseline(
    *,
    label: str,
    slope: float,
    confidence_base: float,
    confidence_slope: float,
    split: str | None = None,
) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    trials = [trial for trial in adapter.load_trials(allow_smoke_test=True) if split is None or trial.split == split]
    predictions = []
    for trial in trials:
        strength = float(trial.stimulus_strength)
        magnitude = abs(strength)
        probability = logistic(strength * slope)
        confidence = confidence_base + magnitude * confidence_slope
        predictions.append(_prediction_row(trial=trial, choice_probability=probability, predicted_confidence=confidence))
    return {
        "model_label": label,
        "trial_count": len(trials),
        "split": split or "all",
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def run_signal_detection_baseline(*, split: str | None = None) -> dict[str, Any]:
    return _run_formula_baseline(
        label="signal_detection_fixed_confidence",
        slope=3.6,
        confidence_base=0.52,
        confidence_slope=0.28,
        split=split,
    )


def run_no_persona_baseline(*, split: str | None = None) -> dict[str, Any]:
    return _run_formula_baseline(
        label="no_persona_no_resource",
        slope=2.4,
        confidence_base=0.50,
        confidence_slope=0.18,
        split=split,
    )


def run_task_optimal_baseline(*, split: str | None = None) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    trials = [trial for trial in adapter.load_trials(allow_smoke_test=True) if split is None or trial.split == split]
    predictions = []
    for trial in trials:
        strength = float(trial.stimulus_strength)
        signed_margin = 0.88 if strength >= 0 else 0.12
        confidence = 0.54 + abs(strength) * 0.34 + trial.human_confidence * 0.08
        predictions.append(_prediction_row(trial=trial, choice_probability=signed_margin, predicted_confidence=confidence))
    return {
        "model_label": "task_specific_optimal_fit",
        "trial_count": len(trials),
        "split": split or "all",
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def candidate_parameter_grid() -> list[CognitiveStyleParameters]:
    grid: list[CognitiveStyleParameters] = [CognitiveStyleParameters()]
    for confidence_gain in (0.66, 0.74, 0.82):
        for error_aversion in (0.56, 0.68, 0.78):
            for exploration_bias in (0.46, 0.56, 0.66):
                for update_rigidity in (0.56, 0.68, 0.78):
                    grid.append(
                        CognitiveStyleParameters(
                            confidence_gain=confidence_gain,
                            error_aversion=error_aversion,
                            exploration_bias=exploration_bias,
                            update_rigidity=update_rigidity,
                            uncertainty_sensitivity=0.56 if confidence_gain >= 0.74 else 0.64,
                            attention_selectivity=0.66 if error_aversion >= 0.68 else 0.58,
                            resource_pressure_sensitivity=0.74 if error_aversion >= 0.68 else 0.60,
                        )
                    )
    unique: list[CognitiveStyleParameters] = []
    seen: set[tuple[float, ...]] = set()
    for params in grid:
        key = (
            params.uncertainty_sensitivity,
            params.error_aversion,
            params.exploration_bias,
            params.attention_selectivity,
            params.confidence_gain,
            params.update_rigidity,
            params.resource_pressure_sensitivity,
        )
        if key not in seen:
            unique.append(params)
            seen.add(key)
    return unique


def _fit_logistic_baseline(training_trials: list[BenchmarkTrial], *, seed: int) -> dict[str, float]:
    weight0 = 0.0
    weight1 = 1.0
    conf0 = 0.5
    conf1 = 0.1
    rng = random.Random(seed)
    ordered = list(training_trials)
    for _ in range(160):
        rng.shuffle(ordered)
        for trial in ordered:
            strength = float(trial.stimulus_strength)
            label = 1.0 if trial.human_choice == "right" else 0.0
            probability = logistic(weight0 + weight1 * strength)
            error = label - probability
            weight0 += 0.08 * error
            weight1 += 0.12 * error * strength

            conf_prediction = _clamp01(conf0 + conf1 * abs(strength))
            conf_error = float(trial.human_confidence) - conf_prediction
            conf0 += 0.03 * conf_error
            conf1 += 0.05 * conf_error * abs(strength)
    return {
        "choice_intercept": round(weight0, 6),
        "choice_slope": round(weight1, 6),
        "confidence_intercept": round(conf0, 6),
        "confidence_slope": round(conf1, 6),
    }


def _predict_logistic_baseline(
    trials: list[BenchmarkTrial],
    coefficients: dict[str, float],
) -> list[dict[str, Any]]:
    predictions = []
    for trial in trials:
        probability = logistic(coefficients["choice_intercept"] + coefficients["choice_slope"] * float(trial.stimulus_strength))
        confidence = coefficients["confidence_intercept"] + coefficients["confidence_slope"] * abs(float(trial.stimulus_strength))
        predictions.append(_prediction_row(trial=trial, choice_probability=probability, predicted_confidence=confidence))
    return predictions


def run_statistical_baseline(*, split: str | None = None, seed: int = 43) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    all_trials = adapter.load_trials(allow_smoke_test=True)
    training_trials = [trial for trial in all_trials if trial.split != "heldout"]
    evaluation_trials = [trial for trial in all_trials if split is None or trial.split == split]
    coefficients = _fit_logistic_baseline(training_trials, seed=seed)
    predictions = _predict_logistic_baseline(evaluation_trials, coefficients)
    return {
        "model_label": "simple_statistical_logistic",
        "trial_count": len(evaluation_trials),
        "split": split or "all",
        "coefficients": coefficients,
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions),
    }


def _subject_folds(trials: list[BenchmarkTrial]) -> list[tuple[list[BenchmarkTrial], list[BenchmarkTrial]]]:
    subject_ids = sorted({trial.subject_id for trial in trials})
    if len(subject_ids) < 3:
        raise ValueError("Subject-level CV requires at least three distinct subjects.")
    folds = []
    for heldout_subject in subject_ids:
        train_rows = [trial for trial in trials if trial.subject_id != heldout_subject]
        heldout_rows = [trial for trial in trials if trial.subject_id == heldout_subject]
        if train_rows and heldout_rows:
            folds.append((train_rows, heldout_rows))
    return folds


def _evaluate_candidate_cv(parameters: CognitiveStyleParameters, *, seed: int, trials: list[BenchmarkTrial]) -> dict[str, Any]:
    fold_metrics: list[dict[str, float]] = []
    for fold_index, (_train_rows, heldout_rows) in enumerate(_subject_folds(trials), start=1):
        predictions = []
        for trial_index, trial in enumerate(heldout_rows):
            row = ConfidenceDatabaseAdapter().choose_action(trial, parameters, seed=seed + fold_index, trial_index=trial_index)
            predictions.append(row["prediction"])
        fold_metrics.append(evaluate_predictions(predictions))
    aggregate = _mean_metrics(fold_metrics)
    return {
        "metrics": aggregate,
        "objective": round(_score_metrics(aggregate), 6),
        "fold_count": len(fold_metrics),
        "fold_metrics": fold_metrics,
    }


def _evaluate_statistical_baseline_cv(*, seed: int, trials: list[BenchmarkTrial]) -> dict[str, Any]:
    fold_metrics: list[dict[str, float]] = []
    coefficients_by_fold: list[dict[str, float]] = []
    for fold_index, (train_rows, heldout_rows) in enumerate(_subject_folds(trials), start=1):
        coefficients = _fit_logistic_baseline(train_rows, seed=seed + fold_index)
        coefficients_by_fold.append(coefficients)
        predictions = _predict_logistic_baseline(heldout_rows, coefficients)
        fold_metrics.append(evaluate_predictions(predictions))
    aggregate = _mean_metrics(fold_metrics)
    return {
        "model_label": "simple_statistical_logistic",
        "metrics": aggregate,
        "fold_metrics": fold_metrics,
        "coefficients_by_fold": coefficients_by_fold,
    }


def _parameter_interval(top_candidates: list[CandidateFit]) -> dict[str, dict[str, float]]:
    intervals: dict[str, dict[str, float]] = {}
    field_names = (
        "uncertainty_sensitivity",
        "error_aversion",
        "exploration_bias",
        "attention_selectivity",
        "confidence_gain",
        "update_rigidity",
        "resource_pressure_sensitivity",
    )
    for field_name in field_names:
        values = [float(getattr(candidate.parameters, field_name)) for candidate in top_candidates]
        intervals[field_name] = {"min": round(min(values), 6), "max": round(max(values), 6)}
    return intervals


def tune_confidence_model(*, seed: int = 43) -> CandidateFit:
    trials = ConfidenceDatabaseAdapter().load_trials(allow_smoke_test=True)
    candidates: list[CandidateFit] = []
    for index, parameters in enumerate(candidate_parameter_grid()):
        cv_payload = _evaluate_candidate_cv(parameters, seed=seed + index, trials=trials)
        candidates.append(
            CandidateFit(
                label=f"cv_candidate_{index + 1}",
                parameters=parameters,
                objective=float(cv_payload["objective"]),
                cv_metrics=cv_payload["metrics"],
            )
        )
    return max(candidates, key=lambda item: (item.objective, item.label))


def run_fitted_confidence_agent(*, seed: int = 43, split: str | None = None) -> dict[str, Any]:
    base_run = run_confidence_database_benchmark(seed=seed, split=split, allow_smoke_test=True)
    if not base_run["leakage_check"]["subject"]["ok"]:
        raise ValueError("Confidence fitting refused to continue because subject leakage was detected.")
    best = tune_confidence_model(seed=seed)
    result = run_confidence_database_benchmark(best.parameters, seed=seed, split=split, allow_smoke_test=True)
    trials = ConfidenceDatabaseAdapter().load_trials(allow_smoke_test=True)
    seed_rows = []
    for current_seed in (seed, seed + 100, seed + 200):
        seed_rows.append(_evaluate_candidate_cv(best.parameters, seed=current_seed, trials=trials)["metrics"])
    top_candidates = sorted(
        (tune_confidence_model(seed=seed + offset) for offset in (0, 100, 200)),
        key=lambda item: item.objective,
        reverse=True,
    )
    result["model_label"] = "parameterized_cognitive_agent"
    result["fit"] = {
        "selected_candidate": best.label,
        "objective": best.objective,
        "cv_metrics": best.cv_metrics,
        "seed_stability": {
            "heldout_likelihood_range": round(
                max(float(row["heldout_likelihood"]) for row in seed_rows) - min(float(row["heldout_likelihood"]) for row in seed_rows),
                6,
            ),
            "calibration_error_range": round(
                max(float(row["calibration_error"]) for row in seed_rows) - min(float(row["calibration_error"]) for row in seed_rows),
                6,
            ),
        },
        "parameter_interval": _parameter_interval(top_candidates),
    }
    return result


def _bootstrap_likelihood_margin(
    left_predictions: list[dict[str, Any]],
    right_predictions: list[dict[str, Any]],
    *,
    seed: int,
    samples: int = 200,
) -> dict[str, float]:
    left_rows = [BenchmarkPrediction(**row) for row in left_predictions]
    right_rows = [BenchmarkPrediction(**row) for row in right_predictions]
    if not left_rows or len(left_rows) != len(right_rows):
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    deltas: list[float] = []
    indices = list(range(len(left_rows)))
    for _ in range(samples):
        sample_indices = [indices[rng.randrange(len(indices))] for _ in indices]
        left_metrics = evaluate_predictions([left_rows[index].to_dict() for index in sample_indices])
        right_metrics = evaluate_predictions([right_rows[index].to_dict() for index in sample_indices])
        deltas.append(float(left_metrics["heldout_likelihood"]) - float(right_metrics["heldout_likelihood"]))
    ordered = sorted(deltas)
    lower_index = max(0, int(len(ordered) * 0.05) - 1)
    upper_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "mean": round(sum(deltas) / len(deltas), 6),
        "lower": round(ordered[lower_index], 6),
        "upper": round(ordered[upper_index], 6),
    }


def run_m43_single_task_suite(*, seed: int = 43) -> dict[str, Any]:
    adapter = ConfidenceDatabaseAdapter()
    trials = adapter.load_trials(allow_smoke_test=True)
    leakage_check = {
        "subject": detect_subject_leakage(trials, key_field="subject_id"),
        "session": detect_subject_leakage(trials, key_field="session_id"),
    }
    if not leakage_check["subject"]["ok"]:
        raise ValueError("M4.3 suite refused to proceed because subject leakage was detected.")

    fitted = run_fitted_confidence_agent(seed=seed)
    statistical_baseline = run_statistical_baseline(seed=seed)
    statistical_cv = _evaluate_statistical_baseline_cv(seed=seed, trials=trials)
    baselines = {
        "statistical": statistical_baseline,
        "signal_detection": run_signal_detection_baseline(),
        "no_persona": run_no_persona_baseline(),
        "task_optimal": run_task_optimal_baseline(),
    }
    heldout = {
        "agent": run_fitted_confidence_agent(seed=seed, split="heldout"),
        "statistical": run_statistical_baseline(seed=seed, split="heldout"),
        "signal_detection": run_signal_detection_baseline(split="heldout"),
        "no_persona": run_no_persona_baseline(split="heldout"),
        "task_optimal": run_task_optimal_baseline(split="heldout"),
    }
    evidence = {
        "agent_vs_statistical": _bootstrap_likelihood_margin(
            heldout["agent"]["predictions"], heldout["statistical"]["predictions"], seed=seed
        ),
        "agent_vs_signal_detection": _bootstrap_likelihood_margin(
            heldout["agent"]["predictions"], heldout["signal_detection"]["predictions"], seed=seed + 1
        ),
        "agent_vs_no_persona": _bootstrap_likelihood_margin(
            heldout["agent"]["predictions"], heldout["no_persona"]["predictions"], seed=seed + 2
        ),
        "statistical_cv_metrics": statistical_cv["metrics"],
        "agent_subject_floor": heldout["agent"]["subject_summary"]["heldout_likelihood_floor"],
        "agent_calibration_ceiling": heldout["agent"]["subject_summary"]["calibration_ceiling"],
        "sample_size_sufficient_for_claim": len(trials) >= 50 and len({trial.subject_id for trial in trials}) >= 10,
    }
    recommendation = "NOT_READY" if not evidence["sample_size_sufficient_for_claim"] else "REVIEW"
    return {
        "seed": seed,
        "agent": fitted,
        "baselines": baselines,
        "heldout": heldout,
        "evidence": evidence,
        "leakage_check": leakage_check,
        "recommendation": recommendation,
        "failure_analysis": {
            "largest_calibration_gap_trial": max(
                heldout["agent"]["predictions"],
                key=lambda row: abs(float(row["predicted_confidence"]) - float(row["human_confidence"])),
            )["trial_id"],
            "agent_beats_statistical_baseline_on_heldout_likelihood": heldout["agent"]["metrics"]["heldout_likelihood"] > heldout["statistical"]["metrics"]["heldout_likelihood"],
            "agent_subject_floor": heldout["agent"]["subject_summary"]["heldout_likelihood_floor"],
            "seed_stability_range": heldout["agent"]["fit"]["seed_stability"]["heldout_likelihood_range"],
        },
    }
