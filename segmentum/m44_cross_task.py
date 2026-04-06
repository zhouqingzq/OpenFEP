from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from .m43_modeling import (
    CONFIDENCE_EVAL_HELDOUT_MAX_TRIALS,
    CONFIDENCE_FIT_TRAIN_MAX_TRIALS,
    CONFIDENCE_FIT_VALIDATION_MAX_TRIALS,
    CONFIDENCE_MAX_SUBJECTS,
    _balanced_subject_sample,
    _load_confidence_data,
    _load_igt_data,
    _resolve_benchmark_root,
    _score_confidence_metrics,
    _score_igt_metrics,
    _set_parameter,
    _simulate_confidence_trials,
    _simulate_igt_trials,
    run_fitted_confidence_agent,
    run_fitted_igt_agent,
    run_parameter_sensitivity_analysis,
)
from .m4_benchmarks import _safe_round
from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE
from .m44_igt_aggregate import PHASE_WINDOWS, compute_igt_aggregate_metrics


DEFAULT_JOINT_WEIGHTS = {"w_conf": 1.0, "w_igt": 0.5, "w_igt_agg": 0.8}
WEIGHT_CONFIGS = {
    "default": DEFAULT_JOINT_WEIGHTS,
    "igt_heavy": {"w_conf": 0.5, "w_igt": 1.0, "w_igt_agg": 1.0},
    "confidence_heavy": {"w_conf": 1.0, "w_igt": 0.2, "w_igt_agg": 0.3},
}
JOINT_PARAMETER_NAMES = [
    "uncertainty_sensitivity",
    "error_aversion",
    "exploration_bias",
    "attention_selectivity",
    "confidence_gain",
    "update_rigidity",
    "virtual_prediction_error_gain",
]
TASK_SPECIFIC_THRESHOLD = 0.05
INERT_PARAMETER_NAMES = {"resource_pressure_sensitivity"}


@dataclass(frozen=True)
class JointCandidateFit:
    parameters: CognitiveStyleParameters
    objective: float
    confidence_metrics: dict[str, float]
    igt_metrics: dict[str, float]
    objective_components: dict[str, float]
    label: str


def _claim_envelope_for_sources(*, confidence_source_type: str, igt_source_type: str) -> str:
    return "benchmark_eval" if confidence_source_type == "external_bundle" and igt_source_type == "external_bundle" else "synthetic_diagnostic"


def _confidence_trials_by_split(
    *,
    benchmark_root: Path | str | None,
    allow_smoke_test: bool,
    sample_limits: dict[str, int] | None,
    seed: int,
) -> dict[str, Any]:
    limits = sample_limits or {}
    data = _load_confidence_data(benchmark_root=benchmark_root, allow_smoke_test=allow_smoke_test)
    return {
        "data": data,
        "train": _balanced_subject_sample(
            data["splits"].get("train", data["all_trials"]),
            max_trials=limits.get("confidence_train_max_trials", CONFIDENCE_FIT_TRAIN_MAX_TRIALS),
            max_subjects=limits.get("confidence_train_max_subjects", CONFIDENCE_MAX_SUBJECTS),
            seed=seed,
        ),
        "validation": _balanced_subject_sample(
            data["splits"].get("validation", data["all_trials"]),
            max_trials=limits.get("confidence_validation_max_trials", CONFIDENCE_FIT_VALIDATION_MAX_TRIALS),
            max_subjects=limits.get("confidence_validation_max_subjects", CONFIDENCE_MAX_SUBJECTS),
            seed=seed + 1,
        ),
        "heldout": _balanced_subject_sample(
            data["splits"].get("heldout", data["all_trials"]),
            max_trials=limits.get("confidence_heldout_max_trials", CONFIDENCE_EVAL_HELDOUT_MAX_TRIALS),
            max_subjects=limits.get("confidence_heldout_max_subjects", CONFIDENCE_MAX_SUBJECTS * 2),
            seed=seed + 2,
        ),
    }


def _igt_trials_by_split(
    *,
    benchmark_root: Path | str | None,
    allow_smoke_test: bool,
    sample_limits: dict[str, int] | None = None,
    seed: int = 44,
) -> dict[str, Any]:
    limits = sample_limits or {}
    data = _load_igt_data(benchmark_root=benchmark_root, allow_smoke_test=allow_smoke_test)
    return {
        "data": data,
        "train": _balanced_subject_sample(
            list(data["splits"].get("train", data["all_trials"])),
            max_trials=limits.get("igt_train_max_trials"),
            max_subjects=limits.get("igt_train_max_subjects"),
            seed=seed,
        ),
        "validation": _balanced_subject_sample(
            list(data["splits"].get("validation", data["all_trials"])),
            max_trials=limits.get("igt_validation_max_trials"),
            max_subjects=limits.get("igt_validation_max_subjects"),
            seed=seed + 1,
        ),
        "heldout": _balanced_subject_sample(
            list(data["splits"].get("heldout", data["all_trials"])),
            max_trials=limits.get("igt_heldout_max_trials"),
            max_subjects=limits.get("igt_heldout_max_subjects"),
            seed=seed + 2,
        ),
    }


def _confidence_payload(
    trials: list[Any],
    parameters: CognitiveStyleParameters,
    *,
    seed: int,
    include_predictions: bool = False,
) -> dict[str, Any]:
    payload = _simulate_confidence_trials(trials, parameters, seed=seed, include_predictions=include_predictions)
    payload["metrics"] = {
        **dict(payload["metrics"]),
        "primary_metric": float(payload["metrics"]["heldout_likelihood"]),
    }
    return payload


def _igt_payload(
    trials: list[Any],
    parameters: CognitiveStyleParameters,
    *,
    seed: int,
    include_predictions: bool = False,
) -> dict[str, Any]:
    payload = _simulate_igt_trials(trials, parameters, seed=seed, include_predictions=include_predictions)
    aggregate_metrics = compute_igt_aggregate_metrics(list(payload["trial_trace"]))
    payload["aggregate_metrics"] = aggregate_metrics
    payload["metrics"] = {
        **dict(payload["metrics"]),
        **{key: value for key, value in aggregate_metrics.items() if isinstance(value, (int, float))},
        "primary_metric": float(payload["metrics"]["deck_match_rate"]),
    }
    return payload


def _joint_objective_from_metrics(
    confidence_metrics: dict[str, float],
    igt_metrics: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    active_weights = dict(DEFAULT_JOINT_WEIGHTS if weights is None else weights)
    confidence_obj = float(_score_confidence_metrics(confidence_metrics))
    igt_obj = float(_score_igt_metrics(igt_metrics))
    igt_agg_obj = float(igt_metrics["igt_behavioral_similarity"])
    joint = (
        active_weights["w_conf"] * confidence_obj
        + active_weights["w_igt"] * igt_obj
        + active_weights["w_igt_agg"] * igt_agg_obj
    )
    return {
        "confidence_objective": _safe_round(confidence_obj),
        "igt_objective": _safe_round(igt_obj),
        "igt_aggregate_objective": _safe_round(igt_agg_obj),
        "joint_objective": _safe_round(joint),
        "weights": {key: _safe_round(float(value)) for key, value in active_weights.items()},
    }


def _joint_candidate_summary(candidate: JointCandidateFit) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "objective": _safe_round(candidate.objective),
        "parameters": candidate.parameters.to_dict(),
        "objective_components": dict(candidate.objective_components),
        "confidence_metrics": dict(candidate.confidence_metrics),
        "igt_metrics": dict(candidate.igt_metrics),
    }


def _coordinate_descent_fit_joint(
    *,
    confidence_train: list[Any],
    confidence_validation: list[Any],
    igt_train: list[Any],
    igt_validation: list[Any],
    seed: int,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    current = CognitiveStyleParameters()
    history: list[JointCandidateFit] = []
    cache: dict[str, JointCandidateFit] = {}

    def evaluate(
        parameters: CognitiveStyleParameters,
        *,
        label: str,
        confidence_trials: list[Any],
        igt_trials: list[Any],
        confidence_seed: int,
        igt_seed: int,
    ) -> JointCandidateFit:
        cache_key = json.dumps(
            {
                "parameters": parameters.to_dict(),
                "confidence_trial_count": len(confidence_trials),
                "igt_trial_count": len(igt_trials),
                "confidence_seed": confidence_seed,
                "igt_seed": igt_seed,
                "weights": dict(DEFAULT_JOINT_WEIGHTS if weights is None else weights),
            },
            sort_keys=True,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        confidence_payload = _confidence_payload(confidence_trials, parameters, seed=confidence_seed, include_predictions=False)
        igt_payload = _igt_payload(igt_trials, parameters, seed=igt_seed, include_predictions=False)
        objective_components = _joint_objective_from_metrics(
            confidence_payload["metrics"],
            igt_payload["metrics"],
            weights=weights,
        )
        cached = JointCandidateFit(
            parameters=parameters,
            objective=float(objective_components["joint_objective"]),
            confidence_metrics=dict(confidence_payload["metrics"]),
            igt_metrics=dict(igt_payload["metrics"]),
            objective_components=objective_components,
            label=label,
        )
        cache[cache_key] = cached
        return cached

    current_fit = evaluate(
        current,
        label="default",
        confidence_trials=confidence_train,
        igt_trials=igt_train,
        confidence_seed=seed,
        igt_seed=seed + 100,
    )
    history.append(current_fit)
    for step in (0.20, 0.12, 0.06):
        improved = True
        while improved:
            improved = False
            for parameter_name in JOINT_PARAMETER_NAMES:
                best_candidate = current_fit
                for delta in (-step, step):
                    candidate_parameters = _set_parameter(current, parameter_name, getattr(current, parameter_name) + delta)
                    candidate = evaluate(
                        candidate_parameters,
                        label=f"{parameter_name}_{'plus' if delta > 0 else 'minus'}_{step:.2f}",
                        confidence_trials=confidence_train,
                        igt_trials=igt_train,
                        confidence_seed=seed + 10,
                        igt_seed=seed + 110,
                    )
                    history.append(candidate)
                    if candidate.objective > best_candidate.objective + 1e-6:
                        best_candidate = candidate
                if best_candidate.parameters != current:
                    current = best_candidate.parameters
                    current_fit = best_candidate
                    improved = True

    unique_candidates: dict[str, JointCandidateFit] = {}
    for candidate in history:
        unique_candidates[json.dumps(candidate.parameters.to_dict(), sort_keys=True)] = candidate

    validation_candidates: list[JointCandidateFit] = []
    for index, candidate in enumerate(unique_candidates.values(), start=1):
        validation_candidates.append(
            evaluate(
                candidate.parameters,
                label=f"validation_{index}",
                confidence_trials=confidence_validation,
                igt_trials=igt_validation,
                confidence_seed=seed + 200 + index,
                igt_seed=seed + 400 + index,
            )
        )
    selected = max(validation_candidates, key=lambda item: (item.objective, item.label))
    return {
        "search_strategy": "coordinate_descent_joint_top7",
        "search_parameters": list(JOINT_PARAMETER_NAMES),
        "selected_parameters": selected.parameters.to_dict(),
        "validation_metrics": {
            "confidence": dict(selected.confidence_metrics),
            "igt": dict(selected.igt_metrics),
        },
        "validation_objective": _safe_round(selected.objective),
        "objective_components": dict(selected.objective_components),
        "history": [_joint_candidate_summary(candidate) for candidate in history[:60]],
        "selected_candidate": _joint_candidate_summary(selected),
        "parameters": selected.parameters,
    }


def _task_fit_with_fixed_parameter(
    *,
    task_id: str,
    training_trials: list[Any],
    validation_trials: list[Any],
    seed: int,
    fixed_values: dict[str, float],
    start_parameters: CognitiveStyleParameters,
) -> dict[str, Any]:
    parameter_names = [name for name in JOINT_PARAMETER_NAMES if name not in fixed_values]
    current = CognitiveStyleParameters.from_dict({**start_parameters.to_dict(), **fixed_values})
    history: list[dict[str, Any]] = []
    cache: dict[str, tuple[float, dict[str, float]]] = {}

    def evaluate(parameters: CognitiveStyleParameters, *, trials: list[Any], active_seed: int) -> tuple[float, dict[str, float]]:
        cache_key = json.dumps(
            {
                "task": task_id,
                "parameters": parameters.to_dict(),
                "trial_count": len(trials),
                "seed": active_seed,
            },
            sort_keys=True,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if task_id == "confidence":
            payload = _confidence_payload(trials, parameters, seed=active_seed, include_predictions=False)
            metrics = dict(payload["metrics"])
            objective = float(_score_confidence_metrics(metrics))
        else:
            payload = _igt_payload(trials, parameters, seed=active_seed, include_predictions=False)
            metrics = dict(payload["metrics"])
            objective = float(_score_igt_metrics(metrics)) * 0.5 + float(metrics["igt_behavioral_similarity"]) * 0.8
        cached = (_safe_round(objective), metrics)
        cache[cache_key] = cached
        return cached

    objective, metrics = evaluate(current, trials=training_trials, active_seed=seed)
    history.append({"label": "default", "objective": objective, "parameters": current.to_dict(), "metrics": metrics})
    current_objective = float(objective)
    for step in (0.20, 0.12, 0.06):
        improved = True
        while improved:
            improved = False
            for parameter_name in parameter_names:
                best_parameters = current
                best_objective = current_objective
                best_metrics = metrics
                for delta in (-step, step):
                    candidate_parameters = CognitiveStyleParameters.from_dict(
                        {**current.to_dict(), **fixed_values, parameter_name: getattr(current, parameter_name) + delta}
                    )
                    candidate_objective, candidate_metrics = evaluate(
                        candidate_parameters,
                        trials=training_trials,
                        active_seed=seed,
                    )
                    history.append(
                        {
                            "label": f"{parameter_name}_{'plus' if delta > 0 else 'minus'}_{step:.2f}",
                            "objective": candidate_objective,
                            "parameters": candidate_parameters.to_dict(),
                            "metrics": candidate_metrics,
                        }
                    )
                    if float(candidate_objective) > best_objective + 1e-6:
                        best_parameters = candidate_parameters
                        best_objective = float(candidate_objective)
                        best_metrics = candidate_metrics
                if best_parameters != current:
                    current = best_parameters
                    current_objective = best_objective
                    metrics = best_metrics
                    improved = True

    validation_objective, validation_metrics = evaluate(current, trials=validation_trials, active_seed=seed + 200)
    return {
        "task": task_id,
        "parameters": current,
        "selected_parameters": current.to_dict(),
        "validation_objective": validation_objective,
        "validation_metrics": validation_metrics,
        "search_parameters": parameter_names,
        "fixed_values": {key: _safe_round(float(value)) for key, value in fixed_values.items()},
        "history": history[:40],
    }


def _relative_degradation(reference_value: float, candidate_value: float, *, higher_is_better: bool = True) -> float:
    if higher_is_better:
        delta = float(reference_value) - float(candidate_value)
    else:
        delta = float(candidate_value) - float(reference_value)
    return _safe_round(max(0.0, delta) / max(abs(float(reference_value)), 1e-6))


def compute_degradation_matrix(
    *,
    confidence_specific_cell: dict[str, Any],
    igt_specific_cell: dict[str, Any],
    joint_cell_confidence: dict[str, Any],
    joint_cell_igt: dict[str, Any],
) -> dict[str, Any]:
    confidence_specific_metric = float(confidence_specific_cell["metrics"]["heldout_likelihood"])
    joint_confidence_metric = float(joint_cell_confidence["metrics"]["heldout_likelihood"])
    igt_specific_similarity = float(igt_specific_cell["metrics"]["igt_behavioral_similarity"])
    joint_igt_similarity = float(joint_cell_igt["metrics"]["igt_behavioral_similarity"])
    confidence_degradation = _relative_degradation(confidence_specific_metric, joint_confidence_metric, higher_is_better=True)
    igt_degradation = _relative_degradation(igt_specific_similarity, joint_igt_similarity, higher_is_better=True)
    return {
        "confidence_joint_vs_specific": {
            "reference_metric": _safe_round(confidence_specific_metric),
            "candidate_metric": _safe_round(joint_confidence_metric),
            "relative_degradation": confidence_degradation,
            "meaningful_threshold": 0.05,
            "meaningful_degradation": confidence_degradation > 0.05,
        },
        "igt_joint_vs_specific": {
            "reference_metric": _safe_round(igt_specific_similarity),
            "candidate_metric": _safe_round(joint_igt_similarity),
            "relative_degradation": igt_degradation,
            "meaningful_threshold": 0.10,
            "meaningful_degradation": igt_degradation > 0.10,
        },
    }


def _parameter_sets_on_heldout(
    *,
    confidence_heldout: list[Any],
    igt_heldout: list[Any],
    parameter_sets: dict[str, CognitiveStyleParameters],
    seed: int,
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for index, (label, parameters) in enumerate(parameter_sets.items(), start=1):
        confidence_payload = _confidence_payload(confidence_heldout, parameters, seed=seed + index, include_predictions=True)
        igt_payload = _igt_payload(igt_heldout, parameters, seed=seed + 100 + index, include_predictions=True)
        matrix[label] = {
            "confidence": {
                "task": "confidence_database",
                "parameters": parameters.to_dict(),
                "trial_count": len(confidence_heldout),
                "subject_count": int(confidence_payload["subject_summary"]["subject_count"]),
                "metrics": dict(confidence_payload["metrics"]),
                "predictions": list(confidence_payload["predictions"]),
                "trial_trace": list(confidence_payload["trial_trace"]),
            },
            "igt": {
                "task": "iowa_gambling_task",
                "parameters": parameters.to_dict(),
                "trial_count": len(igt_heldout),
                "subject_count": int(igt_payload["subject_summary"]["subject_count"]),
                "metrics": dict(igt_payload["metrics"]),
                "aggregate_metrics": dict(igt_payload["aggregate_metrics"]),
                "predictions": list(igt_payload["predictions"]),
                "trial_trace": list(igt_payload["trial_trace"]),
            },
        }
    return matrix


def _confidence_failure_examples(
    confidence_specific_predictions: list[dict[str, Any]],
    joint_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_trial = {str(row["trial_id"]): dict(row) for row in confidence_specific_predictions}
    examples: list[dict[str, Any]] = []
    for row in joint_predictions:
        trial_id = str(row["trial_id"])
        reference = by_trial.get(trial_id)
        if reference is None:
            continue
        human_choice = str(row["human_choice"])
        joint_probability = float(row["predicted_probability_right"]) if human_choice == "right" else 1.0 - float(row["predicted_probability_right"])
        reference_probability = (
            float(reference["predicted_probability_right"])
            if human_choice == "right"
            else 1.0 - float(reference["predicted_probability_right"])
        )
        examples.append(
            {
                "trial_id": trial_id,
                "subject_id": str(row["subject_id"]),
                "stimulus_strength": _safe_round(float(row.get("stimulus_strength", reference.get("stimulus_strength", 0.0)))),
                "human_choice": human_choice,
                "joint_probability_on_human_choice": _safe_round(joint_probability),
                "confidence_specific_probability_on_human_choice": _safe_round(reference_probability),
                "confidence_gap": _safe_round(float(reference["predicted_confidence"]) - float(row["predicted_confidence"])),
                "probability_drop": _safe_round(reference_probability - joint_probability),
            }
        )
    return sorted(examples, key=lambda item: (item["probability_drop"], abs(item["confidence_gap"])), reverse=True)[:5]


def _igt_failure_examples(
    igt_specific_trace: list[dict[str, Any]],
    joint_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_joint_phase: dict[str, list[dict[str, Any]]] = {}
    by_specific_phase: dict[str, list[dict[str, Any]]] = {}
    for start, end in PHASE_WINDOWS:
        phase_name = f"{start}-{end}"
        by_joint_phase[phase_name] = [row for row in joint_trace if start <= int(row["trial_index"]) <= end]
        by_specific_phase[phase_name] = [row for row in igt_specific_trace if start <= int(row["trial_index"]) <= end]
    for phase_name in by_joint_phase:
        joint_rows = by_joint_phase[phase_name]
        specific_rows = by_specific_phase[phase_name]
        if not joint_rows or not specific_rows:
            continue
        examples.append(
            {
                "phase": phase_name,
                "joint_advantageous_rate": _safe_round(mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in joint_rows)),
                "igt_specific_advantageous_rate": _safe_round(
                    mean(1.0 if bool(row["advantageous_choice"]) else 0.0 for row in specific_rows)
                ),
                "human_advantageous_rate": _safe_round(mean(1.0 if bool(row["actual_advantageous"]) else 0.0 for row in joint_rows)),
                "joint_deck_match_rate": _safe_round(mean(1.0 if bool(row["deck_match"]) else 0.0 for row in joint_rows)),
                "igt_specific_deck_match_rate": _safe_round(mean(1.0 if bool(row["deck_match"]) else 0.0 for row in specific_rows)),
            }
        )
    return sorted(
        examples,
        key=lambda item: abs(float(item["igt_specific_advantageous_rate"]) - float(item["joint_advantageous_rate"])),
        reverse=True,
    )[:5]


def classify_parameter_stability(
    *,
    confidence_train: list[Any],
    confidence_validation: list[Any],
    confidence_heldout: list[Any],
    igt_train: list[Any],
    igt_validation: list[Any],
    igt_heldout: list[Any],
    confidence_specific: CognitiveStyleParameters,
    igt_specific: CognitiveStyleParameters,
    joint_parameters: CognitiveStyleParameters,
    sensitivity_payload: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    sensitivity_active = {str(row["parameter"]): bool(row["active"]) for row in sensitivity_payload.get("parameters", [])}
    rows: list[dict[str, Any]] = []
    stable_count = 0
    task_sensitive_count = 0
    inert_count = 0
    indeterminate_count = 0
    for index, parameter_name in enumerate(PARAMETER_REFERENCE, start=1):
        confidence_value = float(getattr(confidence_specific, parameter_name))
        igt_value = float(getattr(igt_specific, parameter_name))
        joint_value = float(getattr(joint_parameters, parameter_name))
        evidence = {
            "confidence_specific_value": _safe_round(confidence_value),
            "igt_specific_value": _safe_round(igt_value),
            "joint_value": _safe_round(joint_value),
            "gap": _safe_round(abs(confidence_value - igt_value)),
            "joint_shift_conf": _safe_round(abs(joint_value - confidence_value)),
            "joint_shift_igt": _safe_round(abs(joint_value - igt_value)),
        }
        if parameter_name in INERT_PARAMETER_NAMES or not sensitivity_active.get(parameter_name, True):
            classification = "inert"
            evidence["ablation"] = None
            evidence["rationale"] = "inert_carryover"
            inert_count += 1
        else:
            fixed_value = {parameter_name: joint_value}
            confidence_ablation = _task_fit_with_fixed_parameter(
                task_id="confidence",
                training_trials=confidence_train,
                validation_trials=confidence_validation,
                seed=seed + index,
                fixed_values=fixed_value,
                start_parameters=confidence_specific,
            )
            igt_ablation = _task_fit_with_fixed_parameter(
                task_id="igt",
                training_trials=igt_train,
                validation_trials=igt_validation,
                seed=seed + 100 + index,
                fixed_values=fixed_value,
                start_parameters=igt_specific,
            )
            confidence_eval = _confidence_payload(
                confidence_heldout,
                confidence_ablation["parameters"],
                seed=seed + 200 + index,
                include_predictions=False,
            )
            igt_eval = _igt_payload(
                igt_heldout,
                igt_ablation["parameters"],
                seed=seed + 300 + index,
                include_predictions=False,
            )
            confidence_anchor = _confidence_payload(
                confidence_heldout,
                confidence_specific,
                seed=seed + 200 + index,
                include_predictions=False,
            )
            igt_anchor = _igt_payload(
                igt_heldout,
                igt_specific,
                seed=seed + 300 + index,
                include_predictions=False,
            )
            confidence_deg = _relative_degradation(
                float(confidence_anchor["metrics"]["heldout_likelihood"]),
                float(confidence_eval["metrics"]["heldout_likelihood"]),
                higher_is_better=True,
            )
            igt_deg = _relative_degradation(
                float(igt_anchor["metrics"]["igt_behavioral_similarity"]),
                float(igt_eval["metrics"]["igt_behavioral_similarity"]),
                higher_is_better=True,
            )
            evidence["ablation"] = {
                "confidence_fixed_fit_parameters": confidence_ablation["selected_parameters"],
                "igt_fixed_fit_parameters": igt_ablation["selected_parameters"],
                "confidence_relative_degradation": confidence_deg,
                "igt_relative_degradation": igt_deg,
                "confidence_threshold": _safe_round(TASK_SPECIFIC_THRESHOLD),
                "igt_threshold": _safe_round(TASK_SPECIFIC_THRESHOLD),
                "heldout_seeds": {
                    "confidence": seed + 200 + index,
                    "igt": seed + 300 + index,
                },
            }
            if confidence_deg < TASK_SPECIFIC_THRESHOLD and igt_deg < TASK_SPECIFIC_THRESHOLD:
                classification = "stable"
                evidence["rationale"] = "two_task_tolerance"
                stable_count += 1
            elif confidence_deg >= TASK_SPECIFIC_THRESHOLD or igt_deg >= TASK_SPECIFIC_THRESHOLD:
                classification = "task_sensitive"
                degraded_tasks: list[str] = []
                if confidence_deg >= TASK_SPECIFIC_THRESHOLD:
                    degraded_tasks.append("confidence")
                if igt_deg >= TASK_SPECIFIC_THRESHOLD:
                    degraded_tasks.append("igt")
                evidence["rationale"] = f"task_degradation:{'+'.join(degraded_tasks)}"
                task_sensitive_count += 1
            else:
                classification = "indeterminate"
                evidence["rationale"] = "insufficient_signal"
                indeterminate_count += 1
        rows.append(
            {
                "parameter": parameter_name,
                "classification": classification,
                "evidence": evidence,
            }
        )
    return {
        "source_type": sensitivity_payload.get("source_type"),
        "claim_envelope": sensitivity_payload.get("claim_envelope"),
        "external_validation": False,
        "parameter_count": len(rows),
        "stable_parameter_count": stable_count,
        "task_sensitive_count": task_sensitive_count,
        "inert_parameter_count": inert_count,
        "indeterminate_parameter_count": indeterminate_count,
        "parameters": rows,
    }


def run_weight_sensitivity_check(
    *,
    confidence_train: list[Any],
    confidence_validation: list[Any],
    igt_train: list[Any],
    igt_validation: list[Any],
    benchmark_root: Path | str | None,
    allow_smoke_test: bool,
    sample_limits: dict[str, int] | None,
    seed: int,
) -> dict[str, Any]:
    fits: dict[str, dict[str, Any]] = {}
    deltas: dict[str, float] = {name: 0.0 for name in PARAMETER_REFERENCE}
    default_parameters: dict[str, float] | None = None
    for index, (label, weights) in enumerate(WEIGHT_CONFIGS.items(), start=1):
        fit = _coordinate_descent_fit_joint(
            confidence_train=confidence_train,
            confidence_validation=confidence_validation,
            igt_train=igt_train,
            igt_validation=igt_validation,
            seed=seed + index * 17,
            weights=weights,
        )
        fits[label] = {
            "weights": {key: _safe_round(float(value)) for key, value in weights.items()},
            "selected_parameters": fit["selected_parameters"],
            "validation_objective": fit["validation_objective"],
            "objective_components": fit["objective_components"],
        }
        if label == "default":
            default_parameters = dict(fit["selected_parameters"])
    assert default_parameters is not None
    for parameter_name in PARAMETER_REFERENCE:
        deltas[parameter_name] = _safe_round(
            max(
                abs(float(payload["selected_parameters"][parameter_name]) - float(default_parameters[parameter_name]))
                for payload in fits.values()
            )
        )
    weight_sensitive_parameters = sorted(name for name, delta in deltas.items() if float(delta) > 0.15)
    return {
        "source_type": "external_bundle" if benchmark_root is not None and not allow_smoke_test else "synthetic_protocol",
        "claim_envelope": "benchmark_eval" if benchmark_root is not None and not allow_smoke_test else "synthetic_diagnostic",
        "external_validation": False,
        "fits": fits,
        "max_parameter_deltas": deltas,
        "weight_sensitive_parameters": weight_sensitive_parameters,
        "robust_to_weighting": not any(float(delta) >= 0.10 for delta in deltas.values()),
    }


def assess_igt_architecture(
    *,
    igt_heldout: list[Any],
    confidence_specific: CognitiveStyleParameters,
    igt_specific: CognitiveStyleParameters,
    joint_payload: dict[str, Any],
    weight_sensitivity: dict[str, Any],
    random_baseline_deck_match_rate: float,
    source_type: str,
    claim_envelope: str,
    seed: int,
    sample_limits: dict[str, int] | None,
) -> dict[str, Any]:
    candidate_count = int((sample_limits or {}).get("architecture_candidate_count", 12))
    rng = random.Random(seed)
    candidate_map: dict[str, CognitiveStyleParameters] = {
        "default": CognitiveStyleParameters(),
        "confidence_specific": confidence_specific,
        "igt_specific": igt_specific,
        "joint_default": CognitiveStyleParameters.from_dict(dict(joint_payload["selected_parameters"])),
    }
    for label, payload in weight_sensitivity["fits"].items():
        candidate_map[f"joint_{label}"] = CognitiveStyleParameters.from_dict(dict(payload["selected_parameters"]))
    anchors = list(candidate_map.values())
    for index in range(candidate_count):
        base = rng.choice(anchors)
        payload = dict(base.to_dict())
        for parameter_name in JOINT_PARAMETER_NAMES:
            payload[parameter_name] = max(0.0, min(1.0, float(payload[parameter_name]) + rng.uniform(-0.35, 0.35)))
        candidate_map[f"sweep_{index + 1:02d}"] = CognitiveStyleParameters.from_dict(payload)

    evaluated: list[dict[str, Any]] = []
    for index, (label, parameters) in enumerate(candidate_map.items(), start=1):
        payload = _igt_payload(igt_heldout, parameters, seed=seed + index, include_predictions=False)
        evaluated.append(
            {
                "label": label,
                "parameters": parameters.to_dict(),
                "deck_match_rate": _safe_round(float(payload["metrics"]["deck_match_rate"])),
                "igt_behavioral_similarity": _safe_round(float(payload["metrics"]["igt_behavioral_similarity"])),
            }
        )
    best_deck_match = max(evaluated, key=lambda item: (float(item["deck_match_rate"]), float(item["igt_behavioral_similarity"])))
    best_behavioral_similarity = max(
        evaluated,
        key=lambda item: (float(item["igt_behavioral_similarity"]), float(item["deck_match_rate"])),
    )
    aggregate_recommended = (
        float(best_deck_match["deck_match_rate"]) - float(random_baseline_deck_match_rate) < 0.10
        or float(best_deck_match["deck_match_rate"]) < 0.35
    )
    return {
        "source_type": source_type,
        "claim_envelope": claim_envelope,
        "external_validation": False,
        "candidate_count": len(evaluated),
        "random_baseline_deck_match_rate": _safe_round(random_baseline_deck_match_rate),
        "best_deck_match_rate": dict(best_deck_match),
        "best_behavioral_similarity": dict(best_behavioral_similarity),
        "aggregate_metrics_recommended": aggregate_recommended,
        "finding": (
            "Per-trial deck matching appears ceiling-limited under the current shared scoring architecture."
            if aggregate_recommended
            else "Per-trial deck matching still carries some usable signal under the current architecture."
        ),
        "recommendation_for_m45": (
            "Use aggregate IGT behavioral metrics as the primary lens and keep deck_match_rate as a supplemental diagnostic."
            if aggregate_recommended
            else "Keep reporting deck_match_rate, but pair it with aggregate behavioral metrics."
        ),
        "evaluated_candidates": evaluated[:20],
    }


def _build_failure_analysis(
    *,
    degradation: dict[str, Any],
    stability: dict[str, Any],
    architecture_assessment: dict[str, Any],
    cross_application_matrix: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    compromise_candidates = [
        {
            "parameter": row["parameter"],
            "classification": row["classification"],
            "gap": row["evidence"]["gap"],
        }
        for row in stability["parameters"]
        if row["classification"] == "task_sensitive"
    ]
    dominant_sources = ["parameter_compromise"] if compromise_candidates else []
    if architecture_assessment["aggregate_metrics_recommended"]:
        dominant_sources.append("igt_architecture_limit")
    if int(stability["indeterminate_parameter_count"]) > 0:
        dominant_sources.append("insufficient_signal")
    return {
        "dominant_sources": dominant_sources,
        "parameter_compromise_candidates": compromise_candidates[:5],
        "confidence_examples": _confidence_failure_examples(
            cross_application_matrix["confidence_specific"]["confidence"]["predictions"],
            cross_application_matrix["joint"]["confidence"]["predictions"],
        ),
        "igt_examples": _igt_failure_examples(
            cross_application_matrix["igt_specific"]["igt"]["trial_trace"],
            cross_application_matrix["joint"]["igt"]["trial_trace"],
        ),
        "joint_degradation": degradation,
    }


def fit_joint_parameters(
    *,
    seed: int = 44,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    confidence_splits = _confidence_trials_by_split(
        benchmark_root=benchmark_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        seed=seed,
    )
    igt_splits = _igt_trials_by_split(
        benchmark_root=benchmark_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        seed=seed + 10,
    )
    fit = _coordinate_descent_fit_joint(
        confidence_train=confidence_splits["train"],
        confidence_validation=confidence_splits["validation"],
        igt_train=igt_splits["train"],
        igt_validation=igt_splits["validation"],
        seed=seed,
        weights=weights,
    )
    heldout = _parameter_sets_on_heldout(
        confidence_heldout=confidence_splits["heldout"],
        igt_heldout=igt_splits["heldout"],
        parameter_sets={"joint": fit["parameters"]},
        seed=seed + 500,
    )["joint"]
    return {
        "benchmark_id": "m44_joint_fit",
        "source_type": "external_bundle"
        if confidence_splits["data"]["source_type"] == "external_bundle" and igt_splits["data"]["source_type"] == "external_bundle"
        else "synthetic_protocol",
        "claim_envelope": _claim_envelope_for_sources(
            confidence_source_type=confidence_splits["data"]["source_type"],
            igt_source_type=igt_splits["data"]["source_type"],
        ),
        "external_validation": False,
        "weights": dict(DEFAULT_JOINT_WEIGHTS if weights is None else weights),
        "training_trial_count": {
            "confidence": len(confidence_splits["train"]),
            "igt": len(igt_splits["train"]),
        },
        "validation_trial_count": {
            "confidence": len(confidence_splits["validation"]),
            "igt": len(igt_splits["validation"]),
        },
        "heldout_trial_count": {
            "confidence": len(confidence_splits["heldout"]),
            "igt": len(igt_splits["heldout"]),
        },
        **fit,
        "heldout_metrics": {
            "confidence": heldout["confidence"]["metrics"],
            "igt": heldout["igt"]["metrics"],
        },
    }


def _blocked_suite(*, benchmark_root: Path | str | None = None) -> dict[str, Any]:
    return {
        "blocked": True,
        "acceptance_state": "blocked_missing_external_bundle",
        "benchmark_root": str(benchmark_root) if benchmark_root else None,
        "joint_fit": {"mode": "blocked", "status": "blocked"},
        "degradation": {"mode": "blocked", "status": "blocked"},
        "parameter_stability": {"mode": "blocked", "status": "blocked"},
        "weight_sensitivity": {"mode": "blocked", "status": "blocked"},
        "igt_aggregate": {"mode": "blocked", "status": "blocked"},
        "architecture_assessment": {"mode": "blocked", "status": "blocked"},
        "failure_analysis": {"mode": "blocked", "status": "blocked", "confidence_examples": [], "igt_examples": []},
    }


def run_m44_cross_task_suite(
    *,
    seed: int = 44,
    benchmark_root: Path | str | None = None,
    allow_smoke_test: bool = False,
    sample_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    resolved_root = None if allow_smoke_test and benchmark_root is None else _resolve_benchmark_root(benchmark_root)
    if resolved_root is None and not allow_smoke_test:
        return _blocked_suite(benchmark_root=benchmark_root)

    confidence_splits = _confidence_trials_by_split(
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        seed=seed,
    )
    igt_splits = _igt_trials_by_split(
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        seed=seed + 10,
    )
    confidence_fit = run_fitted_confidence_agent(
        seed=seed,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
    )
    igt_fit = run_fitted_igt_agent(
        seed=seed + 1,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
    )
    sensitivity = run_parameter_sensitivity_analysis(
        seed=seed + 2,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
    )
    joint_fit = fit_joint_parameters(
        seed=seed + 3,
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        weights=DEFAULT_JOINT_WEIGHTS,
    )

    confidence_specific_parameters = CognitiveStyleParameters.from_dict(dict(confidence_fit["fit"]["selected_parameters"]))
    igt_specific_parameters = CognitiveStyleParameters.from_dict(dict(igt_fit["fit"]["selected_parameters"]))
    joint_parameters = CognitiveStyleParameters.from_dict(dict(joint_fit["selected_parameters"]))

    cross_application_matrix = _parameter_sets_on_heldout(
        confidence_heldout=confidence_splits["heldout"],
        igt_heldout=igt_splits["heldout"],
        parameter_sets={
            "confidence_specific": confidence_specific_parameters,
            "igt_specific": igt_specific_parameters,
            "joint": joint_parameters,
        },
        seed=seed + 600,
    )
    degradation = compute_degradation_matrix(
        confidence_specific_cell=cross_application_matrix["confidence_specific"]["confidence"],
        igt_specific_cell=cross_application_matrix["igt_specific"]["igt"],
        joint_cell_confidence=cross_application_matrix["joint"]["confidence"],
        joint_cell_igt=cross_application_matrix["joint"]["igt"],
    )
    stability = classify_parameter_stability(
        confidence_train=confidence_splits["train"],
        confidence_validation=confidence_splits["validation"],
        confidence_heldout=confidence_splits["heldout"],
        igt_train=igt_splits["train"],
        igt_validation=igt_splits["validation"],
        igt_heldout=igt_splits["heldout"],
        confidence_specific=confidence_specific_parameters,
        igt_specific=igt_specific_parameters,
        joint_parameters=joint_parameters,
        sensitivity_payload=sensitivity,
        seed=seed + 700,
    )
    weight_sensitivity = run_weight_sensitivity_check(
        confidence_train=confidence_splits["train"],
        confidence_validation=confidence_splits["validation"],
        igt_train=igt_splits["train"],
        igt_validation=igt_splits["validation"],
        benchmark_root=resolved_root,
        allow_smoke_test=allow_smoke_test,
        sample_limits=sample_limits,
        seed=seed + 800,
    )
    igt_aggregate = {
        "source_type": igt_fit["source_type"],
        "claim_envelope": igt_fit["claim_envelope"],
        "external_validation": False,
        "submetrics": [
            "learning_curve_distance",
            "post_loss_switch_gap",
            "deck_distribution_l1",
            "exploration_exploitation_entropy_gap",
        ],
        "parameter_sets": {
            label: {
                "metrics": dict(cell["igt"]["metrics"]),
                "aggregate_metrics": dict(cell["igt"]["aggregate_metrics"]),
            }
            for label, cell in cross_application_matrix.items()
        },
    }
    architecture_assessment = assess_igt_architecture(
        igt_heldout=igt_splits["heldout"],
        confidence_specific=confidence_specific_parameters,
        igt_specific=igt_specific_parameters,
        joint_payload=joint_fit,
        weight_sensitivity=weight_sensitivity,
        random_baseline_deck_match_rate=float(igt_fit["baselines"]["random"]["metrics"]["deck_match_rate"]),
        source_type=igt_fit["source_type"],
        claim_envelope=igt_fit["claim_envelope"],
        seed=seed + 900,
        sample_limits=sample_limits,
    )
    failure_analysis = _build_failure_analysis(
        degradation=degradation,
        stability=stability,
        architecture_assessment=architecture_assessment,
        cross_application_matrix=cross_application_matrix,
    )
    mode = "benchmark_eval" if resolved_root is not None else "smoke_only"
    degradation_payload = {
        "mode": mode,
        "source_type": "external_bundle"
        if confidence_fit["source_type"] == "external_bundle" and igt_fit["source_type"] == "external_bundle"
        else "synthetic_protocol",
        "claim_envelope": _claim_envelope_for_sources(
            confidence_source_type=confidence_fit["source_type"],
            igt_source_type=igt_fit["source_type"],
        ),
        "external_validation": False,
        "cross_application_matrix": cross_application_matrix,
        "joint_degradation": degradation,
        "interpretation": {
            "confidence": (
                "Joint fitting meaningfully degrades Confidence performance"
                if degradation["confidence_joint_vs_specific"]["meaningful_degradation"]
                else "Joint fitting preserves Confidence within the diagnostic tolerance."
            ),
            "igt": (
                "Joint fitting meaningfully degrades IGT performance"
                if degradation["igt_joint_vs_specific"]["meaningful_degradation"]
                else "Joint fitting preserves IGT behavioral similarity within the diagnostic tolerance."
            ),
            "shared_parameters_acceptable": (
                not degradation["confidence_joint_vs_specific"]["meaningful_degradation"]
                and not degradation["igt_joint_vs_specific"]["meaningful_degradation"]
            ),
        },
    }
    return {
        "blocked": False,
        "acceptance_state": mode,
        "benchmark_root": str(resolved_root) if resolved_root else None,
        "confidence_fit": confidence_fit,
        "igt_fit": igt_fit,
        "parameter_sensitivity": sensitivity,
        "joint_fit": joint_fit,
        "degradation": degradation_payload,
        "parameter_stability": stability,
        "weight_sensitivity": weight_sensitivity,
        "igt_aggregate": igt_aggregate,
        "architecture_assessment": architecture_assessment,
        "failure_analysis": failure_analysis,
    }


__all__ = [
    "DEFAULT_JOINT_WEIGHTS",
    "JOINT_PARAMETER_NAMES",
    "WEIGHT_CONFIGS",
    "classify_parameter_stability",
    "compute_degradation_matrix",
    "fit_joint_parameters",
    "run_m44_cross_task_suite",
    "run_weight_sensitivity_check",
]
