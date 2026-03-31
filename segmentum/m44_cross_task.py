from __future__ import annotations

from dataclasses import dataclass

from .m4_benchmarks import run_confidence_database_benchmark, run_iowa_gambling_benchmark
from .m4_cognitive_style import CognitiveStyleParameters
from .m43_modeling import candidate_parameter_grid


def _confidence_objective(metrics: dict[str, float]) -> float:
    return float(metrics["heldout_likelihood"]) + float(metrics["confidence_alignment"]) * 0.4 - float(metrics["calibration_error"]) * 0.5


def _igt_objective(metrics: dict[str, float]) -> float:
    return (
        float(metrics["advantageous_choice_rate"]) * 0.7
        + float(metrics["deck_match_rate"]) * 0.8
        + float(metrics["policy_alignment_rate"]) * 0.8
        + float(metrics["late_advantageous_rate"]) * 0.7
        + float(metrics["confidence_advantage_alignment"]) * 0.3
    )


@dataclass(frozen=True)
class SharedParameterResult:
    parameters: CognitiveStyleParameters
    confidence_metrics: dict[str, float]
    igt_metrics: dict[str, float]
    shared_objective: float


def fit_shared_parameters(*, seed: int = 44) -> SharedParameterResult:
    candidates: list[SharedParameterResult] = []
    for index, parameters in enumerate(candidate_parameter_grid()):
        confidence_validation = run_confidence_database_benchmark(parameters, seed=seed + index, split="validation", allow_smoke_test=True)
        igt_validation = run_iowa_gambling_benchmark(parameters, seed=seed + index, split="validation", allow_smoke_test=True)
        objective = _confidence_objective(confidence_validation["metrics"]) + _igt_objective(igt_validation["metrics"])
        candidates.append(
            SharedParameterResult(
                parameters=parameters,
                confidence_metrics=confidence_validation["metrics"],
                igt_metrics=igt_validation["metrics"],
                shared_objective=round(objective, 6),
            )
        )
    return max(candidates, key=lambda item: (item.shared_objective, item.parameters.confidence_gain))


def fit_task_specific_parameters(*, seed: int = 44) -> dict[str, object]:
    best_confidence = None
    best_igt = None
    for index, parameters in enumerate(candidate_parameter_grid()):
        confidence_validation = run_confidence_database_benchmark(parameters, seed=seed + index, split="validation", allow_smoke_test=True)
        igt_validation = run_iowa_gambling_benchmark(parameters, seed=seed + index, split="validation", allow_smoke_test=True)
        confidence_score = _confidence_objective(confidence_validation["metrics"])
        igt_score = _igt_objective(igt_validation["metrics"])
        if best_confidence is None or confidence_score > best_confidence["objective"]:
            best_confidence = {"parameters": parameters, "metrics": confidence_validation["metrics"], "objective": round(confidence_score, 6)}
        if best_igt is None or igt_score > best_igt["objective"]:
            best_igt = {"parameters": parameters, "metrics": igt_validation["metrics"], "objective": round(igt_score, 6)}
    return {"confidence": best_confidence, "igt": best_igt}


def compare_shared_vs_independent(*, seed: int = 44) -> dict[str, object]:
    shared = fit_shared_parameters(seed=seed)
    task_specific = fit_task_specific_parameters(seed=seed)
    shared_confidence = run_confidence_database_benchmark(shared.parameters, seed=seed, split="heldout", allow_smoke_test=True)
    shared_igt = run_iowa_gambling_benchmark(shared.parameters, seed=seed, split="heldout", allow_smoke_test=True)
    task_confidence = run_confidence_database_benchmark(task_specific["confidence"]["parameters"], seed=seed, split="heldout", allow_smoke_test=True)
    task_igt = run_iowa_gambling_benchmark(task_specific["igt"]["parameters"], seed=seed, split="heldout", allow_smoke_test=True)

    stable_parameters = []
    task_specific_parameters = []
    for name in (
        "uncertainty_sensitivity",
        "error_aversion",
        "exploration_bias",
        "attention_selectivity",
        "confidence_gain",
        "update_rigidity",
        "resource_pressure_sensitivity",
    ):
        left = float(getattr(task_specific["confidence"]["parameters"], name))
        right = float(getattr(task_specific["igt"]["parameters"], name))
        if abs(left - right) <= 0.10:
            stable_parameters.append(name)
        else:
            task_specific_parameters.append(name)
    return {
        "shared": {
            "parameters": shared.parameters.to_dict(),
            "heldout": {"confidence": shared_confidence["metrics"], "igt": shared_igt["metrics"]},
            "evidence": {
                "confidence_trial_count": shared_confidence["trial_count"],
                "confidence_subject_count": shared_confidence["subject_summary"]["subject_count"],
                "igt_trial_count": shared_igt["trial_count"],
                "igt_subject_count": shared_igt["subject_summary"]["subject_count"],
            },
            "objective": shared.shared_objective,
        },
        "task_specific": {
            "confidence": {
                "parameters": task_specific["confidence"]["parameters"].to_dict(),
                "heldout": task_confidence["metrics"],
                "trial_count": task_confidence["trial_count"],
                "subject_count": task_confidence["subject_summary"]["subject_count"],
            },
            "igt": {
                "parameters": task_specific["igt"]["parameters"].to_dict(),
                "heldout": task_igt["metrics"],
                "trial_count": task_igt["trial_count"],
                "subject_count": task_igt["subject_summary"]["subject_count"],
            },
        },
        "stability_analysis": {
            "stable_parameters": stable_parameters,
            "task_specific_parameters": task_specific_parameters,
            "shared_parameter_count": len(stable_parameters),
            "parameter_distance_mean": round(
                sum(
                    abs(
                        float(getattr(task_specific["confidence"]["parameters"], name))
                        - float(getattr(task_specific["igt"]["parameters"], name))
                    )
                    for name in (
                        "uncertainty_sensitivity",
                        "error_aversion",
                        "exploration_bias",
                        "attention_selectivity",
                        "confidence_gain",
                        "update_rigidity",
                        "resource_pressure_sensitivity",
                    )
                )
                / 7.0,
                6,
            ),
        },
    }
