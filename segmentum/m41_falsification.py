from __future__ import annotations

"""Legacy same-framework sensitivity checks.

This suite probes whether internal synthetic observables move when individual
synthetic parameters are perturbed. It is not a falsification of the latent
ontology on external data.
"""

import random
from statistics import mean, pvariance
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE, parameter_probe_registry, run_cognitive_style_trial


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _metric_series(parameters: CognitiveStyleParameters, metric_name: str, *, seeds: list[int], stress: bool = False) -> list[float]:
    values: list[float] = []
    for seed in seeds:
        payload = run_cognitive_style_trial(parameters, seed=seed, stress=stress)
        metric_payload = payload["observable_metrics"][metric_name]
        if not metric_payload.get("insufficient_data") and metric_payload.get("value") is not None:
            values.append(float(metric_payload["value"]))
    return values


def _control_series(parameters: CognitiveStyleParameters, metric_name: str, *, seeds: list[int], stress: bool = False) -> list[float]:
    values: list[float] = []
    for seed in seeds:
        payload = run_cognitive_style_trial(parameters, seed=seed, stress=stress)
        if metric_name == "session_length":
            values.append(float(len(payload["logs"])))
            continue
        if metric_name in payload.get("summary", {}):
            values.append(float(payload["summary"][metric_name]))
    return values


def _cohens_d(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    left_var = pvariance(left) if len(left) > 1 else 0.0
    right_var = pvariance(right) if len(right) > 1 else 0.0
    pooled = (((len(left) - 1) * left_var) + ((len(right) - 1) * right_var)) / max(1, (len(left) + len(right) - 2))
    if pooled <= 1e-9:
        return 0.0 if abs(left_mean - right_mean) < 1e-9 else 1.0
    return (left_mean - right_mean) / (pooled ** 0.5)


def _series_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": _round(mean(values)) if values else None,
        "min": _round(min(values)) if values else None,
        "max": _round(max(values)) if values else None,
    }


def run_same_framework_sensitivity_suite(*, seeds: list[int] | None = None) -> dict[str, Any]:
    active_seeds = seeds or [41, 42, 43, 44, 45, 46]
    baseline = CognitiveStyleParameters()
    rng = random.Random(41)
    competitor_map = {
        "uncertainty_sensitivity": "resource_pressure_sensitivity",
        "error_aversion": "attention_selectivity",
        "exploration_bias": "error_aversion",
        "attention_selectivity": "update_rigidity",
        "confidence_gain": "resource_pressure_sensitivity",
        "update_rigidity": "exploration_bias",
        "resource_pressure_sensitivity": "exploration_bias",
        "virtual_prediction_error_gain": "uncertainty_sensitivity",
    }
    experiments = {}
    failures = []
    for parameter_name, probe in parameter_probe_registry().items():
        metric_name = probe["metric"]
        stress = bool(probe.get("stress", False))
        high = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.95})
        low = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.05})
        competitor_name = competitor_map[parameter_name]
        competitor = CognitiveStyleParameters.from_dict({**baseline.to_dict(), competitor_name: 0.95})
        random_payload = baseline.to_dict()
        random_payload[competitor_name] = round(rng.uniform(0.35, 0.65), 6)
        random_parameters = CognitiveStyleParameters.from_dict(random_payload)

        high_values = _metric_series(high, metric_name, seeds=active_seeds, stress=stress)
        low_values = _metric_series(low, metric_name, seeds=active_seeds, stress=stress)
        competitor_values = _metric_series(competitor, metric_name, seeds=active_seeds, stress=stress)
        random_values = _metric_series(random_parameters, metric_name, seeds=active_seeds, stress=stress)

        cohens_d = abs(_cohens_d(high_values, low_values))
        competitor_d = abs(_cohens_d(competitor_values, low_values))
        random_d = abs(_cohens_d(random_values, low_values))
        supported = bool(cohens_d >= 0.5 and cohens_d > competitor_d)
        if not supported:
            failures.append({"parameter": parameter_name, "cohens_d": _round(cohens_d)})

        experiments[parameter_name] = {
            "analysis_type": "parameter_sensitivity_probe",
            "parameter": parameter_name,
            "preregistered_observable": metric_name,
            "presence_condition": {"parameter_value": 0.95, "observed_series": [_round(value) for value in high_values]},
            "absence_condition": {"parameter_value": 0.05, "observed_series": [_round(value) for value in low_values]},
            "alternative_explanation_test": {
                "competitor_parameter": competitor_name,
                "competitor_cohens_d": _round(competitor_d),
                "random_parameter_cohens_d": _round(random_d),
            },
            "failure_condition": "supported is False when the target effect size is too small or not specific to the target parameter",
            "cohens_d": _round(cohens_d),
            "supported": supported,
        }

    control_metric_name = "session_length"
    control_high_values = _control_series(
        CognitiveStyleParameters.from_dict({**baseline.to_dict(), "resource_pressure_sensitivity": 0.95}),
        control_metric_name,
        seeds=active_seeds,
        stress=True,
    )
    control_low_values = _control_series(
        CognitiveStyleParameters.from_dict({**baseline.to_dict(), "resource_pressure_sensitivity": 0.05}),
        control_metric_name,
        seeds=active_seeds,
        stress=True,
    )
    control_metric = {
        "metric": control_metric_name,
        "source_parameter": "resource_pressure_sensitivity",
        "source_reason": "session_length is collected directly from each generated trial and is structurally independent from the scored observable metrics because the internal generator always emits a fixed episode schedule.",
        "control_strength": "weak",
        "control_limitation": "session_length is near-constant by construction in the synthetic generator, so passing this control is not strong evidence",
        "high_condition_series": [_round(value) for value in control_high_values],
        "low_condition_series": [_round(value) for value in control_low_values],
        "high_condition_summary": _series_summary(control_high_values),
        "low_condition_summary": _series_summary(control_low_values),
        "cohens_d": _round(abs(_cohens_d(control_high_values, control_low_values))),
    }
    return {
        "analysis_type": "same_framework_sensitivity_suite",
        "benchmark_scope": "same-framework parameter sensitivity sidecar",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "validation_type": "synthetic_holdout_same_framework",
        "interpretation": "internal generator sensitivity only; not external falsification or M4 acceptance evidence",
        "seeds": list(active_seeds),
        "experiments": experiments,
        "falsification_failures": failures,
        "control_metric": control_metric,
    }


def run_parameter_falsification_suite(*, seeds: list[int] | None = None) -> dict[str, Any]:
    return run_same_framework_sensitivity_suite(seeds=seeds)
