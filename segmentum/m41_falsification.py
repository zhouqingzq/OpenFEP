from __future__ import annotations

from statistics import mean
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, parameter_probe_registry, run_cognitive_style_trial


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _metric_mean(parameters: CognitiveStyleParameters, metric_name: str, *, seeds: list[int], stress: bool = False) -> float | None:
    values = []
    for seed in seeds:
        payload = run_cognitive_style_trial(parameters, seed=seed, stress=stress)
        metric_payload = payload["observable_metrics"][metric_name]
        if not metric_payload.get("insufficient_data") and metric_payload.get("value") is not None:
            values.append(float(metric_payload["value"]))
    return mean(values) if values else None


def _custom_readout(parameters: CognitiveStyleParameters, parameter_name: str, *, seeds: list[int], stress: bool = False) -> float | None:
    values = []
    for seed in seeds:
        payload = run_cognitive_style_trial(parameters, seed=seed, stress=stress)
        logs = payload["logs"]
        if parameter_name == "confidence_gain":
            samples = [
                float(row["internal_confidence"])
                for row in logs
                if float(row["observation_evidence"].get("evidence_strength", 0.0)) >= 0.70
            ]
        elif parameter_name == "resource_pressure_sensitivity":
            samples = [
                1.0 if row["selected_action"] in {"rest", "conserve"} else 0.0
                for row in logs
                if float(row["resource_state"].get("energy", 1.0)) <= 0.35
                or float(row["resource_state"].get("time_remaining", 1.0)) <= 0.30
            ]
        else:
            samples = []
        if samples:
            values.append(mean(samples))
    return mean(values) if values else None


def run_parameter_falsification_suite(*, seeds: list[int] | None = None) -> dict[str, Any]:
    active_seeds = seeds or [41, 42, 43]
    baseline = CognitiveStyleParameters()
    experiments = {}
    for parameter_name, probe in parameter_probe_registry().items():
        metric_name = probe["metric"]
        stress = bool(probe.get("stress", False))
        present = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.95})
        absent = CognitiveStyleParameters.from_dict({**baseline.to_dict(), parameter_name: 0.05})
        competitor_name = next(name for name in baseline.to_dict() if name not in {"schema_version", parameter_name})
        competitor = CognitiveStyleParameters.from_dict({**baseline.to_dict(), competitor_name: 0.95})

        present_value = _custom_readout(present, parameter_name, seeds=active_seeds, stress=stress)
        if present_value is None:
            present_value = _metric_mean(present, metric_name, seeds=active_seeds, stress=stress)
        absent_value = _custom_readout(absent, parameter_name, seeds=active_seeds, stress=stress)
        if absent_value is None:
            absent_value = _metric_mean(absent, metric_name, seeds=active_seeds, stress=stress)
        competitor_value = _custom_readout(competitor, parameter_name, seeds=active_seeds, stress=stress)
        if competitor_value is None:
            competitor_value = _metric_mean(competitor, metric_name, seeds=active_seeds, stress=stress)
        disappearance = None if present_value is None or absent_value is None else present_value - absent_value
        competitor_gap = None if competitor_value is None or absent_value is None else competitor_value - absent_value
        supported = bool(disappearance is not None and disappearance >= float(probe["min_effect"]))

        experiments[parameter_name] = {
            "analysis_type": "preregistered_falsification",
            "parameter": parameter_name,
            "preregistered_observable": metric_name,
            "presence_condition": {"parameter_value": 0.95, "expected_pattern": f"{metric_name} increases"},
            "absence_condition": {"parameter_value": 0.05, "expected_pattern": f"{metric_name} decreases or disappears"},
            "alternative_explanation_test": {"competitor_parameter": competitor_name, "expected_weaker_effect": True},
            "observed": {
                "present_metric": _round(present_value),
                "absent_metric": _round(absent_value),
                "competitor_metric": _round(competitor_value),
                "disappearance_effect": _round(disappearance),
                "competitor_effect": _round(competitor_gap),
            },
            "minimum_effect": probe["min_effect"],
            "supported": supported,
            "failure_condition": f"If {metric_name} does not drop by at least {probe['min_effect']} after ablating {parameter_name}, the mechanism claim weakens.",
        }
    return {"analysis_type": "parameter_falsification_suite", "seeds": list(active_seeds), "experiments": experiments}
