from __future__ import annotations

from statistics import mean
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE, PROFILE_REGISTRY, run_cognitive_style_trial
from .m41_external_dataset import load_external_behavior_dataset
from .m41_inference import infer_cognitive_style, summarize_parameter_recovery


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_corr(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _safe_mean(left)
    right_mean = _safe_mean(right)
    numerator = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right))
    left_var = sum((l - left_mean) ** 2 for l in left)
    right_var = sum((r - right_mean) ** 2 for r in right)
    if left_var <= 0.0 or right_var <= 0.0:
        return 0.0
    return numerator / ((left_var ** 0.5) * (right_var ** 0.5))


def _synthetic_identifiability_rows(*, seeds: tuple[int, ...] = (41, 42, 43)) -> list[dict[str, Any]]:
    rows = []
    for profile_name, parameters in PROFILE_REGISTRY.items():
        for seed in seeds:
            trial = run_cognitive_style_trial(parameters, seed=seed, stress=profile_name == "low_exploration_high_caution")
            inference = infer_cognitive_style(trial["logs"])
            recovery = summarize_parameter_recovery(inference["inferred_parameters"], parameters.to_dict())
            rows.append(
                {
                    "source": "synthetic_internal",
                    "profile_name": profile_name,
                    "seed": seed,
                    "true_parameters": parameters.to_dict(),
                    "inferred_parameters": inference["inferred_parameters"],
                    "parameter_recovery": recovery,
                    "fit_confidence": float(inference["fit_confidence"]),
                    "parameter_estimates": inference["parameter_estimates"],
                }
            )
    return rows


def _external_identifiability_rows() -> list[dict[str, Any]]:
    rows = []
    dataset = load_external_behavior_dataset()
    for session in dataset["sessions"]:
        if not session.get("ground_truth_parameters"):
            continue
        inference = infer_cognitive_style(session["records"], subject_id=session["subject_id"], source_name=session["source_name"])
        recovery = summarize_parameter_recovery(inference["inferred_parameters"], session["ground_truth_parameters"])
        rows.append(
            {
                "source": "external_holdout",
                "profile_name": session.get("profile_label"),
                "session_id": session["session_id"],
                "subject_id": session["subject_id"],
                "true_parameters": session["ground_truth_parameters"],
                "inferred_parameters": inference["inferred_parameters"],
                "parameter_recovery": recovery,
                "fit_confidence": float(inference["fit_confidence"]),
                "parameter_estimates": inference["parameter_estimates"],
            }
        )
    return rows


def build_identifiability_report() -> dict[str, Any]:
    synthetic_rows = _synthetic_identifiability_rows()
    external_rows = _external_identifiability_rows()
    all_rows = [*synthetic_rows, *external_rows]

    parameter_recovery = {}
    coupling_inputs: dict[str, list[float]] = {name: [] for name in PARAMETER_REFERENCE}
    for parameter_name in PARAMETER_REFERENCE:
        errors = [float(row["parameter_recovery"]["per_parameter_error"][parameter_name]) for row in all_rows]
        confidences = [float(row["parameter_estimates"][parameter_name]["confidence"]) for row in all_rows]
        identifiable_rows = [row for row in all_rows if row["parameter_estimates"][parameter_name]["identifiable"]]
        weak_rows = [row for row in all_rows if not row["parameter_estimates"][parameter_name]["identifiable"]]
        for row in all_rows:
            coupling_inputs[parameter_name].append(float(row["inferred_parameters"][parameter_name]))

        if weak_rows:
            interval_min = min(float(row["inferred_parameters"][parameter_name]) for row in weak_rows)
            interval_max = max(float(row["inferred_parameters"][parameter_name]) for row in weak_rows)
        else:
            interval_min = interval_max = None
        parameter_recovery[parameter_name] = {
            "mean_abs_error": _round(_safe_mean(errors)),
            "max_abs_error": _round(max(errors) if errors else 0.0),
            "mean_confidence": _round(_safe_mean(confidences)),
            "identifiable_rate": _round(len(identifiable_rows) / max(1, len(all_rows))),
            "non_identifiable_interval": [_round(interval_min), _round(interval_max)] if interval_min is not None else None,
            "coupled_with": [],
        }

    parameter_coupling = {}
    for left in PARAMETER_REFERENCE:
        parameter_coupling[left] = {}
        for right in PARAMETER_REFERENCE:
            corr = _safe_corr(coupling_inputs[left], coupling_inputs[right])
            parameter_coupling[left][right] = _round(corr)
    for parameter_name in PARAMETER_REFERENCE:
        coupled = sorted(
            (
                (other_name, abs(float(parameter_coupling[parameter_name][other_name])))
                for other_name in PARAMETER_REFERENCE
                if other_name != parameter_name
            ),
            key=lambda item: (-item[1], item[0]),
        )[:2]
        parameter_recovery[parameter_name]["coupled_with"] = [name for name, _score in coupled]

    return {
        "analysis_type": "parameter_identifiability_report",
        "sample_counts": {"synthetic": len(synthetic_rows), "external": len(external_rows), "total": len(all_rows)},
        "parameter_recovery": parameter_recovery,
        "parameter_coupling": parameter_coupling,
        "non_identifiable_parameters": sorted(
            name for name, payload in parameter_recovery.items() if float(payload["identifiable_rate"]) < 0.75
        ),
        "summary": {
            "mean_parameter_error": _round(
                _safe_mean([float(payload["mean_abs_error"]) for payload in parameter_recovery.values()])
            ),
            "best_identified_parameters": sorted(
                parameter_recovery,
                key=lambda name: (-float(parameter_recovery[name]["identifiable_rate"]), float(parameter_recovery[name]["mean_abs_error"])),
            )[:3],
            "fragile_parameters": sorted(
                parameter_recovery,
                key=lambda name: (float(parameter_recovery[name]["identifiable_rate"]), -float(parameter_recovery[name]["mean_abs_error"])),
            )[:3],
        },
        "evidence_rows": all_rows,
    }
