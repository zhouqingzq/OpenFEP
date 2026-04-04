from __future__ import annotations

"""Legacy same-framework recoverability sidecar.

The reports produced here are useful only for understanding whether synthetic
parameters can be recovered within the repository's own latent family. They are
not valid evidence for external identifiability on human data, do not define
M4.1 acceptance, and do not prove M4.2 benchmark recovery-on-task.
"""

from statistics import mean
from typing import Any

from .m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE
from .m41_external_dataset import assert_inference_path_blinded, load_same_framework_behavior_dataset
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


def synthetic_family_recoverability_report() -> dict[str, Any]:
    dataset = load_same_framework_behavior_dataset()
    rows = []
    inferred_by_parameter: dict[str, list[float]] = {name: [] for name in PARAMETER_REFERENCE}
    baseline_by_parameter: dict[str, list[float]] = {name: [] for name in PARAMETER_REFERENCE}
    blindness_checks = []
    primary_model = None
    for session in dataset["sessions"]:
        if not session.get("ground_truth_parameters"):
            continue
        blindness = assert_inference_path_blinded(session["records"])
        inference = infer_cognitive_style(session["records"], subject_id=session["subject_id"], source_name=session["source_name"])
        if primary_model is None:
            primary_model = inference.get("primary_recovery_model")
        candidate_vector = {
            "schema_version": CognitiveStyleParameters().schema_version,
            **{
                parameter_name: inference["candidate_bank_baseline"]["candidate_estimates"][parameter_name]["estimate"]
                for parameter_name in PARAMETER_REFERENCE
            },
        }
        recovery = summarize_parameter_recovery(inference["inferred_parameters"], session["ground_truth_parameters"])
        baseline_recovery = summarize_parameter_recovery(
            candidate_vector,
            session["ground_truth_parameters"],
        )
        rows.append(
            {
                "session_id": session["session_id"],
                "subject_id": session["subject_id"],
                "true_profile": session["profile_label"],
                "inferred_parameters": inference["inferred_parameters"],
                "parameter_recovery": recovery,
                "candidate_bank_recovery": baseline_recovery,
                "fit_confidence": float(inference["fit_confidence"]),
                "inference_path_blinded": blindness["inference_path_blinded"],
            }
        )
        blindness_checks.append(blindness)
        for parameter_name in PARAMETER_REFERENCE:
            inferred_by_parameter[parameter_name].append(float(inference["inferred_parameters"][parameter_name]))
            baseline_value = inference["candidate_bank_baseline"]["candidate_estimates"][parameter_name]["estimate"]
            baseline_by_parameter[parameter_name].append(float(baseline_value))

    parameter_recovery = {}
    candidate_bank_baseline = {}
    primary_vs_baseline_delta = {}
    unrecoverable = []
    for parameter_name in PARAMETER_REFERENCE:
        mae_values = [float(row["parameter_recovery"]["per_parameter_error"][parameter_name]) for row in rows]
        baseline_mae_values = [float(row["candidate_bank_recovery"]["per_parameter_error"][parameter_name]) for row in rows]
        mae = _safe_mean(mae_values)
        baseline_mae = _safe_mean(baseline_mae_values)
        identifiable = mae < 0.25
        parameter_recovery[parameter_name] = {
            "mae": _round(mae),
            "mean_abs_error": _round(mae),
            "identifiable": identifiable,
            "identifiable_rate": 1.0 if identifiable else 0.0,
            "non_identifiable_interval": None if identifiable else [0.0, 1.0],
            "coupled_with": [],
        }
        candidate_bank_baseline[parameter_name] = {
            "mae": _round(baseline_mae),
            "mean_abs_error": _round(baseline_mae),
            "recoverable": baseline_mae < 0.25,
        }
        primary_vs_baseline_delta[parameter_name] = {
            "primary_minus_baseline_mae": _round(mae - baseline_mae),
            "primary_better_or_equal": mae <= baseline_mae,
        }
        if not identifiable:
            unrecoverable.append({"parameter": parameter_name, "mae": _round(mae)})

    parameter_coupling = {}
    high_corr_pairs = []
    for left in PARAMETER_REFERENCE:
        parameter_coupling[left] = {}
        for right in PARAMETER_REFERENCE:
            corr = _safe_corr(inferred_by_parameter[left], inferred_by_parameter[right]) if left != right else 1.0
            parameter_coupling[left][right] = _round(corr)
            if left < right and abs(corr) > 0.6:
                high_corr_pairs.append({"left": left, "right": right, "correlation": _round(corr)})

    for parameter_name in PARAMETER_REFERENCE:
        ranked = sorted(
            (
                (other_name, abs(float(parameter_coupling[parameter_name][other_name])))
                for other_name in PARAMETER_REFERENCE
                if other_name != parameter_name
            ),
            key=lambda item: (-item[1], item[0]),
        )[:2]
        parameter_recovery[parameter_name]["coupled_with"] = [name for name, _score in ranked]

    return {
        "analysis_type": "same_framework_synthetic_recoverability",
        "legacy_analysis_type": "cross_generator_identifiability",
        "benchmark_scope": "same-framework synthetic recoverability sidecar",
        "claim_envelope": "sidecar_synthetic_diagnostic",
        "legacy_status": "m42_plus_preresearch_sidecar",
        "generator_family": "same_framework_synthetic_holdout",
        "validation_type": "synthetic_holdout_same_framework",
        "not_acceptance_evidence": True,
        "interpretation": "within synthetic family recoverability only",
        "sample_counts": {"total": len(rows)},
        "inference_path_blinded": all(check["inference_path_blinded"] for check in blindness_checks) if blindness_checks else True,
        "blindness_checks": blindness_checks,
        "primary_recovery_model": primary_model,
        "parameter_recovery": parameter_recovery,
        "candidate_bank_baseline": candidate_bank_baseline,
        "primary_vs_baseline_delta": primary_vs_baseline_delta,
        "parameter_coupling": parameter_coupling,
        "recoverable_parameters": sorted(name for name, payload in parameter_recovery.items() if payload["mae"] < 0.25),
        "unrecoverable_parameters": unrecoverable,
        "high_correlation_pairs": high_corr_pairs,
        "train_test_seed_overlap": 0,
        "validation_limits": [
            "training and evaluation remain inside the repository's synthetic latent family",
            "ground truth comes from synthetic parameter snapshots rather than independent external labels",
            "the report should not be used as evidence for external identifiability",
            "the report is a sidecar diagnostic only and does not count as M4.1 acceptance or M4.2 task-level recovery evidence",
        ],
        "evidence_rows": rows,
    }


def synthetic_recoverability_summary() -> dict[str, Any]:
    report = synthetic_family_recoverability_report()
    return {
        "analysis_type": "same_framework_recoverability_summary",
        "legacy_analysis_type": "parameter_identifiability_report",
        "legacy_status": report["legacy_status"],
        "claim_envelope": report["claim_envelope"],
        "validation_type": report["validation_type"],
        "interpretation": report["interpretation"],
        "sample_counts": report["sample_counts"],
        "parameter_recovery": {
            name: {
                "mean_abs_error": payload["mean_abs_error"],
                "identifiable_rate": payload["identifiable_rate"],
                "coupled_with": payload["coupled_with"],
                "non_identifiable_interval": payload["non_identifiable_interval"],
            }
            for name, payload in report["parameter_recovery"].items()
        },
        "primary_recovery_model": report["primary_recovery_model"],
        "candidate_bank_baseline": report["candidate_bank_baseline"],
        "primary_vs_baseline_delta": report["primary_vs_baseline_delta"],
        "parameter_coupling": report["parameter_coupling"],
        "validation_limits": report["validation_limits"],
        "evidence_rows": report["evidence_rows"],
    }


def cross_generator_identifiability_report() -> dict[str, Any]:
    return synthetic_family_recoverability_report()


def build_identifiability_report() -> dict[str, Any]:
    return synthetic_recoverability_summary()
