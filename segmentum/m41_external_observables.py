from __future__ import annotations

"""Same-framework synthetic holdout observables.

This module keeps the historical ``external`` compatibility names, but the
observable computations here are driven by a repository-owned synthetic
holdout generator. They are useful as sidecar diagnostics only and should not
be framed as external human-data validation.
"""

import math
from typing import Any

from .m4_cognitive_style import DecisionLogRecord


SAME_FRAMEWORK_HOLDOUT_OBSERVABLES_IMPLEMENTATION_FAMILY = "same_framework_holdout_observables_v1"
EXTERNAL_OBSERVABLES_IMPLEMENTATION_FAMILY = SAME_FRAMEWORK_HOLDOUT_OBSERVABLES_IMPLEMENTATION_FAMILY

SAME_FRAMEWORK_HOLDOUT_MEASUREMENT_MISMATCHES = [
    "uncertainty- and pressure-based cohorts use shifted thresholds relative to the internal observables",
    "confidence/evidence coupling uses centered-and-clipped terms instead of direct products for several metrics",
    "repeat suppression and recovery metrics use softer normalization to preserve generator-specific measurement mismatch",
]
EXTERNAL_MEASUREMENT_MISMATCHES = SAME_FRAMEWORK_HOLDOUT_MEASUREMENT_MISMATCHES


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _proportion(records: list[DecisionLogRecord], predicate) -> float | None:
    if not records:
        return None
    matches = sum(1 for record in records if predicate(record))
    return matches / len(records)


def _metric_result(
    *,
    value: float | None,
    sample_size: int,
    min_samples: int,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "value": _round(value) if sample_size >= min_samples and value is not None else None,
        "sample_size": int(sample_size),
        "min_samples": int(min_samples),
        "insufficient_data": sample_size < min_samples or value is None,
        "implementation_family": SAME_FRAMEWORK_HOLDOUT_OBSERVABLES_IMPLEMENTATION_FAMILY,
        "measurement_mismatch": list(notes or []),
    }


def metric_values_from_payload(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric_name, payload in metrics.items():
        if isinstance(payload, dict) and not payload.get("insufficient_data", False) and payload.get("value") is not None:
            values[metric_name] = float(payload["value"])
    return values


def _candidate_action_score(record: DecisionLogRecord, action_names: set[str]) -> float | None:
    matching_scores = [
        float(candidate.get("total_score", 0.0))
        for candidate in record.candidate_actions
        if isinstance(candidate, dict)
        and isinstance(candidate.get("action"), dict)
        and str(candidate["action"].get("name", "")) in action_names
    ]
    return max(matching_scores) if matching_scores else None


def _normalized_attention_entropy(attention_allocation: dict[str, float]) -> float | None:
    values = [float(value) for value in attention_allocation.values() if float(value) > 0.0]
    if len(values) <= 1:
        return 0.0 if values else None
    entropy = -sum(value * math.log(value, 2) for value in values)
    return entropy / math.log(len(values), 2)


def compute_same_framework_holdout_observable_metrics(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = [record if isinstance(record, DecisionLogRecord) else DecisionLogRecord.from_dict(record) for record in records]

    inspect_actions = {"scan", "inspect", "query"}
    protective_actions = {"rest", "conserve", "recover"}
    conservative_actions = protective_actions | {"scan", "inspect", "query", "plan"}
    low_cost_actions = {"rest", "conserve", "recover", "scan", "inspect"}
    risky_actions = {"commit", "guess", "retry"}

    high_uncertainty = [record for record in normalized if record.observation_evidence.get("uncertainty", 0.0) >= 0.57]
    medium_uncertainty = [record for record in normalized if record.observation_evidence.get("uncertainty", 0.0) >= 0.46]
    high_error = [record for record in normalized if record.observation_evidence.get("expected_error", 0.0) >= 0.60]
    high_pred_error = [record for record in normalized if record.prediction_error >= 0.40]
    high_evidence = [record for record in normalized if record.observation_evidence.get("evidence_strength", 0.0) >= 0.66]
    high_pressure = [record for record in normalized if record.resource_state.get("stress", 0.0) >= 0.56]
    low_resource = [
        record
        for record in normalized
        if record.resource_state.get("energy", 0.0) <= 0.38 or record.resource_state.get("time_remaining", 0.0) <= 0.34
    ]
    conflict_cases = [
        record
        for record in normalized
        if record.prediction_error_vector.get("virtual_error", 0.0) > record.prediction_error_vector.get("direct_error", 0.0) + 0.02
    ]
    inspect_margin_cases = [
        record
        for record in high_uncertainty
        if _candidate_action_score(record, inspect_actions) is not None and _candidate_action_score(record, risky_actions) is not None
    ]
    expected_error_margin_cases = [
        record
        for record in high_error
        if _candidate_action_score(record, conservative_actions) is not None and _candidate_action_score(record, risky_actions) is not None
    ]
    medium_uncertainty_margin_cases = [
        record
        for record in medium_uncertainty
        if _candidate_action_score(record, inspect_actions) is not None and _candidate_action_score(record, {"commit"}) is not None
    ]
    attention_entropy_values = [
        1.0 - entropy
        for record in normalized
        for entropy in [_normalized_attention_entropy(record.attention_allocation)]
        if entropy is not None
    ]

    results = {
        "uncertainty_confidence_drop_rate": _metric_result(
            value=_mean(
                [
                    max(0.0, record.observation_evidence.get("uncertainty", 0.0) - 0.15)
                    * max(0.0, 1.0 - record.internal_confidence)
                    for record in high_uncertainty
                ]
            ),
            sample_size=len(high_uncertainty),
            min_samples=3,
            notes=["shifted uncertainty threshold", "confidence drop is centered before aggregation"],
        ),
        "high_uncertainty_inspect_ratio": _metric_result(
            value=_mean(
                [
                    1.0 / (
                        1.0
                        + math.exp(
                            -1.8
                            * (
                                _candidate_action_score(record, inspect_actions)
                                - _candidate_action_score(record, risky_actions)
                            )
                        )
                    )
                    for record in inspect_margin_cases
                ]
            ),
            sample_size=len(inspect_margin_cases),
            min_samples=3,
            notes=["shifted uncertainty threshold", "inspect preference is evaluated from candidate-score margins"],
        ),
        "high_expected_error_rejection_rate": _metric_result(
            value=_mean(
                [
                    1.0 / (
                        1.0
                        + math.exp(
                            -1.8
                            * (
                                _candidate_action_score(record, conservative_actions)
                                - _candidate_action_score(record, risky_actions)
                            )
                        )
                    )
                    for record in expected_error_margin_cases
                ]
            ),
            sample_size=len(expected_error_margin_cases),
            min_samples=3,
            notes=["expected-error cohort threshold differs from the internal evaluator", "rejection is measured from candidate-score margins"],
        ),
        "post_error_conservative_shift": _metric_result(
            value=_proportion(high_pred_error, lambda record: record.selected_action in protective_actions),
            sample_size=len(high_pred_error),
            min_samples=3,
            notes=["prediction-error trigger threshold is softened", "only strict protective actions count as post-error recovery"],
        ),
        "novel_action_ratio": _metric_result(
            value=_proportion(medium_uncertainty, lambda record: record.selected_action in inspect_actions),
            sample_size=len(medium_uncertainty),
            min_samples=4,
            notes=["medium-uncertainty cohort uses an external threshold"],
        ),
        "choice_repeat_suppression": _metric_result(
            value=_mean(
                [
                    1.0 / (
                        1.0
                        + math.exp(
                            -1.8
                            * (
                                _candidate_action_score(record, inspect_actions)
                                - _candidate_action_score(record, {"commit"})
                            )
                        )
                    )
                    for record in medium_uncertainty_margin_cases
                ]
            ),
            sample_size=len(medium_uncertainty_margin_cases),
            min_samples=4,
            notes=["medium-uncertainty cohort uses an external threshold", "exploration preference is measured from inspect-vs-commit score margins"],
        ),
        "dominant_attention_share": _metric_result(
            value=_mean(
                [
                    max(record.attention_allocation.values()) * 0.92 + min(record.attention_allocation.values()) * 0.08
                    for record in normalized
                    if record.attention_allocation
                ]
            ),
            sample_size=len(normalized),
            min_samples=6,
            notes=["attention concentration is smoothed before averaging"],
        ),
        "evidence_aligned_choice_rate": _metric_result(
            value=_mean(attention_entropy_values),
            sample_size=len(attention_entropy_values),
            min_samples=6,
            notes=["attention selectivity is measured from inverse normalized entropy"],
        ),
        "confidence_evidence_slope": _metric_result(
            value=_mean(
                [
                    max(0.0, record.observation_evidence.get("evidence_strength", 0.0) - 0.10)
                    * max(0.0, record.internal_confidence - 0.05)
                    for record in normalized
                ]
            ),
            sample_size=len(normalized),
            min_samples=6,
            notes=["evidence and confidence are centered prior to multiplication"],
        ),
        "high_evidence_commit_rate": _metric_result(
            value=_proportion(high_evidence, lambda record: record.selected_action == "commit"),
            sample_size=len(high_evidence),
            min_samples=3,
            notes=["high-evidence threshold differs from the internal evaluator"],
        ),
        "mean_update_inverse": _metric_result(
            value=(1.0 - _mean([record.model_update.get("magnitude", 0.0) * 0.94 for record in normalized])) if normalized else None,
            sample_size=len(normalized),
            min_samples=6,
            notes=["update magnitude is scaled before inversion"],
        ),
        "strategy_persistence_after_error": _metric_result(
            value=_mean([1.0 - min(1.0, record.model_update.get("strategy_shift", 0.0) * 0.92) for record in high_pred_error]),
            sample_size=len(high_pred_error),
            min_samples=3,
            notes=["strategy-shift penalty is softened under external dynamics"],
        ),
        "high_pressure_low_cost_ratio": _metric_result(
            value=_proportion(high_pressure, lambda record: record.selected_action in low_cost_actions),
            sample_size=len(high_pressure),
            min_samples=3,
            notes=["pressure cohort and low-cost action set differ slightly from the internal evaluator"],
        ),
        "recovery_trigger_rate": _metric_result(
            value=_proportion(low_resource, lambda record: record.selected_action in {"rest", "recover", "conserve", "plan"}),
            sample_size=len(low_resource),
            min_samples=3,
            notes=["plan counts as a weak recovery action for same-framework synthetic workloads"],
        ),
        "conflict_avoidance_shift": _metric_result(
            value=_proportion(conflict_cases, lambda record: record.selected_action not in {"commit", "guess"}),
            sample_size=len(conflict_cases),
            min_samples=3,
            notes=["conflict avoidance treats guess as a risky commit-like action"],
        ),
        "counterfactual_loss_sensitivity": _metric_result(
            value=_mean(
                [
                    max(
                        0.0,
                        record.prediction_error_vector.get("virtual_error", 0.0)
                        - record.prediction_error_vector.get("direct_error", 0.0),
                    )
                    * (1.0 if record.selected_action in conservative_actions else 0.25 if record.selected_action == "retry" else 0.0)
                    for record in conflict_cases
                ]
            ),
            sample_size=len(conflict_cases),
            min_samples=3,
            notes=["counterfactual response gives retry a partial conservative weight"],
        ),
    }
    return results


def compute_external_observable_metrics(records: list[DecisionLogRecord | dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return compute_same_framework_holdout_observable_metrics(records)
