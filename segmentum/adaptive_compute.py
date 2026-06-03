from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .environment import clamp


LOW_CONFLICT_FLATNESS_MAX = 0.24
HIGH_CONFLICT_DENSITY_MIN = 0.42
LOW_ERROR_FE_MAX = 0.24
HIGH_ERROR_FE_MIN = 0.42
HIGH_URGENCY_GOAL_MIN = 0.72


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AdaptiveComputeDecision:
    confidence_regime: str
    retrieval_k: int
    path_neighborhood_k: int
    verification_target_limit: int
    candidate_action_limit: int
    counterfactual_max_depth: int
    counterfactual_energy_budget: float
    field_refinement_enabled: bool
    escalation_reason_codes: tuple[str, ...]
    field_flatness: float
    conflict_density: float
    prediction_error_surrogate: float
    goal_urgency: float
    projected_subsequent_fe: float
    escalation_no_gain: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_regime": self.confidence_regime,
            "retrieval_k": int(self.retrieval_k),
            "path_neighborhood_k": int(self.path_neighborhood_k),
            "verification_target_limit": int(self.verification_target_limit),
            "candidate_action_limit": int(self.candidate_action_limit),
            "counterfactual_max_depth": int(self.counterfactual_max_depth),
            "counterfactual_energy_budget": round(float(self.counterfactual_energy_budget), 6),
            "field_refinement_enabled": bool(self.field_refinement_enabled),
            "escalation_reason_codes": list(self.escalation_reason_codes),
            "field_flatness": round(float(self.field_flatness), 6),
            "conflict_density": round(float(self.conflict_density), 6),
            "prediction_error_surrogate": round(float(self.prediction_error_surrogate), 6),
            "goal_urgency": round(float(self.goal_urgency), 6),
            "projected_subsequent_fe": round(float(self.projected_subsequent_fe), 6),
            "escalation_no_gain": bool(self.escalation_no_gain),
        }


def _field_metric(field: Mapping[str, object] | None, name: str, default: float = 0.0) -> float:
    if not isinstance(field, Mapping):
        return default
    return _coerce_float(field.get(name), default)


def fixed_budget_decision(
    *,
    base_retrieval_k: int = 3,
    base_path_k: int = 2,
    candidate_action_limit: int = 6,
) -> AdaptiveComputeDecision:
    return AdaptiveComputeDecision(
        confidence_regime="medium",
        retrieval_k=max(1, int(base_retrieval_k)),
        path_neighborhood_k=max(1, int(base_path_k)),
        verification_target_limit=2,
        candidate_action_limit=max(2, int(candidate_action_limit)),
        counterfactual_max_depth=3,
        counterfactual_energy_budget=0.08,
        field_refinement_enabled=True,
        escalation_reason_codes=("fixed_budget_fallback",),
        field_flatness=0.0,
        conflict_density=0.0,
        prediction_error_surrogate=0.0,
        goal_urgency=0.0,
        projected_subsequent_fe=0.0,
    )


def decide_adaptive_compute(
    *,
    field: Mapping[str, object] | None,
    goal_context: Mapping[str, object] | None,
    prediction_error_surrogate: float,
    base_retrieval_k: int,
    base_path_k: int,
    candidate_action_count: int,
) -> AdaptiveComputeDecision:
    field_flatness = clamp(_field_metric(field, "field_flatness", 0.0))
    conflict_density = clamp(_field_metric(field, "conflict_density", 0.0))
    projected_subsequent_fe = max(
        0.0,
        _field_metric(
            field.get("counterfactual_audit") if isinstance(field, Mapping) else None,
            "chosen_decision_subsequent_fe",
            prediction_error_surrogate,
        ),
    )
    goal_name = str(goal_context.get("active_goal", "")) if isinstance(goal_context, Mapping) else ""
    urgency_scores = goal_context.get("urgency_scores") if isinstance(goal_context, Mapping) else {}
    goal_urgency = clamp(
        _coerce_float(urgency_scores.get(goal_name), 0.0) if isinstance(urgency_scores, Mapping) else 0.0
    )
    reasons: list[str] = []
    high_conflict = conflict_density >= HIGH_CONFLICT_DENSITY_MIN
    high_flatness = field_flatness >= 0.62
    high_error = prediction_error_surrogate >= HIGH_ERROR_FE_MIN
    urgent_borderline = goal_urgency >= HIGH_URGENCY_GOAL_MIN and (
        conflict_density >= 0.28 or field_flatness >= 0.38 or prediction_error_surrogate >= 0.32
    )
    low_conflict = conflict_density <= 0.16
    low_flatness = field_flatness <= LOW_CONFLICT_FLATNESS_MAX
    low_error = prediction_error_surrogate <= LOW_ERROR_FE_MAX

    if high_conflict or high_flatness or high_error or urgent_borderline:
        regime = "high"
        retrieval_k = max(base_retrieval_k, 5)
        path_k = max(base_path_k, 4)
        verification_limit = 4
        candidate_limit = max(6, min(8, int(candidate_action_count)))
        cf_depth = 4
        cf_energy = 0.14
        field_refinement_enabled = True
        if high_conflict:
            reasons.append("conflict_dense_field")
        if high_flatness:
            reasons.append("flat_field_requires_more_evidence")
        if high_error:
            reasons.append("high_prediction_error_surrogate")
        if urgent_borderline:
            reasons.append("goal_urgency_escalation")
    elif low_conflict and low_flatness and low_error:
        regime = "low"
        retrieval_k = max(1, min(base_retrieval_k, 2))
        path_k = 1
        verification_limit = 1
        candidate_limit = max(2, min(4, int(candidate_action_count)))
        cf_depth = 2
        cf_energy = 0.04
        field_refinement_enabled = False
        reasons.append("stable_low_conflict_basin")
    else:
        regime = "medium"
        retrieval_k = max(2, min(4, base_retrieval_k + (1 if conflict_density >= 0.26 else 0)))
        path_k = max(2, min(3, base_path_k + (1 if field_flatness >= 0.40 else 0)))
        verification_limit = 2
        candidate_limit = max(4, min(6, int(candidate_action_count)))
        cf_depth = 3
        cf_energy = 0.08
        field_refinement_enabled = True
        reasons.append("bounded_medium_compute")

    audit = field.get("counterfactual_audit") if isinstance(field, Mapping) else {}
    no_gain = False
    if isinstance(audit, Mapping) and audit.get("status") == "field_divergent_no_gain" and regime == "high":
        reasons.append("escalation_no_gain")
        no_gain = True

    return AdaptiveComputeDecision(
        confidence_regime=regime,
        retrieval_k=int(retrieval_k),
        path_neighborhood_k=int(path_k),
        verification_target_limit=int(verification_limit),
        candidate_action_limit=int(candidate_limit),
        counterfactual_max_depth=int(cf_depth),
        counterfactual_energy_budget=float(cf_energy),
        field_refinement_enabled=bool(field_refinement_enabled),
        escalation_reason_codes=tuple(dict.fromkeys(reasons)),
        field_flatness=round(field_flatness, 6),
        conflict_density=round(conflict_density, 6),
        prediction_error_surrogate=round(float(prediction_error_surrogate), 6),
        goal_urgency=round(goal_urgency, 6),
        projected_subsequent_fe=round(projected_subsequent_fe, 6),
        escalation_no_gain=no_gain,
    )
