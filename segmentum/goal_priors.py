from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .environment import clamp
from .preferences import Goal


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mapping_floats(payload: Mapping[str, object] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


@dataclass(frozen=True)
class GoalPriorAdjustment:
    active_goal: str
    prior_channel_shifts: dict[str, float]
    modality_shifts: dict[str, float]
    confidence: float
    urgency: float
    contradiction_guard: float
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "active_goal": self.active_goal,
            "prior_channel_shifts": {
                key: round(float(value), 6)
                for key, value in sorted(self.prior_channel_shifts.items())
            },
            "modality_shifts": {
                key: round(float(value), 6)
                for key, value in sorted(self.modality_shifts.items())
            },
            "confidence": round(float(self.confidence), 6),
            "urgency": round(float(self.urgency), 6),
            "contradiction_guard": round(float(self.contradiction_guard), 6),
            "reason_codes": list(self.reason_codes),
        }


_CHANNEL_ORDER = (
    "safety_prior_shift",
    "energy_preservation_shift",
    "exploration_shift",
    "social_stability_shift",
    "repair_or_seek_shift",
)


_GOAL_CHANNEL_SHIFTS: dict[Goal, dict[str, float]] = {
    Goal.SURVIVAL: {
        "safety_prior_shift": 0.24,
        "energy_preservation_shift": 0.08,
        "exploration_shift": -0.18,
        "social_stability_shift": -0.04,
        "repair_or_seek_shift": -0.06,
    },
    Goal.INTEGRITY: {
        "safety_prior_shift": 0.14,
        "energy_preservation_shift": 0.14,
        "exploration_shift": -0.08,
        "social_stability_shift": 0.08,
        "repair_or_seek_shift": 0.18,
    },
    Goal.CONTROL: {
        "safety_prior_shift": 0.06,
        "energy_preservation_shift": 0.04,
        "exploration_shift": 0.10,
        "social_stability_shift": 0.02,
        "repair_or_seek_shift": 0.14,
    },
    Goal.RESOURCES: {
        "safety_prior_shift": 0.04,
        "energy_preservation_shift": 0.24,
        "exploration_shift": 0.08,
        "social_stability_shift": 0.02,
        "repair_or_seek_shift": 0.06,
    },
    Goal.SOCIAL: {
        "safety_prior_shift": -0.02,
        "energy_preservation_shift": 0.02,
        "exploration_shift": 0.04,
        "social_stability_shift": 0.24,
        "repair_or_seek_shift": 0.18,
    },
}


def _coerce_goal(active_goal: Goal | str | None) -> Goal | None:
    if isinstance(active_goal, Goal):
        return active_goal
    token = str(active_goal or "").strip().upper()
    if not token:
        return None
    try:
        return Goal[token]
    except KeyError:
        return None


def _goal_urgency(
    goal: Goal,
    goal_context: Mapping[str, object] | None,
) -> float:
    if isinstance(goal_context, Mapping):
        raw = goal_context.get("urgency_scores")
        if isinstance(raw, Mapping):
            urgency = _coerce_float(raw.get(goal.name), 0.0)
            return clamp(urgency)
    return 0.5


def _contradiction_guard(
    goal: Goal,
    current_state: Mapping[str, object] | None,
) -> tuple[float, tuple[str, ...]]:
    body_state = _mapping_floats(
        current_state.get("body_state") if isinstance(current_state, Mapping) else None
    )
    observation = _mapping_floats(
        current_state.get("observation") if isinstance(current_state, Mapping) else None
    )
    danger = clamp(observation.get("danger", 0.0))
    stress = clamp(body_state.get("stress", 0.0))
    fatigue = clamp(body_state.get("fatigue", 0.0))
    low_energy = clamp(1.0 - body_state.get("energy", 0.5))
    social_fragility = clamp(1.0 - observation.get("social", 0.5))
    reasons: list[str] = []
    guard = 1.0
    if goal in {Goal.CONTROL, Goal.RESOURCES, Goal.SOCIAL} and danger >= 0.65:
        guard *= 0.45
        reasons.append("strong_observed_danger_limits_goal_prior")
    if goal in {Goal.CONTROL, Goal.RESOURCES} and (stress >= 0.70 or fatigue >= 0.72):
        guard *= 0.70
        reasons.append("high_internal_load_limits_goal_prior")
    if goal == Goal.SOCIAL and (danger >= 0.55 or social_fragility >= 0.60):
        guard *= 0.72
        reasons.append("social_seek_guarded_by_instability")
    if goal == Goal.SURVIVAL and low_energy >= 0.70:
        reasons.append("survival_prior_reinforced_by_low_energy")
    if goal == Goal.INTEGRITY and stress >= 0.60:
        reasons.append("integrity_prior_reinforced_by_stress")
    return clamp(guard, 0.20, 1.0), tuple(reasons)


def _modality_shifts(channel_shifts: Mapping[str, float]) -> dict[str, float]:
    safety = _coerce_float(channel_shifts.get("safety_prior_shift"), 0.0)
    energy = _coerce_float(channel_shifts.get("energy_preservation_shift"), 0.0)
    exploration = _coerce_float(channel_shifts.get("exploration_shift"), 0.0)
    social = _coerce_float(channel_shifts.get("social_stability_shift"), 0.0)
    repair = _coerce_float(channel_shifts.get("repair_or_seek_shift"), 0.0)
    return {
        "danger": clamp((-0.28 * safety) + (0.08 * exploration) - (0.06 * social), -0.14, 0.14),
        "food": clamp((0.24 * energy) + (0.06 * exploration), -0.14, 0.14),
        "novelty": clamp((0.30 * exploration) - (0.10 * safety), -0.14, 0.14),
        "social": clamp((0.28 * social) + (0.10 * repair) - (0.04 * safety), -0.14, 0.14),
        "shelter": clamp((0.24 * safety) + (0.08 * repair), -0.14, 0.14),
    }


def build_goal_prior_adjustment(
    *,
    active_goal: Goal | str | None,
    current_state: Mapping[str, object] | None,
    goal_context: Mapping[str, object] | None = None,
) -> GoalPriorAdjustment | None:
    goal = _coerce_goal(active_goal)
    if goal is None:
        return None
    base_shifts = _GOAL_CHANNEL_SHIFTS.get(goal, {})
    urgency = _goal_urgency(goal, goal_context)
    guard, guard_reasons = _contradiction_guard(goal, current_state)
    scale = (0.55 + (0.45 * urgency)) * guard
    channel_shifts = {
        key: round(clamp(_coerce_float(base_shifts.get(key), 0.0) * scale, -0.30, 0.30), 6)
        for key in _CHANNEL_ORDER
    }
    modality_shifts = {
        key: round(value, 6)
        for key, value in _modality_shifts(channel_shifts).items()
    }
    reasons = [f"active_goal:{goal.name.lower()}", f"goal_urgency:{urgency:.2f}"]
    reasons.extend(guard_reasons)
    return GoalPriorAdjustment(
        active_goal=goal.name,
        prior_channel_shifts=channel_shifts,
        modality_shifts=modality_shifts,
        confidence=round(clamp(scale), 6),
        urgency=round(urgency, 6),
        contradiction_guard=round(guard, 6),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )
