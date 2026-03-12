from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import copysign, exp, log, log1p
from typing import Callable


CANONICAL_OUTCOMES = (
    "survival_threat",
    "integrity_loss",
    "resource_loss",
    "neutral",
    "resource_gain",
)

LABEL_ALIASES = {
    "survival": "survival_threat",
    "survival_threat": "survival_threat",
    "integrity": "integrity_loss",
    "integrity_loss": "integrity_loss",
    "resource": "resource_loss",
    "resource_loss": "resource_loss",
    "neutral": "neutral",
    "resource_gain": "resource_gain",
}


def _coerce_float_dict(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


@dataclass(frozen=True)
class PreferenceEvaluation:
    outcome: str
    log_preference: float
    value_score: float
    preferred_probability: float
    log_preferred_probability: float
    risk: float


@dataclass(frozen=True)
class PreferenceModel:
    """Probabilistic C-matrix style preferences over discrete outcomes."""

    survival_threat: float = -1000.0
    integrity_loss: float = -100.0
    resource_loss: float = -10.0
    neutral: float = 0.0
    resource_gain: float = 5.0

    @property
    def outcomes(self) -> tuple[str, ...]:
        return CANONICAL_OUTCOMES

    @property
    def survival(self) -> float:
        return self.survival_threat

    @property
    def integrity(self) -> float:
        return self.integrity_loss

    def _canonical_label(self, label: str) -> str:
        try:
            return LABEL_ALIASES[label]
        except KeyError as exc:
            raise ValueError(f"unknown preference label: {label}") from exc

    @property
    def log_preferences(self) -> tuple[float, ...]:
        return tuple(self.score(label) for label in self.outcomes)

    @property
    def log_probability_distribution(self) -> dict[str, float]:
        preferences = self.log_preferences
        anchor = max(preferences)
        partition = anchor + log(sum(exp(value - anchor) for value in preferences))
        return {
            label: value - partition
            for label, value in zip(self.outcomes, preferences)
        }

    @property
    def probability_distribution(self) -> dict[str, float]:
        return {
            label: exp(log_probability)
            for label, log_probability in self.log_probability_distribution.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "outcomes": list(self.outcomes),
            "log_preferences": {
                label: self.score(label) for label in self.outcomes
            },
            "probabilities": self.probability_distribution,
            "log_probabilities": self.log_probability_distribution,
        }

    def legacy_value_hierarchy_dict(self) -> dict[str, float]:
        return {
            "survival": self.survival_threat,
            "integrity": self.integrity_loss,
            "resource_loss": self.resource_loss,
            "resource": self.resource_loss,
            "neutral": self.neutral,
            "resource_gain": self.resource_gain,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> PreferenceModel:
        if not payload:
            return cls()

        if "log_preferences" in payload:
            raw_preferences = payload.get("log_preferences")
            if isinstance(raw_preferences, dict):
                payload = raw_preferences
            elif isinstance(raw_preferences, list):
                payload = {
                    label: value
                    for label, value in zip(
                        payload.get("outcomes", CANONICAL_OUTCOMES),
                        raw_preferences,
                    )
                }

        canonical: dict[str, float] = {
            "survival_threat": cls().survival_threat,
            "integrity_loss": cls().integrity_loss,
            "resource_loss": cls().resource_loss,
            "neutral": cls().neutral,
            "resource_gain": cls().resource_gain,
        }
        for label, value in payload.items():
            if label in {"outcomes", "probabilities", "log_probabilities"}:
                continue
            try:
                canonical[LABEL_ALIASES[str(label)]] = float(value)
            except (KeyError, TypeError, ValueError):
                continue
        return cls(**canonical)

    def score(self, label: str) -> float:
        return float(getattr(self, self._canonical_label(label)))

    def normalized_score(self, label: str) -> float:
        raw = self.score(label)
        scale = max(abs(value) for value in self.log_preferences) or 1.0
        if raw == 0.0:
            return 0.0
        return copysign(log1p(abs(raw)) / log1p(scale), raw)

    def preferred_probability(self, label: str) -> float:
        return self.probability_distribution[self._canonical_label(label)]

    def log_preferred_probability(self, label: str) -> float:
        return self.log_probability_distribution[self._canonical_label(label)]

    def risk(self, label: str) -> float:
        return -self.log_preferred_probability(label)

    def hierarchy_weight(self, label: str) -> float:
        weights = {
            "survival_threat": 1.00,
            "integrity_loss": 0.80,
            "resource_loss": 0.60,
            "neutral": 0.25,
            "resource_gain": 0.10,
        }
        return weights[self._canonical_label(label)]

    def goal_weight(self, goal: object | None) -> float:
        goal_name = getattr(goal, "name", str(goal or "")).upper()
        weights = {
            "SURVIVAL": 1.00,
            "INTEGRITY": 0.85,
            "CONTROL": 0.70,
            "RESOURCES": 0.55,
            "SOCIAL": 0.40,
        }
        return weights.get(goal_name, 0.60)

    def weighted_risk(
        self,
        label: str,
        goal: object | None = None,
        *,
        baseline: float | None = None,
    ) -> float:
        risk = self.risk(label) if baseline is None else float(baseline)
        return risk * (1.0 + 0.35 * self.hierarchy_weight(label) + 0.20 * self.goal_weight(goal))

    def expected_free_energy(
        self,
        *,
        outcome: str,
        predicted_error: float,
        action_ambiguity: float,
        goal: object | None = None,
        baseline_risk: float | None = None,
    ) -> float:
        return self.weighted_risk(outcome, goal, baseline=baseline_risk) + predicted_error + action_ambiguity

    def map_state_to_outcome(self, predicted_state: dict[str, object]) -> str:
        body_state = _coerce_float_dict(predicted_state.get("body_state"))
        observation = _coerce_float_dict(predicted_state.get("observation"))
        predicted_outcome = _coerce_float_dict(
            predicted_state.get("predicted_outcome", predicted_state.get("outcome"))
        )
        energy = body_state.get("energy", 0.5)
        stress = body_state.get("stress", 0.0)
        fatigue = body_state.get("fatigue", 0.0)
        temperature = body_state.get("temperature", 0.5)
        danger = observation.get("danger", 0.0)
        free_energy_drop = float(predicted_outcome.get("free_energy_drop", 0.0))
        energy_delta = float(predicted_outcome.get("energy_delta", 0.0))
        stress_delta = float(predicted_outcome.get("stress_delta", 0.0))
        fatigue_delta = float(predicted_outcome.get("fatigue_delta", 0.0))
        temperature_delta = float(predicted_outcome.get("temperature_delta", 0.0))

        if energy <= 0.15 or danger >= 0.85 or free_energy_drop <= -0.30:
            return "survival_threat"
        if (
            stress >= 0.70
            or fatigue >= 0.80
            or abs(temperature - 0.5) >= 0.20
            or stress_delta >= 0.20
            or fatigue_delta >= 0.20
            or abs(temperature_delta) >= 0.15
        ):
            return "integrity_loss"
        if free_energy_drop > 0.0 or energy_delta > 0.0:
            return "resource_gain"
        if free_energy_drop < 0.0 or energy_delta < 0.0 or stress_delta > 0.0:
            return "resource_loss"
        return "neutral"

    def evaluate_state(self, predicted_state: dict[str, object]) -> PreferenceEvaluation:
        outcome = self.map_state_to_outcome(predicted_state)
        return PreferenceEvaluation(
            outcome=outcome,
            log_preference=self.score(outcome),
            value_score=self.normalized_score(outcome),
            preferred_probability=self.preferred_probability(outcome),
            log_preferred_probability=self.log_preferred_probability(outcome),
            risk=self.risk(outcome),
        )

    def classify(
        self,
        *,
        state_snapshot: dict[str, object],
        outcome: dict[str, float],
    ) -> str:
        return self.map_state_to_outcome(
            {
                **state_snapshot,
                "predicted_outcome": dict(outcome),
            }
        )

    def evaluate_details(
        self,
        *,
        state_snapshot: dict[str, object],
        outcome: dict[str, float],
    ) -> tuple[str, float, float]:
        evaluation = self.evaluate_state(
            {
                **state_snapshot,
                "predicted_outcome": dict(outcome),
            }
        )
        return evaluation.outcome, evaluation.log_preference, evaluation.value_score

    def evaluate(
        self,
        *,
        state_snapshot: dict[str, object],
        outcome: dict[str, float],
    ) -> float:
        return self.evaluate_details(
            state_snapshot=state_snapshot,
            outcome=outcome,
        )[2]


ValueHierarchy = PreferenceModel


class Goal(Enum):
    SURVIVAL = 1
    INTEGRITY = 2
    CONTROL = 3
    RESOURCES = 4
    SOCIAL = 5


CONFLICT_THRESHOLD = 0.10
MAX_CONFLICTS = 200
MAX_WEIGHT_ADJUSTMENTS = 200
HIGH_CONFLICT_SURPRISE = 3.0


@dataclass
class ValueConflict:
    """An explicit record of a goal conflict and its arbitration."""

    conflict_id: str
    tick: int
    competing_goals: list[tuple[str, float]]
    winner: str
    resolution_reason: str
    context: dict[str, object]
    action_chosen: str = ""
    outcome_surprise: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "conflict_id": self.conflict_id,
            "tick": self.tick,
            "competing_goals": [[name, score] for name, score in self.competing_goals],
            "winner": self.winner,
            "resolution_reason": self.resolution_reason,
            "context": dict(self.context),
            "action_chosen": self.action_chosen,
            "outcome_surprise": self.outcome_surprise,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> ValueConflict:
        if not payload:
            return cls(
                conflict_id="vc_0000",
                tick=0,
                competing_goals=[],
                winner="CONTROL",
                resolution_reason="",
                context={},
            )
        competing_goals: list[tuple[str, float]] = []
        for item in payload.get("competing_goals", []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                competing_goals.append((str(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
        context = payload.get("context")
        if not isinstance(context, dict):
            context = {}
        raw_surprise = payload.get("outcome_surprise")
        return cls(
            conflict_id=str(payload.get("conflict_id", "vc_0000")),
            tick=int(payload.get("tick", 0)),
            competing_goals=competing_goals,
            winner=str(payload.get("winner", "CONTROL")),
            resolution_reason=str(payload.get("resolution_reason", "")),
            context=dict(context),
            action_chosen=str(payload.get("action_chosen", "")),
            outcome_surprise=(
                float(raw_surprise) if isinstance(raw_surprise, (int, float)) else None
            ),
        )

    def to_log_string(self) -> str:
        competing = " vs ".join(
            f"{name}({score:.2f})" for name, score in self.competing_goals
        ) or "n/a"
        lines = [
            f"[VALUE_CONFLICT] tick={self.tick} conflict_id={self.conflict_id}",
            f"  competing: {competing}",
            f"  context: {self.context}",
            f"  resolution: {self.winner} wins",
            f"  reason: {self.resolution_reason}",
            f"  action_chosen: {self.action_chosen or 'pending'}",
        ]
        if self.outcome_surprise is not None:
            lines.append(f"  outcome_surprise: {self.outcome_surprise:.2f}")
        return "\n".join(lines)


@dataclass
class WeightAdjustment:
    tick: int
    goal: str
    old_weight: float
    new_weight: float
    direction: str
    trigger_conflict_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "goal": self.goal,
            "old_weight": self.old_weight,
            "new_weight": self.new_weight,
            "direction": self.direction,
            "trigger_conflict_ids": list(self.trigger_conflict_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> WeightAdjustment:
        if not payload:
            return cls(tick=0, goal="CONTROL", old_weight=0.0, new_weight=0.0, direction="increased")
        return cls(
            tick=int(payload.get("tick", 0)),
            goal=str(payload.get("goal", "CONTROL")),
            old_weight=float(payload.get("old_weight", 0.0)),
            new_weight=float(payload.get("new_weight", 0.0)),
            direction=str(payload.get("direction", "increased")),
            trigger_conflict_ids=[str(item) for item in payload.get("trigger_conflict_ids", [])],
            reason=str(payload.get("reason", "")),
        )


@dataclass
class GoalStack:
    """Explicit long-horizon goal prioritization layered above outcome preferences."""

    base_weights: dict[Goal, float] = field(
        default_factory=lambda: {
            Goal.SURVIVAL: 0.90,
            Goal.INTEGRITY: 0.80,
            Goal.CONTROL: 0.65,
            Goal.RESOURCES: 0.55,
            Goal.SOCIAL: 0.20,
        }
    )
    active_goal: Goal = Goal.CONTROL
    goal_history: list[tuple[int, Goal]] = field(default_factory=list)
    free_energy_window: int = 5
    conflict_history: list[ValueConflict] = field(default_factory=list)
    weight_adjustments: list[WeightAdjustment] = field(default_factory=list)
    conflict_threshold: float = CONFLICT_THRESHOLD
    last_review_tick: int = 0
    conflict_counter: int = 0
    last_chapter_signal: str | None = None
    log_sink: Callable[[str], None] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "base_weights": {
                goal.name: float(weight) for goal, weight in self.base_weights.items()
            },
            "active_goal": self.active_goal.name,
            "goal_history": [[int(tick), goal.name] for tick, goal in self.goal_history],
            "free_energy_window": self.free_energy_window,
            "conflict_history": [conflict.to_dict() for conflict in self.conflict_history],
            "weight_adjustments": [item.to_dict() for item in self.weight_adjustments],
            "conflict_threshold": self.conflict_threshold,
            "last_review_tick": self.last_review_tick,
            "conflict_counter": self.conflict_counter,
            "last_chapter_signal": self.last_chapter_signal,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> GoalStack:
        stack = cls()
        if not payload:
            return stack
        base_weights: dict[Goal, float] = dict(stack.base_weights)
        raw_base_weights = payload.get("base_weights")
        if isinstance(raw_base_weights, dict):
            for key, value in raw_base_weights.items():
                try:
                    base_weights[Goal[str(key)]] = float(value)
                except (KeyError, TypeError, ValueError):
                    continue
        active_goal = stack.active_goal
        try:
            active_goal = Goal[str(payload.get("active_goal", active_goal.name))]
        except KeyError:
            active_goal = stack.active_goal
        goal_history: list[tuple[int, Goal]] = []
        for item in payload.get("goal_history", []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                goal_history.append((int(item[0]), Goal[str(item[1])]))
            except (KeyError, TypeError, ValueError):
                continue
        conflict_history = [
            ValueConflict.from_dict(item)
            for item in payload.get("conflict_history", [])
            if isinstance(item, dict)
        ]
        weight_adjustments = [
            WeightAdjustment.from_dict(item)
            for item in payload.get("weight_adjustments", [])
            if isinstance(item, dict)
        ]
        return cls(
            base_weights=base_weights,
            active_goal=active_goal,
            goal_history=goal_history,
            free_energy_window=int(payload.get("free_energy_window", stack.free_energy_window)),
            conflict_history=conflict_history[-MAX_CONFLICTS:],
            weight_adjustments=weight_adjustments[-MAX_WEIGHT_ADJUSTMENTS:],
            conflict_threshold=float(payload.get("conflict_threshold", CONFLICT_THRESHOLD)),
            last_review_tick=int(payload.get("last_review_tick", 0)),
            conflict_counter=int(payload.get("conflict_counter", len(conflict_history))),
            last_chapter_signal=(
                str(payload.get("last_chapter_signal"))
                if payload.get("last_chapter_signal")
                else None
            ),
        )

    def evaluate_priority(
        self,
        agent_state: dict[str, object],
        *,
        record_conflict: bool = False,
    ) -> Goal:
        body_state = _coerce_float_dict(agent_state.get("body_state"))
        energy = body_state.get("energy", 0.5)
        free_energy_history = [
            float(value)
            for value in agent_state.get("free_energy_history", [])
            if isinstance(value, (int, float))
        ]
        urgencies = self._goal_urgencies(agent_state)
        if energy < 0.20:
            winner = Goal.SURVIVAL
        elif energy < 0.50:
            winner = Goal.RESOURCES
        elif self._free_energy_is_rising(free_energy_history):
            winner = Goal.CONTROL
        else:
            winner = max(self.base_weights.items(), key=lambda item: (item[1], -item[0].value))[0]
        if record_conflict:
            self._record_conflict(agent_state, urgencies, winner)
        return winner

    def update_active_goal(self, tick: int, agent_state: dict[str, object]) -> Goal:
        snapshot = dict(agent_state)
        snapshot["tick"] = tick
        goal = self.evaluate_priority(snapshot, record_conflict=True)
        self.active_goal = goal
        if not self.goal_history or self.goal_history[-1] != (tick, goal):
            self.goal_history.append((tick, goal))
        return goal

    def get_goal_context_for_decision(
        self,
        agent_state: dict[str, object],
        *,
        tick: int | None = None,
    ) -> dict[str, object]:
        if tick is not None:
            goal = self.update_active_goal(tick, agent_state)
            latest_conflict = next(
                (conflict for conflict in reversed(self.conflict_history) if conflict.tick == tick),
                None,
            )
        else:
            goal = self.evaluate_priority(agent_state, record_conflict=False)
            latest_conflict = None
        urgency_scores = {
            item_goal.name: score
            for item_goal, score in self._goal_urgencies(agent_state).items()
        }
        return {
            "active_goal": goal.name,
            "goal_weight": float(self.base_weights.get(goal, 0.0)),
            "preferred_action_types": self._preferred_actions(goal),
            "conflict_id": latest_conflict.conflict_id if latest_conflict else None,
            "resolution_reason": latest_conflict.resolution_reason if latest_conflict else "",
            "urgency_scores": urgency_scores,
        }

    def note_action_choice(self, tick: int, action: str) -> None:
        conflict = next(
            (
                item
                for item in reversed(self.conflict_history)
                if item.tick == tick and not item.action_chosen
            ),
            None,
        )
        if conflict is not None:
            conflict.action_chosen = action

    def backfill_conflict_outcome(self, tick: int, outcome_surprise: float) -> None:
        conflict = next(
            (
                item
                for item in reversed(self.conflict_history)
                if item.tick == tick and item.outcome_surprise is None
            ),
            None,
        )
        if conflict is None:
            return
        conflict.outcome_surprise = float(outcome_surprise)
        if self.log_sink is not None:
            self.log_sink(conflict.to_log_string())

    def review_conflicts(self, current_tick: int) -> list[WeightAdjustment]:
        recent_conflicts = [
            conflict
            for conflict in self.conflict_history
            if conflict.tick > self.last_review_tick
        ]
        if not recent_conflicts:
            self.last_chapter_signal = None
            self.last_review_tick = current_tick
            return []
        previous_top_two = [goal.name for goal, _score in self._top_goals()[:2]]
        adjustments: list[WeightAdjustment] = []
        repeated_pairs: dict[tuple[str, str], list[ValueConflict]] = {}
        for conflict in recent_conflicts:
            goal_names = sorted(name for name, _score in conflict.competing_goals[:2])
            if len(goal_names) == 2:
                repeated_pairs.setdefault((goal_names[0], goal_names[1]), []).append(conflict)
            if conflict.outcome_surprise is None or conflict.outcome_surprise <= HIGH_CONFLICT_SURPRISE:
                continue
            loser = next(
                (name for name, _score in conflict.competing_goals if name != conflict.winner),
                None,
            )
            if loser is None:
                continue
            adjustments.extend(
                self._apply_weight_shift(
                    winner=conflict.winner,
                    loser=loser,
                    tick=current_tick,
                    trigger_conflict_ids=[conflict.conflict_id],
                    surprise=conflict.outcome_surprise,
                )
            )
        for conflicts in repeated_pairs.values():
            if len(conflicts) > 3:
                self.conflict_threshold = min(0.25, self.conflict_threshold + 0.02)
        if adjustments:
            self.weight_adjustments.extend(adjustments)
            self.weight_adjustments = self.weight_adjustments[-MAX_WEIGHT_ADJUSTMENTS:]
        self.last_review_tick = current_tick
        current_top_two = [goal.name for goal, _score in self._top_goals()[:2]]
        if previous_top_two and current_top_two and previous_top_two[0] != current_top_two[0]:
            ticks = [conflict.tick for conflict in recent_conflicts[:5]]
            self.last_chapter_signal = (
                f"Goal priority shifted: {current_top_two[0]} overtook {previous_top_two[0]} "
                f"due to conflicts at ticks {ticks}"
            )
        else:
            self.last_chapter_signal = None
        return adjustments

    def consume_chapter_signal(self) -> str | None:
        signal = self.last_chapter_signal
        self.last_chapter_signal = None
        return signal

    def goal_alignment_score(
        self,
        *,
        goal: Goal,
        action: str,
        projected_state: dict[str, float],
        predicted_effects: dict[str, float],
        current_state: dict[str, object],
    ) -> float:
        body_state = _coerce_float_dict(current_state.get("body_state"))
        energy = body_state.get("energy", 0.5)
        stress = body_state.get("stress", 0.0)
        fatigue = body_state.get("fatigue", 0.0)
        temperature = body_state.get("temperature", 0.5)
        observation = _coerce_float_dict(current_state.get("observation"))
        danger = observation.get("danger", 0.0)

        score = 0.0
        if goal == Goal.SURVIVAL:
            score += max(0.0, energy - 0.20) * 0.8
            score += max(0.0, 0.35 - projected_state.get("danger", danger)) * 1.2
            if action in {"hide", "rest", "exploit_shelter"}:
                score += 0.2
        elif goal == Goal.INTEGRITY:
            score += max(0.0, stress - projected_state.get("stress", stress)) * 0.8
            score += max(0.0, fatigue - projected_state.get("fatigue", fatigue)) * 0.6
            score += max(
                0.0,
                abs(temperature - 0.5) - abs(projected_state.get("temperature", temperature) - 0.5),
            ) * 0.8
            if action in {"rest", "thermoregulate", "exploit_shelter"}:
                score += 0.2
        elif goal == Goal.CONTROL:
            score += max(0.0, predicted_effects.get("free_energy_drop", 0.0)) * 1.5
            score += max(0.0, 0.30 - projected_state.get("danger", danger)) * 0.4
            if action in {"scan", "hide"}:
                score += 0.15
        elif goal == Goal.RESOURCES:
            score += max(0.0, predicted_effects.get("energy_delta", 0.0)) * 1.4
            score += max(0.0, predicted_effects.get("free_energy_drop", 0.0)) * 0.6
            if action in {"forage", "rest"}:
                score += 0.2
        elif goal == Goal.SOCIAL:
            score += max(0.0, projected_state.get("social", 0.0) - observation.get("social", 0.0)) * 1.2
            if action == "seek_contact":
                score += 0.25
        return max(-1.0, min(1.0, score))

    def _goal_urgencies(self, agent_state: dict[str, object]) -> dict[Goal, float]:
        body_state = _coerce_float_dict(agent_state.get("body_state"))
        energy = body_state.get("energy", 0.5)
        free_energy_history = [
            float(value)
            for value in agent_state.get("free_energy_history", [])
            if isinstance(value, (int, float))
        ]
        rising_steps = self._free_energy_rising_steps(free_energy_history)
        urgencies = dict(self.base_weights)
        urgencies[Goal.SURVIVAL] += max(0.0, 0.20 - energy) * 2.5
        urgencies[Goal.RESOURCES] += max(0.0, 0.50 - energy) * 1.2
        urgencies[Goal.CONTROL] += rising_steps * 0.01
        urgencies[Goal.INTEGRITY] += max(0.0, body_state.get("stress", 0.0) - 0.50) * 0.3
        urgencies[Goal.SOCIAL] += max(0.0, 0.20 - _coerce_float_dict(agent_state.get("observation")).get("social", 0.2)) * 0.2
        return urgencies

    def _record_conflict(
        self,
        agent_state: dict[str, object],
        urgencies: dict[Goal, float],
        winner: Goal,
    ) -> None:
        ranked = sorted(urgencies.items(), key=lambda item: (-item[1], item[0].value))
        if len(ranked) < 2:
            return
        top_score = ranked[0][1]
        contenders = [item for item in ranked if abs(top_score - item[1]) < self.conflict_threshold][:3]
        if len(contenders) < 2:
            return
        tick = int(agent_state.get("tick", -1))
        free_energy_history = [
            float(value)
            for value in agent_state.get("free_energy_history", [])
            if isinstance(value, (int, float))
        ]
        self.conflict_counter += 1
        conflict = ValueConflict(
            conflict_id=f"vc_{self.conflict_counter:04d}",
            tick=tick,
            competing_goals=[(goal.name, round(score, 4)) for goal, score in contenders],
            winner=winner.name,
            resolution_reason=self._resolution_reason(winner, agent_state, free_energy_history),
            context={
                "energy": round(_coerce_float_dict(agent_state.get("body_state")).get("energy", 0.5), 3),
                "free_energy_trend": self._free_energy_context(free_energy_history),
                "threat_level": self._threat_level(agent_state),
            },
        )
        self.conflict_history.append(conflict)
        self.conflict_history = self.conflict_history[-MAX_CONFLICTS:]
        if self.log_sink is not None:
            self.log_sink(conflict.to_log_string())

    def _resolution_reason(
        self,
        winner: Goal,
        agent_state: dict[str, object],
        free_energy_history: list[float],
    ) -> str:
        energy = _coerce_float_dict(agent_state.get("body_state")).get("energy", 0.5)
        if winner == Goal.SURVIVAL and energy < 0.20:
            return "energy below 20% threshold triggers survival priority override"
        if winner == Goal.RESOURCES and energy < 0.50:
            threshold = 40 if energy < 0.40 else 50
            return f"energy below {threshold}% threshold triggers resource priority override"
        if winner == Goal.CONTROL and self._free_energy_is_rising(free_energy_history):
            return (
                f"free energy rising for {self._free_energy_rising_steps(free_energy_history)} steps "
                "triggers control priority override"
            )
        return f"base goal weights favor {winner.name}"

    def _apply_weight_shift(
        self,
        *,
        winner: str,
        loser: str,
        tick: int,
        trigger_conflict_ids: list[str],
        surprise: float,
    ) -> list[WeightAdjustment]:
        winner_goal = Goal[winner]
        loser_goal = Goal[loser]
        delta = min(0.08, 0.02 + max(0.0, surprise - HIGH_CONFLICT_SURPRISE) * 0.01)
        winner_old = float(self.base_weights[winner_goal])
        loser_old = float(self.base_weights[loser_goal])
        self.base_weights[winner_goal] = max(0.05, winner_old - delta)
        self.base_weights[loser_goal] = min(1.25, loser_old + delta)
        return [
            WeightAdjustment(
                tick=tick,
                goal=winner_goal.name,
                old_weight=winner_old,
                new_weight=self.base_weights[winner_goal],
                direction="decreased",
                trigger_conflict_ids=trigger_conflict_ids,
                reason=f"{winner_goal.name} won but produced high surprise, so its base weight was reduced",
            ),
            WeightAdjustment(
                tick=tick,
                goal=loser_goal.name,
                old_weight=loser_old,
                new_weight=self.base_weights[loser_goal],
                direction="increased",
                trigger_conflict_ids=trigger_conflict_ids,
                reason=f"{loser_goal.name} was elevated after {winner_goal.name} underperformed in conflict resolution",
            ),
        ]

    def _top_goals(self) -> list[tuple[Goal, float]]:
        return sorted(self.base_weights.items(), key=lambda item: (-item[1], item[0].value))

    def _free_energy_is_rising(self, free_energy_history: list[float]) -> bool:
        window = free_energy_history[-self.free_energy_window :]
        if len(window) < self.free_energy_window:
            return False
        return all(right > left for left, right in zip(window, window[1:]))

    def _free_energy_rising_steps(self, free_energy_history: list[float]) -> int:
        window = free_energy_history[-self.free_energy_window :]
        if len(window) < 2:
            return 0
        streak = 0
        for left, right in zip(window, window[1:]):
            if right > left:
                streak += 1
            else:
                streak = 0
        return streak

    def _free_energy_context(self, free_energy_history: list[float]) -> str:
        steps = self._free_energy_rising_steps(free_energy_history)
        if steps <= 0:
            return "stable"
        return f"rising({steps} ticks)"

    def _threat_level(self, agent_state: dict[str, object]) -> str:
        danger = _coerce_float_dict(agent_state.get("observation")).get("danger", 0.0)
        if danger >= 0.75:
            return "high"
        if danger >= 0.40:
            return "medium"
        return "low"

    def _preferred_actions(self, goal: Goal) -> list[str]:
        mapping = {
            Goal.SURVIVAL: ["hide", "rest", "exploit_shelter"],
            Goal.INTEGRITY: ["rest", "thermoregulate", "exploit_shelter"],
            Goal.CONTROL: ["scan", "hide"],
            Goal.RESOURCES: ["forage", "rest"],
            Goal.SOCIAL: ["seek_contact"],
        }
        return list(mapping.get(goal, []))
