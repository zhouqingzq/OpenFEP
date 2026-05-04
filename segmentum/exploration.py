"""M10.0: Self-Initiated Exploration Agenda.

Self-thought enters through the same bus path as other cognition and must not
directly edit prompts, memory, or durable self-state.

Architecture:
  MetaObserver / SelfThoughtProducer
  -> SelfThoughtEvent
  -> CognitiveEventBus
  -> AttentionGate
  -> CognitiveLoop
  -> CognitiveStateMVP / SelfAgenda
  -> optional StatePatchProposal or ResponseEvidenceContract update
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .cognitive_events import (
    CognitiveEvent,
    SELF_THOUGHT_TRIGGERS,
    make_self_thought_event,
)


# M10.0: Allowed exploration interventions
ALLOWED_INTERVENTIONS: tuple[str, ...] = (
    "ask_clarifying_question",
    "retrieve_memory_with_cue",
    "mark_claim_as_unverified",
    "defer_durable_update",
    "lower_assertiveness",
    "prefer_repair_strategy",
    "request_external_evidence",
)


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return round(max(lo, min(hi, float(value))), 6)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class ExplorationPolicy:
    """Policy governing what interventions are allowed and how budget is spent.

    Exploration never fabricates missing evidence and respects prompt/attention budgets.
    """

    max_self_thought_per_turn: int = 2
    self_thought_cooldown: int = 3
    priority_threshold: float = 0.35
    budget_cost: float = 0.15
    max_budget_per_turn: float = 0.4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def intervention_allowed(self, intervention: str) -> bool:
        return intervention in ALLOWED_INTERVENTIONS

    def priority_above_threshold(self, priority: float) -> bool:
        return float(priority) >= self.priority_threshold

    def budget_exceeded(self, spent: float) -> bool:
        return spent > self.max_budget_per_turn

    def cooldown_active(self, cooldown_remaining: int) -> bool:
        return cooldown_remaining > 0


@dataclass(frozen=True)
class LoopControl:
    """Prevents runaway reflection through budget, cooldown, and dedup controls."""

    max_self_thought_per_turn: int = 2
    self_thought_cooldown: int = 3
    priority_threshold: float = 0.35
    budget_cost: float = 0.15
    max_budget_per_turn: float = 0.4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def should_produce(
        self,
        *,
        thought_count_this_turn: int,
        cooldown_remaining: int,
        priority: float,
        budget_spent: float,
        gap_id: str,
        prior_gap_ids: tuple[str, ...],
    ) -> tuple[bool, str]:
        """Check whether a new self-thought is allowed.

        Returns (allowed, reason).
        """
        if thought_count_this_turn >= self.max_self_thought_per_turn:
            return False, "max_self_thought_per_turn_exceeded"
        if cooldown_remaining > 0:
            return False, "cooldown_active"
        if priority < self.priority_threshold:
            return False, "below_priority_threshold"
        if budget_spent + self.budget_cost > self.max_budget_per_turn:
            return False, "budget_exceeded"
        if gap_id and gap_id in prior_gap_ids:
            return False, "dedupe_by_gap_id"
        return True, "allowed"

    def thought_count_after(self, current: int) -> int:
        return current + 1

    def cooldown_after(self) -> int:
        return self.self_thought_cooldown

    def budget_after(self, current: float) -> float:
        return _clamp(current + self.budget_cost, 0.0, self.max_budget_per_turn)


class SelfThoughtProducer:
    """Detect triggers and produce SelfThoughtEvents that enter the bus.

    This is NOT a second personality or a free-running inner monologue.
    It is a high-priority event source plus a budgeted agenda over unresolved
    uncertainty.
    """

    def __init__(
        self,
        policy: ExplorationPolicy | None = None,
        loop_control: LoopControl | None = None,
    ) -> None:
        self.policy = policy or ExplorationPolicy()
        self.loop_control = loop_control or LoopControl()

    def detect_triggers(
        self,
        *,
        prediction_error: float = 0.0,
        policy_margin: float = 1.0,
        efe_margin: float = 1.0,
        memory_conflicts: Sequence[str] = (),
        citation_audit_failures: Sequence[str] = (),
        previous_outcomes: Sequence[str] = (),
        identity_tension: float = 0.0,
        commitment_tension: float = 0.0,
        unresolved_questions: Sequence[str] = (),
        open_uncertainty_duration: int = 0,
    ) -> list[dict[str, object]]:
        """Scan internal signals and return detected triggers with metadata."""
        triggers: list[dict[str, object]] = []

        if prediction_error >= 0.55:
            triggers.append({
                "trigger": "high_prediction_error",
                "confidence": _clamp(prediction_error),
                "priority": _clamp(prediction_error * 0.9),
                "details": f"prediction_error={prediction_error:.3f}",
            })

        if policy_margin < 0.12 or efe_margin < 0.05:
            triggers.append({
                "trigger": "low_decision_margin",
                "confidence": _clamp(1.0 - min(policy_margin, efe_margin)),
                "priority": _clamp(0.6 + (0.4 * (1.0 - min(policy_margin, efe_margin)))),
                "details": f"policy_margin={policy_margin:.3f} efe_margin={efe_margin:.3f}",
            })

        if memory_conflicts:
            triggers.append({
                "trigger": "memory_conflict",
                "confidence": _clamp(0.55 + 0.15 * len(memory_conflicts), 0.0, 1.0),
                "priority": _clamp(0.5 + 0.1 * len(memory_conflicts)),
                "details": f"conflicts={len(memory_conflicts)}",
            })

        if citation_audit_failures:
            triggers.append({
                "trigger": "citation_audit_failure",
                "confidence": _clamp(0.6 + 0.1 * len(citation_audit_failures), 0.0, 1.0),
                "priority": _clamp(0.55 + 0.1 * len(citation_audit_failures)),
                "details": f"failures={len(citation_audit_failures)}",
            })

        negative_count = sum(
            1 for o in previous_outcomes if "fail" in o.lower() or "negative" in o.lower()
        )
        if negative_count >= 2:
            triggers.append({
                "trigger": "repeated_negative_outcome",
                "confidence": _clamp(0.6 + 0.1 * negative_count, 0.0, 1.0),
                "priority": _clamp(0.6 + 0.05 * negative_count),
                "details": f"negative_outcomes={negative_count}",
            })

        if identity_tension >= 0.45 or commitment_tension >= 0.45:
            triggers.append({
                "trigger": "identity_or_commitment_tension",
                "confidence": _clamp(max(identity_tension, commitment_tension)),
                "priority": _clamp(0.5 + 0.3 * max(identity_tension, commitment_tension)),
                "details": (
                    f"identity_tension={identity_tension:.3f} "
                    f"commitment_tension={commitment_tension:.3f}"
                ),
            })

        if unresolved_questions:
            triggers.append({
                "trigger": "unresolved_user_question",
                "confidence": _clamp(0.5 + 0.1 * len(unresolved_questions), 0.0, 1.0),
                "priority": _clamp(0.45 + 0.1 * len(unresolved_questions)),
                "details": f"pending={len(unresolved_questions)}",
            })

        if open_uncertainty_duration >= 5:
            triggers.append({
                "trigger": "long_running_open_uncertainty",
                "confidence": _clamp(0.45 + 0.05 * open_uncertainty_duration, 0.0, 1.0),
                "priority": _clamp(0.4 + 0.03 * open_uncertainty_duration),
                "details": f"duration={open_uncertainty_duration}_turns",
            })

        return triggers

    def propose_intervention(
        self,
        trigger: str,
        *,
        has_tools: bool = False,
    ) -> str:
        """Map a trigger to a concrete, allowed intervention."""
        mapping: dict[str, str] = {
            "high_prediction_error": "lower_assertiveness",
            "low_decision_margin": "ask_clarifying_question",
            "memory_conflict": "mark_claim_as_unverified",
            "citation_audit_failure": "mark_claim_as_unverified",
            "repeated_negative_outcome": "prefer_repair_strategy",
            "identity_or_commitment_tension": "defer_durable_update",
            "unresolved_user_question": "ask_clarifying_question",
            "long_running_open_uncertainty": "retrieve_memory_with_cue",
        }
        default_intervention = mapping.get(trigger, "lower_assertiveness")
        if default_intervention == "request_external_evidence" and not has_tools:
            return "mark_claim_as_unverified"
        if not self.policy.intervention_allowed(default_intervention):
            return "mark_claim_as_unverified"
        return default_intervention

    def produce(
        self,
        *,
        turn_id: str,
        cycle: int,
        session_id: str,
        persona_id: str,
        sequence_index: int,
        triggers: Sequence[dict[str, object]],
        gap_id: str = "",
        thought_count_this_turn: int = 0,
        cooldown_remaining: int = 0,
        budget_spent: float = 0.0,
        prior_gap_ids: tuple[str, ...] = (),
        has_tools: bool = False,
    ) -> list[CognitiveEvent]:
        """Produce SelfThoughtEvents from detected triggers, subject to loop control.

        Returns at most one event (M10.0 MVP); the loop control caps total per turn.
        """
        events: list[CognitiveEvent] = []
        accumulated_spent = budget_spent
        accumulated_count = thought_count_this_turn
        seen_gap_ids: list[str] = list(prior_gap_ids)
        for trigger_data in triggers:
            trigger = str(trigger_data.get("trigger", ""))
            if trigger not in SELF_THOUGHT_TRIGGERS:
                continue

            priority = float(trigger_data.get("priority", 0.5))
            confidence = float(trigger_data.get("confidence", 0.5))
            target_gap_id = gap_id or str(trigger_data.get("trigger", ""))

            allowed, reason = self.loop_control.should_produce(
                thought_count_this_turn=accumulated_count,
                cooldown_remaining=cooldown_remaining,
                priority=priority,
                budget_spent=accumulated_spent,
                gap_id=target_gap_id,
                prior_gap_ids=tuple(seen_gap_ids),
            )
            if not allowed:
                continue

            intervention = self.propose_intervention(trigger, has_tools=has_tools)

            # Exploration never fabricates evidence
            evidence_ids: tuple[str, ...] = ()

            event = make_self_thought_event(
                turn_id=turn_id,
                cycle=cycle,
                session_id=session_id,
                persona_id=persona_id,
                source="SelfThoughtProducer",
                sequence_index=sequence_index + len(events),
                trigger=trigger,
                target_gap_id=target_gap_id,
                confidence=confidence,
                salience=priority,
                priority=priority,
                ttl=1,
                proposed_intervention=intervention,
                evidence_event_ids=evidence_ids,
                budget_cost=self.loop_control.budget_cost,
            )
            events.append(event)
            accumulated_spent += self.loop_control.budget_cost
            accumulated_count += 1
            if target_gap_id:
                seen_gap_ids.append(target_gap_id)

        return events[: self.loop_control.max_self_thought_per_turn]


def default_exploration_policy() -> ExplorationPolicy:
    return ExplorationPolicy()


def default_loop_control() -> LoopControl:
    return LoopControl()
