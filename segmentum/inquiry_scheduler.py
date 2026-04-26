from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum  # Python >=3.11
except ImportError:

    class StrEnum(str, Enum):  # Python <3.11
        pass
from typing import Mapping


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _to_str_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(value) for value in values if str(value))


class InquirySchedulingDecision(StrEnum):
    PROMOTE = "promote"
    KEEP_ACTIVE = "keep_active"
    DEFER = "defer"
    SUPPRESS = "suppress"
    EVICT = "evict"
    ESCALATE = "escalate"
    COOLDOWN = "cooldown"


class InquirySuppressionReason(StrEnum):
    LOW_DECISION_RELEVANCE = "low_decision_relevance"
    INSUFFICIENT_INFORMATION_GAIN = "insufficient_information_gain"
    EXCESSIVE_RISK = "excessive_risk"
    EXCESSIVE_COST = "excessive_cost"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    VERIFICATION_SATURATION = "verification_saturation"
    MAINTENANCE_EMERGENCY = "maintenance_emergency"
    CONTINUITY_FRAGILITY = "continuity_fragility"
    OVERSHADOWED_BY_HIGHER_PRIORITY = "overshadowed_by_higher_priority"
    SATURATED_LOW_YIELD = "saturated_low_yield"


@dataclass(frozen=True)
class InquiryCandidate:
    candidate_id: str
    source_subsystem: str
    linked_target_id: str
    linked_unknown_id: str = ""
    linked_hypothesis_ids: tuple[str, ...] = ()
    linked_prediction_id: str = ""
    linked_plan_id: str = ""
    linked_tension: str = ""
    target_channels: tuple[str, ...] = ()
    action_name: str = ""
    uncertainty_level: float = 0.0
    decision_relevance: float = 0.0
    expected_information_gain: float = 0.0
    falsification_importance: float = 0.0
    practical_risk: float = 0.0
    cost: float = 0.0
    urgency: float = 0.0
    chronicity: float = 0.0
    identity_relevance: float = 0.0
    social_relevance: float = 0.0
    continuity_impact: float = 0.0
    active: bool = False
    inconclusive_count: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "source_subsystem": self.source_subsystem,
            "linked_target_id": self.linked_target_id,
            "linked_unknown_id": self.linked_unknown_id,
            "linked_hypothesis_ids": list(self.linked_hypothesis_ids),
            "linked_prediction_id": self.linked_prediction_id,
            "linked_plan_id": self.linked_plan_id,
            "linked_tension": self.linked_tension,
            "target_channels": list(self.target_channels),
            "action_name": self.action_name,
            "uncertainty_level": round(self.uncertainty_level, 6),
            "decision_relevance": round(self.decision_relevance, 6),
            "expected_information_gain": round(self.expected_information_gain, 6),
            "falsification_importance": round(self.falsification_importance, 6),
            "practical_risk": round(self.practical_risk, 6),
            "cost": round(self.cost, 6),
            "urgency": round(self.urgency, 6),
            "chronicity": round(self.chronicity, 6),
            "identity_relevance": round(self.identity_relevance, 6),
            "social_relevance": round(self.social_relevance, 6),
            "continuity_impact": round(self.continuity_impact, 6),
            "active": bool(self.active),
            "inconclusive_count": int(self.inconclusive_count),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryCandidate":
        if not payload:
            return cls(candidate_id="", source_subsystem="", linked_target_id="")
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            source_subsystem=str(payload.get("source_subsystem", "")),
            linked_target_id=str(payload.get("linked_target_id", "")),
            linked_unknown_id=str(payload.get("linked_unknown_id", "")),
            linked_hypothesis_ids=_to_str_tuple(payload.get("linked_hypothesis_ids", [])),
            linked_prediction_id=str(payload.get("linked_prediction_id", "")),
            linked_plan_id=str(payload.get("linked_plan_id", "")),
            linked_tension=str(payload.get("linked_tension", "")),
            target_channels=_to_str_tuple(payload.get("target_channels", [])),
            action_name=str(payload.get("action_name", "")),
            uncertainty_level=float(payload.get("uncertainty_level", 0.0)),
            decision_relevance=float(payload.get("decision_relevance", 0.0)),
            expected_information_gain=float(payload.get("expected_information_gain", 0.0)),
            falsification_importance=float(payload.get("falsification_importance", 0.0)),
            practical_risk=float(payload.get("practical_risk", 0.0)),
            cost=float(payload.get("cost", 0.0)),
            urgency=float(payload.get("urgency", 0.0)),
            chronicity=float(payload.get("chronicity", 0.0)),
            identity_relevance=float(payload.get("identity_relevance", 0.0)),
            social_relevance=float(payload.get("social_relevance", 0.0)),
            continuity_impact=float(payload.get("continuity_impact", 0.0)),
            active=bool(payload.get("active", False)),
            inconclusive_count=int(payload.get("inconclusive_count", 0)),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class InquiryPriorityScore:
    candidate_id: str
    base_value: float = 0.0
    persistence_bonus: float = 0.0
    hysteresis_bonus: float = 0.0
    process_bonus: float = 0.0
    saturation_penalty: float = 0.0
    maintenance_penalty: float = 0.0
    risk_penalty: float = 0.0
    cost_penalty: float = 0.0
    closure_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "base_value": round(self.base_value, 6),
            "persistence_bonus": round(self.persistence_bonus, 6),
            "hysteresis_bonus": round(self.hysteresis_bonus, 6),
            "process_bonus": round(self.process_bonus, 6),
            "saturation_penalty": round(self.saturation_penalty, 6),
            "maintenance_penalty": round(self.maintenance_penalty, 6),
            "risk_penalty": round(self.risk_penalty, 6),
            "cost_penalty": round(self.cost_penalty, 6),
            "closure_penalty": round(self.closure_penalty, 6),
            "total": round(self.total, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryPriorityScore":
        if not payload:
            return cls(candidate_id="")
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            base_value=float(payload.get("base_value", 0.0)),
            persistence_bonus=float(payload.get("persistence_bonus", 0.0)),
            hysteresis_bonus=float(payload.get("hysteresis_bonus", 0.0)),
            process_bonus=float(payload.get("process_bonus", 0.0)),
            saturation_penalty=float(payload.get("saturation_penalty", 0.0)),
            maintenance_penalty=float(payload.get("maintenance_penalty", 0.0)),
            risk_penalty=float(payload.get("risk_penalty", 0.0)),
            cost_penalty=float(payload.get("cost_penalty", 0.0)),
            closure_penalty=float(payload.get("closure_penalty", 0.0)),
            total=float(payload.get("total", 0.0)),
        )


@dataclass(frozen=True)
class PrecisionAllocation:
    candidate_id: str
    precision_weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "precision_weight": round(self.precision_weight, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PrecisionAllocation":
        if not payload:
            return cls(candidate_id="", precision_weight=0.0)
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            precision_weight=float(payload.get("precision_weight", 0.0)),
        )


@dataclass(frozen=True)
class VerificationSlotAssignment:
    candidate_id: str
    prediction_id: str
    slot_index: int
    decision: str
    priority_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "prediction_id": self.prediction_id,
            "slot_index": int(self.slot_index),
            "decision": self.decision,
            "priority_score": round(self.priority_score, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "VerificationSlotAssignment":
        if not payload:
            return cls(candidate_id="", prediction_id="", slot_index=-1, decision="", priority_score=0.0)
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            prediction_id=str(payload.get("prediction_id", "")),
            slot_index=int(payload.get("slot_index", -1)),
            decision=str(payload.get("decision", "")),
            priority_score=float(payload.get("priority_score", 0.0)),
        )


@dataclass(frozen=True)
class WorkspaceInquiryAllocation:
    candidate_id: str
    channel: str
    weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "channel": self.channel,
            "weight": round(self.weight, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "WorkspaceInquiryAllocation":
        if not payload:
            return cls(candidate_id="", channel="", weight=0.0)
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            channel=str(payload.get("channel", "")),
            weight=float(payload.get("weight", 0.0)),
        )


@dataclass(frozen=True)
class ActionBudgetAllocation:
    candidate_id: str
    action_name: str
    budget: float
    granted: bool
    priority_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "action_name": self.action_name,
            "budget": round(self.budget, 6),
            "granted": bool(self.granted),
            "priority_score": round(self.priority_score, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ActionBudgetAllocation":
        if not payload:
            return cls(candidate_id="", action_name="", budget=0.0, granted=False, priority_score=0.0)
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            action_name=str(payload.get("action_name", "")),
            budget=float(payload.get("budget", 0.0)),
            granted=bool(payload.get("granted", False)),
            priority_score=float(payload.get("priority_score", 0.0)),
        )


@dataclass(frozen=True)
class SchedulingDecisionRecord:
    candidate_id: str
    source_subsystem: str
    decision: str
    priority_score: float
    precision_weight: float
    reasons: tuple[str, ...] = ()
    suppression_reason: str = ""
    linked_prediction_id: str = ""
    linked_plan_id: str = ""
    linked_unknown_id: str = ""
    action_name: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "source_subsystem": self.source_subsystem,
            "decision": self.decision,
            "priority_score": round(self.priority_score, 6),
            "precision_weight": round(self.precision_weight, 6),
            "reasons": list(self.reasons),
            "suppression_reason": self.suppression_reason,
            "linked_prediction_id": self.linked_prediction_id,
            "linked_plan_id": self.linked_plan_id,
            "linked_unknown_id": self.linked_unknown_id,
            "action_name": self.action_name,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SchedulingDecisionRecord":
        if not payload:
            return cls(candidate_id="", source_subsystem="", decision="", priority_score=0.0, precision_weight=0.0)
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            source_subsystem=str(payload.get("source_subsystem", "")),
            decision=str(payload.get("decision", "")),
            priority_score=float(payload.get("priority_score", 0.0)),
            precision_weight=float(payload.get("precision_weight", 0.0)),
            reasons=_to_str_tuple(payload.get("reasons", [])),
            suppression_reason=str(payload.get("suppression_reason", "")),
            linked_prediction_id=str(payload.get("linked_prediction_id", "")),
            linked_plan_id=str(payload.get("linked_plan_id", "")),
            linked_unknown_id=str(payload.get("linked_unknown_id", "")),
            action_name=str(payload.get("action_name", "")),
            explanation=str(payload.get("explanation", "")),
        )


@dataclass(frozen=True)
class InquiryCandidateState:
    candidate_id: str
    cumulative_budget: float = 0.0
    inconclusive_streak: int = 0
    cooldown_until_tick: int = 0
    persistence_ticks: int = 0
    last_priority: float = 0.0
    last_decision: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "cumulative_budget": round(self.cumulative_budget, 6),
            "inconclusive_streak": int(self.inconclusive_streak),
            "cooldown_until_tick": int(self.cooldown_until_tick),
            "persistence_ticks": int(self.persistence_ticks),
            "last_priority": round(self.last_priority, 6),
            "last_decision": self.last_decision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryCandidateState":
        if not payload:
            return cls(candidate_id="")
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            cumulative_budget=float(payload.get("cumulative_budget", 0.0)),
            inconclusive_streak=int(payload.get("inconclusive_streak", 0)),
            cooldown_until_tick=int(payload.get("cooldown_until_tick", 0)),
            persistence_ticks=int(payload.get("persistence_ticks", 0)),
            last_priority=float(payload.get("last_priority", 0.0)),
            last_decision=str(payload.get("last_decision", "")),
        )


@dataclass(frozen=True)
class InquiryBudgetState:
    last_tick: int = 0
    max_active_candidates: int = 4
    max_workspace_slots: int = 3
    max_verification_slots: int = 2
    max_action_budget: int = 1
    candidates: tuple[InquiryCandidate, ...] = ()
    priority_scores: tuple[InquiryPriorityScore, ...] = ()
    precision_allocations: tuple[PrecisionAllocation, ...] = ()
    verification_assignments: tuple[VerificationSlotAssignment, ...] = ()
    workspace_allocations: tuple[WorkspaceInquiryAllocation, ...] = ()
    action_allocations: tuple[ActionBudgetAllocation, ...] = ()
    decisions: tuple[SchedulingDecisionRecord, ...] = ()
    active_candidate_ids: tuple[str, ...] = ()
    candidate_states: tuple[InquiryCandidateState, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "last_tick": int(self.last_tick),
            "max_active_candidates": int(self.max_active_candidates),
            "max_workspace_slots": int(self.max_workspace_slots),
            "max_verification_slots": int(self.max_verification_slots),
            "max_action_budget": int(self.max_action_budget),
            "candidates": [item.to_dict() for item in self.candidates],
            "priority_scores": [item.to_dict() for item in self.priority_scores],
            "precision_allocations": [item.to_dict() for item in self.precision_allocations],
            "verification_assignments": [item.to_dict() for item in self.verification_assignments],
            "workspace_allocations": [item.to_dict() for item in self.workspace_allocations],
            "action_allocations": [item.to_dict() for item in self.action_allocations],
            "decisions": [item.to_dict() for item in self.decisions],
            "active_candidate_ids": list(self.active_candidate_ids),
            "candidate_states": [item.to_dict() for item in self.candidate_states],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryBudgetState":
        if not payload:
            return cls()
        return cls(
            last_tick=int(payload.get("last_tick", 0)),
            max_active_candidates=max(1, int(payload.get("max_active_candidates", 4))),
            max_workspace_slots=max(1, int(payload.get("max_workspace_slots", 3))),
            max_verification_slots=max(1, int(payload.get("max_verification_slots", 2))),
            max_action_budget=max(1, int(payload.get("max_action_budget", 1))),
            candidates=tuple(
                InquiryCandidate.from_dict(item)
                for item in payload.get("candidates", [])
                if isinstance(item, Mapping)
            ),
            priority_scores=tuple(
                InquiryPriorityScore.from_dict(item)
                for item in payload.get("priority_scores", [])
                if isinstance(item, Mapping)
            ),
            precision_allocations=tuple(
                PrecisionAllocation.from_dict(item)
                for item in payload.get("precision_allocations", [])
                if isinstance(item, Mapping)
            ),
            verification_assignments=tuple(
                VerificationSlotAssignment.from_dict(item)
                for item in payload.get("verification_assignments", [])
                if isinstance(item, Mapping)
            ),
            workspace_allocations=tuple(
                WorkspaceInquiryAllocation.from_dict(item)
                for item in payload.get("workspace_allocations", [])
                if isinstance(item, Mapping)
            ),
            action_allocations=tuple(
                ActionBudgetAllocation.from_dict(item)
                for item in payload.get("action_allocations", [])
                if isinstance(item, Mapping)
            ),
            decisions=tuple(
                SchedulingDecisionRecord.from_dict(item)
                for item in payload.get("decisions", [])
                if isinstance(item, Mapping)
            ),
            active_candidate_ids=_to_str_tuple(payload.get("active_candidate_ids", [])),
            candidate_states=tuple(
                InquiryCandidateState.from_dict(item)
                for item in payload.get("candidate_states", [])
                if isinstance(item, Mapping)
            ),
        )

    def workspace_focus(self) -> dict[str, float]:
        focus: dict[str, float] = {}
        for item in self.workspace_allocations:
            focus[item.channel] = max(focus.get(item.channel, 0.0), float(item.weight))
        return focus

    def action_bias(self, action: str) -> float:
        bias = 0.0
        for item in self.action_allocations:
            if item.action_name != action:
                continue
            if item.granted:
                bias += 0.06 + item.budget * 0.24
            else:
                bias -= min(0.16, 0.04 + item.priority_score * 0.10)
        return round(max(-0.24, min(0.24, bias)), 6)

    def decision_for_prediction(self, prediction_id: str) -> SchedulingDecisionRecord | None:
        for item in self.decisions:
            if item.linked_prediction_id == prediction_id:
                return item
        return None

    def decision_for_plan(self, plan_id: str) -> SchedulingDecisionRecord | None:
        for item in self.decisions:
            if item.linked_plan_id == plan_id:
                return item
        return None

    def explanation_payload(self) -> dict[str, object]:
        ranked = sorted(self.decisions, key=lambda item: (-item.priority_score, item.candidate_id))
        summary = "Inquiry budget is idle because no candidate currently justifies scarce attention."
        if ranked:
            top = ranked[0]
            summary = (
                f"Inquiry scheduler {top.decision.replace('_', ' ')}d {top.candidate_id} "
                f"because {top.explanation or 'it had the strongest bounded inquiry value'}."
            )
        deferred = [item.to_dict() for item in ranked if item.decision == InquirySchedulingDecision.DEFER.value]
        suppressed = [
            item.to_dict()
            for item in ranked
            if item.decision in {
                InquirySchedulingDecision.SUPPRESS.value,
                InquirySchedulingDecision.COOLDOWN.value,
                InquirySchedulingDecision.EVICT.value,
            }
        ]
        return {
            "summary": summary,
            "active_candidate_ids": list(self.active_candidate_ids),
            "decisions": [item.to_dict() for item in ranked[:8]],
            "workspace_allocations": [item.to_dict() for item in self.workspace_allocations[:8]],
            "verification_assignments": [item.to_dict() for item in self.verification_assignments[:8]],
            "action_allocations": [item.to_dict() for item in self.action_allocations[:8]],
            "deferred": deferred[:6],
            "suppressed": suppressed[:6],
            "candidate_states": [item.to_dict() for item in self.candidate_states[:12]],
        }


class InquiryBudgetScheduler:
    def __init__(
        self,
        *,
        max_active_candidates: int = 4,
        max_workspace_slots: int = 3,
        max_verification_slots: int = 2,
        max_action_budget: int = 1,
        cooldown_ticks: int = 2,
    ) -> None:
        self.max_active_candidates = max(1, int(max_active_candidates))
        self.max_workspace_slots = max(1, int(max_workspace_slots))
        self.max_verification_slots = max(1, int(max_verification_slots))
        self.max_action_budget = max(1, int(max_action_budget))
        self.cooldown_ticks = max(1, int(cooldown_ticks))
        self.state = InquiryBudgetState(
            max_active_candidates=self.max_active_candidates,
            max_workspace_slots=self.max_workspace_slots,
            max_verification_slots=self.max_verification_slots,
            max_action_budget=self.max_action_budget,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_active_candidates": int(self.max_active_candidates),
            "max_workspace_slots": int(self.max_workspace_slots),
            "max_verification_slots": int(self.max_verification_slots),
            "max_action_budget": int(self.max_action_budget),
            "cooldown_ticks": int(self.cooldown_ticks),
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryBudgetScheduler":
        if not payload:
            return cls()
        scheduler = cls(
            max_active_candidates=int(payload.get("max_active_candidates", 4)),
            max_workspace_slots=int(payload.get("max_workspace_slots", 3)),
            max_verification_slots=int(payload.get("max_verification_slots", 2)),
            max_action_budget=int(payload.get("max_action_budget", 1)),
            cooldown_ticks=int(payload.get("cooldown_ticks", 2)),
        )
        scheduler.state = InquiryBudgetState.from_dict(
            payload.get("state") if isinstance(payload.get("state"), Mapping) else None
        )
        return scheduler

    def schedule(
        self,
        *,
        tick: int,
        narrative_uncertainty=None,
        experiment_design=None,
        prediction_ledger=None,
        verification_loop=None,
        subject_state=None,
        reconciliation_engine=None,
        process_valence_state=None,
    ) -> InquiryBudgetState:
        candidates = self._collect_candidates(
            narrative_uncertainty=narrative_uncertainty,
            experiment_design=experiment_design,
            prediction_ledger=prediction_ledger,
            verification_loop=verification_loop,
            subject_state=subject_state,
            reconciliation_engine=reconciliation_engine,
        )
        previous_state = {item.candidate_id: item for item in self.state.candidate_states}
        maintenance_pressure = float(getattr(subject_state, "maintenance_pressure", 0.0))
        continuity_fragile = bool(
            getattr(subject_state, "status_flags", {}).get("continuity_fragile", False)
        )
        priority_scores: list[InquiryPriorityScore] = []
        precision_allocations: list[PrecisionAllocation] = []
        decision_records: list[SchedulingDecisionRecord] = []
        workspace_allocations: list[WorkspaceInquiryAllocation] = []
        verification_assignments: list[VerificationSlotAssignment] = []
        action_allocations: list[ActionBudgetAllocation] = []
        ranked_candidates: list[tuple[float, InquiryCandidate, InquiryCandidateState | None, InquiryPriorityScore]] = []

        for candidate in candidates:
            previous = previous_state.get(candidate.candidate_id)
            score = self._score_candidate(
                candidate=candidate,
                previous=previous,
                maintenance_pressure=maintenance_pressure,
                continuity_fragile=continuity_fragile,
                tick=tick,
                process_valence_state=process_valence_state,
            )
            priority_scores.append(score)
            precision = _clamp(0.08 + score.total * 0.92)
            precision_allocations.append(
                PrecisionAllocation(candidate_id=candidate.candidate_id, precision_weight=precision)
            )
            ranked_candidates.append((score.total, candidate, previous, score))

        ranked_candidates.sort(key=lambda item: (-item[0], item[1].candidate_id))
        verification_rank = [item for item in ranked_candidates if item[1].linked_prediction_id]
        action_rank = [item for item in ranked_candidates if item[1].action_name]
        promoted_ids = {
            item[1].candidate_id
            for item in ranked_candidates[: self.max_active_candidates]
            if item[0] >= 0.30
        }
        verification_ids = {
            item[1].candidate_id
            for item in verification_rank[: self.max_verification_slots]
            if item[0] >= 0.22
        }
        action_ids = {
            item[1].candidate_id
            for item in action_rank[: self.max_action_budget]
            if item[0] >= 0.36
        }
        active_ids: list[str] = []
        next_candidate_states: list[InquiryCandidateState] = []

        for _, candidate, previous, score in ranked_candidates:
            decision, suppression_reason, reasons = self._decision_for_candidate(
                candidate=candidate,
                previous=previous,
                tick=tick,
                promoted_ids=promoted_ids,
                verification_ids=verification_ids,
                action_ids=action_ids,
                maintenance_pressure=maintenance_pressure,
                continuity_fragile=continuity_fragile,
            )
            if decision in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            }:
                active_ids.append(candidate.candidate_id)
            explanation = self._explain_decision(candidate, reasons, suppression_reason)
            precision = next(
                (
                    item.precision_weight
                    for item in precision_allocations
                    if item.candidate_id == candidate.candidate_id
                ),
                0.0,
            )
            decision_records.append(
                SchedulingDecisionRecord(
                    candidate_id=candidate.candidate_id,
                    source_subsystem=candidate.source_subsystem,
                    decision=decision,
                    priority_score=score.total,
                    precision_weight=precision,
                    reasons=tuple(reasons),
                    suppression_reason=suppression_reason,
                    linked_prediction_id=candidate.linked_prediction_id,
                    linked_plan_id=candidate.linked_plan_id,
                    linked_unknown_id=candidate.linked_unknown_id,
                    action_name=candidate.action_name,
                    explanation=explanation,
                )
            )
            next_candidate_states.append(
                self._next_candidate_state(
                    candidate=candidate,
                    previous=previous,
                    decision=decision,
                    score=score.total,
                    tick=tick,
                )
            )

        decision_by_candidate = {item.candidate_id: item for item in decision_records}
        workspace_candidates = [
            item
            for item in ranked_candidates
            if decision_by_candidate[item[1].candidate_id].decision
            in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
                InquirySchedulingDecision.DEFER.value,
            }
        ][: self.max_workspace_slots]
        for _, candidate, _, score in workspace_candidates:
            weight = _clamp(0.18 + score.total * 0.60, high=0.82)
            for channel in candidate.target_channels[:3]:
                workspace_allocations.append(
                    WorkspaceInquiryAllocation(
                        candidate_id=candidate.candidate_id,
                        channel=channel,
                        weight=weight,
                    )
                )

        for slot_index, (_, candidate, _, score) in enumerate(verification_rank[: self.max_verification_slots]):
            decision = decision_by_candidate[candidate.candidate_id].decision
            if decision not in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            }:
                continue
            verification_assignments.append(
                VerificationSlotAssignment(
                    candidate_id=candidate.candidate_id,
                    prediction_id=candidate.linked_prediction_id,
                    slot_index=slot_index,
                    decision=decision,
                    priority_score=score.total,
                )
            )

        for _, candidate, _, score in action_rank[: max(self.max_action_budget * 2, self.max_action_budget)]:
            decision = decision_by_candidate[candidate.candidate_id].decision
            granted = candidate.candidate_id in action_ids and decision in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            }
            action_allocations.append(
                ActionBudgetAllocation(
                    candidate_id=candidate.candidate_id,
                    action_name=candidate.action_name,
                    budget=_clamp(0.12 + score.total * 0.70),
                    granted=granted,
                    priority_score=score.total,
                )
            )

        self.state = InquiryBudgetState(
            last_tick=int(tick),
            max_active_candidates=self.max_active_candidates,
            max_workspace_slots=self.max_workspace_slots,
            max_verification_slots=self.max_verification_slots,
            max_action_budget=self.max_action_budget,
            candidates=tuple(candidate for _, candidate, _, _ in ranked_candidates),
            priority_scores=tuple(priority_scores),
            precision_allocations=tuple(precision_allocations),
            verification_assignments=tuple(verification_assignments),
            workspace_allocations=tuple(workspace_allocations),
            action_allocations=tuple(action_allocations),
            decisions=tuple(decision_records),
            active_candidate_ids=tuple(active_ids[: self.max_active_candidates]),
            candidate_states=tuple(next_candidate_states[:64]),
        )
        return self.state

    def _collect_candidates(
        self,
        *,
        narrative_uncertainty,
        experiment_design,
        prediction_ledger,
        verification_loop,
        subject_state,
        reconciliation_engine,
    ) -> list[InquiryCandidate]:
        candidates: list[InquiryCandidate] = []
        archived_outcomes = getattr(verification_loop, "archived_targets", ()) if verification_loop is not None else ()
        inconclusive_by_prediction: dict[str, int] = {}
        for item in archived_outcomes:
            prediction_id = str(getattr(item, "prediction_id", ""))
            if not prediction_id:
                continue
            outcome = str(getattr(item, "outcome", ""))
            if outcome in {"inconclusive", "deferred", "expired_unverified"}:
                inconclusive_by_prediction[prediction_id] = inconclusive_by_prediction.get(prediction_id, 0) + 1

        for unknown in getattr(narrative_uncertainty, "unknowns", ())[:4]:
            if not getattr(unknown, "action_relevant", False):
                continue
            linked_hypotheses = tuple(str(item) for item in getattr(unknown, "competing_hypothesis_ids", ())[:3])
            decision_relevance = float(getattr(unknown, "decision_relevance", object()).total_score)
            channels = tuple(
                channel
                for channel in (
                    "danger" if "threat" in str(unknown.unknown_type) else "",
                    "social" if "trust" in str(unknown.unknown_type) or "social" in str(unknown.unknown_type) else "",
                )
                if channel
            )
            candidates.append(
                InquiryCandidate(
                    candidate_id=f"inquiry:uncertainty:{unknown.unknown_id}",
                    source_subsystem="narrative_uncertainty",
                    linked_target_id=str(unknown.unknown_id),
                    linked_unknown_id=str(unknown.unknown_id),
                    linked_hypothesis_ids=linked_hypotheses,
                    target_channels=channels,
                    uncertainty_level=float(getattr(unknown, "uncertainty_level", 0.0)),
                    decision_relevance=decision_relevance,
                    expected_information_gain=_clamp(decision_relevance * 0.85 + float(getattr(unknown, "uncertainty_level", 0.0)) * 0.15),
                    falsification_importance=_clamp(float(getattr(unknown.decision_relevance, "verification_urgency", 0.0))),
                    practical_risk=_clamp(0.18 + float(getattr(unknown.decision_relevance, "risk_level", 0.0)) * 0.55),
                    cost=_clamp(0.20 + len(linked_hypotheses) * 0.10),
                    urgency=_clamp(float(getattr(unknown.decision_relevance, "verification_urgency", 0.0))),
                    social_relevance=1.0 if "social" in str(unknown.unknown_type) or "trust" in str(unknown.unknown_type) else 0.0,
                    continuity_impact=_clamp(float(getattr(unknown.decision_relevance, "continuity_impact", 0.0))),
                    summary=str(getattr(unknown, "promotion_reason", "") or getattr(unknown, "unresolved_reason", "")),
                )
            )

        for plan in getattr(experiment_design, "plans", ())[:8]:
            status = str(getattr(plan, "status", ""))
            candidates.append(
                InquiryCandidate(
                    candidate_id=f"inquiry:plan:{plan.plan_id}",
                    source_subsystem="narrative_experiment",
                    linked_target_id=str(plan.plan_id),
                    linked_unknown_id=str(getattr(plan, "target_unknown_id", "")),
                    linked_hypothesis_ids=_to_str_tuple(getattr(plan, "target_hypothesis_ids", ())),
                    linked_plan_id=str(plan.plan_id),
                    target_channels=tuple(
                        sorted(
                            str(item).replace("observe:", "", 1)
                            for item in getattr(plan, "evidence_sought", ())
                            if str(item)
                        )
                    ),
                    action_name=str(getattr(plan, "selected_action", "")),
                    uncertainty_level=_clamp(1.0 - float(getattr(plan, "informative_value", 0.0)) * 0.45),
                    decision_relevance=_clamp(float(getattr(plan, "score", 0.0))),
                    expected_information_gain=_clamp(float(getattr(plan, "informative_value", 0.0))),
                    falsification_importance=_clamp(0.20 + len(getattr(plan, "outcome_differences", ())) * 0.12),
                    practical_risk=_clamp(0.28 if status == "deferred_for_risk" else 0.12),
                    cost=_clamp(0.18 + float(getattr(plan, "expected_horizon", 1)) * 0.12),
                    urgency=_clamp(float(getattr(plan, "score", 0.0)) * 0.80),
                    continuity_impact=_clamp(0.18 if getattr(plan, "target_unknown_id", "") else 0.0),
                    active=status in {"active_experiment", "queued_experiment"},
                    inconclusive_count=int(getattr(plan, "inconclusive_count", 0)),
                    summary=str(getattr(plan, "selected_reason", "")),
                )
            )

        for prediction in getattr(prediction_ledger, "active_predictions", lambda: [])()[:8]:
            target_channels = _to_str_tuple(getattr(prediction, "target_channels", ()))
            source_module = str(getattr(prediction, "source_module", ""))
            prediction_type = str(getattr(prediction, "prediction_type", ""))
            decision_value = _clamp(
                float(getattr(prediction, "decision_relevance", 0.0))
                or float(getattr(prediction, "confidence", 0.0))
            )
            information_value = _clamp(
                0.18 + float(getattr(prediction, "decision_relevance", 0.0)) * 0.60
            )
            falsification_value = _clamp(
                0.24 + float(getattr(prediction, "confidence", 0.0)) * 0.55
            )
            if source_module in {"decision_policy", "decision_cycle"} or prediction_type == "action_consequence":
                decision_value *= 0.45
                information_value *= 0.55
                falsification_value *= 0.55
            candidates.append(
                InquiryCandidate(
                    candidate_id=f"inquiry:prediction:{prediction.prediction_id}",
                    source_subsystem="prediction_ledger",
                    linked_target_id=str(prediction.prediction_id),
                    linked_unknown_id=str(next(iter(getattr(prediction, "linked_unknown_ids", ())), "")),
                    linked_hypothesis_ids=_to_str_tuple(getattr(prediction, "linked_hypothesis_ids", ())),
                    linked_prediction_id=str(prediction.prediction_id),
                    linked_plan_id=str(getattr(prediction, "linked_experiment_plan_id", "")),
                    target_channels=target_channels,
                    uncertainty_level=_clamp(1.0 - float(getattr(prediction, "confidence", 0.0))),
                    decision_relevance=decision_value,
                    expected_information_gain=information_value,
                    falsification_importance=falsification_value,
                    practical_risk=_clamp(0.42 if "danger" in target_channels or "maintenance" in target_channels else 0.18),
                    cost=_clamp(0.14 + float(getattr(prediction, "verification_attempts", 0)) * 0.16),
                    urgency=_clamp(0.18 + min(1.0, float(getattr(prediction, "verification_attempts", 0)) * 0.18)),
                    chronicity=_clamp(float(getattr(prediction, "recurrence_count", 0)) * 0.20),
                    identity_relevance=1.0 if getattr(prediction, "linked_identity_anchors", ()) else 0.0,
                    social_relevance=1.0 if "social" in target_channels else 0.0,
                    continuity_impact=1.0 if "continuity" in str(getattr(prediction, "prediction_type", "")) else 0.0,
                    active=any(str(getattr(item, "prediction_id", "")) == str(prediction.prediction_id) for item in getattr(verification_loop, "active_targets", ())),
                    inconclusive_count=inconclusive_by_prediction.get(str(prediction.prediction_id), 0),
                    summary=str(getattr(prediction, "prediction_type", "")),
                )
            )

        for discrepancy in getattr(prediction_ledger, "top_discrepancies", lambda limit=3: [])(limit=4):
            candidates.append(
                InquiryCandidate(
                    candidate_id=f"inquiry:discrepancy:{discrepancy.discrepancy_id}",
                    source_subsystem="prediction_ledger_discrepancy",
                    linked_target_id=str(discrepancy.discrepancy_id),
                    linked_prediction_id=str(next(iter(getattr(discrepancy, "linked_predictions", ())), "")),
                    linked_tension=str(getattr(discrepancy, "label", "")),
                    target_channels=_to_str_tuple(getattr(discrepancy, "target_channels", ())),
                    uncertainty_level=_clamp(float(getattr(discrepancy, "severity", 0.0)) * 0.75),
                    decision_relevance=_clamp(float(getattr(discrepancy, "severity", 0.0))),
                    expected_information_gain=_clamp(0.22 + float(getattr(discrepancy, "severity", 0.0)) * 0.52),
                    falsification_importance=_clamp(0.38 + float(getattr(discrepancy, "severity", 0.0)) * 0.50),
                    practical_risk=_clamp(float(getattr(discrepancy, "severity", 0.0))),
                    cost=_clamp(0.24 + float(getattr(discrepancy, "repair_attempts", 0)) * 0.12),
                    urgency=_clamp(0.22 + float(getattr(discrepancy, "severity", 0.0)) * 0.58),
                    chronicity=1.0 if getattr(discrepancy, "chronic", False) else 0.0,
                    identity_relevance=1.0 if getattr(discrepancy, "identity_relevant", False) else 0.0,
                    social_relevance=1.0 if "social" in str(getattr(discrepancy, "discrepancy_type", "")) else 0.0,
                    continuity_impact=1.0 if getattr(discrepancy, "subject_critical", False) else 0.0,
                    summary=str(getattr(discrepancy, "label", "")),
                )
            )

        for tension in getattr(subject_state, "unresolved_tensions", ())[:4]:
            channels = tuple(
                channel
                for channel in (
                    "social" if "social" in str(getattr(tension, "tension_type", "")) else "",
                    "danger" if "maintenance" in str(getattr(tension, "tension_type", "")) or "continuity" in str(getattr(tension, "tension_type", "")) else "",
                )
                if channel
            )
            repair_target = str(getattr(tension, "repair_target", ""))
            candidates.append(
                InquiryCandidate(
                    candidate_id=f"inquiry:tension:{str(getattr(tension, 'label', '')).replace(' ', '_')}",
                    source_subsystem="subject_state",
                    linked_target_id=str(getattr(tension, "label", "")),
                    linked_tension=str(getattr(tension, "label", "")),
                    action_name=repair_target if repair_target in {"rest", "scan", "hide", "seek_contact"} else "",
                    target_channels=channels,
                    uncertainty_level=_clamp(float(getattr(tension, "intensity", 0.0)) * 0.60),
                    decision_relevance=_clamp(float(getattr(tension, "intensity", 0.0))),
                    expected_information_gain=_clamp(0.12 + float(getattr(tension, "intensity", 0.0)) * 0.42),
                    falsification_importance=_clamp(0.16 + float(getattr(tension, "intensity", 0.0)) * 0.30),
                    practical_risk=_clamp(0.32 if "maintenance" in str(getattr(tension, "tension_type", "")) else 0.12),
                    cost=_clamp(0.18),
                    urgency=_clamp(float(getattr(tension, "intensity", 0.0))),
                    chronicity=1.0 if "continuity" in str(getattr(tension, "tension_type", "")) else 0.0,
                    identity_relevance=1.0 if "identity" in str(getattr(tension, "tension_type", "")) else 0.0,
                    social_relevance=1.0 if "social" in str(getattr(tension, "tension_type", "")) else 0.0,
                    continuity_impact=1.0 if "continuity" in str(getattr(tension, "tension_type", "")) else 0.0,
                    summary=repair_target or str(getattr(tension, "label", "")),
                )
            )

        for thread in getattr(reconciliation_engine, "active_unresolved_threads", lambda: [])()[:3]:
            candidates.append(
                InquiryCandidate(
                    candidate_id=f"inquiry:reconciliation:{thread.thread_id}",
                    source_subsystem="reconciliation",
                    linked_target_id=str(thread.thread_id),
                    linked_tension=str(getattr(thread, "title", "") or "long-horizon conflict"),
                    uncertainty_level=_clamp(0.36),
                    decision_relevance=_clamp(0.22 + min(1.0, float(getattr(thread, "recurrence_count", 0)) * 0.16)),
                    expected_information_gain=_clamp(0.24 + min(1.0, float(getattr(thread, "recurrence_count", 0)) * 0.12)),
                    falsification_importance=_clamp(0.28),
                    practical_risk=_clamp(0.14),
                    cost=_clamp(0.26),
                    urgency=_clamp(0.20 + min(1.0, float(getattr(thread, "recurrence_count", 0)) * 0.10)),
                    chronicity=1.0,
                    continuity_impact=1.0,
                    summary=str(getattr(thread, "title", "") or getattr(thread, "status", "")),
                )
            )
        return candidates

    def _score_candidate(
        self,
        *,
        candidate: InquiryCandidate,
        previous: InquiryCandidateState | None,
        maintenance_pressure: float,
        continuity_fragile: bool,
        tick: int,
        process_valence_state=None,
    ) -> InquiryPriorityScore:
        del tick
        base = (
            candidate.uncertainty_level * 0.10
            + candidate.decision_relevance * 0.24
            + candidate.expected_information_gain * 0.22
            + candidate.falsification_importance * 0.16
            + candidate.urgency * 0.10
            + candidate.chronicity * 0.05
            + candidate.identity_relevance * 0.04
            + candidate.social_relevance * 0.03
            + candidate.continuity_impact * 0.06
        )
        persistence_bonus = min(0.14, float(previous.persistence_ticks) * 0.03) if previous is not None else 0.0
        hysteresis_bonus = 0.10 if previous is not None and previous.last_decision in {
            InquirySchedulingDecision.PROMOTE.value,
            InquirySchedulingDecision.KEEP_ACTIVE.value,
            InquirySchedulingDecision.ESCALATE.value,
        } else 0.0
        process_bonus, closure_penalty = _process_valence_priority_adjustment(
            candidate=candidate,
            process_valence_state=process_valence_state,
        )
        saturation_penalty = min(
            0.28,
            (float(previous.cumulative_budget) * 0.05 if previous is not None else 0.0)
            + candidate.inconclusive_count * 0.08
            + (float(previous.inconclusive_streak) * 0.05 if previous is not None else 0.0),
        )
        maintenance_penalty = maintenance_pressure * (0.10 + candidate.practical_risk * 0.16)
        if candidate.continuity_impact >= 0.70:
            maintenance_penalty *= 0.55
        risk_penalty = candidate.practical_risk * (0.18 + (0.08 if continuity_fragile else 0.0))
        cost_penalty = candidate.cost * 0.14
        total = _clamp(
            base
            + persistence_bonus
            + hysteresis_bonus
            + process_bonus
            - saturation_penalty
            - maintenance_penalty
            - risk_penalty
            - cost_penalty
            - closure_penalty
        )
        return InquiryPriorityScore(
            candidate_id=candidate.candidate_id,
            base_value=base,
            persistence_bonus=persistence_bonus,
            hysteresis_bonus=hysteresis_bonus,
            process_bonus=process_bonus,
            saturation_penalty=saturation_penalty,
            maintenance_penalty=maintenance_penalty,
            risk_penalty=risk_penalty,
            cost_penalty=cost_penalty,
            closure_penalty=closure_penalty,
            total=total,
        )

    def _decision_for_candidate(
        self,
        *,
        candidate: InquiryCandidate,
        previous: InquiryCandidateState | None,
        tick: int,
        promoted_ids: set[str],
        verification_ids: set[str],
        action_ids: set[str],
        maintenance_pressure: float,
        continuity_fragile: bool,
    ) -> tuple[str, str, list[str]]:
        reasons: list[str] = []
        suppression_reason = ""
        if previous is not None and previous.cooldown_until_tick > tick:
            reasons.append("recent low-yield probing triggered cooldown")
            return InquirySchedulingDecision.COOLDOWN.value, InquirySuppressionReason.SATURATED_LOW_YIELD.value, reasons
        if candidate.inconclusive_count >= 2 and candidate.continuity_impact < 0.70:
            reasons.append("repeated inconclusive probing exhausted local budget")
            return InquirySchedulingDecision.COOLDOWN.value, InquirySuppressionReason.SATURATED_LOW_YIELD.value, reasons
        if maintenance_pressure >= 0.72 and candidate.practical_risk >= 0.42 and candidate.continuity_impact < 0.60:
            reasons.append("maintenance pressure dominates risky inquiry")
            return InquirySchedulingDecision.SUPPRESS.value, InquirySuppressionReason.MAINTENANCE_EMERGENCY.value, reasons
        if continuity_fragile and candidate.practical_risk >= 0.45 and candidate.continuity_impact < 0.75:
            reasons.append("continuity is fragile, so risky ambiguity stays latent")
            return InquirySchedulingDecision.SUPPRESS.value, InquirySuppressionReason.CONTINUITY_FRAGILITY.value, reasons
        if candidate.decision_relevance < 0.16 and candidate.expected_information_gain < 0.20:
            reasons.append("uncertainty is active but low leverage")
            return InquirySchedulingDecision.SUPPRESS.value, InquirySuppressionReason.LOW_DECISION_RELEVANCE.value, reasons
        if candidate.expected_information_gain < 0.14 and candidate.falsification_importance < 0.18:
            reasons.append("expected information gain is too weak for current budget")
            return InquirySchedulingDecision.SUPPRESS.value, InquirySuppressionReason.INSUFFICIENT_INFORMATION_GAIN.value, reasons
        if candidate.practical_risk > 0.82:
            reasons.append("practical risk is too high for bounded inquiry")
            return InquirySchedulingDecision.SUPPRESS.value, InquirySuppressionReason.EXCESSIVE_RISK.value, reasons
        if candidate.cost > 0.88 and candidate.decision_relevance < 0.72:
            reasons.append("resource cost is too high for current leverage")
            return InquirySchedulingDecision.SUPPRESS.value, InquirySuppressionReason.EXCESSIVE_COST.value, reasons
        if candidate.candidate_id in promoted_ids or candidate.candidate_id in verification_ids or candidate.candidate_id in action_ids:
            if previous is not None and previous.last_decision in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            }:
                reasons.append("persistence bonus kept this inquiry stable")
                if candidate.candidate_id in verification_ids and candidate.falsification_importance >= 0.72:
                    reasons.append("falsification value remains high")
                    return InquirySchedulingDecision.ESCALATE.value, suppression_reason, reasons
                return InquirySchedulingDecision.KEEP_ACTIVE.value, suppression_reason, reasons
            reasons.append("shared priority surface promoted this candidate")
            if candidate.candidate_id in verification_ids and candidate.falsification_importance >= 0.72:
                reasons.append("verification slot reserved for high falsification value")
                return InquirySchedulingDecision.ESCALATE.value, suppression_reason, reasons
            if candidate.action_name and candidate.candidate_id in action_ids:
                reasons.append("action budget granted")
            return InquirySchedulingDecision.PROMOTE.value, suppression_reason, reasons
        if candidate.linked_prediction_id and candidate.candidate_id not in verification_ids:
            reasons.append("verification capacity is saturated by stronger candidates")
            return InquirySchedulingDecision.DEFER.value, InquirySuppressionReason.VERIFICATION_SATURATION.value, reasons
        if candidate.action_name and candidate.candidate_id not in action_ids:
            reasons.append("action budget is exhausted by stronger inquiry")
            return InquirySchedulingDecision.DEFER.value, InquirySuppressionReason.BUDGET_EXHAUSTION.value, reasons
        reasons.append("higher-value inquiries occupy the active budget window")
        if previous is not None and previous.last_decision in {
            InquirySchedulingDecision.PROMOTE.value,
            InquirySchedulingDecision.KEEP_ACTIVE.value,
            InquirySchedulingDecision.ESCALATE.value,
        }:
            return InquirySchedulingDecision.EVICT.value, InquirySuppressionReason.OVERSHADOWED_BY_HIGHER_PRIORITY.value, reasons
        return InquirySchedulingDecision.DEFER.value, InquirySuppressionReason.OVERSHADOWED_BY_HIGHER_PRIORITY.value, reasons

    def _next_candidate_state(
        self,
        *,
        candidate: InquiryCandidate,
        previous: InquiryCandidateState | None,
        decision: str,
        score: float,
        tick: int,
    ) -> InquiryCandidateState:
        cumulative_budget = float(previous.cumulative_budget) if previous is not None else 0.0
        inconclusive_streak = int(previous.inconclusive_streak) if previous is not None else 0
        persistence_ticks = int(previous.persistence_ticks) if previous is not None else 0
        cooldown_until_tick = int(previous.cooldown_until_tick) if previous is not None else 0
        if decision in {
            InquirySchedulingDecision.PROMOTE.value,
            InquirySchedulingDecision.KEEP_ACTIVE.value,
            InquirySchedulingDecision.ESCALATE.value,
        }:
            cumulative_budget += 1.0
            persistence_ticks += 1
        else:
            persistence_ticks = 0
        if candidate.inconclusive_count > 0:
            inconclusive_streak = candidate.inconclusive_count
        elif decision in {
            InquirySchedulingDecision.PROMOTE.value,
            InquirySchedulingDecision.KEEP_ACTIVE.value,
            InquirySchedulingDecision.ESCALATE.value,
        }:
            inconclusive_streak = 0
        if decision == InquirySchedulingDecision.COOLDOWN.value:
            cooldown_until_tick = tick + self.cooldown_ticks
        elif decision in {
            InquirySchedulingDecision.PROMOTE.value,
            InquirySchedulingDecision.KEEP_ACTIVE.value,
            InquirySchedulingDecision.ESCALATE.value,
        }:
            cooldown_until_tick = 0
        return InquiryCandidateState(
            candidate_id=candidate.candidate_id,
            cumulative_budget=cumulative_budget,
            inconclusive_streak=inconclusive_streak,
            cooldown_until_tick=cooldown_until_tick,
            persistence_ticks=persistence_ticks,
            last_priority=score,
            last_decision=decision,
        )

    def _explain_decision(
        self,
        candidate: InquiryCandidate,
        reasons: list[str],
        suppression_reason: str,
    ) -> str:
        if reasons:
            return reasons[0]
        if suppression_reason:
            return suppression_reason.replace("_", " ")
        return f"{candidate.source_subsystem} inquiry remained bounded"


def apply_scheduler_to_experiment_design(experiment_design, inquiry_state: InquiryBudgetState):
    from .narrative_experiment import ExperimentDesignResult, ExperimentPlan, InquiryPlanStatus

    if experiment_design is None or not getattr(experiment_design, "plans", ()):
        return experiment_design
    updated_plans: list[ExperimentPlan] = []
    for plan in experiment_design.plans:
        decision = inquiry_state.decision_for_plan(str(plan.plan_id))
        if decision is None:
            updated_plans.append(plan)
            continue
        status = plan.status
        if decision.decision in {
            InquirySchedulingDecision.PROMOTE.value,
            InquirySchedulingDecision.KEEP_ACTIVE.value,
            InquirySchedulingDecision.ESCALATE.value,
        }:
            action_granted = any(
                item.candidate_id == decision.candidate_id and item.granted
                for item in inquiry_state.action_allocations
            )
            status = (
                InquiryPlanStatus.ACTIVE_EXPERIMENT.value
                if action_granted
                else InquiryPlanStatus.QUEUED_EXPERIMENT.value
            )
        elif decision.decision in {
            InquirySchedulingDecision.DEFER.value,
            InquirySchedulingDecision.EVICT.value,
            InquirySchedulingDecision.COOLDOWN.value,
        }:
            status = InquiryPlanStatus.DEFERRED_FOR_BUDGET.value
        elif decision.suppression_reason == InquirySuppressionReason.EXCESSIVE_RISK.value:
            status = InquiryPlanStatus.DEFERRED_FOR_RISK.value
        elif decision.suppression_reason == InquirySuppressionReason.INSUFFICIENT_INFORMATION_GAIN.value:
            status = InquiryPlanStatus.REJECTED_LOW_INFORMATION_GAIN.value
        elif decision.suppression_reason == InquirySuppressionReason.LOW_DECISION_RELEVANCE.value:
            status = InquiryPlanStatus.REJECTED_LOW_VALUE.value
        updated_plans.append(ExperimentPlan(**{**plan.__dict__, "status": status}))
    summary = experiment_design.summary
    scheduler_payload = inquiry_state.explanation_payload()
    if scheduler_payload.get("summary"):
        summary = f"{summary} Scheduler: {scheduler_payload['summary']}".strip()
    return ExperimentDesignResult(
        generated_tick=experiment_design.generated_tick,
        source_episode_id=experiment_design.source_episode_id,
        discrimination_targets=experiment_design.discrimination_targets,
        predictions=experiment_design.predictions,
        candidates=experiment_design.candidates,
        plans=tuple(updated_plans),
        summary=summary,
    )


def semantic_uncertainty_priority_bonus(
    *,
    semantic_grounding: dict[str, object] | None = None,
    semantic_schemas: list[dict[str, object]] | None = None,
) -> float:
    grounding = dict(semantic_grounding or {})
    schemas = list(semantic_schemas or ())
    scores = grounding.get("semantic_direction_scores", {})
    uncertainty_hits = float(scores.get("uncertainty", 0.0)) if isinstance(scores, dict) else 0.0
    active_motifs = {str(item) for item in grounding.get("motifs", []) if str(item)}
    schema_overlap = 0.0
    for schema in schemas:
        motif_signature = {str(item) for item in schema.get("motif_signature", []) if str(item)}
        if motif_signature:
            schema_overlap = max(
                schema_overlap,
                len(active_motifs & motif_signature) / max(1.0, len(active_motifs | motif_signature)),
            )
    return _clamp((uncertainty_hits * 0.18) + (schema_overlap * 0.22))


def _process_valence_priority_adjustment(
    *,
    candidate: InquiryCandidate,
    process_valence_state,
) -> tuple[float, float]:
    if process_valence_state is None:
        return 0.0, 0.0
    if isinstance(process_valence_state, Mapping):
        active_focus_id = str(process_valence_state.get("active_focus_id", ""))
        recent_closed_focus_id = str(process_valence_state.get("recent_closed_focus_id", ""))
        unresolved_tension = _clamp(float(process_valence_state.get("unresolved_tension", 0.0)))
        closure_satisfaction = _clamp(float(process_valence_state.get("closure_satisfaction", 0.0)))
        post_closure_decay = _clamp(float(process_valence_state.get("post_closure_decay", 0.0)))
        boredom_pressure = _clamp(float(process_valence_state.get("boredom_pressure", 0.0)))
        persistence_ticks = int(process_valence_state.get("focus_persistence_ticks", 0))
    else:
        active_focus_id = str(getattr(process_valence_state, "active_focus_id", ""))
        recent_closed_focus_id = str(getattr(process_valence_state, "recent_closed_focus_id", ""))
        unresolved_tension = _clamp(float(getattr(process_valence_state, "unresolved_tension", 0.0)))
        closure_satisfaction = _clamp(float(getattr(process_valence_state, "closure_satisfaction", 0.0)))
        post_closure_decay = _clamp(float(getattr(process_valence_state, "post_closure_decay", 0.0)))
        boredom_pressure = _clamp(float(getattr(process_valence_state, "boredom_pressure", 0.0)))
        persistence_ticks = int(getattr(process_valence_state, "focus_persistence_ticks", 0))

    target_id = (
        str(candidate.linked_unknown_id)
        or str(candidate.linked_target_id)
        or str(candidate.linked_tension)
        or str(candidate.candidate_id)
    )
    process_bonus = 0.0
    closure_penalty = 0.0
    if active_focus_id and target_id == active_focus_id:
        process_bonus += min(
            0.24,
            0.08 + unresolved_tension * 0.18 + min(0.08, persistence_ticks * 0.02),
        )
    elif recent_closed_focus_id and target_id == recent_closed_focus_id:
        closure_penalty += min(
            0.24,
            closure_satisfaction * 0.18 + post_closure_decay * 0.14,
        )
    elif boredom_pressure >= 0.42 and candidate.expected_information_gain >= 0.24:
        process_bonus += min(0.14, boredom_pressure * 0.16)
    if candidate.active and unresolved_tension >= 0.24:
        process_bonus += 0.03
    return round(process_bonus, 6), round(closure_penalty, 6)


def process_valence_priority_adjustment(
    *,
    candidate: InquiryCandidate,
    process_valence_state,
) -> dict[str, float]:
    process_bonus, closure_penalty = _process_valence_priority_adjustment(
        candidate=candidate,
        process_valence_state=process_valence_state,
    )
    return {
        "process_bonus": process_bonus,
        "closure_penalty": closure_penalty,
    }
