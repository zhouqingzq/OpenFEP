from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:
    from enum import StrEnum  # Python >=3.11
except ImportError:

    class StrEnum(str, Enum):  # Python <3.11
        pass
from typing import Mapping

from .action_registry import ActionRegistry
from .action_schema import ActionSchema
from .narrative_uncertainty import (
    CompetingHypothesis,
    NarrativeUnknown,
    UncertaintyDecompositionResult,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _to_str_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(value) for value in values if str(value))


def _to_float_dict(values: object) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    payload: dict[str, float] = {}
    for key, value in values.items():
        try:
            payload[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


class InquiryPlanStatus(StrEnum):
    ACTIVE_EXPERIMENT = "active_experiment"
    QUEUED_EXPERIMENT = "queued_experiment"
    DEFERRED_FOR_RISK = "deferred_for_risk"
    DEFERRED_FOR_BUDGET = "deferred_for_budget"
    REJECTED_LOW_VALUE = "rejected_low_value"
    REJECTED_LOW_INFORMATION_GAIN = "rejected_low_information_gain"
    RESOLVED_NO_LONGER_NEEDED = "resolved_no_longer_needed"
    BLOCKED_BY_GOVERNANCE = "blocked_by_governance"


@dataclass(frozen=True)
class DiscriminationTarget:
    target_id: str
    unknown_id: str
    hypothesis_ids: tuple[str, ...]
    decision_relevance: float
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "unknown_id": self.unknown_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "decision_relevance": round(self.decision_relevance, 6),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "DiscriminationTarget":
        if not payload:
            return cls(target_id="", unknown_id="", hypothesis_ids=(), decision_relevance=0.0, summary="")
        return cls(
            target_id=str(payload.get("target_id", "")),
            unknown_id=str(payload.get("unknown_id", "")),
            hypothesis_ids=_to_str_tuple(payload.get("hypothesis_ids", [])),
            decision_relevance=float(payload.get("decision_relevance", 0.0)),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class HypothesisPrediction:
    prediction_id: str
    parent_hypothesis_id: str
    parent_unknown_id: str
    observable_consequence: str
    expected_horizon: int
    confidence: float
    evidence_channels: tuple[str, ...]
    support_signals: tuple[str, ...]
    contradiction_signals: tuple[str, ...]
    expected_state: dict[str, float] = field(default_factory=dict)
    action_name: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "parent_unknown_id": self.parent_unknown_id,
            "observable_consequence": self.observable_consequence,
            "expected_horizon": int(self.expected_horizon),
            "confidence": round(self.confidence, 6),
            "evidence_channels": list(self.evidence_channels),
            "support_signals": list(self.support_signals),
            "contradiction_signals": list(self.contradiction_signals),
            "expected_state": {str(key): round(float(value), 6) for key, value in self.expected_state.items()},
            "action_name": self.action_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "HypothesisPrediction":
        if not payload:
            return cls(
                prediction_id="",
                parent_hypothesis_id="",
                parent_unknown_id="",
                observable_consequence="",
                expected_horizon=1,
                confidence=0.0,
                evidence_channels=(),
                support_signals=(),
                contradiction_signals=(),
            )
        return cls(
            prediction_id=str(payload.get("prediction_id", "")),
            parent_hypothesis_id=str(payload.get("parent_hypothesis_id", "")),
            parent_unknown_id=str(payload.get("parent_unknown_id", "")),
            observable_consequence=str(payload.get("observable_consequence", "")),
            expected_horizon=max(1, int(payload.get("expected_horizon", 1))),
            confidence=float(payload.get("confidence", 0.0)),
            evidence_channels=_to_str_tuple(payload.get("evidence_channels", [])),
            support_signals=_to_str_tuple(payload.get("support_signals", [])),
            contradiction_signals=_to_str_tuple(payload.get("contradiction_signals", [])),
            expected_state=_to_float_dict(payload.get("expected_state", {})),
            action_name=str(payload.get("action_name", "")),
        )


@dataclass(frozen=True)
class ExpectedInformationGain:
    score: float = 0.0
    separated_hypothesis_count: int = 0
    confirmation_focus: float = 0.0
    falsification_focus: float = 0.0
    ambiguity_remaining: float = 1.0
    weak_signal: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 6),
            "separated_hypothesis_count": int(self.separated_hypothesis_count),
            "confirmation_focus": round(self.confirmation_focus, 6),
            "falsification_focus": round(self.falsification_focus, 6),
            "ambiguity_remaining": round(self.ambiguity_remaining, 6),
            "weak_signal": bool(self.weak_signal),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ExpectedInformationGain":
        if not payload:
            return cls()
        return cls(
            score=float(payload.get("score", 0.0)),
            separated_hypothesis_count=int(payload.get("separated_hypothesis_count", 0)),
            confirmation_focus=float(payload.get("confirmation_focus", 0.0)),
            falsification_focus=float(payload.get("falsification_focus", 0.0)),
            ambiguity_remaining=float(payload.get("ambiguity_remaining", 1.0)),
            weak_signal=bool(payload.get("weak_signal", True)),
        )


@dataclass(frozen=True)
class FalsificationOpportunity:
    score: float = 0.0
    target_hypothesis_ids: tuple[str, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 6),
            "target_hypothesis_ids": list(self.target_hypothesis_ids),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "FalsificationOpportunity":
        if not payload:
            return cls()
        return cls(
            score=float(payload.get("score", 0.0)),
            target_hypothesis_ids=_to_str_tuple(payload.get("target_hypothesis_ids", [])),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class InquiryCostProfile:
    score: float = 0.0
    expected_delay: float = 0.0
    resource_cost: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 6),
            "expected_delay": round(self.expected_delay, 6),
            "resource_cost": round(self.resource_cost, 6),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryCostProfile":
        if not payload:
            return cls()
        return cls(
            score=float(payload.get("score", 0.0)),
            expected_delay=float(payload.get("expected_delay", 0.0)),
            resource_cost=float(payload.get("resource_cost", 0.0)),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class InquiryRiskProfile:
    score: float = 0.0
    safety_margin: float = 1.0
    blockers: tuple[str, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 6),
            "safety_margin": round(self.safety_margin, 6),
            "blockers": list(self.blockers),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryRiskProfile":
        if not payload:
            return cls()
        return cls(
            score=float(payload.get("score", 0.0)),
            safety_margin=float(payload.get("safety_margin", 1.0)),
            blockers=_to_str_tuple(payload.get("blockers", [])),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class InquiryActionCandidate:
    candidate_id: str
    action_name: str
    target_unknown_id: str
    target_hypothesis_ids: tuple[str, ...]
    discrimination_target_id: str
    linked_prediction_ids: tuple[str, ...]
    distinction_summary: str
    information_gain: ExpectedInformationGain
    falsification_opportunity: FalsificationOpportunity
    cost_profile: InquiryCostProfile
    risk_profile: InquiryRiskProfile
    expected_horizon: int
    safe_to_recommend_now: bool
    action_available: bool
    selection_score: float
    governance_note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "action_name": self.action_name,
            "target_unknown_id": self.target_unknown_id,
            "target_hypothesis_ids": list(self.target_hypothesis_ids),
            "discrimination_target_id": self.discrimination_target_id,
            "linked_prediction_ids": list(self.linked_prediction_ids),
            "distinction_summary": self.distinction_summary,
            "information_gain": self.information_gain.to_dict(),
            "falsification_opportunity": self.falsification_opportunity.to_dict(),
            "cost_profile": self.cost_profile.to_dict(),
            "risk_profile": self.risk_profile.to_dict(),
            "expected_horizon": int(self.expected_horizon),
            "safe_to_recommend_now": bool(self.safe_to_recommend_now),
            "action_available": bool(self.action_available),
            "selection_score": round(self.selection_score, 6),
            "governance_note": self.governance_note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryActionCandidate":
        if not payload:
            return cls(
                candidate_id="",
                action_name="",
                target_unknown_id="",
                target_hypothesis_ids=(),
                discrimination_target_id="",
                linked_prediction_ids=(),
                distinction_summary="",
                information_gain=ExpectedInformationGain(),
                falsification_opportunity=FalsificationOpportunity(),
                cost_profile=InquiryCostProfile(),
                risk_profile=InquiryRiskProfile(),
                expected_horizon=1,
                safe_to_recommend_now=False,
                action_available=False,
                selection_score=0.0,
            )
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            action_name=str(payload.get("action_name", "")),
            target_unknown_id=str(payload.get("target_unknown_id", "")),
            target_hypothesis_ids=_to_str_tuple(payload.get("target_hypothesis_ids", [])),
            discrimination_target_id=str(payload.get("discrimination_target_id", "")),
            linked_prediction_ids=_to_str_tuple(payload.get("linked_prediction_ids", [])),
            distinction_summary=str(payload.get("distinction_summary", "")),
            information_gain=ExpectedInformationGain.from_dict(
                payload.get("information_gain") if isinstance(payload.get("information_gain"), Mapping) else None
            ),
            falsification_opportunity=FalsificationOpportunity.from_dict(
                payload.get("falsification_opportunity")
                if isinstance(payload.get("falsification_opportunity"), Mapping)
                else None
            ),
            cost_profile=InquiryCostProfile.from_dict(
                payload.get("cost_profile") if isinstance(payload.get("cost_profile"), Mapping) else None
            ),
            risk_profile=InquiryRiskProfile.from_dict(
                payload.get("risk_profile") if isinstance(payload.get("risk_profile"), Mapping) else None
            ),
            expected_horizon=max(1, int(payload.get("expected_horizon", 1))),
            safe_to_recommend_now=bool(payload.get("safe_to_recommend_now", False)),
            action_available=bool(payload.get("action_available", False)),
            selection_score=float(payload.get("selection_score", 0.0)),
            governance_note=str(payload.get("governance_note", "")),
        )


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    candidate_id: str
    target_unknown_id: str
    target_hypothesis_ids: tuple[str, ...]
    selected_action: str
    selected_reason: str
    evidence_sought: tuple[str, ...]
    outcome_differences: tuple[str, ...]
    fallback_behavior: str
    expected_horizon: int
    status: str
    prediction_ids: tuple[str, ...] = ()
    score: float = 0.0
    informative_value: float = 0.0
    inconclusive_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "target_unknown_id": self.target_unknown_id,
            "target_hypothesis_ids": list(self.target_hypothesis_ids),
            "selected_action": self.selected_action,
            "selected_reason": self.selected_reason,
            "evidence_sought": list(self.evidence_sought),
            "outcome_differences": list(self.outcome_differences),
            "fallback_behavior": self.fallback_behavior,
            "expected_horizon": int(self.expected_horizon),
            "status": self.status,
            "prediction_ids": list(self.prediction_ids),
            "score": round(self.score, 6),
            "informative_value": round(self.informative_value, 6),
            "inconclusive_count": int(self.inconclusive_count),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ExperimentPlan":
        if not payload:
            return cls(
                plan_id="",
                candidate_id="",
                target_unknown_id="",
                target_hypothesis_ids=(),
                selected_action="",
                selected_reason="",
                evidence_sought=(),
                outcome_differences=(),
                fallback_behavior="",
                expected_horizon=1,
                status=InquiryPlanStatus.REJECTED_LOW_VALUE.value,
            )
        return cls(
            plan_id=str(payload.get("plan_id", "")),
            candidate_id=str(payload.get("candidate_id", "")),
            target_unknown_id=str(payload.get("target_unknown_id", "")),
            target_hypothesis_ids=_to_str_tuple(payload.get("target_hypothesis_ids", [])),
            selected_action=str(payload.get("selected_action", "")),
            selected_reason=str(payload.get("selected_reason", "")),
            evidence_sought=_to_str_tuple(payload.get("evidence_sought", [])),
            outcome_differences=_to_str_tuple(payload.get("outcome_differences", [])),
            fallback_behavior=str(payload.get("fallback_behavior", "")),
            expected_horizon=max(1, int(payload.get("expected_horizon", 1))),
            status=str(payload.get("status", InquiryPlanStatus.REJECTED_LOW_VALUE.value)),
            prediction_ids=_to_str_tuple(payload.get("prediction_ids", [])),
            score=float(payload.get("score", 0.0)),
            informative_value=float(payload.get("informative_value", 0.0)),
            inconclusive_count=int(payload.get("inconclusive_count", 0)),
        )


@dataclass(frozen=True)
class ExperimentDesignResult:
    generated_tick: int = 0
    source_episode_id: str = ""
    discrimination_targets: tuple[DiscriminationTarget, ...] = ()
    predictions: tuple[HypothesisPrediction, ...] = ()
    candidates: tuple[InquiryActionCandidate, ...] = ()
    plans: tuple[ExperimentPlan, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_tick": int(self.generated_tick),
            "source_episode_id": self.source_episode_id,
            "discrimination_targets": [item.to_dict() for item in self.discrimination_targets],
            "predictions": [item.to_dict() for item in self.predictions],
            "candidates": [item.to_dict() for item in self.candidates],
            "plans": [item.to_dict() for item in self.plans],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ExperimentDesignResult":
        if not payload:
            return cls()
        return cls(
            generated_tick=int(payload.get("generated_tick", 0)),
            source_episode_id=str(payload.get("source_episode_id", "")),
            discrimination_targets=tuple(
                DiscriminationTarget.from_dict(item)
                for item in payload.get("discrimination_targets", [])
                if isinstance(item, Mapping)
            ),
            predictions=tuple(
                HypothesisPrediction.from_dict(item)
                for item in payload.get("predictions", [])
                if isinstance(item, Mapping)
            ),
            candidates=tuple(
                InquiryActionCandidate.from_dict(item)
                for item in payload.get("candidates", [])
                if isinstance(item, Mapping)
            ),
            plans=tuple(
                ExperimentPlan.from_dict(item)
                for item in payload.get("plans", [])
                if isinstance(item, Mapping)
            ),
            summary=str(payload.get("summary", "")),
        )

    def active_plans(self) -> list[ExperimentPlan]:
        return [item for item in self.plans if item.status == InquiryPlanStatus.ACTIVE_EXPERIMENT.value]

    def action_bias(self, action: str) -> float:
        bias = 0.0
        for plan in self.plans:
            if plan.selected_action != action:
                continue
            if plan.status == InquiryPlanStatus.ACTIVE_EXPERIMENT.value:
                bias += 0.16 + plan.score * 0.10
            elif plan.status == InquiryPlanStatus.QUEUED_EXPERIMENT.value:
                bias += 0.05 + plan.score * 0.04
            elif plan.status in {
                InquiryPlanStatus.DEFERRED_FOR_RISK.value,
                InquiryPlanStatus.BLOCKED_BY_GOVERNANCE.value,
            }:
                bias -= 0.10
        return round(max(-0.30, min(0.30, bias)), 6)

    def workspace_focus(self) -> dict[str, float]:
        focus: dict[str, float] = {}
        for plan in self.plans:
            if plan.status not in {
                InquiryPlanStatus.ACTIVE_EXPERIMENT.value,
                InquiryPlanStatus.QUEUED_EXPERIMENT.value,
            }:
                continue
            weight = 0.18 if plan.status == InquiryPlanStatus.QUEUED_EXPERIMENT.value else 0.32
            for evidence in plan.evidence_sought:
                channel = str(evidence).replace("observe:", "", 1)
                focus[channel] = max(
                    focus.get(channel, 0.0),
                    round(min(0.75, weight + plan.score * 0.28), 6),
                )
        return focus

    def explanation_payload(self, *, chosen_action: str = "") -> dict[str, object]:
        active = self.active_plans()
        summary = self.summary or "No active narrative experiment is currently needed."
        chosen_reason = ""
        if chosen_action:
            matching = next((item for item in self.plans if item.selected_action == chosen_action), None)
            if matching is not None:
                chosen_reason = matching.selected_reason
        return {
            "summary": summary,
            "chosen_action_reason": chosen_reason,
            "active_plans": [item.to_dict() for item in active],
            "deferred_plans": [
                item.to_dict()
                for item in self.plans
                if item.status.startswith("deferred")
                or item.status == InquiryPlanStatus.BLOCKED_BY_GOVERNANCE.value
            ],
            "rejected_plans": [item.to_dict() for item in self.plans if item.status.startswith("rejected")],
            "candidates": [item.to_dict() for item in self.candidates[:8]],
            "predictions": [item.to_dict() for item in self.predictions[:8]],
            "discrimination_targets": [item.to_dict() for item in self.discrimination_targets[:4]],
        }


class NarrativeExperimentDesigner:
    def __init__(
        self,
        *,
        max_targets: int = 2,
        max_candidates_per_target: int = 4,
        max_active_plans: int = 2,
    ) -> None:
        self.max_targets = max(1, int(max_targets))
        self.max_candidates_per_target = max(2, int(max_candidates_per_target))
        self.max_active_plans = max(1, int(max_active_plans))

    def design(
        self,
        *,
        tick: int,
        uncertainty: UncertaintyDecompositionResult,
        action_registry: ActionRegistry,
        active_goal: str = "",
        subject_state=None,
        previous_result: ExperimentDesignResult | None = None,
        verification_loop=None,
    ) -> ExperimentDesignResult:
        if not uncertainty.unknowns or not uncertainty.competing_hypotheses:
            return ExperimentDesignResult(
                generated_tick=tick,
                source_episode_id=uncertainty.source_episode_id,
                summary="No bounded experiment plan was generated because no decision-relevant narrative competition is active.",
            )
        targets = self._select_targets(uncertainty=uncertainty)
        predictions: list[HypothesisPrediction] = []
        candidates: list[InquiryActionCandidate] = []
        plans: list[ExperimentPlan] = []
        active_count = 0
        archived_outcomes = self._archived_outcomes(verification_loop)
        previous_by_unknown = {
            plan.target_unknown_id: plan for plan in (previous_result.plans if previous_result is not None else ())
        }
        for target in targets:
            unknown = next((item for item in uncertainty.unknowns if item.unknown_id == target.unknown_id), None)
            if unknown is None:
                continue
            hypotheses = [item for item in uncertainty.competing_hypotheses if item.hypothesis_id in target.hypothesis_ids]
            target_predictions = self._derive_predictions(unknown=unknown, hypotheses=hypotheses)
            predictions.extend(target_predictions)
            target_candidates = self._generate_candidates(
                target=target,
                unknown=unknown,
                hypotheses=hypotheses,
                predictions=target_predictions,
                action_registry=action_registry,
                subject_state=subject_state,
                active_goal=active_goal,
                previous_plan=previous_by_unknown.get(unknown.unknown_id),
                archived_outcomes=archived_outcomes,
            )[: self.max_candidates_per_target]
            candidates.extend(target_candidates)
            for candidate in target_candidates:
                status = self._plan_status(candidate=candidate, unknown=unknown, active_count=active_count)
                if status == InquiryPlanStatus.ACTIVE_EXPERIMENT.value and active_count >= self.max_active_plans:
                    status = InquiryPlanStatus.QUEUED_EXPERIMENT.value
                if status == InquiryPlanStatus.ACTIVE_EXPERIMENT.value:
                    active_count += 1
                plans.append(
                    ExperimentPlan(
                        plan_id=f"plan:{candidate.candidate_id}",
                        candidate_id=candidate.candidate_id,
                        target_unknown_id=unknown.unknown_id,
                        target_hypothesis_ids=candidate.target_hypothesis_ids,
                        selected_action=candidate.action_name,
                        selected_reason=self._selected_reason(candidate=candidate, unknown=unknown, hypotheses=hypotheses),
                        evidence_sought=tuple(
                            dict.fromkeys([f"observe:{channel}" for channel in self._candidate_channels(candidate, target_predictions)])
                        ),
                        outcome_differences=tuple(
                            dict.fromkeys(
                                [
                                    f"{hypothesis.hypothesis_id}:{self._hypothesis_signature(hypothesis)}"
                                    for hypothesis in hypotheses[:3]
                                ]
                            )
                        ),
                        fallback_behavior=self._fallback_behavior(candidate),
                        expected_horizon=candidate.expected_horizon,
                        status=status,
                        prediction_ids=candidate.linked_prediction_ids,
                        score=candidate.selection_score,
                        informative_value=candidate.information_gain.score,
                        inconclusive_count=int(archived_outcomes.get(f"plan:{candidate.candidate_id}", 0)),
                    )
                )
        return ExperimentDesignResult(
            generated_tick=tick,
            source_episode_id=uncertainty.source_episode_id,
            discrimination_targets=tuple(targets),
            predictions=tuple(predictions),
            candidates=tuple(candidates),
            plans=tuple(plans),
            summary=self._summary(plans=plans, targets=targets),
        )

    def _select_targets(self, *, uncertainty: UncertaintyDecompositionResult) -> list[DiscriminationTarget]:
        targets: list[DiscriminationTarget] = []
        for unknown in uncertainty.unknowns:
            if not unknown.action_relevant:
                continue
            if unknown.decision_relevance.total_score < 0.34:
                continue
            if len(unknown.competing_hypothesis_ids) < 2:
                continue
            targets.append(
                DiscriminationTarget(
                    target_id=f"disc:{unknown.unknown_id}",
                    unknown_id=unknown.unknown_id,
                    hypothesis_ids=tuple(unknown.competing_hypothesis_ids[:3]),
                    decision_relevance=float(unknown.decision_relevance.total_score),
                    summary=unknown.unresolved_reason or unknown.promotion_reason,
                )
            )
        targets.sort(key=lambda item: (-item.decision_relevance, item.target_id))
        return targets[: self.max_targets]

    def _derive_predictions(
        self,
        *,
        unknown: NarrativeUnknown,
        hypotheses: list[CompetingHypothesis],
    ) -> list[HypothesisPrediction]:
        predictions: list[HypothesisPrediction] = []
        for hypothesis in hypotheses:
            channels = tuple(sorted(hypothesis.expected_state_shift)) or self._default_channels(unknown)
            support = tuple(
                f"{channel}>={value:.2f}" if value >= 0.5 else f"{channel}<={value:.2f}"
                for channel, value in sorted(hypothesis.expected_state_shift.items())
            ) or tuple(hypothesis.implied_consequences[:2])
            contradiction = tuple(f"not({signal})" for signal in support[:2]) or ("outcome diverges from expected consequence",)
            consequence = hypothesis.statement
            if hypothesis.implied_consequences:
                consequence += " Observable consequence: " + "; ".join(hypothesis.implied_consequences[:2]) + "."
            predictions.append(
                HypothesisPrediction(
                    prediction_id=f"exp_pred:{hypothesis.hypothesis_id}",
                    parent_hypothesis_id=hypothesis.hypothesis_id,
                    parent_unknown_id=unknown.unknown_id,
                    observable_consequence=consequence,
                    expected_horizon=1 if unknown.unknown_type != "environment_reliability" else 2,
                    confidence=_clamp(
                        hypothesis.prior_plausibility * 0.55
                        + hypothesis.support.support_score * 0.35
                        - hypothesis.support.contradiction_score * 0.15
                    ),
                    evidence_channels=channels,
                    support_signals=support,
                    contradiction_signals=contradiction,
                    expected_state=dict(hypothesis.expected_state_shift),
                )
            )
        return predictions

    def _generate_candidates(
        self,
        *,
        target: DiscriminationTarget,
        unknown: NarrativeUnknown,
        hypotheses: list[CompetingHypothesis],
        predictions: list[HypothesisPrediction],
        action_registry: ActionRegistry,
        subject_state,
        active_goal: str,
        previous_plan: ExperimentPlan | None,
        archived_outcomes: Mapping[str, int],
    ) -> list[InquiryActionCandidate]:
        candidate_actions = [action for action in self._candidate_actions(unknown.unknown_type) if action_registry.contains(action)]
        candidates: list[InquiryActionCandidate] = []
        for action_name in candidate_actions:
            schema = action_registry.get(action_name) or ActionSchema(name=action_name)
            info_gain = self._information_gain(action_name=action_name, hypotheses=hypotheses, unknown=unknown)
            falsification = self._falsification_opportunity(action_name=action_name, hypotheses=hypotheses)
            cost = self._cost_profile(action_name=action_name, schema=schema)
            risk = self._risk_profile(action_name=action_name, unknown=unknown, subject_state=subject_state, schema=schema)
            goal_alignment = self._goal_alignment_bonus(
                action_name=action_name,
                unknown=unknown,
                active_goal=active_goal,
            )
            repeat_penalty = 0.0
            if previous_plan is not None and previous_plan.selected_action == action_name:
                repeat_penalty += 0.08 + min(0.12, previous_plan.inconclusive_count * 0.04)
            repeat_penalty += min(0.10, float(archived_outcomes.get(previous_plan.plan_id if previous_plan else "", 0)) * 0.03)
            selection_score = _clamp(
                info_gain.score * 0.52
                + falsification.score * 0.18
                + float(unknown.decision_relevance.total_score) * 0.20
                + goal_alignment
                - risk.score * 0.20
                - cost.score * 0.14
                - repeat_penalty,
                0.0,
                1.0,
            )
            candidates.append(
                InquiryActionCandidate(
                    candidate_id=f"{target.target_id}:{action_name}",
                    action_name=action_name,
                    target_unknown_id=unknown.unknown_id,
                    target_hypothesis_ids=tuple(hypothesis.hypothesis_id for hypothesis in hypotheses),
                    discrimination_target_id=target.target_id,
                    linked_prediction_ids=tuple(prediction.prediction_id for prediction in predictions),
                    distinction_summary=self._distinction_summary(action_name=action_name, hypotheses=hypotheses),
                    information_gain=info_gain,
                    falsification_opportunity=falsification,
                    cost_profile=cost,
                    risk_profile=risk,
                    expected_horizon=max(1, int(round(cost.expected_delay)) or 1),
                    safe_to_recommend_now=bool(risk.score <= 0.58 and not risk.blockers),
                    action_available=True,
                    selection_score=selection_score,
                    governance_note="compatible with registered action schemas",
                )
            )
        candidates.sort(key=lambda item: (-item.selection_score, item.cost_profile.score, item.action_name))
        return candidates

    def _information_gain(
        self,
        *,
        action_name: str,
        hypotheses: list[CompetingHypothesis],
        unknown: NarrativeUnknown,
    ) -> ExpectedInformationGain:
        pairwise_deltas: list[float] = []
        for index, left in enumerate(hypotheses):
            for right in hypotheses[index + 1 :]:
                channels = set(left.expected_state_shift) | set(right.expected_state_shift)
                pairwise_deltas.append(
                    max(
                        [
                            abs(float(left.expected_state_shift.get(channel, 0.5)) - float(right.expected_state_shift.get(channel, 0.5)))
                            for channel in channels
                        ]
                        or [0.0]
                    )
                )
        separation = max(pairwise_deltas or [0.0])
        affinity = {
            "seek_contact": 0.24 if unknown.unknown_type in {"trust", "motive", "communication"} else 0.02,
            "scan": 0.24 if unknown.unknown_type in {"threat_persistence", "environment_reliability"} else 0.10,
            "wait": 0.18 if unknown.unknown_type in {"threat_persistence", "trust", "environment_reliability"} else 0.08,
            "hide": 0.10,
            "exploit_shelter": 0.08,
            "rest": 0.04,
        }.get(action_name, 0.04)
        score = _clamp(separation * 1.15 + affinity + float(unknown.decision_relevance.verification_urgency) * 0.16)
        separated_hypothesis_count = 1 + sum(1 for value in pairwise_deltas if value >= 0.12) if pairwise_deltas else 0
        falsification_focus = _clamp(separation * 0.85 + (0.12 if action_name in {"scan", "seek_contact"} else 0.04))
        confirmation_focus = _clamp(separation * 0.55 + (0.12 if action_name == "wait" else 0.08))
        return ExpectedInformationGain(
            score=score,
            separated_hypothesis_count=separated_hypothesis_count,
            confirmation_focus=confirmation_focus,
            falsification_focus=falsification_focus,
            ambiguity_remaining=_clamp(1.0 - score),
            weak_signal=score < 0.32,
        )

    def _falsification_opportunity(
        self,
        *,
        action_name: str,
        hypotheses: list[CompetingHypothesis],
    ) -> FalsificationOpportunity:
        sortable = sorted(
            hypotheses,
            key=lambda item: (item.support.support_score - item.support.contradiction_score, item.hypothesis_id),
            reverse=True,
        )
        target_ids = tuple(item.hypothesis_id for item in sortable[1:3]) if len(sortable) > 1 else ()
        return FalsificationOpportunity(
            score=_clamp((0.20 if action_name in {"scan", "seek_contact"} else 0.10) + len(target_ids) * 0.12),
            target_hypothesis_ids=target_ids,
            summary=f"{action_name} can falsify " + (", ".join(target_ids) if target_ids else "little beyond the current ambiguity"),
        )

    def _cost_profile(self, *, action_name: str, schema: ActionSchema) -> InquiryCostProfile:
        delay = {"wait": 3.0, "rest": 2.0, "hide": 2.0, "seek_contact": 2.0, "scan": 1.0, "exploit_shelter": 1.0}.get(action_name, 1.5)
        resource = float(schema.cost_estimate) + sum(float(value) for value in schema.resource_cost.values()) * 0.04
        return InquiryCostProfile(
            score=_clamp(resource * 1.9 + delay * 0.12),
            expected_delay=delay,
            resource_cost=resource,
            summary=f"expected delay {delay:.1f} and cost {resource:.2f}",
        )

    def _risk_profile(
        self,
        *,
        action_name: str,
        unknown: NarrativeUnknown,
        subject_state,
        schema: ActionSchema,
    ) -> InquiryRiskProfile:
        del schema
        score = {
            "seek_contact": 0.48,
            "scan": 0.26,
            "wait": 0.16,
            "hide": 0.10,
            "exploit_shelter": 0.08,
            "rest": 0.05,
        }.get(action_name, 0.18)
        blockers: list[str] = []
        if unknown.unknown_type == "threat_persistence" and action_name == "seek_contact":
            score += 0.20
            blockers.append("contact does not match the threat inquiry channel")
        if unknown.unknown_type in {"trust", "motive"} and action_name == "seek_contact":
            score += 0.10
        if subject_state is not None:
            flags = getattr(subject_state, "status_flags", {})
            if isinstance(flags, Mapping) and flags.get("threatened", False) and action_name in {"seek_contact", "scan"}:
                score += 0.20
                blockers.append("subject currently threatened")
            if isinstance(flags, Mapping) and flags.get("socially_destabilized", False) and action_name == "seek_contact":
                score += 0.12
                blockers.append("social channel currently destabilized")
        score = _clamp(score)
        return InquiryRiskProfile(
            score=score,
            safety_margin=_clamp(1.0 - score),
            blockers=tuple(dict.fromkeys(blockers)),
            summary="safe now" if score <= 0.32 else "risk should be bounded before acting",
        )

    def _plan_status(
        self,
        *,
        candidate: InquiryActionCandidate,
        unknown: NarrativeUnknown,
        active_count: int,
    ) -> str:
        if unknown.decision_relevance.total_score < 0.34:
            return InquiryPlanStatus.REJECTED_LOW_VALUE.value
        if not candidate.action_available:
            return InquiryPlanStatus.BLOCKED_BY_GOVERNANCE.value
        if candidate.information_gain.score < 0.28:
            return InquiryPlanStatus.REJECTED_LOW_INFORMATION_GAIN.value
        if candidate.risk_profile.score >= 0.60 or candidate.risk_profile.blockers:
            return InquiryPlanStatus.DEFERRED_FOR_RISK.value
        if candidate.cost_profile.score >= 0.56 or candidate.cost_profile.expected_delay >= 3.0:
            return InquiryPlanStatus.DEFERRED_FOR_BUDGET.value
        if active_count >= self.max_active_plans:
            return InquiryPlanStatus.QUEUED_EXPERIMENT.value
        return InquiryPlanStatus.ACTIVE_EXPERIMENT.value

    def _candidate_actions(self, unknown_type: str) -> tuple[str, ...]:
        if unknown_type in {"trust", "motive", "communication"}:
            return ("seek_contact", "scan", "wait", "rest")
        if unknown_type in {"threat_persistence", "environment_reliability"}:
            return ("scan", "wait", "exploit_shelter", "hide", "rest")
        return ("scan", "wait", "rest")

    def _goal_alignment_bonus(
        self,
        *,
        action_name: str,
        unknown: NarrativeUnknown,
        active_goal: str,
    ) -> float:
        goal = str(active_goal or "").upper()
        if not goal:
            return 0.0
        if goal == "SOCIAL":
            if unknown.unknown_type in {"trust", "motive", "communication"}:
                return {
                    "seek_contact": 0.07,
                    "scan": 0.01,
                    "wait": 0.02,
                    "rest": 0.0,
                }.get(action_name, 0.0)
            if action_name == "seek_contact":
                return -0.04
        if goal in {"SAFETY", "SURVIVAL"}:
            if unknown.unknown_type in {"threat_persistence", "environment_reliability"}:
                return {
                    "scan": 0.08,
                    "exploit_shelter": 0.05,
                    "hide": 0.04,
                    "wait": 0.02,
                    "rest": -0.01,
                }.get(action_name, 0.0)
            if action_name == "seek_contact":
                return -0.03
        if goal == "RESTORATION":
            return {
                "rest": 0.06,
                "wait": 0.03,
                "scan": -0.01,
                "seek_contact": -0.02,
            }.get(action_name, 0.0)
        return 0.0

    def _distinction_summary(self, *, action_name: str, hypotheses: list[CompetingHypothesis]) -> str:
        labels = [item.hypothesis_id for item in hypotheses[:2]]
        if len(labels) >= 2:
            return f"{action_name} helps separate {labels[0]} from {labels[1]}"
        if labels:
            return f"{action_name} probes whether {labels[0]} remains viable"
        return f"{action_name} probes the remaining ambiguity"

    def _selected_reason(
        self,
        *,
        candidate: InquiryActionCandidate,
        unknown: NarrativeUnknown,
        hypotheses: list[CompetingHypothesis],
    ) -> str:
        left = hypotheses[0].statement if hypotheses else "the leading explanation"
        right = hypotheses[1].statement if len(hypotheses) > 1 else "the remaining alternatives"
        return (
            f"I selected {candidate.action_name} because it best distinguishes {left} from {right} "
            f"for {unknown.unknown_id}, with information gain {candidate.information_gain.score:.2f} "
            f"at risk {candidate.risk_profile.score:.2f} and cost {candidate.cost_profile.score:.2f}."
        )

    def _fallback_behavior(self, candidate: InquiryActionCandidate) -> str:
        if candidate.information_gain.weak_signal:
            return "If evidence stays weak, wait for passive evidence and avoid escalating the inquiry."
        if candidate.risk_profile.score > 0.42:
            return "If the channel becomes unsafe, defer and switch to passive observation."
        return "If outcomes stay inconclusive, fall back to waiting for passive evidence and re-rank alternatives."

    def _default_channels(self, unknown: NarrativeUnknown) -> tuple[str, ...]:
        if unknown.unknown_type in {"trust", "motive", "communication"}:
            return ("social", "danger")
        if unknown.unknown_type in {"threat_persistence", "environment_reliability"}:
            return ("danger", "novelty")
        return ("danger",)

    def _candidate_channels(
        self,
        candidate: InquiryActionCandidate,
        predictions: list[HypothesisPrediction],
    ) -> tuple[str, ...]:
        channels: list[str] = []
        for prediction in predictions:
            if prediction.prediction_id not in candidate.linked_prediction_ids:
                continue
            channels.extend(prediction.evidence_channels)
        if not channels:
            if candidate.action_name == "seek_contact":
                channels.append("social")
            elif candidate.action_name == "scan":
                channels.extend(["danger", "novelty"])
            else:
                channels.append("danger")
        return tuple(dict.fromkeys(channels))

    def _hypothesis_signature(self, hypothesis: CompetingHypothesis) -> str:
        if hypothesis.expected_state_shift:
            key, value = sorted(hypothesis.expected_state_shift.items())[0]
            return f"{key}={value:.2f}"
        return hypothesis.statement[:48]

    def _summary(
        self,
        *,
        plans: list[ExperimentPlan],
        targets: list[DiscriminationTarget],
    ) -> str:
        active = [item for item in plans if item.status == InquiryPlanStatus.ACTIVE_EXPERIMENT.value]
        deferred = [item for item in plans if item.status.startswith("deferred")]
        if active:
            top = active[0]
            return (
                f"The active inquiry is {top.selected_action}, targeting {top.target_unknown_id}, "
                "because it offers the best bounded discrimination among competing narrative hypotheses."
            )
        if deferred:
            top = deferred[0]
            return (
                f"The best currently known inquiry for {top.target_unknown_id} was deferred because "
                "its gain is outweighed by current risk or budget pressure."
            )
        if targets:
            return "Narrative ambiguity was reviewed, but the remaining experiments were rejected as low-value or low-gain."
        return "No narrative discrimination target required experiment design."

    def _archived_outcomes(self, verification_loop) -> dict[str, int]:
        counts: dict[str, int] = {}
        if verification_loop is None:
            return counts
        for target in getattr(verification_loop, "archived_targets", []):
            plan_id = str(getattr(target.plan, "linked_experiment_plan_id", ""))
            if not plan_id:
                continue
            if str(target.outcome) in {"inconclusive", "expired_unverified", "deferred"}:
                counts[plan_id] = counts.get(plan_id, 0) + 1
        return counts
