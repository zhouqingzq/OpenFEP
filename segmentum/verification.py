from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from .prediction_ledger import (
    DiscrepancySource,
    PredictionHypothesis,
    PredictionLedger,
    VerificationStatus as LedgerVerificationStatus,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class VerificationTargetStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COLLECTING_EVIDENCE = "collecting_evidence"
    DEFERRED = "deferred"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class VerificationEvidenceSource(StrEnum):
    RUNTIME_OBSERVATION = "runtime_observation"
    VALIDATION_OBSERVATION = "validation_observation"
    ACTION_ACKNOWLEDGMENT = "action_acknowledgment"
    MISSING_EVIDENCE = "missing_evidence"
    SLEEP_REVIEW = "sleep_review"


class VerificationOutcome(StrEnum):
    CONFIRMED = "confirmed"
    FALSIFIED = "falsified"
    PARTIALLY_SUPPORTED = "partially_supported"
    INCONCLUSIVE = "inconclusive"
    EXPIRED_UNVERIFIED = "expired_unverified"
    DEFERRED = "deferred"
    CONTRADICTED_BY_NEW_EVIDENCE = "contradicted_by_new_evidence"


@dataclass(frozen=True)
class VerificationPlan:
    prediction_id: str
    selected_reason: str
    evidence_sought: tuple[str, ...]
    support_criteria: tuple[str, ...]
    falsification_criteria: tuple[str, ...]
    expected_horizon: int
    created_tick: int
    expires_tick: int
    status: str = VerificationTargetStatus.PENDING.value
    linked_action: str = ""
    attention_channels: tuple[str, ...] = ()
    linked_experiment_plan_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "selected_reason": self.selected_reason,
            "evidence_sought": list(self.evidence_sought),
            "support_criteria": list(self.support_criteria),
            "falsification_criteria": list(self.falsification_criteria),
            "expected_horizon": int(self.expected_horizon),
            "created_tick": int(self.created_tick),
            "expires_tick": int(self.expires_tick),
            "status": self.status,
            "linked_action": self.linked_action,
            "attention_channels": list(self.attention_channels),
            "linked_experiment_plan_id": self.linked_experiment_plan_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "VerificationPlan":
        if not payload:
            return cls(
                prediction_id="",
                selected_reason="",
                evidence_sought=(),
                support_criteria=(),
                falsification_criteria=(),
                expected_horizon=1,
                created_tick=0,
                expires_tick=1,
            )
        return cls(
            prediction_id=str(payload.get("prediction_id", "")),
            selected_reason=str(payload.get("selected_reason", "")),
            evidence_sought=tuple(str(item) for item in payload.get("evidence_sought", [])),
            support_criteria=tuple(str(item) for item in payload.get("support_criteria", [])),
            falsification_criteria=tuple(
                str(item) for item in payload.get("falsification_criteria", [])
            ),
            expected_horizon=max(1, int(payload.get("expected_horizon", 1))),
            created_tick=int(payload.get("created_tick", 0)),
            expires_tick=int(payload.get("expires_tick", 1)),
            status=str(payload.get("status", VerificationTargetStatus.PENDING.value)),
            linked_action=str(payload.get("linked_action", "")),
            attention_channels=tuple(str(item) for item in payload.get("attention_channels", [])),
            linked_experiment_plan_id=str(payload.get("linked_experiment_plan_id", "")),
        )


@dataclass(frozen=True)
class VerificationEvidence:
    evidence_id: str
    tick: int
    target_id: str
    prediction_id: str
    source: str
    summary: str
    observed_state: dict[str, float]
    compared_channels: tuple[str, ...] = ()
    missing_channels: tuple[str, ...] = ()
    support_score: float = 0.0
    contradiction_score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "tick": int(self.tick),
            "target_id": self.target_id,
            "prediction_id": self.prediction_id,
            "source": self.source,
            "summary": self.summary,
            "observed_state": {str(key): float(value) for key, value in self.observed_state.items()},
            "compared_channels": list(self.compared_channels),
            "missing_channels": list(self.missing_channels),
            "support_score": round(self.support_score, 6),
            "contradiction_score": round(self.contradiction_score, 6),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "VerificationEvidence":
        if not payload:
            return cls(
                evidence_id="",
                tick=0,
                target_id="",
                prediction_id="",
                source=VerificationEvidenceSource.RUNTIME_OBSERVATION.value,
                summary="",
                observed_state={},
            )
        return cls(
            evidence_id=str(payload.get("evidence_id", "")),
            tick=int(payload.get("tick", 0)),
            target_id=str(payload.get("target_id", "")),
            prediction_id=str(payload.get("prediction_id", "")),
            source=str(payload.get("source", VerificationEvidenceSource.RUNTIME_OBSERVATION.value)),
            summary=str(payload.get("summary", "")),
            observed_state={
                str(key): float(value)
                for key, value in dict(payload.get("observed_state", {})).items()
                if isinstance(value, (int, float))
            },
            compared_channels=tuple(str(item) for item in payload.get("compared_channels", [])),
            missing_channels=tuple(str(item) for item in payload.get("missing_channels", [])),
            support_score=float(payload.get("support_score", 0.0)),
            contradiction_score=float(payload.get("contradiction_score", 0.0)),
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), dict)
            else {},
        )


@dataclass(frozen=True)
class FalsificationRecord:
    target_id: str
    prediction_id: str
    tick: int
    outcome: str
    reason: str
    evidence_ids: tuple[str, ...]
    severity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "prediction_id": self.prediction_id,
            "tick": int(self.tick),
            "outcome": self.outcome,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "severity": round(self.severity, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "FalsificationRecord":
        if not payload:
            return cls(
                target_id="",
                prediction_id="",
                tick=0,
                outcome=VerificationOutcome.FALSIFIED.value,
                reason="",
                evidence_ids=(),
                severity=0.0,
            )
        return cls(
            target_id=str(payload.get("target_id", "")),
            prediction_id=str(payload.get("prediction_id", "")),
            tick=int(payload.get("tick", 0)),
            outcome=str(payload.get("outcome", VerificationOutcome.FALSIFIED.value)),
            reason=str(payload.get("reason", "")),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", [])),
            severity=float(payload.get("severity", 0.0)),
        )


@dataclass(frozen=True)
class PredictionUpdateResult:
    prediction_id: str
    previous_status: str
    new_status: str
    previous_confidence: float
    new_confidence: float
    ledger_discrepancy_id: str = ""
    discharged_discrepancy_ids: tuple[str, ...] = ()
    subject_pressure_delta: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "previous_confidence": round(self.previous_confidence, 6),
            "new_confidence": round(self.new_confidence, 6),
            "ledger_discrepancy_id": self.ledger_discrepancy_id,
            "discharged_discrepancy_ids": list(self.discharged_discrepancy_ids),
            "subject_pressure_delta": round(self.subject_pressure_delta, 6),
        }


@dataclass(frozen=True)
class VerificationTarget:
    target_id: str
    prediction_id: str
    created_tick: int
    priority_score: float
    selected_reason: str
    plan: VerificationPlan
    status: str = VerificationTargetStatus.PENDING.value
    outcome: str = ""
    outcome_tick: int = 0
    evidence_ids: tuple[str, ...] = ()
    last_evidence_tick: int = 0
    attempts: int = 0
    missed_evidence_count: int = 0
    linked_discrepancy_id: str = ""
    linked_commitments: tuple[str, ...] = ()
    linked_identity_anchors: tuple[str, ...] = ()
    target_channels: tuple[str, ...] = ()
    prediction_type: str = ""
    verification_bias: float = 0.0
    linked_experiment_plan_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "prediction_id": self.prediction_id,
            "created_tick": int(self.created_tick),
            "priority_score": round(self.priority_score, 6),
            "selected_reason": self.selected_reason,
            "plan": self.plan.to_dict(),
            "status": self.status,
            "outcome": self.outcome,
            "outcome_tick": int(self.outcome_tick),
            "evidence_ids": list(self.evidence_ids),
            "last_evidence_tick": int(self.last_evidence_tick),
            "attempts": int(self.attempts),
            "missed_evidence_count": int(self.missed_evidence_count),
            "linked_discrepancy_id": self.linked_discrepancy_id,
            "linked_commitments": list(self.linked_commitments),
            "linked_identity_anchors": list(self.linked_identity_anchors),
            "target_channels": list(self.target_channels),
            "prediction_type": self.prediction_type,
            "verification_bias": round(self.verification_bias, 6),
            "linked_experiment_plan_id": self.linked_experiment_plan_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "VerificationTarget":
        if not payload:
            return cls(
                target_id="",
                prediction_id="",
                created_tick=0,
                priority_score=0.0,
                selected_reason="",
                plan=VerificationPlan.from_dict(None),
            )
        return cls(
            target_id=str(payload.get("target_id", "")),
            prediction_id=str(payload.get("prediction_id", "")),
            created_tick=int(payload.get("created_tick", 0)),
            priority_score=float(payload.get("priority_score", 0.0)),
            selected_reason=str(payload.get("selected_reason", "")),
            plan=VerificationPlan.from_dict(
                payload.get("plan") if isinstance(payload.get("plan"), Mapping) else None
            ),
            status=str(payload.get("status", VerificationTargetStatus.PENDING.value)),
            outcome=str(payload.get("outcome", "")),
            outcome_tick=int(payload.get("outcome_tick", 0)),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", [])),
            last_evidence_tick=int(payload.get("last_evidence_tick", 0)),
            attempts=int(payload.get("attempts", 0)),
            missed_evidence_count=int(payload.get("missed_evidence_count", 0)),
            linked_discrepancy_id=str(payload.get("linked_discrepancy_id", "")),
            linked_commitments=tuple(str(item) for item in payload.get("linked_commitments", [])),
            linked_identity_anchors=tuple(
                str(item) for item in payload.get("linked_identity_anchors", [])
            ),
            target_channels=tuple(str(item) for item in payload.get("target_channels", [])),
            prediction_type=str(payload.get("prediction_type", "")),
            verification_bias=float(payload.get("verification_bias", 0.0)),
            linked_experiment_plan_id=str(payload.get("linked_experiment_plan_id", "")),
        )


@dataclass(frozen=True)
class VerificationLoopUpdate:
    created_targets: tuple[str, ...] = ()
    prioritized_target_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    prediction_updates: tuple[dict[str, object], ...] = ()
    discharged_discrepancies: tuple[str, ...] = ()
    escalated_discrepancies: tuple[str, ...] = ()
    expired_targets: tuple[str, ...] = ()
    deferred_targets: tuple[str, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "created_targets": list(self.created_targets),
            "prioritized_target_id": self.prioritized_target_id,
            "evidence_ids": list(self.evidence_ids),
            "outcomes": list(self.outcomes),
            "prediction_updates": list(self.prediction_updates),
            "discharged_discrepancies": list(self.discharged_discrepancies),
            "escalated_discrepancies": list(self.escalated_discrepancies),
            "expired_targets": list(self.expired_targets),
            "deferred_targets": list(self.deferred_targets),
            "summary": self.summary,
        }


@dataclass
class VerificationLoop:
    active_targets: list[VerificationTarget] = field(default_factory=list)
    archived_targets: list[VerificationTarget] = field(default_factory=list)
    evidence_history: list[VerificationEvidence] = field(default_factory=list)
    falsification_history: list[FalsificationRecord] = field(default_factory=list)
    last_tick: int = 0
    max_active_targets: int = 3
    evidence_history_limit: int = 96
    archive_limit: int = 48

    def to_dict(self) -> dict[str, object]:
        return {
            "active_targets": [item.to_dict() for item in self.active_targets],
            "archived_targets": [item.to_dict() for item in self.archived_targets],
            "evidence_history": [item.to_dict() for item in self.evidence_history],
            "falsification_history": [item.to_dict() for item in self.falsification_history],
            "last_tick": int(self.last_tick),
            "max_active_targets": int(self.max_active_targets),
            "evidence_history_limit": int(self.evidence_history_limit),
            "archive_limit": int(self.archive_limit),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "VerificationLoop":
        if not payload:
            return cls()
        return cls(
            active_targets=[
                VerificationTarget.from_dict(item)
                for item in payload.get("active_targets", [])
                if isinstance(item, Mapping)
            ],
            archived_targets=[
                VerificationTarget.from_dict(item)
                for item in payload.get("archived_targets", [])
                if isinstance(item, Mapping)
            ],
            evidence_history=[
                VerificationEvidence.from_dict(item)
                for item in payload.get("evidence_history", [])
                if isinstance(item, Mapping)
            ],
            falsification_history=[
                FalsificationRecord.from_dict(item)
                for item in payload.get("falsification_history", [])
                if isinstance(item, Mapping)
            ],
            last_tick=int(payload.get("last_tick", 0)),
            max_active_targets=max(1, int(payload.get("max_active_targets", 3))),
            evidence_history_limit=max(8, int(payload.get("evidence_history_limit", 96))),
            archive_limit=max(8, int(payload.get("archive_limit", 48))),
        )

    def refresh_targets(
        self,
        *,
        tick: int,
        ledger: PredictionLedger,
        diagnostics=None,
        subject_state=None,
        narrative_uncertainty=None,
        experiment_design=None,
        workspace_channels: tuple[str, ...] = (),
    ) -> VerificationLoopUpdate:
        expired_updates = self._expire_missing_targets(tick=tick, ledger=ledger)
        active_prediction_ids = {item.prediction_id for item in self.active_targets}
        created_targets: list[str] = []
        candidates: list[tuple[float, PredictionHypothesis]] = []
        for prediction in ledger.active_predictions():
            if not prediction.prediction_id or prediction.prediction_id in active_prediction_ids:
                continue
            score = self._target_score(
                prediction=prediction,
                tick=tick,
                diagnostics=diagnostics,
                subject_state=subject_state,
                narrative_uncertainty=narrative_uncertainty,
                experiment_design=experiment_design,
                workspace_channels=workspace_channels,
            )
            if score < 0.28:
                continue
            candidates.append((score, prediction))
        candidates.sort(key=lambda item: (-item[0], item[1].created_tick, item[1].prediction_id))
        available_slots = max(0, self.max_active_targets - len(self.active_targets))
        for score, prediction in candidates[:available_slots]:
            target = self._make_target(
                prediction=prediction,
                tick=tick,
                diagnostics=diagnostics,
                subject_state=subject_state,
                experiment_design=experiment_design,
            )
            target = VerificationTarget(
                **{
                    **target.__dict__,
                    "priority_score": score,
                    "verification_bias": round(min(0.18, 0.05 + score * 0.12), 6),
                }
            )
            self.active_targets.append(target)
            created_targets.append(target.target_id)
        self._trim()
        prioritized = self.prioritized_target()
        self.last_tick = tick
        summary = "No verification target is currently active."
        if prioritized is not None:
            summary = (
                f"I prioritized verifying {prioritized.prediction_id} because "
                f"{prioritized.selected_reason}."
            )
        return VerificationLoopUpdate(
            created_targets=tuple(created_targets),
            prioritized_target_id=prioritized.target_id if prioritized else "",
            prediction_updates=tuple(expired_updates["prediction_updates"]),
            discharged_discrepancies=tuple(expired_updates["discharged_discrepancies"]),
            escalated_discrepancies=tuple(expired_updates["escalated_discrepancies"]),
            expired_targets=tuple(expired_updates["expired_targets"]),
            deferred_targets=tuple(expired_updates["deferred_targets"]),
            summary=summary,
        )

    def register_action_ack(
        self,
        *,
        tick: int,
        action_name: str,
        success: bool,
    ) -> VerificationLoopUpdate:
        evidence_ids: list[str] = []
        deferred_targets: list[str] = []
        updated_targets: list[VerificationTarget] = []
        for target in self.active_targets:
            if target.plan.linked_action != action_name:
                updated_targets.append(target)
                continue
            evidence = VerificationEvidence(
                evidence_id=f"ack:{target.target_id}:{tick}",
                tick=tick,
                target_id=target.target_id,
                prediction_id=target.prediction_id,
                source=VerificationEvidenceSource.ACTION_ACKNOWLEDGMENT.value,
                summary=(
                    f"Verification-seeking action {action_name} "
                    + ("was acknowledged." if success else "failed to acknowledge.")
                ),
                observed_state={},
                support_score=0.25 if success else 0.0,
                contradiction_score=0.20 if not success else 0.0,
                metadata={"action_name": action_name, "success": bool(success)},
            )
            self.evidence_history.append(evidence)
            evidence_ids.append(evidence.evidence_id)
            updated_targets.append(
                VerificationTarget(
                    **{
                        **target.__dict__,
                        "status": VerificationTargetStatus.COLLECTING_EVIDENCE.value
                        if success
                        else VerificationTargetStatus.DEFERRED.value,
                        "evidence_ids": tuple([*target.evidence_ids, evidence.evidence_id]),
                        "last_evidence_tick": tick,
                        "attempts": target.attempts + 1,
                        "missed_evidence_count": target.missed_evidence_count + (0 if success else 1),
                    }
                )
            )
            if not success:
                deferred_targets.append(target.target_id)
        self.active_targets = updated_targets
        self._trim()
        prioritized = self.prioritized_target()
        return VerificationLoopUpdate(
            evidence_ids=tuple(evidence_ids),
            deferred_targets=tuple(deferred_targets),
            prioritized_target_id=prioritized.target_id if prioritized else "",
            summary="Verification loop recorded action acknowledgement evidence."
            if evidence_ids
            else "No verification-seeking action acknowledgement was relevant.",
        )

    def process_observation(
        self,
        *,
        tick: int,
        observation: Mapping[str, float],
        ledger: PredictionLedger,
        source: str,
        subject_state=None,
    ) -> VerificationLoopUpdate:
        evidence_ids: list[str] = []
        outcomes: list[str] = []
        prediction_updates: list[dict[str, object]] = []
        discharged: list[str] = []
        escalated: list[str] = []
        expired: list[str] = []
        deferred: list[str] = []
        updated_targets: list[VerificationTarget] = []
        for target in self.active_targets:
            prediction = self._prediction_by_id(ledger, target.prediction_id)
            if prediction is None:
                self.archived_targets.append(
                    VerificationTarget(
                        **{
                            **target.__dict__,
                            "status": VerificationTargetStatus.EXPIRED.value,
                            "outcome": VerificationOutcome.EXPIRED_UNVERIFIED.value,
                            "outcome_tick": tick,
                        }
                    )
                )
                expired.append(target.target_id)
                continue
            evidence, outcome = self._evaluate_prediction(
                target=target,
                prediction=prediction,
                observation=observation,
                tick=tick,
                source=source,
            )
            if evidence is not None:
                self.evidence_history.append(evidence)
                evidence_ids.append(evidence.evidence_id)
            if outcome in {
                VerificationOutcome.CONFIRMED.value,
                VerificationOutcome.FALSIFIED.value,
                VerificationOutcome.PARTIALLY_SUPPORTED.value,
                VerificationOutcome.INCONCLUSIVE.value,
                VerificationOutcome.EXPIRED_UNVERIFIED.value,
                VerificationOutcome.DEFERRED.value,
                VerificationOutcome.CONTRADICTED_BY_NEW_EVIDENCE.value,
            }:
                outcomes.append(outcome)
            if outcome in {
                VerificationOutcome.CONFIRMED.value,
                VerificationOutcome.FALSIFIED.value,
                VerificationOutcome.PARTIALLY_SUPPORTED.value,
                VerificationOutcome.INCONCLUSIVE.value,
                VerificationOutcome.EXPIRED_UNVERIFIED.value,
                VerificationOutcome.CONTRADICTED_BY_NEW_EVIDENCE.value,
            }:
                update_result = self._apply_outcome(
                    tick=tick,
                    ledger=ledger,
                    prediction=prediction,
                    target=target,
                    evidence=evidence,
                    outcome=outcome,
                    subject_state=subject_state,
                )
                prediction_updates.append(update_result.to_dict())
                discharged.extend(update_result.discharged_discrepancy_ids)
                if update_result.ledger_discrepancy_id:
                    escalated.append(update_result.ledger_discrepancy_id)
            if outcome == VerificationOutcome.DEFERRED.value:
                deferred.append(target.target_id)
                updated_targets.append(
                    VerificationTarget(
                        **{
                            **target.__dict__,
                            "status": VerificationTargetStatus.DEFERRED.value,
                            "outcome": outcome,
                            "outcome_tick": tick,
                            "evidence_ids": tuple(
                                [*target.evidence_ids, *([evidence.evidence_id] if evidence else [])]
                            ),
                            "last_evidence_tick": tick if evidence else target.last_evidence_tick,
                            "missed_evidence_count": target.missed_evidence_count + 1,
                        }
                    )
                )
                continue
            if outcome in {
                VerificationOutcome.CONFIRMED.value,
                VerificationOutcome.FALSIFIED.value,
                VerificationOutcome.PARTIALLY_SUPPORTED.value,
                VerificationOutcome.INCONCLUSIVE.value,
                VerificationOutcome.EXPIRED_UNVERIFIED.value,
                VerificationOutcome.CONTRADICTED_BY_NEW_EVIDENCE.value,
            }:
                resolved_status = (
                    VerificationTargetStatus.EXPIRED.value
                    if outcome == VerificationOutcome.EXPIRED_UNVERIFIED.value
                    else VerificationTargetStatus.RESOLVED.value
                )
                archived = VerificationTarget(
                    **{
                        **target.__dict__,
                        "status": resolved_status,
                        "outcome": outcome,
                        "outcome_tick": tick,
                        "evidence_ids": tuple(
                            [*target.evidence_ids, *([evidence.evidence_id] if evidence else [])]
                        ),
                        "last_evidence_tick": tick if evidence else target.last_evidence_tick,
                    }
                )
                self.archived_targets.append(archived)
                if outcome == VerificationOutcome.EXPIRED_UNVERIFIED.value:
                    expired.append(target.target_id)
                continue
            updated_targets.append(target)
        self.active_targets = updated_targets
        self._trim()
        self.last_tick = tick
        prioritized = self.prioritized_target()
        summary = "Verification loop did not update any target."
        if outcomes:
            summary = f"Verification outcomes observed: {', '.join(outcomes[:3])}."
        return VerificationLoopUpdate(
            prioritized_target_id=prioritized.target_id if prioritized else "",
            evidence_ids=tuple(evidence_ids),
            outcomes=tuple(outcomes),
            prediction_updates=tuple(prediction_updates),
            discharged_discrepancies=tuple(dict.fromkeys(discharged)),
            escalated_discrepancies=tuple(dict.fromkeys(escalated)),
            expired_targets=tuple(dict.fromkeys(expired)),
            deferred_targets=tuple(dict.fromkeys(deferred)),
            summary=summary,
        )

    def action_bias(self, action: str) -> float:
        bias = 0.0
        for target in self.active_targets[: self.max_active_targets]:
            if target.plan.linked_action == action:
                bias += target.verification_bias
            elif action in {"scan", "observe_world"} and (
                "danger" in target.plan.attention_channels or not target.plan.linked_action
            ):
                bias += target.verification_bias * 0.35
            elif target.plan.linked_action and action != target.plan.linked_action:
                bias -= target.verification_bias * 0.22
        return round(max(-0.24, min(0.24, bias)), 6)

    def workspace_focus(self) -> dict[str, float]:
        focus: dict[str, float] = {}
        for target in self.active_targets:
            weight = max(0.18, target.priority_score * 0.75)
            for channel in target.plan.attention_channels:
                focus[channel] = max(focus.get(channel, 0.0), round(weight, 6))
        return focus

    def memory_threshold_delta(self) -> float:
        if not self.active_targets:
            return 0.0
        strongest = max(target.priority_score for target in self.active_targets)
        return round(-min(0.10, 0.03 + strongest * 0.05), 6)

    def maintenance_signal(self) -> dict[str, object]:
        tasks: list[str] = []
        recommended_action = ""
        priority_gain = 0.0
        for target in self.active_targets:
            priority_gain += target.priority_score * 0.14
            tasks.append(f"verify:{target.prediction_id}")
            if not recommended_action and target.plan.linked_action:
                recommended_action = target.plan.linked_action
        return {
            "priority_gain": round(min(0.22, priority_gain), 6),
            "active_tasks": list(dict.fromkeys(tasks)),
            "recommended_action": recommended_action,
            "suppressed_actions": [],
        }

    def explanation_payload(self, *, chosen_action: str = "") -> dict[str, object]:
        prioritized = self.prioritized_target()
        summary = "I am not actively verifying any prediction right now."
        verification_motive = ""
        unresolved_reason = ""
        if prioritized is not None:
            summary = (
                f"I am currently trying to verify {prioritized.prediction_id} "
                f"because {prioritized.selected_reason}."
            )
            if prioritized.linked_experiment_plan_id:
                summary += " This target came from a narrative experiment plan."
            if chosen_action and chosen_action == prioritized.plan.linked_action:
                verification_motive = (
                    f"This action is partly verification-seeking: {chosen_action} should gather "
                    f"evidence about {prioritized.prediction_id}."
                )
        if prioritized is not None and prioritized.status == VerificationTargetStatus.DEFERRED.value:
            unresolved_reason = (
                "The prediction remains unresolved because the expected evidence has not "
                "arrived in time yet."
            )
        recent_outcomes = [
            item.to_dict()
            for item in self.archived_targets[-4:]
            if item.outcome
        ]
        return {
            "summary": summary,
            "verification_motive": verification_motive,
            "unresolved_reason": unresolved_reason,
            "prioritized_target": prioritized.to_dict() if prioritized else None,
            "active_targets": [item.to_dict() for item in self.active_targets],
            "recent_outcomes": recent_outcomes,
            "recent_evidence": [item.to_dict() for item in self.evidence_history[-6:]],
            "recent_falsifications": [item.to_dict() for item in self.falsification_history[-4:]],
            "counts": {
                "active_targets": len(self.active_targets),
                "archived_targets": len(self.archived_targets),
                "evidence_events": len(self.evidence_history),
            },
        }

    def prioritized_target(self) -> VerificationTarget | None:
        if not self.active_targets:
            return None
        ranked = sorted(
            self.active_targets,
            key=lambda item: (
                -item.priority_score,
                item.missed_evidence_count,
                item.created_tick,
                item.target_id,
            ),
        )
        return ranked[0]

    def _target_score(
        self,
        *,
        prediction: PredictionHypothesis,
        tick: int,
        diagnostics=None,
        subject_state=None,
        narrative_uncertainty=None,
        experiment_design=None,
        workspace_channels: tuple[str, ...],
    ) -> float:
        del narrative_uncertainty
        age = max(0, tick - prediction.created_tick)
        score = 0.18 + prediction.confidence * 0.26
        score += min(0.18, age * 0.05)
        if age >= prediction.expected_horizon:
            score += 0.16
        if prediction.recurrence_count:
            score += min(0.14, prediction.recurrence_count * 0.04)
        if prediction.linked_commitments or prediction.linked_identity_anchors:
            score += 0.12
        if prediction.linked_unknown_ids or prediction.linked_hypothesis_ids:
            score += 0.08
        if prediction.source_module == "narrative_uncertainty":
            score += 0.08 + min(0.10, prediction.decision_relevance * 0.18)
        if prediction.source_module == "narrative_experiment":
            score += 0.14 + min(0.14, prediction.decision_relevance * 0.24)
        if experiment_design is not None and prediction.linked_experiment_plan_id:
            active_plan_ids = {
                item.plan_id
                for item in getattr(experiment_design, "plans", ())
                if getattr(item, "status", "") in {"active_experiment", "queued_experiment"}
            }
            if prediction.linked_experiment_plan_id in active_plan_ids:
                score += 0.18
        if "social" in prediction.target_channels or "social" in prediction.prediction_type:
            score += 0.06
        if "danger" in prediction.target_channels or "maintenance" in prediction.target_channels:
            score += 0.10
        if diagnostics is not None and getattr(diagnostics, "active_goal", ""):
            if str(getattr(diagnostics, "active_goal", "")) == prediction.linked_goal:
                score += 0.08
        if subject_state is not None and getattr(subject_state, "status_flags", {}).get(
            "continuity_fragile", False
        ):
            if prediction.linked_identity_anchors or "continuity" in prediction.prediction_type:
                score += 0.10
        if subject_state is not None:
            slow_biases = getattr(subject_state, "slow_biases", {})
            if isinstance(slow_biases, Mapping):
                score += float(
                    min(
                        0.22,
                        max(
                            -0.04,
                            (
                                max(
                                    0.0,
                                    float(slow_biases.get("threat_sensitivity", 0.5)) - 0.5,
                                )
                                * (0.16 if "danger" in prediction.target_channels else 0.0)
                            )
                            + (
                                max(
                                    0.0,
                                    0.5 - float(slow_biases.get("trust_stance", 0.5)),
                                )
                                * (0.10 if "social" in prediction.target_channels else 0.0)
                            )
                            + (
                                max(
                                    0.0,
                                    0.55 - float(slow_biases.get("continuity_resilience", 0.5)),
                                )
                                * (0.12 if "continuity" in prediction.prediction_type else 0.0)
                            ),
                        ),
                    )
                )
        if any(channel in workspace_channels for channel in prediction.target_channels):
            score += 0.08
        return _clamp(score, 0.0, 1.2)

    def _make_target(
        self,
        *,
        prediction: PredictionHypothesis,
        tick: int,
        diagnostics=None,
        subject_state=None,
        experiment_design=None,
    ) -> VerificationTarget:
        del diagnostics, subject_state
        reason_parts: list[str] = []
        if "danger" in prediction.target_channels:
            reason_parts.append("it is survival-relevant")
        if "maintenance" in prediction.target_channels:
            reason_parts.append("it affects maintenance stability")
        if prediction.linked_commitments or prediction.linked_identity_anchors:
            reason_parts.append("it is identity-relevant")
        if prediction.source_module == "narrative_uncertainty":
            reason_parts.append("it would resolve a narrative ambiguity")
        if prediction.source_module == "narrative_experiment":
            reason_parts.append("it is linked to an active experiment plan")
        if "social" in prediction.target_channels or "social" in prediction.prediction_type:
            reason_parts.append("it is socially consequential")
        if prediction.recurrence_count:
            reason_parts.append("it has recurred")
        if not reason_parts:
            reason_parts.append("it is still unresolved")
        linked_action = self._linked_action(prediction)
        attention_channels = tuple(dict.fromkeys([*prediction.target_channels]))
        if prediction.linked_identity_anchors and "continuity" not in attention_channels:
            attention_channels = tuple([*attention_channels, "continuity"])
        plan = VerificationPlan(
            prediction_id=prediction.prediction_id,
            selected_reason=", ".join(reason_parts[:3]),
            evidence_sought=tuple(
                [f"observe:{channel}" for channel in prediction.target_channels]
                or [f"observe:{prediction.prediction_type}"]
            ),
            support_criteria=tuple(
                f"{channel} stays within {0.10 + (1.0 - prediction.confidence) * 0.10:.2f}"
                for channel in prediction.expected_state
            )
            or ("matching observation arrives",),
            falsification_criteria=tuple(
                f"{channel} diverges by >= {0.28 + prediction.confidence * 0.10:.2f}"
                for channel in prediction.expected_state
            )
            or ("contradicting observation arrives",),
            expected_horizon=max(1, prediction.expected_horizon),
            created_tick=tick,
            expires_tick=tick + max(1, prediction.expected_horizon),
            status=VerificationTargetStatus.ACTIVE.value,
            linked_action=linked_action,
            attention_channels=attention_channels,
            linked_experiment_plan_id=prediction.linked_experiment_plan_id,
        )
        return VerificationTarget(
            target_id=f"verify:{prediction.prediction_id}",
            prediction_id=prediction.prediction_id,
            created_tick=tick,
            priority_score=0.0,
            selected_reason=plan.selected_reason,
            plan=plan,
            status=VerificationTargetStatus.ACTIVE.value,
            linked_commitments=tuple(str(item) for item in prediction.linked_commitments),
            linked_identity_anchors=tuple(str(item) for item in prediction.linked_identity_anchors),
            target_channels=tuple(str(item) for item in prediction.target_channels),
            prediction_type=prediction.prediction_type,
            linked_experiment_plan_id=prediction.linked_experiment_plan_id,
        )

    def _linked_action(self, prediction: PredictionHypothesis) -> str:
        if prediction.source_module == "narrative_experiment" and prediction.maintenance_context:
            return prediction.maintenance_context
        channels = set(prediction.target_channels)
        if "danger" in channels:
            return "scan"
        if "social" in channels:
            return "seek_contact"
        if "maintenance" in channels or prediction.prediction_type == "maintenance_recovery":
            return "rest"
        if "continuity" in channels:
            return "scan"
        if prediction.prediction_type == "action_consequence":
            return "scan"
        return "scan"

    def _prediction_by_id(
        self,
        ledger: PredictionLedger,
        prediction_id: str,
    ) -> PredictionHypothesis | None:
        for item in [*ledger.predictions, *ledger.archived_predictions]:
            if item.prediction_id == prediction_id:
                return item
        return None

    def _evaluate_prediction(
        self,
        *,
        target: VerificationTarget,
        prediction: PredictionHypothesis,
        observation: Mapping[str, float],
        tick: int,
        source: str,
    ) -> tuple[VerificationEvidence | None, str]:
        compared_channels: list[str] = []
        missing_channels: list[str] = []
        support_hits = 0
        contradiction_hits = 0
        total_error = 0.0
        for channel, expected in prediction.expected_state.items():
            if channel not in observation:
                missing_channels.append(channel)
                continue
            observed_value = float(observation[channel])
            delta = abs(observed_value - expected)
            total_error += delta
            compared_channels.append(channel)
            if delta <= 0.12:
                support_hits += 1
            elif delta >= 0.30:
                contradiction_hits += 1
        if not compared_channels:
            if tick > target.plan.expires_tick:
                evidence = VerificationEvidence(
                    evidence_id=f"miss:{target.target_id}:{tick}",
                    tick=tick,
                    target_id=target.target_id,
                    prediction_id=target.prediction_id,
                    source=VerificationEvidenceSource.MISSING_EVIDENCE.value,
                    summary="Expected evidence did not arrive before the verification horizon elapsed.",
                    observed_state={},
                    missing_channels=tuple(missing_channels),
                    metadata={"timeout": True, "source": source},
                )
                return evidence, VerificationOutcome.EXPIRED_UNVERIFIED.value
            return None, VerificationOutcome.DEFERRED.value
        avg_error = total_error / max(1, len(compared_channels))
        support_threshold = max(0.12, 0.30 - prediction.confidence * 0.12)
        falsify_threshold = 0.28 + prediction.confidence * 0.10
        support_score = _clamp(1.0 - avg_error / max(0.01, falsify_threshold))
        contradiction_score = _clamp(avg_error / max(0.01, falsify_threshold))
        evidence = VerificationEvidence(
            evidence_id=f"obs:{target.target_id}:{tick}:{source}",
            tick=tick,
            target_id=target.target_id,
            prediction_id=target.prediction_id,
            source=source,
            summary=f"Observed {', '.join(compared_channels)} while verifying {prediction.prediction_id}.",
            observed_state={channel: float(observation[channel]) for channel in compared_channels},
            compared_channels=tuple(compared_channels),
            missing_channels=tuple(missing_channels),
            support_score=support_score,
            contradiction_score=contradiction_score,
            metadata={"avg_error": round(avg_error, 6)},
        )
        if avg_error <= support_threshold and contradiction_hits == 0 and not missing_channels:
            return evidence, VerificationOutcome.CONFIRMED.value
        if contradiction_hits and support_hits:
            return evidence, VerificationOutcome.PARTIALLY_SUPPORTED.value
        if avg_error >= falsify_threshold:
            if target.outcome in {
                VerificationOutcome.CONFIRMED.value,
                VerificationOutcome.PARTIALLY_SUPPORTED.value,
                VerificationOutcome.INCONCLUSIVE.value,
            }:
                return evidence, VerificationOutcome.CONTRADICTED_BY_NEW_EVIDENCE.value
            return evidence, VerificationOutcome.FALSIFIED.value
        if support_hits:
            return evidence, VerificationOutcome.PARTIALLY_SUPPORTED.value
        return evidence, VerificationOutcome.INCONCLUSIVE.value

    def _apply_outcome(
        self,
        *,
        tick: int,
        ledger: PredictionLedger,
        prediction: PredictionHypothesis,
        target: VerificationTarget,
        evidence: VerificationEvidence | None,
        outcome: str,
        subject_state=None,
    ) -> PredictionUpdateResult:
        del subject_state
        previous_status = prediction.status
        previous_confidence = prediction.confidence
        new_prediction = prediction
        discrepancy_id = ""
        discharged: tuple[str, ...] = ()
        subject_pressure_delta = 0.0
        if outcome == VerificationOutcome.CONFIRMED.value:
            discharged = tuple(
                ledger._discharge_matching(
                    f"disc:{prediction.prediction_type}:{'-'.join(prediction.target_channels) or 'state'}",
                    tick=tick,
                    reason="verification_confirmed",
                )
            )
            new_prediction = PredictionHypothesis(
                **{
                    **prediction.__dict__,
                    "last_updated_tick": tick,
                    "verification_tick": tick,
                    "verification_attempts": prediction.verification_attempts + 1,
                    "confidence": _clamp(prediction.confidence + 0.12),
                    "status": LedgerVerificationStatus.DISCHARGED.value,
                }
            )
            ledger.archived_predictions.append(new_prediction)
            ledger.predictions = [
                item for item in ledger.predictions if item.prediction_id != prediction.prediction_id
            ]
        elif outcome in {
            VerificationOutcome.FALSIFIED.value,
            VerificationOutcome.CONTRADICTED_BY_NEW_EVIDENCE.value,
        }:
            discrepancy_id = f"disc:verify:{prediction.prediction_id}"
            severity = 0.55
            if evidence is not None:
                severity = max(severity, evidence.contradiction_score)
            ledger._record_discrepancy(
                tick=tick,
                discrepancy_id=discrepancy_id,
                label=f"verification contradicted {prediction.prediction_type}",
                source=DiscrepancySource.PREDICTION_ERROR.value,
                discrepancy_type="verification_falsification",
                severity=min(1.0, severity),
                target_channels=prediction.target_channels,
                evidence=tuple(
                    dict.fromkeys(
                        [
                            f"prediction={prediction.prediction_id}",
                            evidence.summary if evidence is not None else "contradicting_observation",
                        ]
                    )
                ),
                identity_relevant=bool(prediction.linked_commitments or prediction.linked_identity_anchors),
                subject_critical=bool(
                    prediction.linked_identity_anchors or "danger" in prediction.target_channels
                ),
                linked_predictions=(prediction.prediction_id,),
                linked_commitments=prediction.linked_commitments,
                linked_goal=prediction.linked_goal,
            )
            subject_pressure_delta = 0.12 if prediction.linked_identity_anchors else 0.05
            new_prediction = PredictionHypothesis(
                **{
                    **prediction.__dict__,
                    "last_updated_tick": tick,
                    "verification_tick": tick,
                    "verification_attempts": prediction.verification_attempts + 1,
                    "recurrence_count": prediction.recurrence_count + 1,
                    "confidence": _clamp(prediction.confidence - 0.28),
                    "status": LedgerVerificationStatus.FALSIFIED.value,
                }
            )
            ledger.archived_predictions.append(new_prediction)
            ledger.predictions = [
                item for item in ledger.predictions if item.prediction_id != prediction.prediction_id
            ]
            self.falsification_history.append(
                FalsificationRecord(
                    target_id=target.target_id,
                    prediction_id=prediction.prediction_id,
                    tick=tick,
                    outcome=outcome,
                    reason=evidence.summary if evidence is not None else "contradicting evidence",
                    evidence_ids=(evidence.evidence_id,) if evidence is not None else (),
                    severity=min(1.0, severity),
                )
            )
        elif outcome == VerificationOutcome.PARTIALLY_SUPPORTED.value:
            new_prediction = PredictionHypothesis(
                **{
                    **prediction.__dict__,
                    "last_updated_tick": tick,
                    "verification_tick": tick,
                    "verification_attempts": prediction.verification_attempts + 1,
                    "confidence": _clamp(prediction.confidence + 0.03),
                    "expected_horizon": prediction.expected_horizon + 1,
                    "status": LedgerVerificationStatus.PARTIALLY_RESOLVED.value,
                }
            )
            ledger.predictions = [
                new_prediction if item.prediction_id == prediction.prediction_id else item
                for item in ledger.predictions
            ]
        elif outcome == VerificationOutcome.INCONCLUSIVE.value:
            new_prediction = PredictionHypothesis(
                **{
                    **prediction.__dict__,
                    "last_updated_tick": tick,
                    "verification_tick": tick,
                    "verification_attempts": prediction.verification_attempts + 1,
                    "confidence": _clamp(prediction.confidence - 0.05),
                    "expected_horizon": prediction.expected_horizon + 1,
                    "status": LedgerVerificationStatus.ACTIVE.value,
                }
            )
            ledger.predictions = [
                new_prediction if item.prediction_id == prediction.prediction_id else item
                for item in ledger.predictions
            ]
        elif outcome == VerificationOutcome.EXPIRED_UNVERIFIED.value:
            discrepancy_id = f"disc:verify_timeout:{prediction.prediction_id}"
            ledger._record_discrepancy(
                tick=tick,
                discrepancy_id=discrepancy_id,
                label="verification timeout",
                source=DiscrepancySource.PREDICTION_ERROR.value,
                discrepancy_type="verification_timeout",
                severity=min(1.0, 0.35 + prediction.confidence * 0.25),
                target_channels=prediction.target_channels,
                evidence=(f"prediction={prediction.prediction_id}", "expected_evidence_missing"),
                identity_relevant=bool(prediction.linked_identity_anchors),
                subject_critical="danger" in prediction.target_channels,
                linked_predictions=(prediction.prediction_id,),
                linked_goal=prediction.linked_goal,
            )
            new_prediction = PredictionHypothesis(
                **{
                    **prediction.__dict__,
                    "last_updated_tick": tick,
                    "verification_tick": tick,
                    "verification_attempts": prediction.verification_attempts + 1,
                    "confidence": _clamp(prediction.confidence - 0.12),
                    "status": LedgerVerificationStatus.ESCALATED.value,
                }
            )
            ledger.predictions = [
                new_prediction if item.prediction_id == prediction.prediction_id else item
                for item in ledger.predictions
            ]
        ledger._trim()
        return PredictionUpdateResult(
            prediction_id=prediction.prediction_id,
            previous_status=previous_status,
            new_status=new_prediction.status,
            previous_confidence=previous_confidence,
            new_confidence=new_prediction.confidence,
            ledger_discrepancy_id=discrepancy_id,
            discharged_discrepancy_ids=discharged,
            subject_pressure_delta=subject_pressure_delta,
        )

    def _expire_missing_targets(
        self,
        *,
        tick: int,
        ledger: PredictionLedger,
    ) -> dict[str, list[object]]:
        prediction_updates: list[dict[str, object]] = []
        discharged: list[str] = []
        escalated: list[str] = []
        expired: list[str] = []
        deferred: list[str] = []
        retained: list[VerificationTarget] = []
        for target in self.active_targets:
            if tick <= target.plan.expires_tick:
                retained.append(target)
                continue
            prediction = self._prediction_by_id(ledger, target.prediction_id)
            if prediction is None:
                expired.append(target.target_id)
                continue
            if target.last_evidence_tick >= target.plan.expires_tick:
                retained.append(target)
                continue
            evidence = VerificationEvidence(
                evidence_id=f"timeout:{target.target_id}:{tick}",
                tick=tick,
                target_id=target.target_id,
                prediction_id=target.prediction_id,
                source=VerificationEvidenceSource.MISSING_EVIDENCE.value,
                summary="Expected evidence did not arrive within the verification horizon.",
                observed_state={},
                missing_channels=target.plan.attention_channels,
                metadata={"timeout": True},
            )
            self.evidence_history.append(evidence)
            update_result = self._apply_outcome(
                tick=tick,
                ledger=ledger,
                prediction=prediction,
                target=target,
                evidence=evidence,
                outcome=VerificationOutcome.EXPIRED_UNVERIFIED.value,
            )
            prediction_updates.append(update_result.to_dict())
            if update_result.ledger_discrepancy_id:
                escalated.append(update_result.ledger_discrepancy_id)
            discharged.extend(update_result.discharged_discrepancy_ids)
            self.archived_targets.append(
                VerificationTarget(
                    **{
                        **target.__dict__,
                        "status": VerificationTargetStatus.EXPIRED.value,
                        "outcome": VerificationOutcome.EXPIRED_UNVERIFIED.value,
                        "outcome_tick": tick,
                        "evidence_ids": tuple([*target.evidence_ids, evidence.evidence_id]),
                        "last_evidence_tick": tick,
                    }
                )
            )
            expired.append(target.target_id)
            deferred.append(target.target_id)
        self.active_targets = retained
        return {
            "prediction_updates": prediction_updates,
            "discharged_discrepancies": discharged,
            "escalated_discrepancies": escalated,
            "expired_targets": expired,
            "deferred_targets": deferred,
        }

    def _trim(self) -> None:
        self.active_targets = sorted(
            self.active_targets,
            key=lambda item: (-item.priority_score, item.created_tick, item.target_id),
        )[: self.max_active_targets]
        self.archived_targets = self.archived_targets[-self.archive_limit :]
        self.evidence_history = self.evidence_history[-self.evidence_history_limit :]
        self.falsification_history = self.falsification_history[-self.archive_limit :]
