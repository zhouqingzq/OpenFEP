from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class VerificationStatus(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    ESCALATED = "escalated"
    PARTIALLY_RESOLVED = "partially_resolved"
    DISCHARGED = "discharged"
    ARCHIVED = "archived"
    FALSIFIED = "falsified"


class DiscrepancySource(StrEnum):
    PREDICTION_ERROR = "prediction_error"
    IDENTITY = "identity"
    MAINTENANCE = "maintenance"
    SOCIAL = "social"
    CONTINUITY = "continuity"
    MEMORY = "memory"
    WORKSPACE = "workspace"
    ACTION = "action"


class LedgerPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PredictionHypothesis:
    prediction_id: str
    created_tick: int
    last_updated_tick: int
    source_module: str
    prediction_type: str
    target_channels: tuple[str, ...]
    expected_state: dict[str, float]
    confidence: float
    expected_horizon: int
    status: str = VerificationStatus.ACTIVE.value
    supporting_evidence: tuple[str, ...] = ()
    linked_commitments: tuple[str, ...] = ()
    linked_identity_anchors: tuple[str, ...] = ()
    linked_unknown_ids: tuple[str, ...] = ()
    linked_hypothesis_ids: tuple[str, ...] = ()
    linked_goal: str = ""
    maintenance_context: str = ""
    decision_relevance: float = 0.0
    verification_tick: int = 0
    recurrence_count: int = 0
    verification_attempts: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "created_tick": self.created_tick,
            "last_updated_tick": self.last_updated_tick,
            "source_module": self.source_module,
            "prediction_type": self.prediction_type,
            "target_channels": list(self.target_channels),
            "expected_state": {str(key): float(value) for key, value in self.expected_state.items()},
            "confidence": round(self.confidence, 6),
            "expected_horizon": int(self.expected_horizon),
            "status": self.status,
            "supporting_evidence": list(self.supporting_evidence),
            "linked_commitments": list(self.linked_commitments),
            "linked_identity_anchors": list(self.linked_identity_anchors),
            "linked_unknown_ids": list(self.linked_unknown_ids),
            "linked_hypothesis_ids": list(self.linked_hypothesis_ids),
            "linked_goal": self.linked_goal,
            "maintenance_context": self.maintenance_context,
            "decision_relevance": round(self.decision_relevance, 6),
            "verification_tick": int(self.verification_tick),
            "recurrence_count": int(self.recurrence_count),
            "verification_attempts": int(self.verification_attempts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PredictionHypothesis":
        if not payload:
            return cls(
                prediction_id="",
                created_tick=0,
                last_updated_tick=0,
                source_module="",
                prediction_type="",
                target_channels=(),
                expected_state={},
                confidence=0.0,
                expected_horizon=1,
            )
        return cls(
            prediction_id=str(payload.get("prediction_id", "")),
            created_tick=int(payload.get("created_tick", 0)),
            last_updated_tick=int(payload.get("last_updated_tick", payload.get("created_tick", 0))),
            source_module=str(payload.get("source_module", "")),
            prediction_type=str(payload.get("prediction_type", "")),
            target_channels=tuple(str(item) for item in payload.get("target_channels", [])),
            expected_state={
                str(key): float(value)
                for key, value in dict(payload.get("expected_state", {})).items()
                if isinstance(value, (int, float))
            },
            confidence=float(payload.get("confidence", 0.0)),
            expected_horizon=max(1, int(payload.get("expected_horizon", 1))),
            status=str(payload.get("status", VerificationStatus.ACTIVE.value)),
            supporting_evidence=tuple(str(item) for item in payload.get("supporting_evidence", [])),
            linked_commitments=tuple(str(item) for item in payload.get("linked_commitments", [])),
            linked_identity_anchors=tuple(
                str(item) for item in payload.get("linked_identity_anchors", [])
            ),
            linked_unknown_ids=tuple(str(item) for item in payload.get("linked_unknown_ids", [])),
            linked_hypothesis_ids=tuple(
                str(item) for item in payload.get("linked_hypothesis_ids", [])
            ),
            linked_goal=str(payload.get("linked_goal", "")),
            maintenance_context=str(payload.get("maintenance_context", "")),
            decision_relevance=float(payload.get("decision_relevance", 0.0)),
            verification_tick=int(payload.get("verification_tick", 0)),
            recurrence_count=int(payload.get("recurrence_count", 0)),
            verification_attempts=int(payload.get("verification_attempts", 0)),
        )


@dataclass(frozen=True)
class LedgerDiscrepancy:
    discrepancy_id: str
    label: str
    source: str
    discrepancy_type: str
    created_tick: int
    last_seen_tick: int
    severity: float
    status: str = VerificationStatus.ACTIVE.value
    recurrence_count: int = 1
    target_channels: tuple[str, ...] = ()
    supporting_evidence: tuple[str, ...] = ()
    acute: bool = True
    chronic: bool = False
    identity_relevant: bool = False
    subject_critical: bool = False
    repair_attempts: int = 0
    archived_reason: str = ""
    linked_predictions: tuple[str, ...] = ()
    linked_commitments: tuple[str, ...] = ()
    linked_goal: str = ""

    def age_at(self, tick: int | None = None) -> int:
        if tick is None:
            tick = self.last_seen_tick
        return max(0, int(tick) - self.created_tick)

    @property
    def age(self) -> int:
        return self.age_at()

    @property
    def priority(self) -> str:
        score = self.severity + min(0.35, self.recurrence_count * 0.06)
        if self.subject_critical or self.identity_relevant:
            score += 0.15
        if self.chronic:
            score += 0.10
        if score >= 0.95:
            return LedgerPriority.CRITICAL.value
        if score >= 0.65:
            return LedgerPriority.HIGH.value
        if score >= 0.35:
            return LedgerPriority.MEDIUM.value
        return LedgerPriority.LOW.value

    def to_dict(self, *, reference_tick: int | None = None) -> dict[str, object]:
        return {
            "discrepancy_id": self.discrepancy_id,
            "label": self.label,
            "source": self.source,
            "discrepancy_type": self.discrepancy_type,
            "created_tick": self.created_tick,
            "last_seen_tick": self.last_seen_tick,
            "severity": round(self.severity, 6),
            "status": self.status,
            "recurrence_count": int(self.recurrence_count),
            "target_channels": list(self.target_channels),
            "supporting_evidence": list(self.supporting_evidence),
            "acute": bool(self.acute),
            "chronic": bool(self.chronic),
            "identity_relevant": bool(self.identity_relevant),
            "subject_critical": bool(self.subject_critical),
            "repair_attempts": int(self.repair_attempts),
            "archived_reason": self.archived_reason,
            "linked_predictions": list(self.linked_predictions),
            "linked_commitments": list(self.linked_commitments),
            "linked_goal": self.linked_goal,
            "age": int(self.age_at(reference_tick)),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "LedgerDiscrepancy":
        if not payload:
            return cls(
                discrepancy_id="",
                label="",
                source=DiscrepancySource.PREDICTION_ERROR.value,
                discrepancy_type="",
                created_tick=0,
                last_seen_tick=0,
                severity=0.0,
            )
        return cls(
            discrepancy_id=str(payload.get("discrepancy_id", "")),
            label=str(payload.get("label", "")),
            source=str(payload.get("source", DiscrepancySource.PREDICTION_ERROR.value)),
            discrepancy_type=str(payload.get("discrepancy_type", "")),
            created_tick=int(payload.get("created_tick", 0)),
            last_seen_tick=int(payload.get("last_seen_tick", payload.get("created_tick", 0))),
            severity=float(payload.get("severity", 0.0)),
            status=str(payload.get("status", VerificationStatus.ACTIVE.value)),
            recurrence_count=max(1, int(payload.get("recurrence_count", 1))),
            target_channels=tuple(str(item) for item in payload.get("target_channels", [])),
            supporting_evidence=tuple(str(item) for item in payload.get("supporting_evidence", [])),
            acute=bool(payload.get("acute", True)),
            chronic=bool(payload.get("chronic", False)),
            identity_relevant=bool(payload.get("identity_relevant", False)),
            subject_critical=bool(payload.get("subject_critical", False)),
            repair_attempts=int(payload.get("repair_attempts", 0)),
            archived_reason=str(payload.get("archived_reason", "")),
            linked_predictions=tuple(str(item) for item in payload.get("linked_predictions", [])),
            linked_commitments=tuple(str(item) for item in payload.get("linked_commitments", [])),
            linked_goal=str(payload.get("linked_goal", "")),
        )


@dataclass(frozen=True)
class PredictionLedgerUpdate:
    verified_predictions: tuple[str, ...] = ()
    falsified_predictions: tuple[str, ...] = ()
    created_predictions: tuple[str, ...] = ()
    escalated_discrepancies: tuple[str, ...] = ()
    discharged_discrepancies: tuple[str, ...] = ()
    active_prediction_count: int = 0
    active_discrepancy_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "verified_predictions": list(self.verified_predictions),
            "falsified_predictions": list(self.falsified_predictions),
            "created_predictions": list(self.created_predictions),
            "escalated_discrepancies": list(self.escalated_discrepancies),
            "discharged_discrepancies": list(self.discharged_discrepancies),
            "active_prediction_count": self.active_prediction_count,
            "active_discrepancy_count": self.active_discrepancy_count,
        }


@dataclass
class PredictionLedger:
    predictions: list[PredictionHypothesis] = field(default_factory=list)
    discrepancies: list[LedgerDiscrepancy] = field(default_factory=list)
    archived_predictions: list[PredictionHypothesis] = field(default_factory=list)
    archived_discrepancies: list[LedgerDiscrepancy] = field(default_factory=list)
    last_tick: int = 0
    max_active_predictions: int = 24
    max_active_discrepancies: int = 24

    def to_dict(self) -> dict[str, object]:
        return {
            "predictions": [item.to_dict() for item in self.predictions],
            "discrepancies": [item.to_dict() for item in self.discrepancies],
            "archived_predictions": [item.to_dict() for item in self.archived_predictions],
            "archived_discrepancies": [item.to_dict() for item in self.archived_discrepancies],
            "last_tick": int(self.last_tick),
            "max_active_predictions": int(self.max_active_predictions),
            "max_active_discrepancies": int(self.max_active_discrepancies),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PredictionLedger":
        if not payload:
            return cls()
        return cls(
            predictions=[
                PredictionHypothesis.from_dict(item)
                for item in payload.get("predictions", [])
                if isinstance(item, Mapping)
            ],
            discrepancies=[
                LedgerDiscrepancy.from_dict(item)
                for item in payload.get("discrepancies", [])
                if isinstance(item, Mapping)
            ],
            archived_predictions=[
                PredictionHypothesis.from_dict(item)
                for item in payload.get("archived_predictions", [])
                if isinstance(item, Mapping)
            ],
            archived_discrepancies=[
                LedgerDiscrepancy.from_dict(item)
                for item in payload.get("archived_discrepancies", [])
                if isinstance(item, Mapping)
            ],
            last_tick=int(payload.get("last_tick", 0)),
            max_active_predictions=max(1, int(payload.get("max_active_predictions", 24))),
            max_active_discrepancies=max(1, int(payload.get("max_active_discrepancies", 24))),
        )

    def active_predictions(self) -> list[PredictionHypothesis]:
        return [
            item
            for item in self.predictions
            if item.status not in {VerificationStatus.DISCHARGED.value, VerificationStatus.ARCHIVED.value}
        ]

    def active_discrepancies(self) -> list[LedgerDiscrepancy]:
        return [
            item
            for item in self.discrepancies
            if item.status not in {VerificationStatus.DISCHARGED.value, VerificationStatus.ARCHIVED.value}
        ]

    def top_discrepancies(self, limit: int = 3) -> list[LedgerDiscrepancy]:
        items = list(self.active_discrepancies())
        items.sort(
            key=lambda item: (
                {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(item.priority, 0),
                item.subject_critical,
                item.identity_relevant,
                item.chronic,
                item.severity,
                item.recurrence_count,
                -item.created_tick,
            ),
            reverse=True,
        )
        return items[:limit]

    def prediction_action_bias(self, action: str) -> float:
        bias = 0.0
        for item in self.top_discrepancies(limit=4):
            if (
                item.severity < 0.55
                and not item.chronic
                and not item.subject_critical
                and not item.identity_relevant
            ):
                continue
            if item.discrepancy_type in {"danger_mismatch", "continuity_instability"}:
                if action in {"hide", "scan", "rest", "exploit_shelter"}:
                    bias += 0.08 * item.severity
                if action in {"forage", "seek_contact"}:
                    bias -= 0.06 * item.severity
            if item.discrepancy_type == "maintenance_overload":
                if action in {"rest", "exploit_shelter", "thermoregulate"}:
                    bias += 0.08 * item.severity
                if action == "forage":
                    bias -= 0.05 * item.severity
            if item.discrepancy_type == "social_rupture":
                if action == "scan":
                    bias += 0.06 * item.severity
                if action == "seek_contact":
                    bias += 0.03 * item.severity
            if item.identity_relevant and action in {"rest", "scan", "hide"}:
                bias += 0.05 * item.severity
        for prediction in self.active_predictions()[:4]:
            if prediction.prediction_type == "social_repair" and action == "seek_contact":
                bias += 0.04 * prediction.confidence
            if prediction.prediction_type == "maintenance_recovery" and action == "rest":
                bias += 0.05 * prediction.confidence
            if prediction.source_module == "narrative_uncertainty":
                if action in {"scan", "hide"}:
                    bias += 0.04 * max(prediction.confidence, prediction.decision_relevance)
                elif action == "forage":
                    bias -= 0.03 * max(prediction.confidence, prediction.decision_relevance)
        return round(max(-0.45, min(0.45, bias)), 6)

    def workspace_focus(self) -> dict[str, float]:
        focus: dict[str, float] = {}
        for item in self.top_discrepancies(limit=4):
            weight = item.severity + (0.12 if item.chronic else 0.0)
            for channel in item.target_channels:
                focus[channel] = max(focus.get(channel, 0.0), round(weight, 6))
        for prediction in self.active_predictions()[:4]:
            for channel in prediction.target_channels:
                focus[channel] = max(
                    focus.get(channel, 0.0),
                    round(prediction.confidence * 0.45, 6),
                )
        return focus

    def memory_threshold_delta(self) -> float:
        delta = 0.0
        for item in self.top_discrepancies(limit=4):
            if item.severity < 0.45 and not item.chronic and not item.identity_relevant and not item.subject_critical:
                continue
            delta -= min(0.06, item.severity * 0.04)
            if item.identity_relevant or item.subject_critical:
                delta -= 0.015
            if item.chronic:
                delta -= 0.01
        return round(max(-0.14, min(0.0, delta)), 6)

    def maintenance_signal(self) -> dict[str, object]:
        tasks: list[str] = []
        recommended_action = ""
        priority_gain = 0.0
        suppressed_actions: list[str] = []
        for item in self.top_discrepancies(limit=4):
            priority_gain += item.severity * (1.20 if item.subject_critical else 0.80)
            if item.discrepancy_type == "maintenance_overload":
                tasks.append("ledger_maintenance_repair")
                recommended_action = recommended_action or "rest"
                suppressed_actions.extend(["forage"])
            elif item.discrepancy_type == "danger_mismatch":
                tasks.append("ledger_verify_threat")
                recommended_action = recommended_action or "hide"
                suppressed_actions.extend(["forage", "seek_contact"])
            elif item.discrepancy_type == "continuity_instability":
                tasks.append("ledger_continuity_guard")
                recommended_action = recommended_action or "scan"
            elif item.discrepancy_type == "social_rupture":
                tasks.append("ledger_social_repair")
                recommended_action = recommended_action or "seek_contact"
        return {
            "priority_gain": round(min(0.40, priority_gain * 0.20), 6),
            "active_tasks": list(dict.fromkeys(tasks)),
            "recommended_action": recommended_action,
            "suppressed_actions": list(dict.fromkeys(suppressed_actions)),
        }

    def explanation_payload(self) -> dict[str, object]:
        reference_tick = self.last_tick
        unresolved_discrepancies = [
            item.to_dict(reference_tick=reference_tick) for item in self.active_discrepancies()
        ]
        top_discrepancies = [
            item.to_dict(reference_tick=reference_tick) for item in self.top_discrepancies(limit=4)
        ]
        active_predictions = [item.to_dict() for item in self.active_predictions()[:4]]
        summary = "No unresolved prediction burden is currently dominant."
        if top_discrepancies:
            top = top_discrepancies[0]
            summary = (
                f"My most urgent unresolved discrepancy is {top['label']} "
                f"({top['priority']}, recurrence={top['recurrence_count']})."
            )
        elif active_predictions:
            top_pred = active_predictions[0]
            if top_pred.get("source_module") == "narrative_uncertainty":
                summary = (
                    f"I am carrying forward narrative uncertainty about {top_pred['prediction_type']}, "
                    "because competing explanations still imply different outcomes."
                )
            else:
                summary = (
                    f"I still expect {top_pred['prediction_type']} on "
                    f"{', '.join(top_pred['target_channels']) or 'current channels'}, "
                    "but it remains unverified."
                )
        return {
            "summary": summary,
            "active_predictions": active_predictions,
            "unresolved_discrepancies": unresolved_discrepancies,
            "top_discrepancies": top_discrepancies,
            "counts": {
                "active_predictions": len(self.active_predictions()),
                "active_discrepancies": len(self.active_discrepancies()),
                "archived_predictions": len(self.archived_predictions),
                "archived_discrepancies": len(self.archived_discrepancies),
            },
        }

    def verify_predictions(
        self,
        *,
        tick: int,
        observation: Mapping[str, float],
    ) -> PredictionLedgerUpdate:
        verified: list[str] = []
        falsified: list[str] = []
        escalated: list[str] = []
        updated_predictions: list[PredictionHypothesis] = []
        for prediction in self.predictions:
            if prediction.status in {
                VerificationStatus.DISCHARGED.value,
                VerificationStatus.ARCHIVED.value,
            }:
                updated_predictions.append(prediction)
                continue
            should_verify = tick > prediction.created_tick and (
                tick - prediction.created_tick >= prediction.expected_horizon
                or any(channel in observation for channel in prediction.target_channels)
            )
            if not should_verify:
                updated_predictions.append(prediction)
                continue
            verification_error = 0.0
            compared = 0
            for channel, expected_value in prediction.expected_state.items():
                if channel not in observation:
                    continue
                verification_error += abs(float(observation[channel]) - expected_value)
                compared += 1
            if compared == 0:
                updated_predictions.append(prediction)
                continue
            verification_error /= compared
            if verification_error <= max(0.12, 0.35 - prediction.confidence * 0.20):
                self.archived_predictions.append(
                    PredictionHypothesis(
                        **{
                            **prediction.__dict__,
                            "last_updated_tick": tick,
                            "verification_tick": tick,
                            "verification_attempts": prediction.verification_attempts + 1,
                            "status": VerificationStatus.DISCHARGED.value,
                        }
                    )
                )
                verified.append(prediction.prediction_id)
                continue
            self.archived_predictions.append(
                PredictionHypothesis(
                    **{
                        **prediction.__dict__,
                        "last_updated_tick": tick,
                        "verification_tick": tick,
                        "verification_attempts": prediction.verification_attempts + 1,
                        "recurrence_count": prediction.recurrence_count + 1,
                        "status": VerificationStatus.FALSIFIED.value,
                    }
                )
            )
            falsified.append(prediction.prediction_id)
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id=f"disc:{prediction.prediction_type}:{'-'.join(prediction.target_channels) or 'state'}",
                label=f"{prediction.prediction_type.replace('_', ' ')} mismatch",
                source=DiscrepancySource.PREDICTION_ERROR.value,
                discrepancy_type=f"{prediction.prediction_type}_mismatch",
                severity=min(1.0, verification_error * 1.8),
                target_channels=prediction.target_channels,
                evidence=(
                    f"prediction={prediction.prediction_id}",
                    f"verification_error={verification_error:.3f}",
                ),
                identity_relevant=bool(prediction.linked_commitments or prediction.linked_identity_anchors),
                subject_critical="continuity" in prediction.prediction_type,
                linked_predictions=(prediction.prediction_id,),
                linked_commitments=prediction.linked_commitments,
                linked_goal=prediction.linked_goal,
            )
            escalated.append(discrepancy.discrepancy_id)
        self.predictions = updated_predictions
        self._trim()
        self.last_tick = tick
        return PredictionLedgerUpdate(
            verified_predictions=tuple(verified),
            falsified_predictions=tuple(falsified),
            escalated_discrepancies=tuple(escalated),
            active_prediction_count=len(self.active_predictions()),
            active_discrepancy_count=len(self.active_discrepancies()),
        )

    def seed_predictions(
        self,
        *,
        tick: int,
        diagnostics,
        prediction: Mapping[str, float],
        subject_state,
        narrative_uncertainty=None,
    ) -> PredictionLedgerUpdate:
        created: list[str] = []
        priorities = sorted(
            prediction.items(),
            key=lambda item: (-abs(float(item[1])), item[0]),
        )[:2]
        for channel, value in priorities:
            created_id = f"pred:env:{channel}"
            self._upsert_prediction(
                PredictionHypothesis(
                    prediction_id=created_id,
                    created_tick=tick,
                    last_updated_tick=tick,
                    source_module="decision_cycle",
                    prediction_type="environment_state",
                    target_channels=(str(channel),),
                    expected_state={str(channel): float(value)},
                    confidence=0.45,
                    expected_horizon=1,
                    supporting_evidence=tuple(diagnostics.workspace_broadcast_channels[:2]),
                    linked_commitments=tuple(diagnostics.commitment_focus[:2]),
                    linked_identity_anchors=tuple(subject_state.continuity_anchors[:2]),
                    linked_goal=diagnostics.active_goal,
                )
            )
            created.append(created_id)
        chosen = diagnostics.chosen
        consequence_fields = {
            key.replace("_delta", ""): float(value)
            for key, value in chosen.predicted_effects.items()
            if key.endswith("_delta") and isinstance(value, (int, float))
        }
        if consequence_fields:
            baseline = dict(prediction)
            expected_state = {
                key: _clamp(float(baseline.get(key, 0.5)) + value, 0.0, 1.0)
                for key, value in consequence_fields.items()
            }
            prediction_id = f"pred:action:{chosen.choice}"
            self._upsert_prediction(
                PredictionHypothesis(
                    prediction_id=prediction_id,
                    created_tick=tick,
                    last_updated_tick=tick,
                    source_module="decision_policy",
                    prediction_type="action_consequence",
                    target_channels=tuple(sorted(expected_state)),
                    expected_state=expected_state,
                    confidence=_clamp(chosen.preferred_probability, 0.25, 0.95),
                    expected_horizon=1,
                    supporting_evidence=(f"action={chosen.choice}", f"goal={diagnostics.active_goal}"),
                    linked_commitments=tuple(diagnostics.commitment_focus[:2]),
                    linked_identity_anchors=tuple(subject_state.continuity_anchors[:2]),
                    linked_goal=diagnostics.active_goal,
                )
            )
            created.append(prediction_id)
        if diagnostics.social_focus:
            social_prediction_id = f"pred:social:{diagnostics.social_focus[0]}"
            self._upsert_prediction(
                PredictionHypothesis(
                    prediction_id=social_prediction_id,
                    created_tick=tick,
                    last_updated_tick=tick,
                    source_module="social_model",
                    prediction_type="social_repair",
                    target_channels=("social",),
                    expected_state={"social": max(0.45, float(prediction.get("social", 0.0)))},
                    confidence=0.40,
                    expected_horizon=2,
                    supporting_evidence=tuple(diagnostics.social_alerts[:2] or diagnostics.social_focus[:2]),
                    linked_goal=diagnostics.active_goal,
                )
            )
            created.append(social_prediction_id)
        uncertainty_unknowns = getattr(narrative_uncertainty, "unknowns", ()) if narrative_uncertainty is not None else ()
        uncertainty_hypotheses = (
            getattr(narrative_uncertainty, "competing_hypotheses", ())
            if narrative_uncertainty is not None
            else ()
        )
        for unknown in uncertainty_unknowns[:2]:
            if not getattr(unknown, "action_relevant", False):
                continue
            hypothesis = next(
                (
                    item
                    for item in uncertainty_hypotheses
                    if item.parent_unknown_id == unknown.unknown_id
                ),
                None,
            )
            expected_state = (
                dict(hypothesis.expected_state_shift)
                if hypothesis is not None
                else {
                    "danger": max(0.35, float(prediction.get("danger", 0.0)))
                    if unknown.unknown_type == "threat_persistence"
                    else max(0.35, float(prediction.get("social", 0.0)))
                }
            )
            if not expected_state:
                continue
            prediction_id = f"pred:narrative:{unknown.unknown_id}"
            self._upsert_prediction(
                PredictionHypothesis(
                    prediction_id=prediction_id,
                    created_tick=tick,
                    last_updated_tick=tick,
                    source_module="narrative_uncertainty",
                    prediction_type=f"narrative_{unknown.unknown_type}",
                    target_channels=tuple(sorted(expected_state)),
                    expected_state=expected_state,
                    confidence=_clamp(
                        0.34
                        + float(unknown.decision_relevance.verification_urgency) * 0.28
                        + (float(hypothesis.prior_plausibility) * 0.18 if hypothesis is not None else 0.0)
                    ),
                    expected_horizon=2,
                    supporting_evidence=tuple(
                        [unknown.unresolved_reason]
                        + list(getattr(hypothesis, "implied_consequences", ())[:2])
                    )[:3],
                    linked_identity_anchors=tuple(subject_state.continuity_anchors[:2]),
                    linked_unknown_ids=(unknown.unknown_id,),
                    linked_hypothesis_ids=(
                        (hypothesis.hypothesis_id,) if hypothesis is not None else ()
                    ),
                    linked_goal=diagnostics.active_goal,
                    decision_relevance=float(unknown.decision_relevance.total_score),
                )
            )
            created.append(prediction_id)
        self._trim()
        self.last_tick = tick
        return PredictionLedgerUpdate(
            created_predictions=tuple(created),
            active_prediction_count=len(self.active_predictions()),
            active_discrepancy_count=len(self.active_discrepancies()),
        )

    def record_runtime_discrepancies(
        self,
        *,
        tick: int,
        diagnostics,
        errors: Mapping[str, float],
        maintenance_agenda,
        continuity_score: float,
        subject_state,
        memory_surprise: float = 0.0,
    ) -> PredictionLedgerUpdate:
        escalated: list[str] = []
        discharged: list[str] = []
        danger_mismatch = abs(float(errors.get("danger", 0.0)))
        if danger_mismatch >= 0.18:
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id="disc:danger_mismatch",
                label="repeated danger mismatch",
                source=DiscrepancySource.PREDICTION_ERROR.value,
                discrepancy_type="danger_mismatch",
                severity=min(1.0, danger_mismatch * 1.6),
                target_channels=("danger",),
                evidence=(f"danger_error={danger_mismatch:.3f}",),
                subject_critical=True,
            )
            escalated.append(discrepancy.discrepancy_id)
        else:
            discharged.extend(self._discharge_matching("disc:danger_mismatch", tick=tick, reason="danger_normalized"))
        if diagnostics.identity_tension >= 0.08 or diagnostics.self_inconsistency_error >= 0.08:
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id="disc:identity_tension",
                label="unresolved self inconsistency",
                source=DiscrepancySource.IDENTITY.value,
                discrepancy_type="identity_tension",
                severity=min(1.0, max(diagnostics.identity_tension, diagnostics.self_inconsistency_error)),
                target_channels=("conflict",),
                evidence=tuple(diagnostics.violated_commitments[:3] or [diagnostics.conflict_type]),
                identity_relevant=True,
                subject_critical=subject_state.status_flags.get("continuity_fragile", False),
                linked_commitments=tuple(diagnostics.commitment_focus[:3]),
                linked_goal=diagnostics.active_goal,
            )
            escalated.append(discrepancy.discrepancy_id)
        elif diagnostics.repair_triggered or not diagnostics.violated_commitments:
            discharged.extend(self._discharge_matching("disc:identity_tension", tick=tick, reason="repair_or_alignment"))
        if maintenance_agenda.chronic_debt_pressure >= 0.08 or maintenance_agenda.policy_shift_strength >= 0.25:
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id="disc:maintenance_overload",
                label="unresolved maintenance discrepancy",
                source=DiscrepancySource.MAINTENANCE.value,
                discrepancy_type="maintenance_overload",
                severity=min(
                    1.0,
                    max(maintenance_agenda.chronic_debt_pressure, maintenance_agenda.policy_shift_strength),
                ),
                target_channels=("maintenance", "stress"),
                evidence=tuple(maintenance_agenda.active_tasks[:3]),
                subject_critical=maintenance_agenda.protected_mode,
                linked_goal=diagnostics.active_goal,
            )
            escalated.append(discrepancy.discrepancy_id)
        else:
            discharged.extend(self._discharge_matching("disc:maintenance_overload", tick=tick, reason="maintenance_normalized"))
        if diagnostics.social_alerts:
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id="disc:social_rupture",
                label="recurring social rupture signal",
                source=DiscrepancySource.SOCIAL.value,
                discrepancy_type="social_rupture",
                severity=min(1.0, 0.40 + 0.10 * len(diagnostics.social_alerts)),
                target_channels=("social",),
                evidence=tuple(diagnostics.social_alerts[:3]),
                linked_goal=diagnostics.active_goal,
            )
            escalated.append(discrepancy.discrepancy_id)
        else:
            discharged.extend(self._discharge_matching("disc:social_rupture", tick=tick, reason="social_normalized"))
        if continuity_score < 0.78:
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id="disc:continuity_instability",
                label="continuity-relevant tension",
                source=DiscrepancySource.CONTINUITY.value,
                discrepancy_type="continuity_instability",
                severity=min(1.0, 1.0 - continuity_score),
                target_channels=("continuity",),
                evidence=tuple(subject_state.continuity_anchors[:3] or ["continuity_score_drop"]),
                identity_relevant=True,
                subject_critical=True,
            )
            escalated.append(discrepancy.discrepancy_id)
        else:
            discharged.extend(self._discharge_matching("disc:continuity_instability", tick=tick, reason="continuity_restored"))
        if memory_surprise >= 0.70:
            discrepancy = self._record_discrepancy(
                tick=tick,
                discrepancy_id="disc:surprise_burden",
                label="repeated surprise burden",
                source=DiscrepancySource.MEMORY.value,
                discrepancy_type="surprise_burden",
                severity=min(1.0, memory_surprise / 1.5),
                target_channels=("novelty",),
                evidence=(f"memory_surprise={memory_surprise:.3f}",),
                identity_relevant=subject_state.identity_tension_level > 0.20,
            )
            escalated.append(discrepancy.discrepancy_id)
        self._trim()
        self.last_tick = tick
        return PredictionLedgerUpdate(
            escalated_discrepancies=tuple(escalated),
            discharged_discrepancies=tuple(discharged),
            active_prediction_count=len(self.active_predictions()),
            active_discrepancy_count=len(self.active_discrepancies()),
        )

    def sleep_review(self, *, tick: int) -> dict[str, object]:
        archived_predictions: list[str] = []
        archived_discrepancies: list[str] = []
        escalated: list[str] = []
        retained: list[str] = []
        active_predictions: list[PredictionHypothesis] = []
        for prediction in self.predictions:
            age = tick - prediction.created_tick
            if prediction.status == VerificationStatus.DISCHARGED.value or age > 6:
                self.archived_predictions.append(
                    PredictionHypothesis(
                        **{**prediction.__dict__, "status": VerificationStatus.ARCHIVED.value}
                    )
                )
                archived_predictions.append(prediction.prediction_id)
                continue
            active_predictions.append(prediction)
        self.predictions = active_predictions
        active_discrepancies: list[LedgerDiscrepancy] = []
        for discrepancy in self.discrepancies:
            age = tick - discrepancy.created_tick
            protect = discrepancy.identity_relevant or discrepancy.subject_critical
            if discrepancy.status == VerificationStatus.DISCHARGED.value and not protect:
                self.archived_discrepancies.append(
                    LedgerDiscrepancy(
                        **{
                            **discrepancy.__dict__,
                            "status": VerificationStatus.ARCHIVED.value,
                            "last_seen_tick": tick,
                        }
                    )
                )
                archived_discrepancies.append(discrepancy.discrepancy_id)
                continue
            chronic = discrepancy.chronic or discrepancy.recurrence_count >= 3 or age >= 3
            if chronic and discrepancy.status != VerificationStatus.ESCALATED.value:
                discrepancy = LedgerDiscrepancy(
                    **{
                        **discrepancy.__dict__,
                        "chronic": True,
                        "acute": False,
                        "status": VerificationStatus.ESCALATED.value,
                        "severity": min(1.0, discrepancy.severity + 0.10),
                        "last_seen_tick": tick,
                    }
                )
                escalated.append(discrepancy.discrepancy_id)
            if protect:
                retained.append(discrepancy.discrepancy_id)
            active_discrepancies.append(discrepancy)
        self.discrepancies = active_discrepancies
        self._trim()
        self.last_tick = tick
        return {
            "archived_predictions": archived_predictions,
            "archived_discrepancies": archived_discrepancies,
            "escalated_discrepancies": escalated,
            "retained_identity_critical": retained,
            "active_predictions": len(self.active_predictions()),
            "active_discrepancies": len(self.active_discrepancies()),
        }

    def _upsert_prediction(self, prediction: PredictionHypothesis) -> None:
        replaced = False
        updated: list[PredictionHypothesis] = []
        for current in self.predictions:
            if current.prediction_id == prediction.prediction_id:
                updated.append(prediction)
                replaced = True
            else:
                updated.append(current)
        if not replaced:
            updated.append(prediction)
        self.predictions = updated

    def _record_discrepancy(
        self,
        *,
        tick: int,
        discrepancy_id: str,
        label: str,
        source: str,
        discrepancy_type: str,
        severity: float,
        target_channels: tuple[str, ...],
        evidence: tuple[str, ...],
        identity_relevant: bool = False,
        subject_critical: bool = False,
        linked_predictions: tuple[str, ...] = (),
        linked_commitments: tuple[str, ...] = (),
        linked_goal: str = "",
    ) -> LedgerDiscrepancy:
        for index, current in enumerate(self.discrepancies):
            if current.discrepancy_id != discrepancy_id:
                continue
            updated = LedgerDiscrepancy(
                discrepancy_id=current.discrepancy_id,
                label=label,
                source=source,
                discrepancy_type=discrepancy_type,
                created_tick=current.created_tick,
                last_seen_tick=tick,
                severity=min(1.0, max(current.severity, severity) + 0.05 * min(4, current.recurrence_count)),
                status=VerificationStatus.ESCALATED.value if current.recurrence_count >= 1 else VerificationStatus.ACTIVE.value,
                recurrence_count=current.recurrence_count + 1,
                target_channels=tuple(dict.fromkeys([*current.target_channels, *target_channels])),
                supporting_evidence=tuple(dict.fromkeys([*current.supporting_evidence, *evidence]))[:6],
                acute=current.recurrence_count < 2,
                chronic=current.chronic or current.recurrence_count + 1 >= 3 or tick - current.created_tick >= 3,
                identity_relevant=current.identity_relevant or identity_relevant,
                subject_critical=current.subject_critical or subject_critical,
                repair_attempts=current.repair_attempts,
                archived_reason="",
                linked_predictions=tuple(dict.fromkeys([*current.linked_predictions, *linked_predictions])),
                linked_commitments=tuple(dict.fromkeys([*current.linked_commitments, *linked_commitments])),
                linked_goal=linked_goal or current.linked_goal,
            )
            self.discrepancies[index] = updated
            return updated
        created = LedgerDiscrepancy(
            discrepancy_id=discrepancy_id,
            label=label,
            source=source,
            discrepancy_type=discrepancy_type,
            created_tick=tick,
            last_seen_tick=tick,
            severity=_clamp(severity),
            target_channels=target_channels,
            supporting_evidence=evidence,
            identity_relevant=identity_relevant,
            subject_critical=subject_critical,
            linked_predictions=linked_predictions,
            linked_commitments=linked_commitments,
            linked_goal=linked_goal,
        )
        self.discrepancies.append(created)
        return created

    def _discharge_matching(self, discrepancy_id: str, *, tick: int, reason: str) -> list[str]:
        discharged: list[str] = []
        updated: list[LedgerDiscrepancy] = []
        for current in self.discrepancies:
            if current.discrepancy_id != discrepancy_id:
                updated.append(current)
                continue
            if current.identity_relevant or current.subject_critical:
                updated.append(
                    LedgerDiscrepancy(
                        **{
                            **current.__dict__,
                            "status": VerificationStatus.PARTIALLY_RESOLVED.value,
                            "last_seen_tick": tick,
                            "archived_reason": reason,
                            "severity": max(0.0, current.severity - 0.15),
                        }
                    )
                )
                continue
            self.archived_discrepancies.append(
                LedgerDiscrepancy(
                    **{
                        **current.__dict__,
                        "status": VerificationStatus.DISCHARGED.value,
                        "last_seen_tick": tick,
                        "archived_reason": reason,
                        "severity": max(0.0, current.severity - 0.20),
                    }
                )
            )
            discharged.append(current.discrepancy_id)
        self.discrepancies = updated
        return discharged

    def _trim(self) -> None:
        self.predictions = self.predictions[-self.max_active_predictions :]
        self.archived_predictions = self.archived_predictions[-self.max_active_predictions :]
        active = self.active_discrepancies()
        protected = [item for item in active if item.identity_relevant or item.subject_critical]
        protected_ids = {item.discrepancy_id for item in protected}
        unprotected = [item for item in active if item.discrepancy_id not in protected_ids]
        unprotected.sort(
            key=lambda item: (item.severity, item.recurrence_count, item.created_tick, item.discrepancy_id)
        )
        keep = protected + unprotected[-max(0, self.max_active_discrepancies - len(protected)) :]
        keep_ids = {item.discrepancy_id for item in keep}
        archived = [item for item in active if item.discrepancy_id not in keep_ids]
        self.archived_discrepancies.extend(archived)
        self.archived_discrepancies = self.archived_discrepancies[-self.max_active_discrepancies * 2 :]
        self.discrepancies = keep
