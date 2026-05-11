"""Append-only M11 user-prediction ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams

PredictionType = Literal[
    "intent_prediction",
    "preference_prediction",
    "reaction_prediction",
    "claim_reliability_prediction",
    "relationship_state_prediction",
    "needed_memory_prediction",
]
ValidationStatus = Literal["pending", "confirmed", "violated", "uncertain"]
ConfidenceBand = Literal["low", "med", "high"]

VALID_PREDICTION_TYPES = {
    "intent_prediction",
    "preference_prediction",
    "reaction_prediction",
    "claim_reliability_prediction",
    "relationship_state_prediction",
    "needed_memory_prediction",
}
VALID_STATUSES = {"pending", "confirmed", "violated", "uncertain"}
VALID_BANDS = {"low", "med", "high"}


@dataclass(frozen=True)
class PredictionEntry:
    prediction_id: str
    turn_id: int
    prediction_type: PredictionType
    predicted_value_summary: str
    confidence_band: ConfidenceBand
    evidence_refs: tuple[str, ...]
    validation_status: ValidationStatus = "pending"
    observed_outcome_summary: str = ""
    calibration_need_band: ConfidenceBand = "low"
    source_proposal_id: str = ""
    event_kind: str = "prediction"

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "turn_id": self.turn_id,
            "prediction_type": self.prediction_type,
            "predicted_value_summary": self.predicted_value_summary,
            "confidence_band": self.confidence_band,
            "evidence_refs": list(self.evidence_refs),
            "validation_status": self.validation_status,
            "observed_outcome_summary": self.observed_outcome_summary,
            "calibration_need_band": self.calibration_need_band,
            "source_proposal_id": self.source_proposal_id,
            "event_kind": self.event_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionEntry":
        return cls(
            prediction_id=str(payload.get("prediction_id", "")),
            turn_id=int(payload.get("turn_id", 0)),
            prediction_type=_prediction_type(payload.get("prediction_type")),
            predicted_value_summary=str(payload.get("predicted_value_summary", "")),
            confidence_band=_band(payload.get("confidence_band")),
            evidence_refs=tuple(str(x) for x in payload.get("evidence_refs", [])),
            validation_status=_status(payload.get("validation_status")),
            observed_outcome_summary=str(payload.get("observed_outcome_summary", "")),
            calibration_need_band=_band(payload.get("calibration_need_band")),
            source_proposal_id=str(payload.get("source_proposal_id", "")),
            event_kind=str(payload.get("event_kind", "prediction")),
        )


@dataclass(frozen=True)
class PredictionProposal:
    proposal_id: str
    proposed_prediction_type: PredictionType
    predicted_value_summary: str
    confidence_band: ConfidenceBand
    source_hypothesis_ids: tuple[str, ...] = ()
    source_judgment_ids: tuple[str, ...] = ()
    expires_after_turns: int = 1
    accepted: bool = False
    rejection_reason: str = ""
    turn_id: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposed_prediction_type": self.proposed_prediction_type,
            "predicted_value_summary": self.predicted_value_summary,
            "confidence_band": self.confidence_band,
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "source_judgment_ids": list(self.source_judgment_ids),
            "expires_after_turns": self.expires_after_turns,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", payload.get("id", ""))),
            proposed_prediction_type=_prediction_type(payload.get("proposed_prediction_type", payload.get("prediction_type"))),
            predicted_value_summary=str(payload.get("predicted_value_summary", ""))[:120],
            confidence_band=_band(payload.get("confidence_band")),
            source_hypothesis_ids=tuple(str(x) for x in payload.get("source_hypothesis_ids", [])),
            source_judgment_ids=tuple(str(x) for x in payload.get("source_judgment_ids", [])),
            expires_after_turns=max(int(payload.get("expires_after_turns", DEFAULT_HYPERPARAMS.default_prediction_expiry_turns)), 1),
            accepted=bool(payload.get("accepted", False)),
            rejection_reason=str(payload.get("rejection_reason", "")),
            turn_id=int(payload.get("turn_id", 0)),
        )


@dataclass(frozen=True)
class UserPredictionLedger:
    entries: tuple[PredictionEntry, ...] = ()
    proposals: tuple[PredictionProposal, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "proposals": [proposal.to_dict() for proposal in self.proposals],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserPredictionLedger":
        return cls(
            entries=tuple(
                PredictionEntry.from_dict(row)
                for row in payload.get("entries", [])
                if isinstance(row, Mapping)
            ),
            proposals=tuple(
                PredictionProposal.from_dict(row)
                for row in payload.get("proposals", [])
                if isinstance(row, Mapping)
            ),
        )

    def latest_status(self, prediction_id: str) -> ValidationStatus | None:
        for entry in reversed(self.entries):
            if entry.prediction_id == prediction_id:
                return entry.validation_status
        return None

    def predictions_by_status(self, *, status: ValidationStatus, current_turn_id: int, last_n_turns: int) -> tuple[PredictionEntry, ...]:
        floor = current_turn_id - last_n_turns
        latest: dict[str, PredictionEntry] = {}
        for entry in self.entries:
            if entry.turn_id >= floor:
                latest[entry.prediction_id] = entry
        return tuple(entry for entry in latest.values() if entry.validation_status == status)


def apply_prediction_updates(
    ledger: UserPredictionLedger,
    *,
    turn_id: int,
    proposals: Sequence[Mapping[str, object]],
    judgments: Sequence[Mapping[str, object]],
    known_hypothesis_ids: set[str],
    known_judgment_ids: set[str],
    calibration_need_band: ConfidenceBand = "low",
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> UserPredictionLedger:
    next_entries = list(ledger.entries)
    next_proposals = list(ledger.proposals)
    open_ids = {
        entry.prediction_id
        for entry in next_entries
        if ledger.latest_status(entry.prediction_id) == "pending"
    }

    for judgment in judgments:
        prediction_id = str(judgment.get("prediction_id", ""))
        if prediction_id not in open_ids:
            continue
        source = _latest_entry(next_entries, prediction_id)
        status = _status(judgment.get("status"))
        evidence_refs = tuple(str(x) for x in judgment.get("evidence_quote_ids", []))
        next_entries.append(
            PredictionEntry(
                prediction_id=prediction_id,
                turn_id=turn_id,
                prediction_type=source.prediction_type,
                predicted_value_summary=source.predicted_value_summary,
                confidence_band=source.confidence_band,
                evidence_refs=evidence_refs,
                validation_status=status,
                observed_outcome_summary=status,
                calibration_need_band=calibration_need_band,
                source_proposal_id=source.source_proposal_id,
                event_kind="judgment",
            )
        )
        known_judgment_ids.add(prediction_id)

    for entry in tuple(next_entries):
        if entry.validation_status != "pending":
            continue
        proposal = next((p for p in next_proposals if p.proposal_id == entry.source_proposal_id), None)
        if proposal is None:
            continue
        if turn_id - entry.turn_id > proposal.expires_after_turns:
            next_entries.append(
                PredictionEntry(
                    prediction_id=entry.prediction_id,
                    turn_id=turn_id,
                    prediction_type=entry.prediction_type,
                    predicted_value_summary=entry.predicted_value_summary,
                    confidence_band=entry.confidence_band,
                    evidence_refs=entry.evidence_refs,
                    validation_status="uncertain",
                    observed_outcome_summary="expired",
                    calibration_need_band="med",
                    source_proposal_id=entry.source_proposal_id,
                    event_kind="expiration",
                )
            )
            next_proposals.append(
                PredictionProposal(
                    proposal_id=proposal.proposal_id,
                    proposed_prediction_type=proposal.proposed_prediction_type,
                    predicted_value_summary=proposal.predicted_value_summary,
                    confidence_band=proposal.confidence_band,
                    source_hypothesis_ids=proposal.source_hypothesis_ids,
                    source_judgment_ids=proposal.source_judgment_ids,
                    expires_after_turns=proposal.expires_after_turns,
                    accepted=proposal.accepted,
                    rejection_reason="expired",
                    turn_id=turn_id,
                )
            )

    admitted = 0
    for raw in proposals:
        proposal = PredictionProposal.from_dict({**dict(raw), "turn_id": turn_id})
        rejection = ""
        if admitted >= hyperparams.proposal_quota_per_turn:
            rejection = "proposal_quota_exceeded"
        elif any(source_id not in known_hypothesis_ids for source_id in proposal.source_hypothesis_ids):
            rejection = "unknown_source_id"
        elif any(source_id not in known_judgment_ids for source_id in proposal.source_judgment_ids):
            rejection = "unknown_source_id"
        accepted = rejection == ""
        if accepted:
            admitted += 1
        gated = PredictionProposal(
            proposal_id=proposal.proposal_id,
            proposed_prediction_type=proposal.proposed_prediction_type,
            predicted_value_summary=proposal.predicted_value_summary,
            confidence_band=proposal.confidence_band,
            source_hypothesis_ids=proposal.source_hypothesis_ids,
            source_judgment_ids=proposal.source_judgment_ids,
            expires_after_turns=proposal.expires_after_turns,
            accepted=accepted,
            rejection_reason=rejection,
            turn_id=turn_id,
        )
        next_proposals.append(gated)
        if accepted:
            next_entries.append(
                PredictionEntry(
                    prediction_id=f"pred:{proposal.proposal_id}",
                    turn_id=turn_id,
                    prediction_type=proposal.proposed_prediction_type,
                    predicted_value_summary=proposal.predicted_value_summary,
                    confidence_band=proposal.confidence_band,
                    evidence_refs=(*proposal.source_hypothesis_ids, *proposal.source_judgment_ids),
                    validation_status="pending",
                    observed_outcome_summary="",
                    calibration_need_band=calibration_need_band,
                    source_proposal_id=proposal.proposal_id,
                    event_kind="prediction",
                )
            )
    return UserPredictionLedger(entries=tuple(next_entries), proposals=tuple(next_proposals))


def _latest_entry(entries: Sequence[PredictionEntry], prediction_id: str) -> PredictionEntry:
    for entry in reversed(entries):
        if entry.prediction_id == prediction_id:
            return entry
    raise KeyError(prediction_id)


def _prediction_type(value: object) -> PredictionType:
    text = str(value or "intent_prediction")
    return text if text in VALID_PREDICTION_TYPES else "intent_prediction"  # type: ignore[return-value]


def _status(value: object) -> ValidationStatus:
    text = str(value or "pending")
    return text if text in VALID_STATUSES else "pending"  # type: ignore[return-value]


def _band(value: object) -> ConfidenceBand:
    text = str(value or "low")
    return text if text in VALID_BANDS else "low"  # type: ignore[return-value]
