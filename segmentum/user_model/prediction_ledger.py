"""Append-only M11/M17 user-prediction ledger."""

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
VALID_EVIDENCE_BASIS = {
    "no_evidence",
    "current_user_question",
    "current_user_request",
    "current_user_statement",
    "recent_topic_continuity",
    "direct_user_statement",
    "repeated_evidence",
    "prior_confirmation",
    "system_invariant",
    "product_invariant",
}
LEGACY_BAND_DEFAULTS = {
    "low": 0.55,
    "med": 0.65,
    "high": 0.75,
}


@dataclass(frozen=True)
class NormalizedConfidence:
    raw_confidence: float
    committed_confidence: float
    confidence_band: ConfidenceBand
    cap_reason: str


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
    raw_confidence: float = 0.55
    committed_confidence: float = 0.55
    confidence_cap_reason: str = "legacy_confidence_band_default"
    evidence_basis: tuple[str, ...] = ()
    expires_after_turns: int = DEFAULT_HYPERPARAMS.default_prediction_expiry_turns
    created_at_turn: int = 0
    created_before_response: bool = False
    response_turn_id: int = 0
    settlement_outcome: str = ""
    settlement_confidence: float = 0.0
    settlement_id: str = ""
    m17_prediction_error: float | None = None
    m17_brier_score: float | None = None
    source_episode_id: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "prediction_id": self.prediction_id,
            "turn_id": self.turn_id,
            "prediction_type": self.prediction_type,
            "predicted_value_summary": self.predicted_value_summary,
            "confidence_band": self.confidence_band,
            "raw_confidence": round(self.raw_confidence, 6),
            "committed_confidence": round(self.committed_confidence, 6),
            "confidence_cap_reason": self.confidence_cap_reason,
            "evidence_basis": list(self.evidence_basis),
            "evidence_refs": list(self.evidence_refs),
            "expires_after_turns": self.expires_after_turns,
            "created_at_turn": self.created_at_turn,
            "created_before_response": self.created_before_response,
            "response_turn_id": self.response_turn_id,
            "validation_status": self.validation_status,
            "observed_outcome_summary": self.observed_outcome_summary,
            "settlement_outcome": self.settlement_outcome,
            "settlement_confidence": round(self.settlement_confidence, 6),
            "settlement_id": self.settlement_id,
            "m17_prediction_error": self.m17_prediction_error,
            "m17_brier_score": self.m17_brier_score,
            "calibration_need_band": self.calibration_need_band,
            "source_proposal_id": self.source_proposal_id,
            "event_kind": self.event_kind,
            "source_episode_id": self.source_episode_id,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionEntry":
        band = _band(payload.get("confidence_band"))
        raw_confidence = _legacy_or_numeric_confidence(payload.get("raw_confidence"), band)
        committed_confidence = _legacy_or_numeric_confidence(payload.get("committed_confidence"), band)
        evidence_basis = tuple(
            value for value in (str(x) for x in payload.get("evidence_basis", [])) if value in VALID_EVIDENCE_BASIS
        )
        return cls(
            prediction_id=str(payload.get("prediction_id", "")),
            turn_id=int(payload.get("turn_id", 0)),
            prediction_type=_prediction_type(payload.get("prediction_type")),
            predicted_value_summary=str(payload.get("predicted_value_summary", ""))[:200],
            confidence_band=band,
            raw_confidence=raw_confidence,
            committed_confidence=committed_confidence,
            confidence_cap_reason=str(
                payload.get("confidence_cap_reason")
                or ("legacy_confidence_band_default" if "committed_confidence" not in payload else "")
            ),
            evidence_basis=evidence_basis,
            evidence_refs=tuple(str(x) for x in payload.get("evidence_refs", [])),
            expires_after_turns=max(int(payload.get("expires_after_turns", DEFAULT_HYPERPARAMS.default_prediction_expiry_turns) or DEFAULT_HYPERPARAMS.default_prediction_expiry_turns), 1),
            created_at_turn=max(int(payload.get("created_at_turn", payload.get("turn_id", 0)) or 0), 0),
            created_before_response=bool(payload.get("created_before_response", False)),
            response_turn_id=max(int(payload.get("response_turn_id", payload.get("turn_id", 0)) or 0), 0),
            validation_status=_status(payload.get("validation_status")),
            observed_outcome_summary=str(payload.get("observed_outcome_summary", "")),
            settlement_outcome=str(payload.get("settlement_outcome", "")),
            settlement_confidence=_bounded_confidence(payload.get("settlement_confidence"), default=0.0, low=0.0),
            settlement_id=str(payload.get("settlement_id", "")),
            m17_prediction_error=_optional_float(payload.get("m17_prediction_error")),
            m17_brier_score=_optional_float(payload.get("m17_brier_score")),
            calibration_need_band=_band(payload.get("calibration_need_band")),
            source_proposal_id=str(payload.get("source_proposal_id", "")),
            event_kind=str(payload.get("event_kind", "prediction")),
            source_episode_id=str(payload.get("source_episode_id", "")),
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
    raw_confidence: float = 0.55
    evidence_basis: tuple[str, ...] = ()
    evidence_quote_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposed_prediction_type": self.proposed_prediction_type,
            "predicted_value_summary": self.predicted_value_summary,
            "confidence_band": self.confidence_band,
            "raw_confidence": round(self.raw_confidence, 6),
            "evidence_basis": list(self.evidence_basis),
            "evidence_quote_ids": list(self.evidence_quote_ids),
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "source_judgment_ids": list(self.source_judgment_ids),
            "expires_after_turns": self.expires_after_turns,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PredictionProposal":
        band = _band(payload.get("confidence_band"))
        evidence_basis = tuple(
            value for value in (str(x) for x in payload.get("evidence_basis", [])) if value in VALID_EVIDENCE_BASIS
        )
        return cls(
            proposal_id=str(payload.get("proposal_id", payload.get("id", ""))),
            proposed_prediction_type=_prediction_type(payload.get("proposed_prediction_type", payload.get("prediction_type"))),
            predicted_value_summary=str(payload.get("predicted_value_summary", ""))[:200],
            confidence_band=band,
            raw_confidence=_legacy_or_numeric_confidence(payload.get("raw_confidence"), band),
            evidence_basis=evidence_basis,
            evidence_quote_ids=tuple(str(x) for x in payload.get("evidence_quote_ids", [])),
            source_hypothesis_ids=tuple(str(x) for x in payload.get("source_hypothesis_ids", [])),
            source_judgment_ids=tuple(str(x) for x in payload.get("source_judgment_ids", [])),
            expires_after_turns=max(int(payload.get("expires_after_turns", DEFAULT_HYPERPARAMS.default_prediction_expiry_turns) or DEFAULT_HYPERPARAMS.default_prediction_expiry_turns), 1),
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

    def latest_entry(self, prediction_id: str) -> PredictionEntry | None:
        for entry in reversed(self.entries):
            if entry.prediction_id == prediction_id:
                return entry
        return None

    def pending_entries(self) -> tuple[PredictionEntry, ...]:
        latest: dict[str, PredictionEntry] = {}
        for entry in self.entries:
            latest[entry.prediction_id] = entry
        return tuple(
            entry
            for entry in latest.values()
            if entry.validation_status == "pending"
        )

    def predictions_by_status(self, *, status: ValidationStatus, current_turn_id: int, last_n_turns: int) -> tuple[PredictionEntry, ...]:
        floor = current_turn_id - last_n_turns
        latest: dict[str, PredictionEntry] = {}
        for entry in self.entries:
            if entry.turn_id >= floor:
                latest[entry.prediction_id] = entry
        return tuple(entry for entry in latest.values() if entry.validation_status == status)


def attach_prediction_source_episode(
    ledger: UserPredictionLedger,
    *,
    created_at_turn: int,
    source_episode_id: str,
) -> UserPredictionLedger:
    episode_id = str(source_episode_id or "").strip()
    if not episode_id:
        return ledger
    updated_entries = tuple(
        PredictionEntry(
            prediction_id=entry.prediction_id,
            turn_id=entry.turn_id,
            prediction_type=entry.prediction_type,
            predicted_value_summary=entry.predicted_value_summary,
            confidence_band=entry.confidence_band,
            evidence_refs=entry.evidence_refs,
            validation_status=entry.validation_status,
            observed_outcome_summary=entry.observed_outcome_summary,
            calibration_need_band=entry.calibration_need_band,
            source_proposal_id=entry.source_proposal_id,
            event_kind=entry.event_kind,
            raw_confidence=entry.raw_confidence,
            committed_confidence=entry.committed_confidence,
            confidence_cap_reason=entry.confidence_cap_reason,
            evidence_basis=entry.evidence_basis,
            expires_after_turns=entry.expires_after_turns,
            created_at_turn=entry.created_at_turn,
            created_before_response=entry.created_before_response,
            response_turn_id=entry.response_turn_id,
            settlement_outcome=entry.settlement_outcome,
            settlement_confidence=entry.settlement_confidence,
            settlement_id=entry.settlement_id,
            m17_prediction_error=entry.m17_prediction_error,
            m17_brier_score=entry.m17_brier_score,
            source_episode_id=episode_id if entry.created_at_turn == created_at_turn and not entry.source_episode_id else entry.source_episode_id,
        )
        for entry in ledger.entries
    )
    return UserPredictionLedger(entries=updated_entries, proposals=ledger.proposals)


def normalize_prediction_confidence(
    raw_confidence: object,
    prediction_type: object,
    evidence_basis: Sequence[object] | None,
    *,
    type_precision: float | None = None,
    type_sample_count: int | None = None,
) -> NormalizedConfidence:
    del prediction_type
    normalized_basis = [
        basis for basis in (str(item or "") for item in (evidence_basis or ())) if basis in VALID_EVIDENCE_BASIS
    ]
    raw = _bounded_confidence(raw_confidence, default=0.55)
    cap = 0.55
    cap_reason = "no_evidence"
    if any(basis in {"product_invariant", "system_invariant"} for basis in normalized_basis):
        cap = 0.90
        cap_reason = "system_or_product_invariant"
    elif any(basis in {"repeated_evidence", "prior_confirmation"} for basis in normalized_basis):
        cap = 0.85
        cap_reason = "repeated_evidence_or_prior_confirmation"
    elif any(basis in {"direct_user_statement", "current_user_statement"} for basis in normalized_basis):
        cap = 0.80
        cap_reason = "direct_user_statement"
    elif any(
        basis in {"current_user_question", "current_user_request", "recent_topic_continuity"}
        for basis in normalized_basis
    ):
        cap = 0.72
        cap_reason = "current_turn_or_topic_continuity"
    if type_precision is not None:
        precision_cap = _bounded_confidence(type_precision + 0.15, default=cap)
        if type_sample_count is not None and int(type_sample_count) < 20:
            precision_cap = min(precision_cap, 0.78)
        if precision_cap < cap:
            cap = precision_cap
            cap_reason = "type_precision_cap"
    committed = min(raw, cap)
    return NormalizedConfidence(
        raw_confidence=round(raw, 6),
        committed_confidence=round(committed, 6),
        confidence_band=_band_from_confidence(committed),
        cap_reason=cap_reason,
    )


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
    type_precision_by_type: Mapping[str, float] | None = None,
    type_sample_count_by_type: Mapping[str, int] | None = None,
    source_episode_id: str = "",
) -> UserPredictionLedger:
    next_entries = list(ledger.entries)
    next_proposals = list(ledger.proposals)
    latest_by_id = {entry.prediction_id: entry for entry in next_entries}
    open_ids = {
        prediction_id
        for prediction_id, entry in latest_by_id.items()
        if entry.validation_status == "pending"
    }

    for judgment in judgments:
        prediction_id = str(judgment.get("prediction_id", "") or "")
        if prediction_id not in open_ids:
            continue
        source = _latest_entry(next_entries, prediction_id)
        outcome = _normalize_outcome(judgment.get("outcome", judgment.get("status")))
        settlement_confidence = _bounded_confidence(
            judgment.get("settlement_confidence"),
            default=0.81 if outcome in {"confirmed", "violated"} else 0.0,
            low=0.0,
        )
        evidence_span = str(judgment.get("evidence_span", "") or "").strip()[:200]
        evidence_refs = _dedupe_strings(
            judgment.get("evidence_quote_ids", []),
            judgment.get("evidence_refs", []),
            limit=8,
        )
        reason_codes = _dedupe_strings(judgment.get("reason_codes", []), limit=8)
        should_update = outcome in {"confirmed", "violated"} and settlement_confidence >= 0.55 and bool(evidence_span or evidence_refs)
        if not should_update and outcome in {"confirmed", "violated"}:
            outcome = "unclear"
        status = _compatible_status(outcome)
        m17_prediction_error = _prediction_error_from_outcome(source.committed_confidence, outcome) if should_update else None
        m17_brier_score = _prediction_brier_from_outcome(source.committed_confidence, outcome) if should_update else None
        observed = evidence_span or outcome
        if reason_codes:
            observed = f"{observed} [{'|'.join(reason_codes[:3])}]"
        next_entries.append(
            PredictionEntry(
                prediction_id=prediction_id,
                turn_id=turn_id,
                prediction_type=source.prediction_type,
                predicted_value_summary=source.predicted_value_summary,
                confidence_band=source.confidence_band,
                raw_confidence=source.raw_confidence,
                committed_confidence=source.committed_confidence,
                confidence_cap_reason=source.confidence_cap_reason,
                evidence_basis=source.evidence_basis,
                evidence_refs=tuple(evidence_refs),
                expires_after_turns=source.expires_after_turns,
                created_at_turn=source.created_at_turn,
                created_before_response=source.created_before_response,
                response_turn_id=source.response_turn_id,
                validation_status=status,
                observed_outcome_summary=observed,
                settlement_outcome=outcome,
                settlement_confidence=settlement_confidence,
                settlement_id=str(judgment.get("settlement_id", f"settle:{prediction_id}:{turn_id}")),
                m17_prediction_error=m17_prediction_error,
                m17_brier_score=m17_brier_score,
                calibration_need_band=calibration_need_band,
                source_proposal_id=source.source_proposal_id,
                event_kind="judgment" if outcome != "expired" else "expiration",
                source_episode_id=source.source_episode_id,
            )
        )
        known_judgment_ids.add(prediction_id)
        open_ids.discard(prediction_id)

    latest_by_id = {entry.prediction_id: entry for entry in next_entries}
    for prediction_id, entry in tuple(latest_by_id.items()):
        if entry.validation_status != "pending":
            continue
        proposal = next((p for p in reversed(next_proposals) if p.proposal_id == entry.source_proposal_id), None)
        if proposal is None:
            continue
        if turn_id - entry.created_at_turn > proposal.expires_after_turns:
            next_entries.append(
                PredictionEntry(
                    prediction_id=entry.prediction_id,
                    turn_id=turn_id,
                    prediction_type=entry.prediction_type,
                    predicted_value_summary=entry.predicted_value_summary,
                    confidence_band=entry.confidence_band,
                    raw_confidence=entry.raw_confidence,
                    committed_confidence=entry.committed_confidence,
                    confidence_cap_reason=entry.confidence_cap_reason,
                    evidence_basis=entry.evidence_basis,
                    evidence_refs=entry.evidence_refs,
                    expires_after_turns=entry.expires_after_turns,
                    created_at_turn=entry.created_at_turn,
                    created_before_response=entry.created_before_response,
                    response_turn_id=entry.response_turn_id,
                    validation_status="uncertain",
                    observed_outcome_summary="expired",
                    settlement_outcome="expired",
                    settlement_confidence=0.0,
                    settlement_id=f"expire:{entry.prediction_id}:{turn_id}",
                    calibration_need_band="med",
                    source_proposal_id=entry.source_proposal_id,
                    event_kind="expiration",
                    source_episode_id=entry.source_episode_id,
                )
            )
            next_proposals.append(
                PredictionProposal(
                    proposal_id=proposal.proposal_id,
                    proposed_prediction_type=proposal.proposed_prediction_type,
                    predicted_value_summary=proposal.predicted_value_summary,
                    confidence_band=proposal.confidence_band,
                    raw_confidence=proposal.raw_confidence,
                    evidence_basis=proposal.evidence_basis,
                    evidence_quote_ids=proposal.evidence_quote_ids,
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
        elif not (proposal.source_hypothesis_ids or proposal.source_judgment_ids or proposal.evidence_quote_ids):
            rejection = "missing_evidence_refs"
        accepted = rejection == ""
        if accepted:
            admitted += 1
        gated = PredictionProposal(
            proposal_id=proposal.proposal_id,
            proposed_prediction_type=proposal.proposed_prediction_type,
            predicted_value_summary=proposal.predicted_value_summary,
            confidence_band=proposal.confidence_band,
            raw_confidence=proposal.raw_confidence,
            evidence_basis=proposal.evidence_basis,
            evidence_quote_ids=proposal.evidence_quote_ids,
            source_hypothesis_ids=proposal.source_hypothesis_ids,
            source_judgment_ids=proposal.source_judgment_ids,
            expires_after_turns=proposal.expires_after_turns,
            accepted=accepted,
            rejection_reason=rejection,
            turn_id=turn_id,
        )
        next_proposals.append(gated)
        if accepted:
            normalized = normalize_prediction_confidence(
                proposal.raw_confidence,
                proposal.proposed_prediction_type,
                proposal.evidence_basis,
                type_precision=_mapping_float(type_precision_by_type).get(proposal.proposed_prediction_type),
                type_sample_count=_mapping_int(type_sample_count_by_type).get(proposal.proposed_prediction_type),
            )
            next_entries.append(
                PredictionEntry(
                    prediction_id=f"pred:{proposal.proposal_id}",
                    turn_id=turn_id,
                    prediction_type=proposal.proposed_prediction_type,
                    predicted_value_summary=proposal.predicted_value_summary,
                    confidence_band=normalized.confidence_band,
                    raw_confidence=normalized.raw_confidence,
                    committed_confidence=normalized.committed_confidence,
                    confidence_cap_reason=normalized.cap_reason,
                    evidence_basis=proposal.evidence_basis,
                    evidence_refs=tuple(
                        _dedupe_strings(
                            proposal.source_hypothesis_ids,
                            proposal.source_judgment_ids,
                            proposal.evidence_quote_ids,
                            limit=12,
                        )
                    ),
                    expires_after_turns=proposal.expires_after_turns,
                    created_at_turn=turn_id,
                    created_before_response=True,
                    response_turn_id=turn_id,
                    validation_status="pending",
                    observed_outcome_summary="",
                    settlement_outcome="",
                    settlement_confidence=0.0,
                    settlement_id="",
                    calibration_need_band=calibration_need_band,
                    source_proposal_id=proposal.proposal_id,
                    event_kind="prediction",
                    source_episode_id=source_episode_id,
                )
            )
    return UserPredictionLedger(entries=tuple(next_entries), proposals=tuple(next_proposals))


def _mapping_float(payload: Mapping[str, float] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): float(value) for key, value in payload.items()}


def _mapping_int(payload: Mapping[str, int] | None) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): int(value) for key, value in payload.items()}


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


def _band_from_confidence(value: float) -> ConfidenceBand:
    if value >= 0.75:
        return "high"
    if value >= 0.60:
        return "med"
    return "low"


def _legacy_or_numeric_confidence(value: object, band: ConfidenceBand) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        return LEGACY_BAND_DEFAULTS[band]
    return _bounded_confidence(parsed, default=LEGACY_BAND_DEFAULTS[band])


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _bounded_confidence(value: object, *, default: float = 0.55, low: float = 0.50, high: float = 0.90) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(max(low, min(high, parsed)), 6)


def _dedupe_strings(*values: object, limit: int = 8) -> list[str]:
    flattened: list[object] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            flattened.extend(list(value))
        else:
            flattened.append(value)
    result: list[str] = []
    seen: set[str] = set()
    for item in flattened:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text[:160])
        if len(result) >= limit:
            break
    return result


def _normalize_outcome(value: object) -> str:
    from segmentum.user_model.m17_prediction_error import normalize_prediction_outcome

    return normalize_prediction_outcome(value)


def _compatible_status(outcome: str) -> ValidationStatus:
    if outcome == "confirmed":
        return "confirmed"
    if outcome == "violated":
        return "violated"
    return "uncertain"


def _prediction_error_from_outcome(committed_confidence: float, outcome: str) -> float | None:
    from segmentum.user_model.m17_prediction_error import prediction_error_from_outcome

    return prediction_error_from_outcome(committed_confidence, outcome)


def _prediction_brier_from_outcome(committed_confidence: float, outcome: str) -> float | None:
    from segmentum.user_model.m17_prediction_error import prediction_brier_from_outcome

    return prediction_brier_from_outcome(committed_confidence, outcome)
