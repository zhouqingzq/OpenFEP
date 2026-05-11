"""Bounded M11 user-model state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams

ConfidenceBand = Literal["low", "med", "high"]
PermittedUse = Literal["explicit_fact", "cautious_hypothesis", "strategy_only", "forbidden"]
ClaimKind = Literal["verified_fact", "user_stated_claim", "inferred_hypothesis", "contradiction", "unknown"]
Domain = Literal[
    "self_reported_preferences",
    "self_reported_history",
    "task_requirements",
    "emotional_state",
    "technical_claims",
    "social_relationship_claims",
]

VALID_CONFIDENCE_BANDS = {"low", "med", "high"}
VALID_PERMITTED_USES = {"explicit_fact", "cautious_hypothesis", "strategy_only", "forbidden"}
VALID_DOMAINS = {
    "self_reported_preferences",
    "self_reported_history",
    "task_requirements",
    "emotional_state",
    "technical_claims",
    "social_relationship_claims",
}


@dataclass(frozen=True)
class EvidenceRef:
    turn_id: str
    quote_id: str

    def to_dict(self) -> dict[str, str]:
        return {"turn_id": self.turn_id, "quote_id": self.quote_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceRef":
        return cls(turn_id=str(payload.get("turn_id", "")), quote_id=str(payload.get("quote_id", "")))


@dataclass(frozen=True)
class UserHypothesis:
    hypothesis_id: str
    content_summary: str
    domain: Domain
    confidence_band: ConfidenceBand = "low"
    evidence_refs: tuple[EvidenceRef, ...] = ()
    contradiction_refs: tuple[EvidenceRef, ...] = ()
    last_confirmed_turn: str = ""
    last_violated_turn: str = ""
    permitted_use: PermittedUse = "cautious_hypothesis"
    claim_kind: ClaimKind = "inferred_hypothesis"

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "content_summary": self.content_summary,
            "domain": self.domain,
            "confidence_band": self.confidence_band,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "contradiction_refs": [ref.to_dict() for ref in self.contradiction_refs],
            "last_confirmed_turn": self.last_confirmed_turn,
            "last_violated_turn": self.last_violated_turn,
            "permitted_use": self.permitted_use,
            "claim_kind": self.claim_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserHypothesis":
        return cls(
            hypothesis_id=str(payload.get("hypothesis_id", "")),
            content_summary=str(payload.get("content_summary", "")),
            domain=_domain(payload.get("domain")),
            confidence_band=_band(payload.get("confidence_band")),
            evidence_refs=tuple(
                EvidenceRef.from_dict(ref)
                for ref in payload.get("evidence_refs", [])
                if isinstance(ref, Mapping)
            ),
            contradiction_refs=tuple(
                EvidenceRef.from_dict(ref)
                for ref in payload.get("contradiction_refs", [])
                if isinstance(ref, Mapping)
            ),
            last_confirmed_turn=str(payload.get("last_confirmed_turn", "")),
            last_violated_turn=str(payload.get("last_violated_turn", "")),
            permitted_use=_permitted_use(payload.get("permitted_use")),
            claim_kind=_claim_kind(payload.get("claim_kind")),
        )


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    content_summary: str
    domain: Domain
    claim_kind: ClaimKind
    evidence_refs: tuple[EvidenceRef, ...] = ()
    confidence_band: ConfidenceBand = "low"

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "content_summary": self.content_summary,
            "domain": self.domain,
            "claim_kind": self.claim_kind,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence_band": self.confidence_band,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ClaimRecord":
        return cls(
            claim_id=str(payload.get("claim_id", "")),
            content_summary=str(payload.get("content_summary", "")),
            domain=_domain(payload.get("domain")),
            claim_kind=_claim_kind(payload.get("claim_kind")),
            evidence_refs=tuple(
                EvidenceRef.from_dict(ref)
                for ref in payload.get("evidence_refs", [])
                if isinstance(ref, Mapping)
            ),
            confidence_band=_band(payload.get("confidence_band")),
        )


@dataclass(frozen=True)
class UserModel:
    user_id: str
    display_name: str = ""
    cognitive_style_hypotheses: tuple[UserHypothesis, ...] = ()
    personality_hypotheses: tuple[UserHypothesis, ...] = ()
    preference_hypotheses: tuple[UserHypothesis, ...] = ()
    boundaries_and_dislikes: tuple[UserHypothesis, ...] = ()
    relationship_state: tuple[UserHypothesis, ...] = ()
    source_reliability_by_domain: dict[str, float] = field(default_factory=dict)
    claim_history_summary: tuple[ClaimRecord, ...] = ()
    prediction_history_summary: tuple[str, ...] = ()
    open_uncertainties: tuple[str, ...] = ()
    last_updated_turn_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "cognitive_style_hypotheses": [h.to_dict() for h in self.cognitive_style_hypotheses],
            "personality_hypotheses": [h.to_dict() for h in self.personality_hypotheses],
            "preference_hypotheses": [h.to_dict() for h in self.preference_hypotheses],
            "boundaries_and_dislikes": [h.to_dict() for h in self.boundaries_and_dislikes],
            "relationship_state": [h.to_dict() for h in self.relationship_state],
            "source_reliability_by_domain": dict(sorted(self.source_reliability_by_domain.items())),
            "claim_history_summary": [c.to_dict() for c in self.claim_history_summary],
            "prediction_history_summary": list(self.prediction_history_summary),
            "open_uncertainties": list(self.open_uncertainties),
            "last_updated_turn_id": self.last_updated_turn_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "UserModel":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("UserModel JSON must decode to an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserModel":
        return cls(
            user_id=str(payload.get("user_id", "")),
            display_name=str(payload.get("display_name", "")),
            cognitive_style_hypotheses=_hyp_tuple(payload.get("cognitive_style_hypotheses")),
            personality_hypotheses=_hyp_tuple(payload.get("personality_hypotheses")),
            preference_hypotheses=_hyp_tuple(payload.get("preference_hypotheses")),
            boundaries_and_dislikes=_hyp_tuple(payload.get("boundaries_and_dislikes")),
            relationship_state=_hyp_tuple(payload.get("relationship_state")),
            source_reliability_by_domain={
                str(k): float(v)
                for k, v in dict(payload.get("source_reliability_by_domain", {})).items()
            },
            claim_history_summary=tuple(
                ClaimRecord.from_dict(row)
                for row in payload.get("claim_history_summary", [])
                if isinstance(row, Mapping)
            ),
            prediction_history_summary=tuple(str(x) for x in payload.get("prediction_history_summary", [])),
            open_uncertainties=tuple(str(x) for x in payload.get("open_uncertainties", [])),
            last_updated_turn_id=str(payload.get("last_updated_turn_id", "")),
        )

    def all_hypotheses(self) -> tuple[UserHypothesis, ...]:
        return (
            *self.cognitive_style_hypotheses,
            *self.personality_hypotheses,
            *self.preference_hypotheses,
            *self.boundaries_and_dislikes,
            *self.relationship_state,
        )

    def with_reliability(self, reliability_by_domain: Mapping[str, float]) -> "UserModel":
        data = self.to_dict()
        data["source_reliability_by_domain"] = {
            str(k): round(float(v), DEFAULT_HYPERPARAMS.float_round_digits)
            for k, v in reliability_by_domain.items()
        }
        return UserModel.from_dict(data)


def apply_claims_to_user_model(
    model: UserModel,
    claims: Sequence[Mapping[str, object]],
    *,
    turn_id: str,
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> UserModel:
    """Return an updated model from bounded extractor claims.

    This function inspects no raw text; it consumes only extractor enum fields
    and bounded summaries.
    """

    preferences = list(model.preference_hypotheses)
    boundaries = list(model.boundaries_and_dislikes)
    relationships = list(model.relationship_state)
    cognitive = list(model.cognitive_style_hypotheses)
    personality = list(model.personality_hypotheses)
    history = list(model.claim_history_summary)
    uncertainties = list(model.open_uncertainties)

    for claim in claims:
        claim_id = str(claim.get("id", ""))
        domain = _domain(claim.get("domain"))
        summary = _bounded_summary(claim.get("content_summary"))
        refs = tuple(
            EvidenceRef(turn_id=turn_id, quote_id=str(qid))
            for qid in _string_seq(claim.get("evidence_quote_ids"))
        )[: hyperparams.max_evidence_refs_per_hypothesis]
        band = _band(claim.get("confidence_band"))
        modality = str(claim.get("modality", "factual"))
        claim_kind: ClaimKind = "user_stated_claim" if modality == "factual" else "unknown"
        history.append(
            ClaimRecord(
                claim_id=claim_id,
                content_summary=summary,
                domain=domain,
                claim_kind=claim_kind,
                evidence_refs=refs,
                confidence_band=band,
            )
        )
        if modality in {"roleplay", "joke", "hypothetical"}:
            uncertainties.append(claim_id or summary)
            continue
        bucket = preferences
        permitted: PermittedUse = "cautious_hypothesis"
        if domain == "self_reported_preferences":
            bucket = boundaries if _is_boundary_claim(claim) else preferences
        elif domain == "social_relationship_claims":
            bucket = relationships
        elif domain == "emotional_state":
            bucket = cognitive
            permitted = "strategy_only"
        elif domain in {"self_reported_history", "technical_claims", "task_requirements"}:
            bucket = cognitive
            permitted = "strategy_only" if domain != "task_requirements" else "explicit_fact"
        hyp_id = str(claim.get("hypothesis_id") or claim_id or f"hyp:{domain}:{len(bucket)}")
        bucket[:] = _upsert_hypothesis(
            bucket,
            hypothesis_id=hyp_id,
            content_summary=summary,
            domain=domain,
            evidence_refs=refs,
            requested_band=band,
            turn_id=turn_id,
            permitted_use=permitted,
            hyperparams=hyperparams,
        )

    return UserModel(
        user_id=model.user_id,
        display_name=model.display_name,
        cognitive_style_hypotheses=tuple(cognitive[: hyperparams.max_hypotheses_per_bucket]),
        personality_hypotheses=tuple(personality[: hyperparams.max_hypotheses_per_bucket]),
        preference_hypotheses=tuple(preferences[: hyperparams.max_hypotheses_per_bucket]),
        boundaries_and_dislikes=tuple(boundaries[: hyperparams.max_hypotheses_per_bucket]),
        relationship_state=tuple(relationships[: hyperparams.max_hypotheses_per_bucket]),
        source_reliability_by_domain=dict(model.source_reliability_by_domain),
        claim_history_summary=tuple(history[-hyperparams.max_hypotheses_per_bucket :]),
        prediction_history_summary=model.prediction_history_summary,
        open_uncertainties=tuple(uncertainties[-hyperparams.max_hypotheses_per_bucket :]),
        last_updated_turn_id=turn_id,
    )


def apply_contradictions(
    model: UserModel,
    contradictions: Sequence[Mapping[str, object]],
    *,
    turn_id: str,
) -> UserModel:
    if not contradictions:
        return model
    all_ids = {h.hypothesis_id for h in model.all_hypotheses()}
    contradiction_ids = {
        str(item.get("conflicts_with_memory_id", ""))
        for item in contradictions
        if str(item.get("conflicts_with_memory_id", "")) in all_ids
    }

    def patch(hyp: UserHypothesis) -> UserHypothesis:
        if hyp.hypothesis_id not in contradiction_ids:
            return hyp
        refs = (*hyp.contradiction_refs, EvidenceRef(turn_id=turn_id, quote_id=str(hyp.hypothesis_id)))
        return UserHypothesis(
            hypothesis_id=hyp.hypothesis_id,
            content_summary=hyp.content_summary,
            domain=hyp.domain,
            confidence_band="low" if hyp.confidence_band == "med" else hyp.confidence_band,
            evidence_refs=hyp.evidence_refs,
            contradiction_refs=refs,
            last_confirmed_turn=hyp.last_confirmed_turn,
            last_violated_turn=turn_id,
            permitted_use="cautious_hypothesis",
            claim_kind="contradiction",
        )

    data = model.to_dict()
    for key in (
        "cognitive_style_hypotheses",
        "personality_hypotheses",
        "preference_hypotheses",
        "boundaries_and_dislikes",
        "relationship_state",
    ):
        data[key] = [patch(UserHypothesis.from_dict(row)).to_dict() for row in data[key]]
    data["last_updated_turn_id"] = turn_id
    return UserModel.from_dict(data)


def _upsert_hypothesis(
    bucket: list[UserHypothesis],
    *,
    hypothesis_id: str,
    content_summary: str,
    domain: Domain,
    evidence_refs: tuple[EvidenceRef, ...],
    requested_band: ConfidenceBand,
    turn_id: str,
    permitted_use: PermittedUse,
    hyperparams: Hyperparams,
) -> list[UserHypothesis]:
    existing = next((h for h in bucket if h.hypothesis_id == hypothesis_id), None)
    refs = tuple(evidence_refs)
    contradictions: tuple[EvidenceRef, ...] = ()
    if existing is not None:
        refs = (*existing.evidence_refs, *evidence_refs)
        contradictions = existing.contradiction_refs
    bounded_refs = refs[-hyperparams.max_evidence_refs_per_hypothesis :]
    band = _promoted_band(requested_band, bounded_refs, hyperparams)
    hyp = UserHypothesis(
        hypothesis_id=hypothesis_id,
        content_summary=content_summary,
        domain=domain,
        confidence_band=band,
        evidence_refs=bounded_refs,
        contradiction_refs=contradictions,
        last_confirmed_turn=turn_id,
        last_violated_turn=existing.last_violated_turn if existing else "",
        permitted_use=permitted_use if band == "high" else "cautious_hypothesis",
        claim_kind="user_stated_claim" if permitted_use == "explicit_fact" else "inferred_hypothesis",
    )
    return [hyp if h.hypothesis_id == hypothesis_id else h for h in bucket] if existing else [*bucket, hyp]


def _promoted_band(
    requested_band: ConfidenceBand,
    refs: Sequence[EvidenceRef],
    hyperparams: Hyperparams,
) -> ConfidenceBand:
    distinct_turns = {ref.turn_id for ref in refs if ref.turn_id}
    if (
        len(refs) >= hyperparams.high_confidence_min_evidence_refs
        and len(distinct_turns) >= hyperparams.high_confidence_min_distinct_turns
        and requested_band == "high"
    ):
        return "high"
    if len(refs) >= hyperparams.med_confidence_min_evidence_refs or requested_band == "med":
        return "med"
    return "low"


def _hyp_tuple(payload: object) -> tuple[UserHypothesis, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return ()
    return tuple(UserHypothesis.from_dict(row) for row in payload if isinstance(row, Mapping))


def _bounded_summary(value: object) -> str:
    return str(value or "")[:120]


def _string_seq(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


def _band(value: object) -> ConfidenceBand:
    text = str(value or "low")
    return text if text in VALID_CONFIDENCE_BANDS else "low"  # type: ignore[return-value]


def _domain(value: object) -> Domain:
    text = str(value or "task_requirements")
    return text if text in VALID_DOMAINS else "task_requirements"  # type: ignore[return-value]


def _permitted_use(value: object) -> PermittedUse:
    text = str(value or "cautious_hypothesis")
    return text if text in VALID_PERMITTED_USES else "cautious_hypothesis"  # type: ignore[return-value]


def _claim_kind(value: object) -> ClaimKind:
    text = str(value or "unknown")
    return text if text in {"verified_fact", "user_stated_claim", "inferred_hypothesis", "contradiction", "unknown"} else "unknown"  # type: ignore[return-value]


def _is_boundary_claim(claim: Mapping[str, object]) -> bool:
    tags = {str(item) for item in _string_seq(claim.get("tags"))}
    return "boundary" in tags or str(claim.get("permitted_use", "")) == "forbidden"
