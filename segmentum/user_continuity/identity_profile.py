"""M12.0 identity profile and bounded deterministic update helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

Band = str
IdentityState = str
SupportKind = str


@dataclass(frozen=True)
class AliasObservation:
    alias_observation_id: str
    alias_text: str
    first_seen_turn: str
    last_seen_turn: str
    seen_count: int
    modality: str
    permitted_use: str

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_observation_id": self.alias_observation_id,
            "alias_text": self.alias_text,
            "first_seen_turn": self.first_seen_turn,
            "last_seen_turn": self.last_seen_turn,
            "seen_count": int(self.seen_count),
            "modality": self.modality,
            "permitted_use": self.permitted_use,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AliasObservation":
        return cls(
            alias_observation_id=str(payload.get("alias_observation_id", "")),
            alias_text=str(payload.get("alias_text", "")),
            first_seen_turn=str(payload.get("first_seen_turn", "")),
            last_seen_turn=str(payload.get("last_seen_turn", "")),
            seen_count=int(payload.get("seen_count", 0)),
            modality=str(payload.get("modality", "factual")),
            permitted_use=str(payload.get("permitted_use", "observe")),
        )


@dataclass(frozen=True)
class ContinuityCue:
    cue_id: str
    cue_kind: str
    content_summary: str
    evidence_refs: tuple[str, ...]
    confidence_band: Band
    supports: SupportKind
    source_turn_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "cue_kind": self.cue_kind,
            "content_summary": self.content_summary,
            "evidence_refs": list(self.evidence_refs),
            "confidence_band": self.confidence_band,
            "supports": self.supports,
            "source_turn_id": self.source_turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContinuityCue":
        refs = payload.get("evidence_refs", [])
        if not isinstance(refs, list):
            refs = []
        return cls(
            cue_id=str(payload.get("cue_id", "")),
            cue_kind=str(payload.get("cue_kind", "history")),
            content_summary=str(payload.get("content_summary", ""))[:120],
            evidence_refs=tuple(str(item) for item in refs),
            confidence_band=str(payload.get("confidence_band", "low")),
            supports=str(payload.get("supports", "weakens")),
            source_turn_id=str(payload.get("source_turn_id", "")),
        )


@dataclass(frozen=True)
class IdentityProfile:
    user_id: str
    display_name: str
    aliases_observed: tuple[AliasObservation, ...] = ()
    continuity_evidence: tuple[ContinuityCue, ...] = ()
    style_habit_cues: tuple[ContinuityCue, ...] = ()
    known_relationship_facts: tuple[str, ...] = ()
    contradiction_refs: tuple[str, ...] = ()
    binding_confidence_band: Band = "low"
    identity_state: IdentityState = "unverified"
    last_updated_turn_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "aliases_observed": [item.to_dict() for item in self.aliases_observed],
            "continuity_evidence": [item.to_dict() for item in self.continuity_evidence],
            "style_habit_cues": [item.to_dict() for item in self.style_habit_cues],
            "known_relationship_facts": list(self.known_relationship_facts),
            "contradiction_refs": list(self.contradiction_refs),
            "binding_confidence_band": self.binding_confidence_band,
            "identity_state": self.identity_state,
            "last_updated_turn_id": self.last_updated_turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdentityProfile":
        raw_aliases = payload.get("aliases_observed", [])
        if not isinstance(raw_aliases, list):
            raw_aliases = []
        raw_cues = payload.get("continuity_evidence", [])
        if not isinstance(raw_cues, list):
            raw_cues = []
        raw_style = payload.get("style_habit_cues", [])
        if not isinstance(raw_style, list):
            raw_style = []
        rel = payload.get("known_relationship_facts", [])
        if not isinstance(rel, list):
            rel = []
        refs = payload.get("contradiction_refs", [])
        if not isinstance(refs, list):
            refs = []
        return cls(
            user_id=str(payload.get("user_id", "")),
            display_name=str(payload.get("display_name", "")),
            aliases_observed=tuple(
                AliasObservation.from_dict(item)
                for item in raw_aliases
                if isinstance(item, Mapping)
            ),
            continuity_evidence=tuple(
                ContinuityCue.from_dict(item)
                for item in raw_cues
                if isinstance(item, Mapping)
            ),
            style_habit_cues=tuple(
                ContinuityCue.from_dict(item)
                for item in raw_style
                if isinstance(item, Mapping)
            ),
            known_relationship_facts=tuple(str(item) for item in rel),
            contradiction_refs=tuple(str(item) for item in refs),
            binding_confidence_band=str(payload.get("binding_confidence_band", "low")),
            identity_state=str(payload.get("identity_state", "unverified")),
            last_updated_turn_id=str(payload.get("last_updated_turn_id", "")),
        )


def profile_state_hash(profile: IdentityProfile) -> str:
    raw = json.dumps(profile.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
