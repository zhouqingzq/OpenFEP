"""M12.0 deterministic cross-user alias conflict detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .identity_profile import IdentityProfile


@dataclass(frozen=True)
class ConflictRecord:
    conflict_id: str
    turn_id: str
    asserted_alias: str
    claimant_user_id: str
    incumbent_user_id: str
    severity_band: str
    contradiction_cue_ids: tuple[str, ...]
    incumbent_binding_confidence: str
    resolution_status: str = "open"

    def to_dict(self) -> dict[str, object]:
        return {
            "conflict_id": self.conflict_id,
            "turn_id": self.turn_id,
            "asserted_alias": self.asserted_alias,
            "claimant_user_id": self.claimant_user_id,
            "incumbent_user_id": self.incumbent_user_id,
            "severity_band": self.severity_band,
            "contradiction_cue_ids": list(self.contradiction_cue_ids),
            "incumbent_binding_confidence": self.incumbent_binding_confidence,
            "resolution_status": self.resolution_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConflictRecord":
        raw_ids = payload.get("contradiction_cue_ids", [])
        if not isinstance(raw_ids, list):
            raw_ids = []
        return cls(
            conflict_id=str(payload.get("conflict_id", "")),
            turn_id=str(payload.get("turn_id", "")),
            asserted_alias=str(payload.get("asserted_alias", "")),
            claimant_user_id=str(payload.get("claimant_user_id", "")),
            incumbent_user_id=str(payload.get("incumbent_user_id", "")),
            severity_band=str(payload.get("severity_band", "minor")),
            contradiction_cue_ids=tuple(str(item) for item in raw_ids),
            incumbent_binding_confidence=str(payload.get("incumbent_binding_confidence", "low")),
            resolution_status=str(payload.get("resolution_status", "open")),
        )


def detect_identity_conflicts(
    *,
    turn_id: str,
    claim_id: str,
    claimant_user_id: str,
    asserted_alias: str,
    modality: str,
    profiles_by_user: Mapping[str, IdentityProfile],
) -> tuple[ConflictRecord, ...]:
    alias_key = asserted_alias.strip().lower()
    if not alias_key:
        return ()
    conflicts: list[ConflictRecord] = []
    for user_id, profile in profiles_by_user.items():
        if user_id == claimant_user_id:
            continue
        incumbent_aliases = {
            obs.alias_text.strip().lower()
            for obs in profile.aliases_observed
            if obs.alias_text.strip()
        }
        if alias_key not in incumbent_aliases:
            continue
        if profile.identity_state != "corroborated":
            severity = "minor"
        elif modality in {"roleplay", "joke", "hypothetical"}:
            severity = "minor"
        elif profile.binding_confidence_band == "high":
            severity = "major"
        else:
            severity = "minor"
        conflicts.append(
            ConflictRecord(
                conflict_id=f"conflict:{turn_id}:{claim_id}:{user_id}",
                turn_id=turn_id,
                asserted_alias=asserted_alias,
                claimant_user_id=claimant_user_id,
                incumbent_user_id=user_id,
                severity_band=severity,
                contradiction_cue_ids=tuple(
                    cue.cue_id for cue in profile.continuity_evidence if cue.supports == "contradicts"
                ),
                incumbent_binding_confidence=profile.binding_confidence_band,
                resolution_status="open",
            )
        )
    return tuple(conflicts)
