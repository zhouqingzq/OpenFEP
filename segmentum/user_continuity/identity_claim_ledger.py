"""M12.0 append-only identity claim ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class IdentityClaimLedgerEntry:
    claim_id: str
    turn_id: str
    claimant_user_id: str
    asserted_alias: str
    modality: str
    evidence_quote_ids: tuple[str, ...]
    band_at_claim_time: str
    prior_state_snapshot_ref: str
    post_state_snapshot_ref: str
    validation_status: str
    conflict_record_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "turn_id": self.turn_id,
            "claimant_user_id": self.claimant_user_id,
            "asserted_alias": self.asserted_alias,
            "modality": self.modality,
            "evidence_quote_ids": list(self.evidence_quote_ids),
            "band_at_claim_time": self.band_at_claim_time,
            "prior_state_snapshot_ref": self.prior_state_snapshot_ref,
            "post_state_snapshot_ref": self.post_state_snapshot_ref,
            "validation_status": self.validation_status,
            "conflict_record_id": self.conflict_record_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdentityClaimLedgerEntry":
        refs = payload.get("evidence_quote_ids", [])
        if not isinstance(refs, list):
            refs = []
        return cls(
            claim_id=str(payload.get("claim_id", "")),
            turn_id=str(payload.get("turn_id", "")),
            claimant_user_id=str(payload.get("claimant_user_id", "")),
            asserted_alias=str(payload.get("asserted_alias", "")),
            modality=str(payload.get("modality", "factual")),
            evidence_quote_ids=tuple(str(item) for item in refs),
            band_at_claim_time=str(payload.get("band_at_claim_time", "low")),
            prior_state_snapshot_ref=str(payload.get("prior_state_snapshot_ref", "")),
            post_state_snapshot_ref=str(payload.get("post_state_snapshot_ref", "")),
            validation_status=str(payload.get("validation_status", "pending")),
            conflict_record_id=str(payload.get("conflict_record_id", "")),
        )


@dataclass(frozen=True)
class IdentityClaimLedger:
    entries: tuple[IdentityClaimLedgerEntry, ...] = ()

    def append(self, entry: IdentityClaimLedgerEntry) -> "IdentityClaimLedger":
        return IdentityClaimLedger(entries=(*self.entries, entry))

    def claims_for_user(self, user_id: str) -> tuple[IdentityClaimLedgerEntry, ...]:
        return tuple(entry for entry in self.entries if entry.claimant_user_id == user_id)

    def to_dict(self) -> dict[str, object]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdentityClaimLedger":
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raw_entries = []
        return cls(
            entries=tuple(
                IdentityClaimLedgerEntry.from_dict(item)
                for item in raw_entries
                if isinstance(item, Mapping)
            )
        )

    @classmethod
    def from_entries(cls, entries: Sequence[IdentityClaimLedgerEntry]) -> "IdentityClaimLedger":
        return cls(entries=tuple(entries))
