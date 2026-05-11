"""M11 source-reliability ledger.

This is the only durable latent-trust float state in M11. It is a domain-scoped
Beta posterior over enum judgments with decay toward the documented prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams

ReliabilityStatus = Literal["confirmed", "violated", "uncertain", "pending"]
Modality = Literal["factual", "roleplay", "joke", "hypothetical", "request", "command"]

FACTUAL_MODALITIES = {"factual", "request", "command"}


@dataclass(frozen=True)
class ReliabilityJudgment:
    judgment_id: str
    turn_id: int
    domain: str
    status: ReliabilityStatus
    modality: Modality = "factual"
    evidence_refs: tuple[str, ...] = ()
    evidence_text: str = ""


@dataclass(frozen=True)
class SourceReliability:
    domain: str
    alpha: float
    beta: float
    reliability: float
    last_updated_turn: int
    update_reason: str = "prior"

    def to_dict(self, *, hyperparams: Hyperparams = DEFAULT_HYPERPARAMS) -> dict[str, object]:
        digits = hyperparams.float_round_digits
        return {
            "domain": self.domain,
            "alpha": round(self.alpha, digits),
            "beta": round(self.beta, digits),
            "reliability": round(self.reliability, digits),
            "last_updated_turn": self.last_updated_turn,
            "update_reason": self.update_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceReliability":
        return cls(
            domain=str(payload.get("domain", "")),
            alpha=float(payload.get("alpha", DEFAULT_HYPERPARAMS.prior_alpha)),
            beta=float(payload.get("beta", DEFAULT_HYPERPARAMS.prior_beta)),
            reliability=float(payload.get("reliability", DEFAULT_HYPERPARAMS.prior_mean)),
            last_updated_turn=int(payload.get("last_updated_turn", 0)),
            update_reason=str(payload.get("update_reason", "prior")),
        )


@dataclass(frozen=True)
class ReliabilityUpdate:
    domain: str
    previous: SourceReliability
    updated: SourceReliability
    applied_statuses: tuple[str, ...]
    ignored_judgment_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "previous": self.previous.to_dict(),
            "updated": self.updated.to_dict(),
            "applied_statuses": list(self.applied_statuses),
            "ignored_judgment_ids": list(self.ignored_judgment_ids),
        }


@dataclass(frozen=True)
class SourceReliabilityLedger:
    entries_by_domain: dict[str, SourceReliability]

    @classmethod
    def empty(cls) -> "SourceReliabilityLedger":
        return cls(entries_by_domain={})

    def reliability_for(
        self,
        domain: str,
        *,
        hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
    ) -> SourceReliability:
        existing = self.entries_by_domain.get(domain)
        if existing is not None:
            return existing
        return SourceReliability(
            domain=domain,
            alpha=hyperparams.prior_alpha,
            beta=hyperparams.prior_beta,
            reliability=hyperparams.prior_mean,
            last_updated_turn=0,
            update_reason="prior",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            domain: entry.to_dict()
            for domain, entry in sorted(self.entries_by_domain.items())
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceReliabilityLedger":
        return cls(
            entries_by_domain={
                str(domain): SourceReliability.from_dict(row)
                for domain, row in payload.items()
                if isinstance(row, Mapping)
            }
        )


def update_reliability(
    ledger: SourceReliabilityLedger,
    judgments: Sequence[ReliabilityJudgment],
    *,
    current_turn_id: int,
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> tuple[SourceReliabilityLedger, tuple[ReliabilityUpdate, ...]]:
    grouped: dict[str, list[ReliabilityJudgment]] = {}
    ignored_by_domain: dict[str, list[str]] = {}
    for judgment in judgments:
        grouped.setdefault(judgment.domain, [])
        ignored_by_domain.setdefault(judgment.domain, [])
        if judgment.modality not in FACTUAL_MODALITIES or judgment.status not in {"confirmed", "violated"}:
            ignored_by_domain[judgment.domain].append(judgment.judgment_id)
            continue
        grouped[judgment.domain].append(judgment)

    domains = set(ledger.entries_by_domain) | set(grouped) | set(ignored_by_domain)
    next_entries = dict(ledger.entries_by_domain)
    updates: list[ReliabilityUpdate] = []
    for domain in sorted(domains):
        previous = ledger.reliability_for(domain, hyperparams=hyperparams)
        factual = tuple(grouped.get(domain, ()))
        alpha, beta = _decayed_counts(previous, current_turn_id, hyperparams)
        for judgment in factual:
            if judgment.status == "confirmed":
                alpha += hyperparams.confirm_weight
            elif judgment.status == "violated":
                beta += hyperparams.violate_weight
        raw_reliability = alpha / (alpha + beta)
        reliability = (
            _clamp_delta(previous.reliability, raw_reliability, hyperparams.max_delta_per_turn)
            if factual
            else raw_reliability
        )
        reason = "judgment_update" if factual else "silence_decay" if previous.last_updated_turn != current_turn_id else "no_change"
        updated = SourceReliability(
            domain=domain,
            alpha=alpha,
            beta=beta,
            reliability=reliability,
            last_updated_turn=current_turn_id,
            update_reason=reason,
        )
        next_entries[domain] = updated
        updates.append(
            ReliabilityUpdate(
                domain=domain,
                previous=previous,
                updated=updated,
                applied_statuses=tuple(j.status for j in factual),
                ignored_judgment_ids=tuple(ignored_by_domain.get(domain, ())),
            )
        )
    return SourceReliabilityLedger(next_entries), tuple(updates)


def _decayed_counts(
    previous: SourceReliability,
    current_turn_id: int,
    hyperparams: Hyperparams,
) -> tuple[float, float]:
    gap = max(current_turn_id - previous.last_updated_turn, 0)
    if gap == 0:
        return previous.alpha, previous.beta
    retention = 0.5 ** (gap / hyperparams.reliability_half_life_turns)
    alpha = hyperparams.prior_alpha + (previous.alpha - hyperparams.prior_alpha) * retention
    beta = hyperparams.prior_beta + (previous.beta - hyperparams.prior_beta) * retention
    return alpha, beta


def _clamp_delta(previous: float, candidate: float, max_delta: float) -> float:
    lower = previous - max_delta
    upper = previous + max_delta
    return min(max(candidate, lower), upper)
