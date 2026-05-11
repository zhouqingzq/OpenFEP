"""M11 user-model hyperparameters.

The M11 deterministic layer is intentionally boring: all thresholds, weights,
quotas, and priors live here so ledger code can be audited without hunting for
inline constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hyperparams:
    """Operational constants for M11 deterministic updates."""

    max_evidence_refs_per_hypothesis: int = field(
        default=6,
        metadata={"doc": "Maximum bounded turn/quote references retained per hypothesis."},
    )
    max_hypotheses_per_bucket: int = field(
        default=12,
        metadata={"doc": "Maximum active hypotheses retained in each user-model bucket."},
    )
    high_confidence_min_evidence_refs: int = field(
        default=3,
        metadata={"doc": "Minimum supporting references before a hypothesis may reach high confidence."},
    )
    high_confidence_min_distinct_turns: int = field(
        default=2,
        metadata={"doc": "Minimum distinct turns before one utterance can no longer dominate promotion."},
    )
    med_confidence_min_evidence_refs: int = field(
        default=2,
        metadata={"doc": "Minimum supporting references before low evidence may become medium confidence."},
    )
    proposal_quota_per_turn: int = field(
        default=3,
        metadata={"doc": "Maximum prediction proposals the deterministic gate admits for review each turn."},
    )
    default_prediction_expiry_turns: int = field(
        default=1,
        metadata={"doc": "Fallback lifetime for accepted prediction proposals when no expiry is supplied."},
    )
    prior_alpha: float = field(
        default=2.0,
        metadata={"doc": "Beta prior confirmed-equivalent count for a new source-reliability domain."},
    )
    prior_beta: float = field(
        default=2.0,
        metadata={"doc": "Beta prior violated-equivalent count for a new source-reliability domain."},
    )
    confirm_weight: float = field(
        default=0.7,
        metadata={"doc": "Confirmed judgment increment for the source-reliability Beta posterior."},
    )
    violate_weight: float = field(
        default=1.4,
        metadata={"doc": "Violated judgment increment; larger than confirmations so detected lies cost more."},
    )
    reliability_half_life_turns: int = field(
        default=20,
        metadata={"doc": "Turns of silence needed to halve distance from the source-reliability prior."},
    )
    max_delta_per_turn: float = field(
        default=0.12,
        metadata={"doc": "Clamp on source-reliability movement attributable to a single turn update."},
    )
    band_prior_low: float = field(
        default=0.2,
        metadata={"doc": "Composer prior value for low memory-value band."},
    )
    band_prior_med: float = field(
        default=0.5,
        metadata={"doc": "Composer prior value for medium memory-value band."},
    )
    band_prior_high: float = field(
        default=0.8,
        metadata={"doc": "Composer prior value for high memory-value band."},
    )
    trust_factor_base: float = field(
        default=0.5,
        metadata={"doc": "Minimum trust multiplier applied before reliability contributes."},
    )
    trust_factor_scale: float = field(
        default=0.5,
        metadata={"doc": "Reliability multiplier scale, keeping trust factor inside a bounded range."},
    )
    contradiction_penalty: float = field(
        default=0.4,
        metadata={"doc": "Value penalty for unresolved contradictions."},
    )
    privacy_penalty: float = field(
        default=0.6,
        metadata={"doc": "Dominant value penalty when privacy or safety policy flags a candidate."},
    )
    long_term_threshold: float = field(
        default=0.62,
        metadata={"doc": "Minimum composed value for long-term user-model promotion."},
    )
    patch_threshold: float = field(
        default=0.35,
        metadata={"doc": "Minimum composed value for a bounded user-model patch."},
    )
    short_term_threshold: float = field(
        default=0.12,
        metadata={"doc": "Minimum composed value for short-term retention."},
    )
    recency_half_life_turns: int = field(
        default=8,
        metadata={"doc": "Turns needed for recency weight to halve when composing value."},
    )
    low_reliability_threshold: float = field(
        default=0.45,
        metadata={"doc": "Below this reliability, reply policy softens factual language for the domain."},
    )
    strong_reliability_threshold: float = field(
        default=0.65,
        metadata={"doc": "Above this reliability, stable preferences can affect reply policy."},
    )
    brevity_hypothesis_min_refs: int = field(
        default=3,
        metadata={"doc": "Minimum evidence refs before brevity preference can shorten replies."},
    )
    float_round_digits: int = field(
        default=6,
        metadata={"doc": "Canonical decimal precision for reproducible diagnostic floats."},
    )

    @property
    def prior_mean(self) -> float:
        return self.prior_alpha / (self.prior_alpha + self.prior_beta)

    def doc_table(self) -> dict[str, str]:
        return {
            name: str(field_info.metadata.get("doc", ""))
            for name, field_info in self.__dataclass_fields__.items()
        }


DEFAULT_HYPERPARAMS = Hyperparams()
