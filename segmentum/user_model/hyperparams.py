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
    reaction_expectation_unverified_fe: float = field(
        default=0.64,
        metadata={"doc": "Free energy carried by an unverified expectation about another user's reaction."},
    )
    reaction_expectation_verified_fe: float = field(
        default=0.22,
        metadata={"doc": "Residual free energy when the reaction expectation is already well verified."},
    )
    reaction_expectation_violated_fe: float = field(
        default=0.82,
        metadata={"doc": "Free energy after a recent violation of the predicted social reaction."},
    )
    reaction_expectation_incomprehensible_fe: float = field(
        default=0.95,
        metadata={"doc": "Free energy for reaction evidence that current cognition cannot explain."},
    )
    direct_share_resolution_ratio: float = field(
        default=0.62,
        metadata={"doc": "Fraction of reaction-expectation free energy expected to resolve after direct sharing."},
    )
    abstract_share_resolution_ratio: float = field(
        default=0.38,
        metadata={"doc": "Fraction of reaction-expectation free energy expected to resolve after abstract sharing."},
    )
    abstract_boundary_cost_ratio: float = field(
        default=0.25,
        metadata={"doc": "Remaining boundary cost after removing identifying details through abstraction."},
    )
    default_social_boundary_cost: float = field(
        default=0.03,
        metadata={"doc": "Small boundary cost when the source user declared no sharing constraint."},
    )
    restricted_implicit_boundary_cost: float = field(
        default=0.32,
        metadata={"doc": "Boundary cost for soft or implicit constraints where abstraction is usually preferred."},
    )
    restricted_explicit_boundary_cost: float = field(
        default=1.0,
        metadata={"doc": "Hard-blocking boundary cost for explicit secrecy or forbidden visibility."},
    )
    social_share_relationship_cost_base: float = field(
        default=0.10,
        metadata={"doc": "Baseline relationship free-energy cost before boundary and regret feedback are applied."},
    )
    direct_share_fe_reduction_threshold: float = field(
        default=0.0,
        metadata={"doc": "Minimum net expected free-energy reduction required for direct retelling."},
    )
    abstract_share_fe_reduction_threshold: float = field(
        default=0.0,
        metadata={"doc": "Minimum net expected free-energy reduction required for abstract reference."},
    )
    sharing_regret_feedback_increment: float = field(
        default=0.18,
        metadata={"doc": "Relationship-cost increase after negative feedback on cross-user sharing."},
    )
    sharing_regret_feedback_decay: float = field(
        default=0.03,
        metadata={"doc": "Small relationship-cost decay when a turn does not reinforce the sharing concern."},
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
