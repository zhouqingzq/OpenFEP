"""M12.0 deterministic constants.

All decision constants for identity continuity live here to avoid hidden
behavior changes and inline magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class M12Hyperparams:
    """Operational constants for M12 identity continuity."""

    max_alias_observations: int = 32
    max_continuity_cues: int = 48
    max_relationship_facts: int = 32
    max_claims_per_turn: int = 12
    max_cues_per_turn: int = 24
    bind_promotion_threshold: int = 3
    """Minimum med/high `binds` cue count to raise binding band."""
    min_distinct_turns_for_binding_from_cues: int = 2
    """Binds must appear on at least this many distinct turns before cue-only promotion."""
    contradict_demote_threshold: int = 2
    high_contradiction_window: int = 3
    strangeness_signal_rate_limit_per_turn: int = 1
    strangeness_ttl_turns: int = 2
    strangeness_budget_cost: int = 1
    forbidden_user_facing_tokens: tuple[str, ...] = field(
        default_factory=lambda: (
            "prediction error",
            "free energy",
            "bayesian",
            "posterior",
            "预测误差",
            "自由能",
            "贝叶斯",
            "后验",
            "identity_conflict",
            "identitystrangenesssignal",
            "conflictrecord",
        )
    )


DEFAULT_HYPERPARAMS = M12Hyperparams()
