"""M12.1 deterministic constants.

All constants that affect personality-profile patching, trigger decisions,
plain-language linting, and evidence-card ordering live here.  The module is
data-only so replay fixtures remain byte-identical across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class M121Hyperparams:
    """Operational constants for M12.1 user personality modeling."""

    hyperparams_version: str = "m12.1.v1"
    """Version stamped onto profiles, reports, cards, and acceptance traces."""

    max_transcript_quote_refs: int = 12
    """Maximum turn-window quote references visible to any step extractor."""

    max_m11_hypothesis_refs: int = 12
    """Maximum read-only M11 hypothesis summaries copied into a step snapshot."""

    max_m12_continuity_cues: int = 12
    """Maximum read-only M12.0 continuity cues copied into a step snapshot."""

    max_evidence_refs_per_claim: int = 6
    """Maximum quote references retained for a single profile claim."""

    max_evidence_items: int = 12
    """Maximum Step-2 evidence items retained per profile."""

    max_defenses: int = 6
    """Maximum Step-5 defenses retained per profile."""

    max_card_count: int = 8
    """Maximum M12.1 evidence cards exposed to generation."""

    max_summary_chars: int = 180
    """Hard bound for user-facing summary fields."""

    max_reason_chars: int = 160
    """Hard bound for insufficient-evidence and linter diagnostic reasons."""

    cadence_turn_interval: int = 8
    """Turn-count cadence for slow refreshes when enough evidence exists."""

    new_evidence_threshold: int = 3
    """New continuity-cue count that can trigger a refresh before cadence."""

    strangeness_high_turn_threshold: int = 2
    """Consecutive high-strangeness turns required for follow-up refresh."""

    max_personality_runs_per_hour: int = 2
    """Per-user cap for trigger-approved M12.1 runs."""

    step1_insufficient_suspend_count: int = 2
    """Cadence is suspended after this many consecutive Step-1 sparse runs."""

    high_confidence_min_evidence_refs: int = 3
    """High confidence requires at least this many evidence references."""

    high_confidence_min_distinct_turns: int = 2
    """High confidence requires evidence from at least this many turns."""

    med_confidence_min_evidence_refs: int = 2
    """Medium confidence requires at least this many evidence references."""

    per_run_band_promotion_levels: int = 1
    """A single run may raise confidence by at most this many ordinal levels."""

    card_confidence_float_low: float = 0.33
    """Compatibility float mirrored from low confidence-band cards."""

    card_confidence_float_med: float = 0.66
    """Compatibility float mirrored from med confidence-band cards."""

    card_confidence_float_high: float = 0.9
    """Compatibility float mirrored from high confidence-band cards."""

    trigger_calendar_window: tuple[int, int] | None = None
    """Optional inclusive weekday window; default None means no calendar gate."""

    forbidden_user_facing_tokens_extra: tuple[str, ...] = field(
        default_factory=lambda: (
            "predict",
            "prediction",
            "model",
            "prediction error",
            "free energy",
            "bayesian",
            "posterior",
            "prior update",
            "predictive system",
            "compress error",
            "预测",
            "模型",
            "误差",
            "预测误差",
            "自由能",
            "贝叶斯",
            "后验",
            "先验更新",
            "预测系统",
            "棰勬祴",
            "妯″瀷",
            "璇樊",
            "棰勬祴璇樊",
            "鑷敱鑳",
            "璐濆彾鏂",
            "鍚庨獙",
            "棰勬祴绯荤粺",
        )
    )
    """User-facing engineering jargon forbidden by Principle 7."""

    forbidden_clinical_label_tokens: tuple[str, ...] = field(
        default_factory=lambda: (
            "narcissist",
            "narcissistic personality",
            "borderline",
            "borderline personality",
            "autistic",
            "schizoid",
            "schizotypal",
            "bipolar",
            "adhd",
            "ptsd",
            "ocd",
            "depression disorder",
            "anxiety disorder",
            "自恋型",
            "边缘型",
            "自闭",
            "抑郁症",
            "焦虑症",
            "双相",
            "强迫症",
            "鑷亱鍨",
            "杈圭紭鎬",
            "鑷棴",
            "鎶戦儊鐥",
            "鐒﹁檻鐥",
            "鍙岀浉",
            "寮鸿揩鐥",
        )
    )
    """Clinical and DSM/ICD-style labels banned from user-facing surfaces."""

    forbidden_moral_or_chicken_soup_tokens: tuple[str, ...] = field(
        default_factory=lambda: (
            "should",
            "ought",
            "shouldn't",
            "everything will be fine",
            "needs to love himself",
            "needs to love herself",
            "应该",
            "本该",
            "都会好起来",
            "需要爱自己",
            "搴旇",
            "鏈",
            "閮戒細濂借捣鏉",
            "闇€瑕佺埍鑷繁",
        )
    )
    """Moral verdicts and chicken-soup filler banned from visible text."""


DEFAULT_HYPERPARAMS = M121Hyperparams()

SECTION_KINDS: tuple[str, ...] = (
    "step_1",
    "step_2",
    "step_3",
    "step_4",
    "step_5",
    "step_6",
    "step_7",
    "step_8",
)

CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "med": 1, "high": 2}
CONFIDENCE_BY_LEVEL: tuple[str, ...] = ("low", "med", "high")
PERMITTED_PRIORITY: dict[str, int] = {
    "explicit_fact": 0,
    "cautious_hypothesis": 1,
    "strategy_only": 2,
    "forbidden": 3,
}
CARD_CONFIDENCE_PRIORITY: dict[str, int] = {"high": 0, "med": 1, "low": 2}
