"""Deterministic M11 memory-value composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .hyperparams import DEFAULT_HYPERPARAMS, Hyperparams

ConfidenceBand = Literal["low", "med", "high"]
WriteTarget = Literal["none", "short_term", "user_model_patch", "long_term_user_model"]


@dataclass(frozen=True)
class ValueComposition:
    value_score: float
    write_target: WriteTarget
    gating_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "value_score": self.value_score,
            "write_target": self.write_target,
            "gating_reasons": list(self.gating_reasons),
        }


def recency_weight(
    *,
    current_turn_id: int,
    evidence_turn_id: int,
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> float:
    distance = max(current_turn_id - evidence_turn_id, 0)
    return round(0.5 ** (distance / hyperparams.recency_half_life_turns), hyperparams.float_round_digits)


def compose_value(
    *,
    memory_value_band: ConfidenceBand,
    confidence_band: ConfidenceBand,
    source_reliability: float,
    recency_weight: float,
    contradiction_unresolved: bool,
    privacy_or_safety_flag: bool,
    hyperparams: Hyperparams = DEFAULT_HYPERPARAMS,
) -> ValueComposition:
    band_prior = _band_prior(memory_value_band, hyperparams)
    trust_factor = hyperparams.trust_factor_base + (hyperparams.trust_factor_scale * _clamp(source_reliability))
    penalty = 0.0
    reasons = [
        f"band_prior:{memory_value_band}",
        "source_reliability_factor",
        "recency_weight",
    ]
    if contradiction_unresolved:
        penalty += hyperparams.contradiction_penalty
        reasons.append("contradiction_penalty")
    if privacy_or_safety_flag:
        penalty += hyperparams.privacy_penalty
        reasons.append("privacy_or_safety_penalty")
    score = _clamp((band_prior * trust_factor * _clamp(recency_weight)) - penalty)
    score = round(score, hyperparams.float_round_digits)
    if privacy_or_safety_flag:
        target: WriteTarget = "user_model_patch" if score >= hyperparams.patch_threshold else "short_term" if score >= hyperparams.short_term_threshold else "none"
    elif score >= hyperparams.long_term_threshold and confidence_band == "high":
        target = "long_term_user_model"
    elif score >= hyperparams.patch_threshold:
        target = "user_model_patch"
    elif score >= hyperparams.short_term_threshold:
        target = "short_term"
    else:
        target = "none"
    reasons.append(f"write_target:{target}")
    return ValueComposition(value_score=score, write_target=target, gating_reasons=tuple(reasons))


def _band_prior(memory_value_band: ConfidenceBand, hyperparams: Hyperparams) -> float:
    if memory_value_band == "high":
        return hyperparams.band_prior_high
    if memory_value_band == "med":
        return hyperparams.band_prior_med
    return hyperparams.band_prior_low


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
