"""VIA Character Strengths Projection.

M2.7 Phase A — Projects the internal PersonalityParameterSpace onto the
VIA 24-strength taxonomy.  This is a *diagnostic / readability* output,
not an input.  The 24 strengths are derived from the underlying OCEAN traits,
cognitive style, and narrative priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# The 24 VIA strengths grouped by virtue
VIA_STRENGTHS: tuple[str, ...] = (
    # Wisdom
    "creativity", "curiosity", "judgment", "love_of_learning", "perspective",
    # Courage
    "bravery", "perseverance", "honesty", "zest",
    # Humanity
    "love", "kindness", "social_intelligence",
    # Justice
    "teamwork", "fairness", "leadership",
    # Temperance
    "forgiveness", "humility", "prudence", "self_regulation",
    # Transcendence
    "appreciation_of_beauty", "gratitude", "hope", "humor", "spirituality",
)


@dataclass(frozen=True)
class VIAProfile:
    """All 24 VIA strengths on [0, 1]."""
    strengths: dict[str, float]

    def top_strengths(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.strengths.items(), key=lambda kv: -kv[1])[:n]

    def bottom_strengths(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.strengths.items(), key=lambda kv: kv[1])[:n]

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.strengths.items()}


class VIAProjection:
    """Project personality parameters onto the VIA 24-strength space.

    All mappings are deterministic linear combinations of:
    - Big Five traits (O, C, E, A, N)
    - Narrative priors (trust_prior, controllability_prior, etc.)
    - Cognitive style extensions (meaning_construction_tendency,
      emotional_regulation_style)
    """

    def project(
        self,
        *,
        openness: float = 0.5,
        conscientiousness: float = 0.5,
        extraversion: float = 0.5,
        agreeableness: float = 0.5,
        neuroticism: float = 0.5,
        trust_prior: float = 0.0,
        controllability_prior: float = 0.0,
        meaning_construction_tendency: float = 0.5,
        emotional_regulation_style: float = 0.5,
    ) -> VIAProfile:
        o = openness
        c = conscientiousness
        e = extraversion
        a = agreeableness
        n_inv = 1.0 - neuroticism
        tp = (trust_prior + 1.0) / 2.0   # map [-1,1] → [0,1]
        cp = (controllability_prior + 1.0) / 2.0
        mct = meaning_construction_tendency
        ers = emotional_regulation_style

        strengths: dict[str, float] = {}

        # --- Wisdom ---
        strengths["creativity"] = _clamp01(o * 0.5 + e * 0.15 + n_inv * 0.1 + 0.1)
        strengths["curiosity"] = _clamp01(o * 0.45 + e * 0.15 + n_inv * 0.1 + 0.15)
        strengths["judgment"] = _clamp01(o * 0.3 + c * 0.25 + n_inv * 0.15 + 0.15)
        strengths["love_of_learning"] = _clamp01(o * 0.5 + c * 0.2 + 0.15)
        strengths["perspective"] = _clamp01(o * 0.3 + cp * 0.15 + n_inv * 0.2 + mct * 0.1 + 0.1)

        # --- Courage ---
        strengths["bravery"] = _clamp01(n_inv * 0.35 + e * 0.2 + cp * 0.15 + 0.1)
        strengths["perseverance"] = _clamp01(c * 0.45 + n_inv * 0.15 + 0.2)
        strengths["honesty"] = _clamp01(a * 0.3 + c * 0.25 + n_inv * 0.1 + 0.2)
        strengths["zest"] = _clamp01(e * 0.4 + n_inv * 0.2 + o * 0.1 + 0.15)

        # --- Humanity ---
        strengths["love"] = _clamp01(a * 0.3 + e * 0.2 + tp * 0.25 + 0.1)
        strengths["kindness"] = _clamp01(a * 0.45 + tp * 0.2 + 0.15)
        strengths["social_intelligence"] = _clamp01(e * 0.25 + a * 0.2 + o * 0.15 + tp * 0.1 + 0.15)

        # --- Justice ---
        strengths["teamwork"] = _clamp01(a * 0.3 + c * 0.2 + e * 0.15 + tp * 0.1 + 0.1)
        strengths["fairness"] = _clamp01(a * 0.35 + c * 0.2 + n_inv * 0.1 + 0.2)
        strengths["leadership"] = _clamp01(e * 0.3 + c * 0.2 + n_inv * 0.15 + cp * 0.1 + 0.1)

        # --- Temperance ---
        strengths["forgiveness"] = _clamp01(a * 0.4 + n_inv * 0.2 + tp * 0.1 + 0.15)
        strengths["humility"] = _clamp01((1.0 - e) * 0.25 + a * 0.25 + c * 0.1 + 0.2)
        strengths["prudence"] = _clamp01(c * 0.4 + n_inv * 0.1 + (1.0 - o) * 0.1 + 0.2)
        strengths["self_regulation"] = _clamp01(c * 0.4 + n_inv * 0.25 + 0.15)

        # --- Transcendence ---
        strengths["appreciation_of_beauty"] = _clamp01(o * 0.45 + mct * 0.2 + 0.15)
        strengths["gratitude"] = _clamp01(a * 0.3 + n_inv * 0.15 + tp * 0.2 + 0.2)
        strengths["hope"] = _clamp01(n_inv * 0.3 + e * 0.15 + tp * 0.15 + cp * 0.1 + 0.15)
        strengths["humor"] = _clamp01(ers * 0.4 + e * 0.2 + o * 0.1 + 0.15)
        strengths["spirituality"] = _clamp01(mct * 0.45 + o * 0.15 + n_inv * 0.1 + 0.15)

        return VIAProfile(strengths=strengths)
