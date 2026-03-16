"""Precision Manipulation Engine: computational basis for defense mechanisms.

M2.7 Phase A — When prediction error is too large and direct model update
(accommodate) is too costly, the system manipulates signal-channel precision
weights to reduce free energy.  Different manipulation modes correspond to
different classical defense mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SUPPRESS_INTENSITY: float = 0.3
PRECISION_DEBT_DECAY: float = 0.95
PRECISION_DEBT_ACCUMULATION: float = 0.1

# Channel names mirroring predictive_coding modalities + social extensions
DEFAULT_CHANNELS: tuple[str, ...] = (
    "food", "danger", "novelty", "shelter", "temperature", "social",
    "attachment", "self_worth", "threat",
)


class ManipulationType(str, Enum):
    """Kinds of precision manipulation — each maps to a defense family."""
    SUPPRESS = "suppress"    # Denial / Repression: lower target precision
    AMPLIFY = "amplify"      # Anxious hyper-attention: raise target precision
    REDIRECT = "redirect"    # Projection / Sublimation: shift precision
    REFRAME = "reframe"      # Rationalization: change valence encoding


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreeEnergyCost:
    """Short-term and projected long-term free-energy cost of a manipulation."""
    short_term: float
    long_term_projected: float


@dataclass(frozen=True)
class PrecisionManipulationResult:
    """Outcome of a single precision manipulation operation."""
    manipulation_type: str
    target_channel: str
    intensity: float
    precision_before: float
    precision_after: float
    free_energy_delta: float  # negative ⇒ FE decreased (beneficial short-term)
    cost: FreeEnergyCost
    source_channel: str | None = None  # only for redirect


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PrecisionManipulator:
    """Precision manipulation engine — the unified computational basis for
    defense mechanisms.

    Each ``apply_manipulation`` call adjusts a channel's precision weight and
    logs the operation so that MetaCognitiveLayer can later observe the
    pattern.
    """

    def __init__(
        self,
        *,
        neuroticism: float = 0.5,
        openness: float = 0.5,
        extraversion: float = 0.5,
        agreeableness: float = 0.5,
        conscientiousness: float = 0.5,
        trust_prior: float = 0.0,
    ) -> None:
        self.neuroticism = neuroticism
        self.openness = openness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.conscientiousness = conscientiousness
        self.trust_prior = trust_prior

        # Current precision weights per channel (all start at 1.0)
        self.channel_precisions: dict[str, float] = {
            ch: 1.0 for ch in DEFAULT_CHANNELS
        }

        # Accumulated precision debt from suppress operations
        self.precision_debt: float = 0.0

        # History of manipulations (for metacognitive observation)
        self.manipulation_history: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_channel_precisions(
        self,
        personality_state: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute personality-modulated precision weights.

        High Neuroticism → threat/danger precision ↑, safety signals ↓
        High Openness    → novelty precision ↑
        High trust_prior → social/attachment precision ↑
        """
        n = self.neuroticism - 0.5
        o = self.openness - 0.5
        e = self.extraversion - 0.5
        tp = max(-1.0, min(1.0, self.trust_prior))

        base = dict(self.channel_precisions)
        # Neuroticism effects
        base["danger"] = max(0.05, base.get("danger", 1.0) + n * 0.4)
        base["threat"] = max(0.05, base.get("threat", 1.0) + n * 0.35)
        base["social"] = max(0.05, base.get("social", 1.0) - n * 0.2)
        base["self_worth"] = max(0.05, base.get("self_worth", 1.0) - n * 0.15)

        # Openness effects
        base["novelty"] = max(0.05, base.get("novelty", 1.0) + o * 0.3)

        # Trust / social
        base["attachment"] = max(0.05, base.get("attachment", 1.0) + tp * 0.3)
        base["social"] = max(0.05, base["social"] + e * 0.15 + tp * 0.2)

        return base

    def apply_manipulation(
        self,
        manipulation_type: str | ManipulationType,
        target_channel: str,
        intensity: float,
        *,
        source_channel: str | None = None,
        cycle: int = 0,
    ) -> PrecisionManipulationResult:
        """Execute a precision manipulation and return the result.

        Parameters
        ----------
        manipulation_type
            One of suppress / amplify / redirect / reframe.
        target_channel
            Channel whose precision is being manipulated.
        intensity
            Strength of manipulation in [0, 1].
        source_channel
            For ``redirect`` only — precision is *moved* from source to target.
        cycle
            Current decision cycle (for history logging).
        """
        mt = ManipulationType(manipulation_type)
        intensity = max(0.0, min(1.0, intensity))

        # Ensure target exists
        if target_channel not in self.channel_precisions:
            self.channel_precisions[target_channel] = 1.0

        old_precision = self.channel_precisions[target_channel]

        if mt is ManipulationType.SUPPRESS:
            delta = -intensity * DEFAULT_SUPPRESS_INTENSITY
            self.channel_precisions[target_channel] = max(
                0.05, old_precision + delta
            )
            self.precision_debt += intensity * PRECISION_DEBT_ACCUMULATION

        elif mt is ManipulationType.AMPLIFY:
            delta = intensity * 0.25
            self.channel_precisions[target_channel] = min(
                2.0, old_precision + delta
            )

        elif mt is ManipulationType.REDIRECT:
            if source_channel is None:
                source_channel = target_channel
            if source_channel not in self.channel_precisions:
                self.channel_precisions[source_channel] = 1.0
            transfer = intensity * 0.2
            self.channel_precisions[source_channel] = max(
                0.05, self.channel_precisions[source_channel] - transfer
            )
            self.channel_precisions[target_channel] = min(
                2.0, old_precision + transfer
            )

        elif mt is ManipulationType.REFRAME:
            # Reframe doesn't change precision magnitude — it adjusts the
            # effective valence.  We model this as a small precision nudge
            # reflecting reduced salience of the original signal.
            delta = -intensity * 0.1
            self.channel_precisions[target_channel] = max(
                0.05, old_precision + delta
            )

        new_precision = self.channel_precisions[target_channel]
        cost = self.compute_manipulation_cost(mt, intensity)

        # Free energy delta estimate: lowering precision on a high-PE channel
        # reduces weighted PE (beneficial short-term).
        fe_delta = (new_precision - old_precision) * 0.5  # simplified proxy

        result = PrecisionManipulationResult(
            manipulation_type=mt.value,
            target_channel=target_channel,
            intensity=intensity,
            precision_before=round(old_precision, 6),
            precision_after=round(new_precision, 6),
            free_energy_delta=round(fe_delta, 6),
            cost=cost,
            source_channel=source_channel if mt is ManipulationType.REDIRECT else None,
        )

        self.manipulation_history.append({
            "cycle": cycle,
            "type": mt.value,
            "target": target_channel,
            "intensity": round(intensity, 4),
            "precision_before": round(old_precision, 6),
            "precision_after": round(new_precision, 6),
            "precision_debt": round(self.precision_debt, 6),
        })

        return result

    def compute_manipulation_cost(
        self,
        manipulation_type: str | ManipulationType,
        intensity: float,
    ) -> FreeEnergyCost:
        """Cost of a precision manipulation.

        Suppress is cheapest short-term but most expensive long-term due to
        precision debt accumulation.
        """
        mt = ManipulationType(manipulation_type)
        intensity = max(0.0, min(1.0, intensity))

        if mt is ManipulationType.SUPPRESS:
            short = intensity * 0.05
            long = intensity * 0.3 + self.precision_debt * 0.15
        elif mt is ManipulationType.AMPLIFY:
            short = intensity * 0.15
            long = intensity * 0.10
        elif mt is ManipulationType.REDIRECT:
            short = intensity * 0.12
            long = intensity * 0.12
        else:  # REFRAME
            short = intensity * 0.10
            long = intensity * 0.08

        return FreeEnergyCost(
            short_term=round(short, 6),
            long_term_projected=round(long, 6),
        )

    def decay_precision_debt(self) -> float:
        """Apply natural decay to precision debt (called once per cycle)."""
        self.precision_debt *= PRECISION_DEBT_DECAY
        return self.precision_debt

    def update_personality(
        self,
        *,
        neuroticism: float | None = None,
        openness: float | None = None,
        extraversion: float | None = None,
        agreeableness: float | None = None,
        conscientiousness: float | None = None,
        trust_prior: float | None = None,
    ) -> None:
        """Sync personality parameters (called when PersonalityProfile updates)."""
        if neuroticism is not None:
            self.neuroticism = neuroticism
        if openness is not None:
            self.openness = openness
        if extraversion is not None:
            self.extraversion = extraversion
        if agreeableness is not None:
            self.agreeableness = agreeableness
        if conscientiousness is not None:
            self.conscientiousness = conscientiousness
        if trust_prior is not None:
            self.trust_prior = trust_prior

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_precisions": dict(self.channel_precisions),
            "precision_debt": round(self.precision_debt, 6),
            "neuroticism": self.neuroticism,
            "openness": self.openness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "conscientiousness": self.conscientiousness,
            "trust_prior": self.trust_prior,
            "manipulation_history": list(self.manipulation_history[-64:]),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PrecisionManipulator":
        if not payload:
            return cls()
        pm = cls(
            neuroticism=float(payload.get("neuroticism", 0.5)),
            openness=float(payload.get("openness", 0.5)),
            extraversion=float(payload.get("extraversion", 0.5)),
            agreeableness=float(payload.get("agreeableness", 0.5)),
            conscientiousness=float(payload.get("conscientiousness", 0.5)),
            trust_prior=float(payload.get("trust_prior", 0.0)),
        )
        raw_precisions = payload.get("channel_precisions")
        if isinstance(raw_precisions, dict):
            pm.channel_precisions = {
                str(k): float(v) for k, v in raw_precisions.items()
            }
        pm.precision_debt = float(payload.get("precision_debt", 0.0))
        raw_history = payload.get("manipulation_history")
        if isinstance(raw_history, list):
            pm.manipulation_history = [
                dict(entry) for entry in raw_history if isinstance(entry, dict)
            ]
        return pm
