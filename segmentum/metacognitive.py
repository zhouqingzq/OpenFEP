"""MetaCognitive Layer: observing internal precision manipulation patterns.

M2.7 Phase B — The metacognitive layer's observation target is not the
external world but the agent's own precision-weighting patterns and defense
strategy selections.  It detects repetitive defense patterns, computes
meta-prediction errors, and generates cognitive dissociation signals that
can loosen rigidified beliefs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_PE_THRESHOLD: float = 0.5
DISSOCIATION_BELIEF_VARIANCE_BOOST: float = 0.005
OBSERVATION_WINDOW: int = 20
PATTERN_DETECTION_MIN_FREQUENCY: float = 0.6


# ---------------------------------------------------------------------------
# Observation containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionPatternObservation:
    """Result of observing precision manipulation patterns."""
    channel_suppression_counts: dict[str, int]
    chronic_suppression_channels: list[str]   # channels suppressed ≥ threshold
    suppression_frequency: float              # fraction of recent cycles with suppress
    debt_trend: float                         # positive ⇒ debt growing
    total_manipulations: int


@dataclass(frozen=True)
class StrategyPatternObservation:
    """Result of observing defense strategy selection patterns."""
    strategy_counts: dict[str, int]
    dominant_strategy: str
    dominant_frequency: float
    diversity: float           # 0 = always same strategy, 1 = uniform
    accommodate_frequency: float
    suppress_frequency: float


@dataclass(frozen=True)
class MetaPredictionError:
    """Meta-PE: discrepancy between 'I should be learning & growing'
    and the actual observed pattern of repeated suppression / rigidity."""
    magnitude: float
    pattern_description: str
    dominant_maladaptive_pattern: str
    accommodate_deficit: float     # how far accommodate% is below healthy baseline
    suggested_intervention: str


@dataclass(frozen=True)
class DissociationSignal:
    """Cognitive dissociation: 'I notice I am doing X' rather than
    'X is just how reality is.'"""
    strength: float               # [0, 1]
    affected_channels: list[str]  # channels whose beliefs could be loosened
    suppress_efe_penalty: float   # extra EFE cost added to suppress strategy
    belief_variance_boost: float  # amount to add to belief variance
    description: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetaCognitiveLayer:
    """Metacognitive layer — observes the agent's own precision manipulation
    and defense strategy patterns.

    Provides the computational substrate for:
    - ACT-style cognitive defusion / dissociation
    - Mindfulness: non-judgmental monitoring of internal PE flow
    - Breaking vicious cycles by making suppress costly once pattern detected
    """

    def __init__(self, observation_window_size: int = OBSERVATION_WINDOW) -> None:
        self.observation_window_size = observation_window_size
        self.precision_history: list[dict[str, object]] = []
        self.strategy_history: list[dict[str, object]] = []
        self.meta_beliefs: dict[str, object] = {}
        self.dissociation_level: float = 0.0
        self._meta_pe_history: list[float] = []
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_precision_pattern(
        self,
        manipulation_record: dict[str, object],
    ) -> PrecisionPatternObservation:
        """Record and analyze a precision manipulation event."""
        self.precision_history.append(dict(manipulation_record))
        # Trim to window
        if len(self.precision_history) > self.observation_window_size * 3:
            self.precision_history = self.precision_history[-self.observation_window_size * 3:]

        window = self.precision_history[-self.observation_window_size:]

        # Count suppressions per channel
        suppression_counts: dict[str, int] = {}
        total_suppress = 0
        for record in window:
            if record.get("type") == "suppress":
                ch = str(record.get("target", "unknown"))
                suppression_counts[ch] = suppression_counts.get(ch, 0) + 1
                total_suppress += 1

        n = max(1, len(window))
        suppress_freq = total_suppress / n

        # Chronic suppression: channel suppressed in ≥ threshold of window
        threshold_count = int(self.observation_window_size * PATTERN_DETECTION_MIN_FREQUENCY)
        chronic = [
            ch for ch, count in suppression_counts.items()
            if count >= max(2, threshold_count)
        ]

        # Debt trend: compare first half to second half
        debts = [float(r.get("precision_debt", 0.0)) for r in window if "precision_debt" in r]
        if len(debts) >= 4:
            mid = len(debts) // 2
            debt_trend = (sum(debts[mid:]) / max(1, len(debts[mid:]))) - \
                         (sum(debts[:mid]) / max(1, len(debts[:mid])))
        else:
            debt_trend = 0.0

        return PrecisionPatternObservation(
            channel_suppression_counts=suppression_counts,
            chronic_suppression_channels=chronic,
            suppression_frequency=round(suppress_freq, 4),
            debt_trend=round(debt_trend, 6),
            total_manipulations=len(window),
        )

    def observe_strategy_pattern(
        self,
        strategy_record: dict[str, object],
    ) -> StrategyPatternObservation:
        """Record and analyze a defense strategy selection."""
        self.strategy_history.append(dict(strategy_record))
        if len(self.strategy_history) > self.observation_window_size * 3:
            self.strategy_history = self.strategy_history[-self.observation_window_size * 3:]

        window = self.strategy_history[-self.observation_window_size:]
        n = max(1, len(window))

        counts: dict[str, int] = Counter(
            str(r.get("strategy", "unknown")) for r in window
        )
        total = sum(counts.values())

        dominant = max(counts, key=counts.get) if counts else "unknown"
        dominant_freq = counts.get(dominant, 0) / max(1, total)

        # Shannon diversity (normalised)
        import math
        diversity = 0.0
        if total > 0:
            for c in counts.values():
                p = c / total
                if p > 0:
                    diversity -= p * math.log2(p)
            max_entropy = math.log2(max(1, len(counts)))
            diversity = diversity / max(1e-9, max_entropy) if max_entropy > 0 else 0.0

        accommodate_freq = counts.get("accommodate", 0) / max(1, total)
        suppress_freq = counts.get("suppress", 0) / max(1, total)

        return StrategyPatternObservation(
            strategy_counts=dict(counts),
            dominant_strategy=dominant,
            dominant_frequency=round(dominant_freq, 4),
            diversity=round(diversity, 4),
            accommodate_frequency=round(accommodate_freq, 4),
            suppress_frequency=round(suppress_freq, 4),
        )

    # ------------------------------------------------------------------
    # Meta-PE computation
    # ------------------------------------------------------------------

    def compute_meta_prediction_error(self) -> MetaPredictionError:
        """Compute the meta-level prediction error.

        Meta-prediction: "I should be learning and growing" (≈ accommodate
        at a healthy baseline of ~25%).
        Meta-observation: actual strategy distribution.
        Meta-PE: discrepancy between the two.
        """
        window = self.strategy_history[-self.observation_window_size:]
        n = max(1, len(window))

        counts = Counter(str(r.get("strategy", "unknown")) for r in window)
        total = sum(counts.values())

        accommodate_freq = counts.get("accommodate", 0) / max(1, total)
        suppress_freq = counts.get("suppress", 0) / max(1, total)

        # Healthy baseline: ~25% accommodate
        accommodate_deficit = max(0.0, 0.25 - accommodate_freq)

        # Meta-PE driven by suppress dominance + accommodate deficit
        magnitude = suppress_freq * 0.6 + accommodate_deficit * 0.8

        # Identify dominant maladaptive pattern
        if suppress_freq >= PATTERN_DETECTION_MIN_FREQUENCY:
            pattern = "chronic_suppression"
            desc = (
                f"Suppress used {suppress_freq:.0%} of the time "
                f"(accommodate only {accommodate_freq:.0%}). "
                f"Chronic precision suppression detected."
            )
            intervention = "Increase suppress EFE cost to encourage accommodate."
        elif accommodate_freq < 0.05 and n >= 5:
            pattern = "belief_rigidity"
            desc = (
                f"Accommodate at {accommodate_freq:.0%} — beliefs are rigidified. "
                f"No model updating occurring."
            )
            intervention = "Loosen belief variance to make accommodate viable."
        else:
            pattern = "none"
            desc = f"Strategy distribution appears balanced (accommodate={accommodate_freq:.0%})."
            intervention = "No intervention needed."

        meta_pe = MetaPredictionError(
            magnitude=round(magnitude, 6),
            pattern_description=desc,
            dominant_maladaptive_pattern=pattern,
            accommodate_deficit=round(accommodate_deficit, 6),
            suggested_intervention=intervention,
        )

        self._meta_pe_history.append(meta_pe.magnitude)
        if len(self._meta_pe_history) > 100:
            self._meta_pe_history = self._meta_pe_history[-100:]

        return meta_pe

    # ------------------------------------------------------------------
    # Dissociation signal
    # ------------------------------------------------------------------

    def generate_dissociation_signal(
        self,
        meta_pe: MetaPredictionError,
    ) -> DissociationSignal | None:
        """Generate a cognitive dissociation signal when meta-PE exceeds threshold.

        Dissociation = "I notice I am suppressing" rather than
        "the signal just isn't there."

        Effects:
        - Raises suppress strategy's EFE cost (next cycle)
        - Applies upward pressure on belief variance (loosening)
        - Does NOT directly change belief means
        """
        if meta_pe.magnitude < META_PE_THRESHOLD:
            # Sub-threshold: decay dissociation level
            self.dissociation_level = max(0.0, self.dissociation_level - 0.05)
            return None

        # Increase dissociation level — ramps up significantly once pattern detected
        increment = min(0.2, (meta_pe.magnitude - META_PE_THRESHOLD) * 0.5)
        self.dissociation_level = min(1.0, self.dissociation_level + increment)

        # Identify affected channels from precision history
        window = self.precision_history[-self.observation_window_size:]
        suppressed_channels: list[str] = []
        for record in window:
            if record.get("type") == "suppress":
                ch = str(record.get("target", ""))
                if ch and ch not in suppressed_channels:
                    suppressed_channels.append(ch)

        suppress_penalty = self.dissociation_level * 0.4
        variance_boost = self.dissociation_level * DISSOCIATION_BELIEF_VARIANCE_BOOST

        signal = DissociationSignal(
            strength=round(self.dissociation_level, 4),
            affected_channels=suppressed_channels[:5],
            suppress_efe_penalty=round(suppress_penalty, 6),
            belief_variance_boost=round(variance_boost, 6),
            description=(
                f"Metacognitive awareness: {meta_pe.pattern_description} "
                f"Dissociation level: {self.dissociation_level:.2f}."
            ),
        )
        return signal

    # ------------------------------------------------------------------
    # Meta-beliefs
    # ------------------------------------------------------------------

    def update_meta_beliefs(
        self,
        precision_obs: PrecisionPatternObservation | None = None,
        strategy_obs: StrategyPatternObservation | None = None,
        meta_pe: MetaPredictionError | None = None,
    ) -> dict[str, object]:
        """Update beliefs about own patterns."""
        if precision_obs is not None:
            if precision_obs.chronic_suppression_channels:
                self.meta_beliefs["chronic_suppression"] = {
                    "channels": precision_obs.chronic_suppression_channels,
                    "frequency": precision_obs.suppression_frequency,
                }
            self.meta_beliefs["debt_trend"] = precision_obs.debt_trend

        if strategy_obs is not None:
            self.meta_beliefs["dominant_strategy"] = strategy_obs.dominant_strategy
            self.meta_beliefs["strategy_diversity"] = strategy_obs.diversity
            self.meta_beliefs["accommodate_rate"] = strategy_obs.accommodate_frequency
            self.meta_beliefs["suppress_rate"] = strategy_obs.suppress_frequency

        if meta_pe is not None:
            self.meta_beliefs["meta_pe_magnitude"] = meta_pe.magnitude
            self.meta_beliefs["maladaptive_pattern"] = meta_pe.dominant_maladaptive_pattern
            self.meta_beliefs["accommodate_deficit"] = meta_pe.accommodate_deficit

        self.meta_beliefs["dissociation_level"] = self.dissociation_level
        return dict(self.meta_beliefs)

    # ------------------------------------------------------------------
    # Full observation cycle
    # ------------------------------------------------------------------

    def observe_cycle(
        self,
        manipulation_records: list[dict[str, object]],
        strategy_records: list[dict[str, object]],
    ) -> tuple[
        PrecisionPatternObservation | None,
        StrategyPatternObservation | None,
        MetaPredictionError,
        DissociationSignal | None,
    ]:
        """Run a complete metacognitive observation cycle.

        Called at the end of each decision cycle.  Returns all observations
        and, if warranted, a dissociation signal.
        """
        prec_obs = None
        strat_obs = None

        for record in manipulation_records:
            prec_obs = self.observe_precision_pattern(record)

        for record in strategy_records:
            strat_obs = self.observe_strategy_pattern(record)

        meta_pe = self.compute_meta_prediction_error()
        dissociation = self.generate_dissociation_signal(meta_pe)
        self.update_meta_beliefs(prec_obs, strat_obs, meta_pe)

        return prec_obs, strat_obs, meta_pe, dissociation

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_window_size": self.observation_window_size,
            "precision_history": list(self.precision_history[-self.observation_window_size * 2:]),
            "strategy_history": list(self.strategy_history[-self.observation_window_size * 2:]),
            "meta_beliefs": dict(self.meta_beliefs),
            "dissociation_level": round(self.dissociation_level, 6),
            "meta_pe_history": [round(v, 6) for v in self._meta_pe_history[-50:]],
            "enabled": self._enabled,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "MetaCognitiveLayer":
        if not payload:
            return cls()
        layer = cls(
            observation_window_size=int(payload.get("observation_window_size", OBSERVATION_WINDOW)),
        )
        raw = payload.get("precision_history")
        if isinstance(raw, list):
            layer.precision_history = [dict(e) for e in raw if isinstance(e, dict)]
        raw = payload.get("strategy_history")
        if isinstance(raw, list):
            layer.strategy_history = [dict(e) for e in raw if isinstance(e, dict)]
        raw = payload.get("meta_beliefs")
        if isinstance(raw, dict):
            layer.meta_beliefs = dict(raw)
        layer.dissociation_level = float(payload.get("dissociation_level", 0.0))
        raw = payload.get("meta_pe_history")
        if isinstance(raw, list):
            layer._meta_pe_history = [float(v) for v in raw]
        layer._enabled = bool(payload.get("enabled", True))
        return layer
