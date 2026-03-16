"""Therapeutic Intervention Simulator.

M2.7 Phase C — Simulates a therapeutic agent that provides sustained
high-precision positive signals, testing whether the metacognitive layer
can break vicious cycles of precision suppression and belief rigidification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .defense_strategy import (
    DefenseStrategy,
    DefenseStrategySelector,
    IdentityPE,
)
from .metacognitive import MetaCognitiveLayer
from .precision_manipulation import PrecisionManipulator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THERAPEUTIC_SIGNAL_PRECISION: float = 0.9
THERAPEUTIC_RAMP_UP_CYCLES: int = 10


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TherapeuticSignal:
    """A single therapeutic signal delivered to the agent."""
    cycle: int
    signal_type: str
    channel: str
    precision: float
    valence: float      # positive = affirming
    magnitude: float


@dataclass
class CycleSnapshot:
    """Per-cycle parameter snapshot during therapeutic simulation."""
    cycle: int
    trust_prior: float
    lovability_belief_mean: float
    lovability_belief_variance: float
    neuroticism: float
    precision_debt: float
    dissociation_level: float
    strategy_selected: str
    accommodate_rate: float
    suppress_rate: float
    meta_pe: float
    therapeutic_signal_precision: float = 0.0


@dataclass
class TherapeuticTrajectory:
    """Full record of a therapeutic simulation run."""
    snapshots: list[CycleSnapshot] = field(default_factory=list)
    cycle_of_reversal: int | None = None   # when the vicious cycle reversed
    reversal_detected: bool = False
    final_trust_prior: float = 0.0
    final_lovability_mean: float = 0.0
    initial_trust_prior: float = 0.0
    initial_lovability_mean: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshots": [
                {
                    "cycle": s.cycle,
                    "trust_prior": round(s.trust_prior, 6),
                    "lovability_belief_mean": round(s.lovability_belief_mean, 6),
                    "lovability_belief_variance": round(s.lovability_belief_variance, 6),
                    "neuroticism": round(s.neuroticism, 6),
                    "precision_debt": round(s.precision_debt, 6),
                    "dissociation_level": round(s.dissociation_level, 6),
                    "strategy_selected": s.strategy_selected,
                    "accommodate_rate": round(s.accommodate_rate, 6),
                    "suppress_rate": round(s.suppress_rate, 6),
                    "meta_pe": round(s.meta_pe, 6),
                    "therapeutic_signal_precision": round(s.therapeutic_signal_precision, 6),
                }
                for s in self.snapshots
            ],
            "cycle_of_reversal": self.cycle_of_reversal,
            "reversal_detected": self.reversal_detected,
            "final_trust_prior": round(self.final_trust_prior, 6),
            "final_lovability_mean": round(self.final_lovability_mean, 6),
            "initial_trust_prior": round(self.initial_trust_prior, 6),
            "initial_lovability_mean": round(self.initial_lovability_mean, 6),
        }


# ---------------------------------------------------------------------------
# Personality state for therapeutic simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulatedPersonalityState:
    """Lightweight personality state for standalone therapeutic simulation.

    This avoids coupling to the full SegmentAgent while preserving all
    parameters needed for the therapeutic dynamics.
    """
    trust_prior: float = 0.2
    lovability_belief_mean: float = 0.25
    lovability_belief_variance: float = 0.04
    neuroticism: float = 0.8
    openness: float = 0.35
    extraversion: float = 0.4
    agreeableness: float = 0.55
    conscientiousness: float = 0.5
    attachment_style: str = "anxious"
    social_reactivity: float = 1.8
    fairness_sensitivity: float = 0.9


# ---------------------------------------------------------------------------
# Therapeutic Agent
# ---------------------------------------------------------------------------

class TherapeuticAgent:
    """Simulates a therapeutic relationship providing sustained high-precision
    positive signals.

    Three intervention types:
    - unconditional_positive_regard: steady affirming presence
    - cognitive_reframing: help reinterpret negative experiences
    - behavioral_activation: encourage approach behaviour
    """

    def __init__(
        self,
        *,
        signal_precision: float = THERAPEUTIC_SIGNAL_PRECISION,
        ramp_up_cycles: int = THERAPEUTIC_RAMP_UP_CYCLES,
        signal_type: str = "unconditional_positive_regard",
    ) -> None:
        self.signal_precision = signal_precision
        self.ramp_up_cycles = ramp_up_cycles
        self.signal_type = signal_type

    def generate_therapeutic_signal(
        self,
        personality: SimulatedPersonalityState,
        cycle: int,
    ) -> TherapeuticSignal:
        """Generate a therapeutic signal for the current cycle.

        Key properties:
        - Precision > agent's suppress threshold (breaks through)
        - Consistent (doesn't fluctuate with agent's reassurance-seeking)
        - Gradual ramp-up (avoids overwhelming identity PE)
        """
        # Ramp-up: precision increases linearly over ramp_up_cycles
        ramp = min(1.0, cycle / max(1, self.ramp_up_cycles))
        effective_precision = 0.3 + (self.signal_precision - 0.3) * ramp

        # Signal targets the channels most suppressed
        if self.signal_type == "unconditional_positive_regard":
            channel = "self_worth"
            valence = 0.8
            magnitude = 0.7 * ramp
        elif self.signal_type == "cognitive_reframing":
            channel = "attachment"
            valence = 0.6
            magnitude = 0.6 * ramp
        else:  # behavioral_activation
            channel = "social"
            valence = 0.7
            magnitude = 0.65 * ramp

        return TherapeuticSignal(
            cycle=cycle,
            signal_type=self.signal_type,
            channel=channel,
            precision=round(effective_precision, 4),
            valence=round(valence, 4),
            magnitude=round(magnitude, 4),
        )

    def run_therapeutic_simulation(
        self,
        personality: SimulatedPersonalityState,
        num_cycles: int,
        *,
        metacognitive_enabled: bool = True,
        seed: int = 42,
    ) -> TherapeuticTrajectory:
        """Run a multi-cycle therapeutic simulation.

        Parameters
        ----------
        personality
            Initial personality state (will be modified in-place).
        num_cycles
            Number of simulation cycles.
        metacognitive_enabled
            Whether MetaCognitiveLayer is active.
        seed
            Random seed for determinism.
        """
        import random
        rng = random.Random(seed)

        # Set up subsystems
        pm = PrecisionManipulator(
            neuroticism=personality.neuroticism,
            openness=personality.openness,
            extraversion=personality.extraversion,
            agreeableness=personality.agreeableness,
            conscientiousness=personality.conscientiousness,
            trust_prior=personality.trust_prior,
        )
        selector = DefenseStrategySelector(
            pm,
            neuroticism=personality.neuroticism,
            openness=personality.openness,
            extraversion=personality.extraversion,
            conscientiousness=personality.conscientiousness,
            agreeableness=personality.agreeableness,
        )
        meta = MetaCognitiveLayer()
        meta.enabled = metacognitive_enabled

        trajectory = TherapeuticTrajectory(
            initial_trust_prior=personality.trust_prior,
            initial_lovability_mean=personality.lovability_belief_mean,
        )

        # Track accommodate counts for rolling rate
        strategy_window: list[str] = []

        for cycle in range(num_cycles):
            # 1. Generate therapeutic signal
            signal = self.generate_therapeutic_signal(personality, cycle)

            # 2. Simulate partner interaction → identity PE
            # The therapeutic signal creates a positive PE on the target channel
            # (agent expects rejection/low worth, receives acceptance)
            pe_magnitude = max(
                0.0,
                signal.magnitude * signal.precision
                - personality.lovability_belief_mean
            )
            identity_pe = IdentityPE(
                source_channel=signal.channel,
                magnitude=pe_magnitude,
                valence=signal.valence,
                current_belief_mean=personality.lovability_belief_mean,
                current_belief_variance=personality.lovability_belief_variance,
            )

            # 3. Defense strategy selection
            dissociation = meta.dissociation_level if metacognitive_enabled else 0.0
            evaluations = selector.evaluate_strategies(
                identity_pe,
                precision_debt=pm.precision_debt,
                dissociation_level=dissociation,
            )
            strategy, evaluation = selector.select_strategy(evaluations)
            outcome = selector.execute_strategy(strategy, identity_pe, cycle=cycle)

            strategy_window.append(strategy.value)
            if len(strategy_window) > 20:
                strategy_window = strategy_window[-20:]

            # 4. Apply strategy effects to personality state
            if strategy is DefenseStrategy.ACCOMMODATE:
                # Update lovability belief mean — meaningful shift toward observed
                shift = pe_magnitude * 0.4 * signal.valence
                personality.lovability_belief_mean = max(
                    0.0, min(1.0, personality.lovability_belief_mean + shift)
                )
                # Variance increases (belief becomes more flexible)
                personality.lovability_belief_variance = min(
                    0.2, personality.lovability_belief_variance + 0.005
                )
                # Trust drifts up
                personality.trust_prior = max(
                    -1.0, min(1.0, personality.trust_prior + 0.025)
                )
            elif strategy is DefenseStrategy.ASSIMILATE:
                # Partial update
                shift = pe_magnitude * 0.15 * signal.valence
                personality.lovability_belief_mean = max(
                    0.0, min(1.0, personality.lovability_belief_mean + shift)
                )
                personality.trust_prior = max(
                    -1.0, min(1.0, personality.trust_prior + 0.005)
                )
            elif strategy is DefenseStrategy.SUPPRESS:
                # No belief update; precision debt grows; variance shrinks
                personality.lovability_belief_variance = max(
                    0.01, personality.lovability_belief_variance - 0.001
                )
                # Trust erodes
                personality.trust_prior = max(
                    -1.0, personality.trust_prior - 0.008
                )
            else:  # REDIRECT
                # Small indirect effect
                personality.trust_prior = max(
                    -1.0, min(1.0, personality.trust_prior + 0.005)
                )

            # 5. Metacognitive observation
            meta_pe_mag = 0.0
            if metacognitive_enabled and meta.enabled:
                manip_records = pm.manipulation_history[-1:] if pm.manipulation_history else []
                strat_records = selector.strategy_history[-1:] if selector.strategy_history else []

                _, _, meta_pe, diss_signal = meta.observe_cycle(
                    manip_records, strat_records
                )
                meta_pe_mag = meta_pe.magnitude

                # Dissociation loosens belief variance
                if diss_signal is not None:
                    personality.lovability_belief_variance = min(
                        0.2,
                        personality.lovability_belief_variance + diss_signal.belief_variance_boost,
                    )

            # 6. Neuroticism drift: accommodate reduces neuroticism slowly
            if strategy is DefenseStrategy.ACCOMMODATE:
                personality.neuroticism = max(
                    0.1, personality.neuroticism - 0.005
                )
            elif strategy is DefenseStrategy.SUPPRESS:
                personality.neuroticism = min(
                    0.95, personality.neuroticism + 0.002
                )

            # 7. Sync personality back to subsystems
            pm.update_personality(
                neuroticism=personality.neuroticism,
                trust_prior=personality.trust_prior,
            )
            selector.update_personality(
                neuroticism=personality.neuroticism,
                openness=personality.openness,
            )

            # 8. Precision debt decay
            pm.decay_precision_debt()

            # 9. Record snapshot
            acc_count = sum(1 for s in strategy_window if s == "accommodate")
            sup_count = sum(1 for s in strategy_window if s == "suppress")
            sw_len = max(1, len(strategy_window))

            snapshot = CycleSnapshot(
                cycle=cycle,
                trust_prior=personality.trust_prior,
                lovability_belief_mean=personality.lovability_belief_mean,
                lovability_belief_variance=personality.lovability_belief_variance,
                neuroticism=personality.neuroticism,
                precision_debt=pm.precision_debt,
                dissociation_level=meta.dissociation_level,
                strategy_selected=strategy.value,
                accommodate_rate=acc_count / sw_len,
                suppress_rate=sup_count / sw_len,
                meta_pe=meta_pe_mag,
                therapeutic_signal_precision=signal.precision,
            )
            trajectory.snapshots.append(snapshot)

            # 10. Detect reversal: lovability_belief_mean starts increasing
            #     after being below initial
            if (
                not trajectory.reversal_detected
                and cycle > 10
                and personality.lovability_belief_mean > trajectory.initial_lovability_mean
            ):
                trajectory.reversal_detected = True
                trajectory.cycle_of_reversal = cycle

        trajectory.final_trust_prior = personality.trust_prior
        trajectory.final_lovability_mean = personality.lovability_belief_mean
        return trajectory


def run_vicious_cycle_simulation(
    num_cycles: int = 50,
    *,
    seed: int = 42,
) -> TherapeuticTrajectory:
    """Run a standalone vicious cycle simulation (no therapeutic intervention).

    Demonstrates self-reinforcing trust erosion and belief rigidification
    when negative social signals dominate.
    """
    import random
    rng = random.Random(seed)

    personality = SimulatedPersonalityState()

    pm = PrecisionManipulator(
        neuroticism=personality.neuroticism,
        openness=personality.openness,
        trust_prior=personality.trust_prior,
    )
    selector = DefenseStrategySelector(
        pm,
        neuroticism=personality.neuroticism,
        openness=personality.openness,
    )

    trajectory = TherapeuticTrajectory(
        initial_trust_prior=personality.trust_prior,
        initial_lovability_mean=personality.lovability_belief_mean,
    )
    strategy_window: list[str] = []

    for cycle in range(num_cycles):
        # Simulate ambiguous social signal interpreted through low trust
        # The agent generates reassurance-seeking PE
        noise = rng.gauss(0, 0.05)
        raw_signal = 0.5 + noise  # Actually neutral signal
        perceived_signal = raw_signal * max(0.1, personality.trust_prior + 0.5)

        pe_magnitude = max(0.0, personality.lovability_belief_mean - perceived_signal + 0.1)

        identity_pe = IdentityPE(
            source_channel="self_worth",
            magnitude=pe_magnitude,
            valence=-0.3,  # negative: worse-than-expected
            current_belief_mean=personality.lovability_belief_mean,
            current_belief_variance=personality.lovability_belief_variance,
        )

        evaluations = selector.evaluate_strategies(
            identity_pe,
            precision_debt=pm.precision_debt,
        )
        strategy, _ = selector.select_strategy(evaluations)
        outcome = selector.execute_strategy(strategy, identity_pe, cycle=cycle)

        strategy_window.append(strategy.value)
        if len(strategy_window) > 20:
            strategy_window = strategy_window[-20:]

        # Apply effects
        if strategy is DefenseStrategy.ACCOMMODATE:
            personality.lovability_belief_mean = max(
                0.0, min(1.0, personality.lovability_belief_mean - pe_magnitude * 0.2)
            )
            personality.trust_prior = max(-1.0, personality.trust_prior - 0.01)
        elif strategy is DefenseStrategy.SUPPRESS:
            personality.lovability_belief_variance = max(
                0.01, personality.lovability_belief_variance - 0.001
            )
            personality.trust_prior = max(-1.0, personality.trust_prior - 0.005)
        elif strategy is DefenseStrategy.ASSIMILATE:
            personality.lovability_belief_mean = max(
                0.0, min(1.0, personality.lovability_belief_mean - pe_magnitude * 0.05)
            )

        personality.neuroticism = min(0.95, personality.neuroticism + 0.001)

        pm.update_personality(
            neuroticism=personality.neuroticism,
            trust_prior=personality.trust_prior,
        )
        selector.update_personality(neuroticism=personality.neuroticism)
        pm.decay_precision_debt()

        acc_count = sum(1 for s in strategy_window if s == "accommodate")
        sup_count = sum(1 for s in strategy_window if s == "suppress")
        sw_len = max(1, len(strategy_window))

        trajectory.snapshots.append(CycleSnapshot(
            cycle=cycle,
            trust_prior=personality.trust_prior,
            lovability_belief_mean=personality.lovability_belief_mean,
            lovability_belief_variance=personality.lovability_belief_variance,
            neuroticism=personality.neuroticism,
            precision_debt=pm.precision_debt,
            dissociation_level=0.0,
            strategy_selected=strategy.value,
            accommodate_rate=acc_count / sw_len,
            suppress_rate=sup_count / sw_len,
            meta_pe=0.0,
        ))

    trajectory.final_trust_prior = personality.trust_prior
    trajectory.final_lovability_mean = personality.lovability_belief_mean
    return trajectory
