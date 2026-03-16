"""Defense Strategy Selector: four-pathway EFE-driven strategy selection.

M2.7 Phase A — Replaces the M2.6 two-pathway (accommodate / assimilate)
identity-PE handling with a full four-pathway selector driven by Expected
Free Energy evaluation and personality bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .precision_manipulation import (
    ManipulationType,
    PrecisionManipulator,
    PrecisionManipulationResult,
)


class DefenseStrategy(str, Enum):
    """The four defense pathways available when identity PE arises."""
    ACCOMMODATE = "accommodate"   # Update self-model trait means
    ASSIMILATE = "assimilate"     # Reinterpret experience valence/salience
    SUPPRESS = "suppress"         # Lower signal precision weight
    REDIRECT = "redirect"         # Substitute drive-satisfaction pathway


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyEvaluation:
    """Expected Free Energy evaluation for a single defense strategy."""
    strategy: str
    efe_short_term: float
    efe_long_term: float
    efe_total: float
    personality_bias: float
    reasoning: str


@dataclass
class StrategyOutcome:
    """Result of executing a selected defense strategy."""
    strategy: str
    belief_changes: dict[str, float] = field(default_factory=dict)
    precision_changes: dict[str, float] = field(default_factory=dict)
    long_term_cost: float = 0.0
    manipulation_result: PrecisionManipulationResult | None = None
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Identity PE representation
# ---------------------------------------------------------------------------

@dataclass
class IdentityPE:
    """An identity prediction error — discrepancy between self-model
    prediction and actual experience."""
    source_channel: str       # Which channel generated the PE
    magnitude: float          # Size of the PE [0, ∞)
    valence: float            # Positive = better-than-expected, negative = worse
    current_belief_mean: float   # Current belief about this dimension
    current_belief_variance: float  # Uncertainty / flexibility of belief
    context: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class DefenseStrategySelector:
    """EFE-driven four-pathway defense strategy selector.

    For each incoming identity PE, evaluates all four strategies' Expected
    Free Energy (short-term + long-term), applies personality bias, and
    selects the best strategy.
    """

    # Weight for combining short/long-term EFE
    SHORT_TERM_WEIGHT: float = 0.6
    LONG_TERM_WEIGHT: float = 0.4

    def __init__(
        self,
        precision_manipulator: PrecisionManipulator,
        *,
        neuroticism: float = 0.5,
        openness: float = 0.5,
        extraversion: float = 0.5,
        conscientiousness: float = 0.5,
        agreeableness: float = 0.5,
    ) -> None:
        self.precision_manipulator = precision_manipulator
        self.neuroticism = neuroticism
        self.openness = openness
        self.extraversion = extraversion
        self.conscientiousness = conscientiousness
        self.agreeableness = agreeableness

        # History of strategy selections (for metacognitive pattern detection)
        self.strategy_history: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Personality bias
    # ------------------------------------------------------------------

    def _personality_bias(self, strategy: DefenseStrategy) -> float:
        """Personality-driven preference for a strategy.

        Returns a value in [-0.3, 0.3] that lowers the effective EFE
        (making the strategy more attractive).
        """
        o = self.openness - 0.5
        c = self.conscientiousness - 0.5
        e = self.extraversion - 0.5
        n = self.neuroticism - 0.5

        if strategy is DefenseStrategy.ACCOMMODATE:
            return max(-0.3, min(0.3, o * 0.4 + (-n) * 0.2))
        elif strategy is DefenseStrategy.ASSIMILATE:
            return max(-0.3, min(0.3, c * 0.35 + o * 0.1))
        elif strategy is DefenseStrategy.SUPPRESS:
            return max(-0.3, min(0.3, n * 0.4 + (-o) * 0.15))
        else:  # REDIRECT
            return max(-0.3, min(0.3, e * 0.35 + (-n) * 0.1))

    # ------------------------------------------------------------------
    # EFE evaluation
    # ------------------------------------------------------------------

    def evaluate_strategies(
        self,
        identity_pe: IdentityPE,
        *,
        precision_debt: float = 0.0,
        dissociation_level: float = 0.0,
    ) -> list[StrategyEvaluation]:
        """Evaluate all four strategies for a given identity PE.

        Returns evaluations sorted by total EFE (lowest = best).

        Parameters
        ----------
        identity_pe
            The identity prediction error to resolve.
        precision_debt
            Current accumulated precision debt (increases suppress cost).
        dissociation_level
            MetaCognitive dissociation level — raises suppress short-term
            EFE when > 0, making the system reconsider accommodation.
        """
        mag = identity_pe.magnitude
        var = identity_pe.current_belief_variance

        evaluations: list[StrategyEvaluation] = []

        # Belief rigidity factor: low variance → higher accommodate cost
        # but also reflects that flexible beliefs are easier to update
        rigidity = max(0.0, 1.0 - var * 10.0)  # 0 when var≥0.1, 1 when var→0

        # --- ACCOMMODATE ---
        # Short-term cost: proportional to PE magnitude, modulated by rigidity
        # Long-term benefit: reduces future PE substantially
        acc_short = mag * (0.4 + rigidity * 0.3)
        acc_long = -mag * 0.5  # negative = beneficial
        evaluations.append(StrategyEvaluation(
            strategy=DefenseStrategy.ACCOMMODATE.value,
            efe_short_term=round(acc_short, 6),
            efe_long_term=round(acc_long, 6),
            efe_total=round(
                acc_short * self.SHORT_TERM_WEIGHT + acc_long * self.LONG_TERM_WEIGHT,
                6,
            ),
            personality_bias=round(self._personality_bias(DefenseStrategy.ACCOMMODATE), 6),
            reasoning=(
                f"Rewrite belief mean to absorb PE={mag:.3f}. "
                f"Short-term cost modulated by rigidity={rigidity:.2f}."
            ),
        ))

        # --- ASSIMILATE ---
        # Reinterpret experience rather than change belief
        assim_short = mag * 0.35
        assim_long = mag * 0.1  # some residual cost
        evaluations.append(StrategyEvaluation(
            strategy=DefenseStrategy.ASSIMILATE.value,
            efe_short_term=round(assim_short, 6),
            efe_long_term=round(assim_long, 6),
            efe_total=round(
                assim_short * self.SHORT_TERM_WEIGHT + assim_long * self.LONG_TERM_WEIGHT,
                6,
            ),
            personality_bias=round(self._personality_bias(DefenseStrategy.ASSIMILATE), 6),
            reasoning=(
                f"Reinterpret experience valence/salience. "
                f"Moderate cost, moderate residual."
            ),
        ))

        # --- SUPPRESS ---
        # Cheapest short-term but precision debt accumulates heavily
        # Dissociation acts as a strong multiplicative penalty
        supp_short = mag * 0.15 + dissociation_level * mag * 1.2
        supp_long = mag * 0.5 + precision_debt * 0.5
        evaluations.append(StrategyEvaluation(
            strategy=DefenseStrategy.SUPPRESS.value,
            efe_short_term=round(supp_short, 6),
            efe_long_term=round(supp_long, 6),
            efe_total=round(
                supp_short * self.SHORT_TERM_WEIGHT + supp_long * self.LONG_TERM_WEIGHT,
                6,
            ),
            personality_bias=round(self._personality_bias(DefenseStrategy.SUPPRESS), 6),
            reasoning=(
                f"Suppress precision on channel '{identity_pe.source_channel}'. "
                f"Cheap now (debt={precision_debt:.3f}) but debt grows."
            ),
        ))

        # --- REDIRECT ---
        # Drive substitution: moderate both short and long
        redir_short = mag * 0.35
        redir_long = mag * 0.15
        evaluations.append(StrategyEvaluation(
            strategy=DefenseStrategy.REDIRECT.value,
            efe_short_term=round(redir_short, 6),
            efe_long_term=round(redir_long, 6),
            efe_total=round(
                redir_short * self.SHORT_TERM_WEIGHT + redir_long * self.LONG_TERM_WEIGHT,
                6,
            ),
            personality_bias=round(self._personality_bias(DefenseStrategy.REDIRECT), 6),
            reasoning=(
                f"Redirect drive satisfaction to alternative channel. "
                f"Moderate cost, moderate residual."
            ),
        ))

        # Sort by effective EFE (total - personality_bias; lower = better)
        evaluations.sort(key=lambda ev: ev.efe_total - ev.personality_bias)
        return evaluations

    def select_strategy(
        self,
        evaluations: list[StrategyEvaluation],
    ) -> tuple[DefenseStrategy, StrategyEvaluation]:
        """Select the best strategy from evaluated options.

        Returns (strategy_enum, evaluation).
        """
        best = evaluations[0]
        return DefenseStrategy(best.strategy), best

    def execute_strategy(
        self,
        strategy: DefenseStrategy,
        identity_pe: IdentityPE,
        *,
        cycle: int = 0,
    ) -> StrategyOutcome:
        """Execute the selected strategy and return state changes."""
        mag = identity_pe.magnitude
        channel = identity_pe.source_channel

        if strategy is DefenseStrategy.ACCOMMODATE:
            # Shift belief mean toward observed value
            shift = mag * 0.5 * (1.0 if identity_pe.valence > 0 else -1.0)
            new_mean = identity_pe.current_belief_mean + shift
            outcome = StrategyOutcome(
                strategy=strategy.value,
                belief_changes={channel: round(shift, 6)},
                long_term_cost=0.0,
                reasoning=f"Belief mean shifted by {shift:+.4f} on '{channel}'.",
            )

        elif strategy is DefenseStrategy.ASSIMILATE:
            # Reduce the PE by reinterpreting: partial belief shift + valence reframe
            shift = mag * 0.2 * (1.0 if identity_pe.valence > 0 else -1.0)
            outcome = StrategyOutcome(
                strategy=strategy.value,
                belief_changes={channel: round(shift, 6)},
                long_term_cost=round(mag * 0.1, 6),
                reasoning=f"Reinterpreted experience; partial belief shift {shift:+.4f}.",
            )

        elif strategy is DefenseStrategy.SUPPRESS:
            # Lower precision on the offending channel
            manip_result = self.precision_manipulator.apply_manipulation(
                ManipulationType.SUPPRESS,
                target_channel=channel,
                intensity=min(1.0, mag),
                cycle=cycle,
            )
            outcome = StrategyOutcome(
                strategy=strategy.value,
                precision_changes={
                    channel: round(
                        manip_result.precision_after - manip_result.precision_before, 6
                    )
                },
                long_term_cost=manip_result.cost.long_term_projected,
                manipulation_result=manip_result,
                reasoning=(
                    f"Suppressed precision on '{channel}' "
                    f"({manip_result.precision_before:.3f} → "
                    f"{manip_result.precision_after:.3f})."
                ),
            )

        else:  # REDIRECT
            # Shift precision from source channel to an alternative
            alt_channel = self._alternative_channel(channel)
            manip_result = self.precision_manipulator.apply_manipulation(
                ManipulationType.REDIRECT,
                target_channel=alt_channel,
                intensity=min(1.0, mag),
                source_channel=channel,
                cycle=cycle,
            )
            outcome = StrategyOutcome(
                strategy=strategy.value,
                precision_changes={
                    channel: round(-min(1.0, mag) * 0.2, 6),
                    alt_channel: round(min(1.0, mag) * 0.2, 6),
                },
                long_term_cost=manip_result.cost.long_term_projected,
                manipulation_result=manip_result,
                reasoning=(
                    f"Redirected drive from '{channel}' to '{alt_channel}'."
                ),
            )

        # Record in history
        self.strategy_history.append({
            "cycle": cycle,
            "strategy": strategy.value,
            "channel": channel,
            "pe_magnitude": round(mag, 6),
            "long_term_cost": outcome.long_term_cost,
        })

        return outcome

    def update_personality(
        self,
        *,
        neuroticism: float | None = None,
        openness: float | None = None,
        extraversion: float | None = None,
        conscientiousness: float | None = None,
        agreeableness: float | None = None,
    ) -> None:
        if neuroticism is not None:
            self.neuroticism = neuroticism
        if openness is not None:
            self.openness = openness
        if extraversion is not None:
            self.extraversion = extraversion
        if conscientiousness is not None:
            self.conscientiousness = conscientiousness
        if agreeableness is not None:
            self.agreeableness = agreeableness

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alternative_channel(channel: str) -> str:
        """Pick a drive-substitution target for redirect."""
        alternatives = {
            "attachment": "social",
            "self_worth": "novelty",
            "social": "novelty",
            "threat": "shelter",
            "danger": "shelter",
        }
        return alternatives.get(channel, "novelty")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "neuroticism": self.neuroticism,
            "openness": self.openness,
            "extraversion": self.extraversion,
            "conscientiousness": self.conscientiousness,
            "agreeableness": self.agreeableness,
            "strategy_history": list(self.strategy_history[-64:]),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object] | None,
        precision_manipulator: PrecisionManipulator,
    ) -> "DefenseStrategySelector":
        if not payload:
            return cls(precision_manipulator)
        selector = cls(
            precision_manipulator,
            neuroticism=float(payload.get("neuroticism", 0.5)),
            openness=float(payload.get("openness", 0.5)),
            extraversion=float(payload.get("extraversion", 0.5)),
            conscientiousness=float(payload.get("conscientiousness", 0.5)),
            agreeableness=float(payload.get("agreeableness", 0.5)),
        )
        raw_history = payload.get("strategy_history")
        if isinstance(raw_history, list):
            selector.strategy_history = [
                dict(e) for e in raw_history if isinstance(e, dict)
            ]
        return selector
