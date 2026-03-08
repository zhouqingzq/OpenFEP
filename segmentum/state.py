from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Strategy(str, Enum):
    """Macro action selected by the brainstem-level controller."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"
    ESCAPE = "escape"


@dataclass
class AgentState:
    """
    Minimal internal state for the daemonized Segment.

    `internal_energy` approximates available compute/token budget.
    `prediction_error` is the current mismatch between expected and sensed state.
    `boredom` rises when prediction error stays too low for too long, creating
    epistemic pressure to seek novel but still learnable signals.
    `surprise_load` accumulates unresolved perturbation across ticks.
    """

    internal_energy: float = 0.85
    prediction_error: float = 0.10
    boredom: float = 0.15
    surprise_load: float = 0.05
    tick_count: int = 0
    last_strategy: Strategy = Strategy.EXPLOIT

    def to_dict(self) -> dict[str, object]:
        return {
            "internal_energy": self.internal_energy,
            "prediction_error": self.prediction_error,
            "boredom": self.boredom,
            "surprise_load": self.surprise_load,
            "tick_count": self.tick_count,
            "last_strategy": self.last_strategy.value,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> AgentState:
        if not payload:
            return cls()

        raw_strategy = str(payload.get("last_strategy", Strategy.EXPLOIT.value))
        try:
            last_strategy = Strategy(raw_strategy)
        except ValueError:
            last_strategy = Strategy.EXPLOIT

        return cls(
            internal_energy=float(payload.get("internal_energy", 0.85)),
            prediction_error=float(payload.get("prediction_error", 0.10)),
            boredom=float(payload.get("boredom", 0.15)),
            surprise_load=float(payload.get("surprise_load", 0.05)),
            tick_count=int(payload.get("tick_count", 0)),
            last_strategy=last_strategy,
        )


@dataclass
class TickInput:
    """
    Bottom-up perturbation passed upward from sensorimotor layers.

    The brainstem receives only error-like summaries rather than raw sensor spam.
    """

    cpu_prediction_error: float = 0.0
    memory_prediction_error: float = 0.0
    resource_pressure: float = 0.0
    surprise_signal: float = 0.0
    boredom_signal: float = 0.0
    energy_drain: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class PolicyTendency:
    """
    Simplified EFE vector over macro strategies.

    Lower EFE is preferred. `epistemic_value` captures information gain from
    exploring uncertainty. `pragmatic_value` captures the value of protecting
    the agent from costly, destabilizing perturbations.
    """

    explore_efe: float
    exploit_efe: float
    escape_efe: float
    epistemic_value: float
    pragmatic_value: float
    chosen_strategy: Strategy


@dataclass
class TickOutcome:
    """Snapshot emitted after each heartbeat tick."""

    state_before: AgentState
    tick_input: TickInput
    policy: PolicyTendency
    state_after: AgentState
    inner_speech: str
