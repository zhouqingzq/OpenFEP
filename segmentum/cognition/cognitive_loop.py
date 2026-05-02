from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..cognitive_events import CognitiveEvent, CognitiveEventBus
from ..cognitive_state import CognitiveStateMVP, update_cognitive_state
from .attention_gate import AttentionGate, AttentionGateResult


StateUpdater = Callable[
    [
        CognitiveStateMVP | None,
    ],
    CognitiveStateMVP,
]


@dataclass(frozen=True)
class CognitiveLoopResult:
    state: CognitiveStateMVP
    consumed_events: tuple[CognitiveEvent, ...]
    selected_events: tuple[CognitiveEvent, ...]
    trace_only_events: tuple[CognitiveEvent, ...]
    dropped_reasons: dict[str, str]


class CognitiveLoop:
    """Minimal bus -> attention gate -> state updater loop."""

    def __init__(
        self,
        bus: CognitiveEventBus,
        *,
        attention_gate: AttentionGate | None = None,
        state_updater: Callable[..., CognitiveStateMVP] = update_cognitive_state,
    ) -> None:
        self.bus = bus
        self.attention_gate = attention_gate or AttentionGate()
        self.state_updater = state_updater

    def consume_and_update(
        self,
        previous_state: CognitiveStateMVP | None,
        *,
        turn_id: str,
        persona_id: str | None,
        diagnostics: object | None,
        observation: Mapping[str, float],
        previous_outcome: str = "",
        self_prior_summary: Mapping[str, object] | str | None = None,
    ) -> CognitiveLoopResult:
        consumed = self.bus.consume(
            turn_id=turn_id,
            persona_id=persona_id,
            include_expired=False,
        )
        gate_result: AttentionGateResult = self.attention_gate.select(consumed)
        state = self.state_updater(
            previous_state,
            events=gate_result.selected_events,
            diagnostics=diagnostics,
            observation=observation,
            previous_outcome=previous_outcome,
            self_prior_summary=self_prior_summary,
        )
        return CognitiveLoopResult(
            state=state,
            consumed_events=consumed,
            selected_events=gate_result.selected_events,
            trace_only_events=gate_result.trace_only_events,
            dropped_reasons=gate_result.dropped_reasons,
        )
