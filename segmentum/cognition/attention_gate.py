from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..cognitive_events import CognitiveEvent


@dataclass(frozen=True)
class AttentionGateConfig:
    min_salience: float = 0.5
    min_priority: float = 0.7
    event_budget: int = 8


@dataclass(frozen=True)
class AttentionGateResult:
    selected_events: tuple[CognitiveEvent, ...]
    trace_only_events: tuple[CognitiveEvent, ...]
    dropped_reasons: dict[str, str] = field(default_factory=dict)


class AttentionGate:
    """Select high-value cognitive events without deleting trace-only events."""

    def __init__(self, config: AttentionGateConfig | None = None) -> None:
        self.config = config or AttentionGateConfig()

    def select(self, events: Sequence[CognitiveEvent]) -> AttentionGateResult:
        reasons: dict[str, str] = {}
        eligible: list[CognitiveEvent] = []
        trace_only: list[CognitiveEvent] = []

        for event in events:
            if int(event.ttl) <= 0:
                trace_only.append(event)
                reasons[event.event_id] = "expired"
                continue
            if (
                float(event.salience) < float(self.config.min_salience)
                and float(event.priority) < float(self.config.min_priority)
            ):
                trace_only.append(event)
                reasons[event.event_id] = "low_salience_priority"
                continue
            eligible.append(event)

        ranked = sorted(
            eligible,
            key=lambda event: (
                -float(event.priority),
                -float(event.salience),
                int(event.cycle),
                str(event.event_id),
            ),
        )
        budget = max(0, int(self.config.event_budget))
        selected = ranked[:budget]
        over_budget = ranked[budget:]
        for event in over_budget:
            trace_only.append(event)
            reasons[event.event_id] = "over_budget"

        return AttentionGateResult(
            selected_events=tuple(selected),
            trace_only_events=tuple(trace_only),
            dropped_reasons=reasons,
        )
