from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from .state import AgentState, PolicyTendency, TickInput


class ConsciousnessLogger:
    """Append-only logger for the agent's internal monologue stream."""

    def __init__(self, log_path: Union[str, Path] = "consciousness_stream.log") -> None:
        self.log_path = Path(log_path)

    def append(
        self,
        state: AgentState,
        tick_input: TickInput,
        policy: PolicyTendency,
        inner_speech: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        notes = ", ".join(tick_input.notes) if tick_input.notes else "stable"
        line = (
            f"{timestamp} "
            f"[tick={state.tick_count:04d}] "
            f"[strategy={policy.chosen_strategy}] "
            f"[energy={state.internal_energy:.2f} "
            f"error={state.prediction_error:.2f} "
            f"boredom={state.boredom:.2f} "
            f"surprise={state.surprise_load:.2f}] "
            f"[input={notes}] "
            f"{inner_speech}\n"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
