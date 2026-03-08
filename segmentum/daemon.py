from __future__ import annotations

from pathlib import Path

from .runtime import SegmentRuntime
from .state import TickOutcome


class HeartbeatDaemon:
    """Compatibility wrapper around the unified SegmentRuntime."""

    def __init__(
        self,
        tick_interval_seconds: float = 2.0,
        runtime: SegmentRuntime | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.tick_interval_seconds = tick_interval_seconds
        self.runtime = runtime or SegmentRuntime.load_or_create(state_path=state_path)

    async def tick(self) -> TickOutcome:
        host_tick = await self.runtime.arun_host_telemetry_step()
        return TickOutcome(
            state_before=host_tick["state_before"],
            tick_input=host_tick["tick_input"],
            policy=host_tick["policy"],
            state_after=host_tick["state_after"],
            inner_speech=host_tick["inner_speech"],
        )

    async def run(self, max_ticks: int | None = None) -> dict[str, object]:
        return await self.runtime.arun(
            cycles=max_ticks,
            verbose=False,
            host_telemetry=True,
            tick_interval_seconds=self.tick_interval_seconds,
        )


async def run_daemon(
    tick_interval_seconds: float = 2.0,
    max_ticks: int | None = None,
) -> dict[str, object]:
    daemon = HeartbeatDaemon(tick_interval_seconds=tick_interval_seconds)
    return await daemon.run(max_ticks=max_ticks)
