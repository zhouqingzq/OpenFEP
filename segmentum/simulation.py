from __future__ import annotations

from pathlib import Path

from .runtime import SegmentRuntime


def run_simulation(
    cycles: int = 20,
    seed: int = 17,
    state_path: str | Path | None = None,
    reset: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    runtime = SegmentRuntime.load_or_create(
        state_path=state_path,
        seed=seed,
        reset=reset,
    )
    return runtime.run(cycles=cycles, verbose=verbose)
