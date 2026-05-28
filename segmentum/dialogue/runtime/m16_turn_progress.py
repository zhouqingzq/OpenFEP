"""Stage-weighted turn progress for M16 gateway clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressPublish = Callable[[int, str, int], None]

# Ordered pipeline for a user turn; each advance() moves one step (run or skip).
PIPELINE_STEPS: tuple[str, ...] = (
    "init",
    "m12_identity_pre",
    "m13_settlement",
    "conscious_loop",
    "query_planner",
    "evidence_judge",
    "memory_recall",
    "user_modeling",
    "m13_eval",
    "thinking_reply",
    "post_reply_observer",
    "finalize",
)


class TurnProgressReporter:
    """Reports monotonic 0–100% progress across the MVP run_turn pipeline."""

    def __init__(self, *, turn_index: int, publish: ProgressPublish | None = None) -> None:
        self.turn_index = int(turn_index)
        self._publish = publish
        self._step = 0
        self._total = len(PIPELINE_STEPS)
        self._last_percent = -1

    def advance(self, step_id: str) -> None:
        if self._step >= self._total:
            return
        expected = PIPELINE_STEPS[self._step]
        if step_id != expected:
            # Keep pipeline order strict; ignore out-of-order calls.
            return
        self._step += 1
        percent = min(100, round(self._step / self._total * 100))
        if percent <= self._last_percent:
            return
        self._last_percent = percent
        if self._publish is not None:
            self._publish(self.turn_index, step_id, percent)


def build_turn_progress_payload(*, turn_index: int, stage: str, percent: int) -> dict[str, Any]:
    return {
        "audit_type": "turn_progress",
        "turn_index": int(turn_index),
        "stage": str(stage or "")[:64],
        "percent": max(0, min(100, int(percent))),
    }
