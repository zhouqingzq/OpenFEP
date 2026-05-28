from __future__ import annotations

from segmentum.dialogue.runtime.m16_turn_progress import (
    PIPELINE_STEPS,
    TurnProgressReporter,
    build_turn_progress_payload,
)


def test_turn_progress_reaches_100_after_pipeline() -> None:
    events: list[tuple[str, int]] = []

    reporter = TurnProgressReporter(
        turn_index=3,
        publish=lambda _turn_index, stage, percent: events.append((stage, percent)),
    )
    for step in PIPELINE_STEPS:
        reporter.advance(step)
    assert events[-1] == ("finalize", 100)
    assert all(events[index][1] <= events[index + 1][1] for index in range(len(events) - 1))


def test_turn_progress_ignores_out_of_order_steps() -> None:
    events: list[tuple[str, int]] = []
    reporter = TurnProgressReporter(
        turn_index=1,
        publish=lambda _turn_index, stage, percent: events.append((stage, percent)),
    )
    reporter.advance("conscious_loop")
    reporter.advance("init")
    reporter.advance("init")
    assert events == [("init", 8)]


def test_build_turn_progress_payload_is_bounded() -> None:
    payload = build_turn_progress_payload(turn_index=2, stage="thinking_reply", percent=150)
    assert payload["audit_type"] == "turn_progress"
    assert payload["turn_index"] == 2
    assert payload["stage"] == "thinking_reply"
    assert payload["percent"] == 100
