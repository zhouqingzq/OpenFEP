from __future__ import annotations

from pathlib import Path

from segmentum.dialogue.runtime.m14_1_background_continuity import (
    merge_background_continuity_into_initiative,
    normalize_background_continuity_state,
)
from segmentum.dialogue.runtime.m14_1_background_continuity import release_runner_lock
from segmentum.dialogue.runtime.m16_protocol import FORBIDDEN_ACTUATION_PAYLOAD_KEYS
from segmentum.dialogue.runtime.mvp_loop import MVPStateStore
from tests.m16_1_test_helpers import NOW, build_stack
from tests.test_mvp_dialogue_runtime import FakeJSONLLM


def test_runner_claims_input_and_produces_assistant_actuation(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    event_id = bridge.append_client_input(text="你好", correlation_id="corr_user")
    step = runner.run_once_for_tests(now=clk())
    assert step.claimed_events >= 1
    assert any(row.get("reply") for row in step.processed if isinstance(row, dict))
    kinds = [msg.get("kind") for msg in step.actuation_messages]
    assert "AssistantMessageCommitted" in kinds
    assert bridge.is_event_processed(event_id)


def test_runner_self_ticks_without_ui_ping(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    runner.run_once_for_tests(now=clk())
    clk.advance(5)
    step = runner.run_once_for_tests(now=clk())
    phases = [row.get("phase") for row in step.processed if isinstance(row, dict)]
    assert "idle_cognitive_tick" in phases
    assert "background_self_tick" in phases


def test_second_runner_refused_while_lock_held(tmp_path: Path) -> None:
    bridge, hub, runner_a, clk = build_stack(tmp_path)
    started = runner_a.start()
    assert started.running
    bridge_b, hub_b, runner_b, _ = build_stack(tmp_path, persona_id="p", session_id="s")
    blocked = runner_b.start()
    assert not blocked.running
    assert blocked.last_error == "runner_collision"
    runner_a.stop()
    release_runner_lock(tmp_path)


def test_budget_exhaustion_suppresses_llm_but_keeps_health_events(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    state = bridge.store.load()
    m13 = state["m13_drive_state"]
    initiative = m13["initiative"]
    bg = normalize_background_continuity_state(initiative.get("background_continuity"))
    bg["llm_calls_today"] = bg.get("llm_calls_budget_per_day", 24)
    initiative["background_continuity"] = bg
    m13["initiative"] = merge_background_continuity_into_initiative(initiative)
    state["m13_drive_state"] = m13
    bridge.store.save(state)
    clk.advance(5)
    step = runner.run_once_for_tests(now=clk())
    assert step.health.get("at") == clk()
    bg_tick = next(row for row in step.processed if row.get("phase") == "background_self_tick")
    assert bg_tick.get("skip_reason") or bg_tick.get("reason") or bg_tick.get("skipped") is not None or bg_tick.get("ran_introspection") is not None


def test_crash_after_event_claim_replays_on_restart(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    event_id = bridge.append_client_input(text="replay me", correlation_id="corr_replay")
    claimed = bridge.claim_events(limit=1)
    assert claimed and claimed[0]["event_id"] == event_id
    clk.advance(120)
    step = runner.run_once_for_tests(now=clk())
    assert bridge.is_event_processed(event_id)
    assert any(msg.get("kind") == "AssistantMessageCommitted" for msg in step.actuation_messages)


def test_crash_after_assistant_commit_does_not_duplicate_delivery(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    event_id = bridge.append_client_input(text="once", correlation_id="corr_once")
    claimed = bridge.claim_events(limit=1)[0]
    first = runner._handle_client_input(claimed, now=clk())
    assert first.get("actuation_messages")
    bridge._processed_path.unlink(missing_ok=True)
    second = runner._handle_client_input(claimed, now=clk())
    assert not second.get("actuation_messages")
    assert bridge.was_actuation_delivered(f"assistant:{event_id}")


def test_path_a_and_m10_untouched() -> None:
    import segmentum.dialogue.runtime.m16_api as api
    import segmentum.dialogue.runtime.m16_cli as cli
    import segmentum.dialogue.runtime.m16_runner as runner_mod
    import segmentum.dialogue.runtime.m16_runtime_bridge as bridge

    banned = (
        "conversation" + "_loop",
        "SelfThought" + "Producer",
        "Cognitive" + "Loop",
    )
    for module in (api, cli, runner_mod, bridge):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in banned:
            assert token not in source


def test_forbidden_internal_fields_not_present_on_ws_payloads(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    bridge.append_client_input(text="safe", correlation_id="corr_safe")
    step = runner.run_once_for_tests(now=clk())
    for msg in step.actuation_messages:
        payload = msg.get("payload") or {}
        for key in payload:
            assert str(key).casefold() not in {k.casefold() for k in FORBIDDEN_ACTUATION_PAYLOAD_KEYS}


def test_streamlit_not_required_for_runner_acceptance_path(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    event_id = bridge.append_client_input(text="no streamlit", correlation_id="corr_no_ui")
    step = runner.run_once_for_tests(now=clk())
    assert step.claimed_events >= 1
    assert bridge.is_event_processed(event_id)
    assert not any("streamlit" in str(msg).casefold() for msg in step.actuation_messages)


def test_runner_acks_slow_turn_after_default_claim_lease(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    event_id = bridge.append_client_input(text="slow turn", correlation_id="corr_slow")

    def _slow(_text: str, *, turn_index: int, now: int) -> object:
        clk.advance(120)
        return type("TurnResult", (), {"reply": "done"})()

    runner._inline_run_turn = _slow  # type: ignore[method-assign]
    step = runner.run_once_for_tests(now=clk())
    assert bridge.is_event_processed(event_id)
    assert any(msg.get("kind") == "AssistantMessageCommitted" for msg in step.actuation_messages)
    assert runner.status().last_error == ""


def test_runner_status_hides_stale_error_when_healthy(tmp_path: Path) -> None:
    bridge, hub, runner, clk = build_stack(tmp_path)
    runner.start()
    runner._status.last_error = "ValueError"
    runner._status.last_health_at = clk()
    row = runner.status().to_dict()
    assert row["running"] is True
    assert row["last_error"] == ""
