from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import (
    default_idle_introspection_state,
    evaluate_idle_structural_pre_filter,
    evaluate_idle_tick,
    gather_idle_structural_signals,
    normalize_idle_introspection_state,
    set_idle_introspection_user_opt_in,
    should_persist_idle_audit_events,
)
from segmentum.dialogue.runtime.m13_reward import normalize_affective_reward_proxy_state
from segmentum.dialogue.runtime.m13_initiative import (
    default_initiative_state,
    normalize_initiative_state,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPIdleResult, MVPStateStore


class _NoLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        raise AssertionError("M13.4 tests must not call LLM")


def _fully_opted_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [],
        "temporal_state": {
            "last_turn_at": 1_700_000_000,
            "last_user_turn_at": 1_700_000_000,
            "last_turn_index": 1,
        },
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    state["m13_drive_state"] = m13
    state.update(overrides)
    return state


def test_idle_introspection_disabled_by_default() -> None:
    idle = default_idle_introspection_state()
    assert idle["enabled"] is False
    assert idle["user_opt_in"] is False
    assert idle["max_per_session"] == 4
    initiative = normalize_initiative_state(default_initiative_state())
    assert normalize_idle_introspection_state(initiative.get("idle_introspection"))["enabled"] is False


def test_idle_introspection_requires_proactive_opt_in_first() -> None:
    state = _fully_opted_state()
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=False)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    idle = normalize_idle_introspection_state(
        normalize_initiative_state(m13.get("initiative")).get("idle_introspection")
    )
    assert idle["enabled"] is False
    assert idle["user_opt_in"] is False


def test_idle_tick_skips_when_user_typing() -> None:
    state = _fully_opted_state()
    _, check = evaluate_idle_tick(
        state,
        now=1_700_000_200,
        turn_index=2,
        user_active=True,
    )
    assert check.skip_reason == "user_active"
    assert any(e.get("type") == "IdleTickEvent" for e in check.events)


def test_idle_tick_skips_when_idle_under_threshold() -> None:
    state = _fully_opted_state(
        temporal_state={
            "last_turn_at": 1_700_000_190,
            "last_user_turn_at": 1_700_000_190,
            "last_turn_index": 1,
        }
    )
    _, check = evaluate_idle_tick(
        state,
        now=1_700_000_200,
        turn_index=2,
        user_active=False,
    )
    assert check.skip_reason == "idle_time_too_short"


def test_idle_tick_skips_when_min_interval_not_passed() -> None:
    state = _fully_opted_state()
    m13 = state["m13_drive_state"]
    from segmentum.dialogue.runtime.m13_initiative import merge_initiative_into_m13_state, normalize_initiative_state
    from segmentum.dialogue.runtime.m13_idle import merge_idle_introspection_into_initiative

    merged = merge_initiative_into_m13_state(m13)  # type: ignore[arg-type]
    initiative = merge_idle_introspection_into_initiative(merged.get("initiative"))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    idle["last_introspection_at"] = 1_700_000_150
    initiative["idle_introspection"] = idle
    merged["initiative"] = initiative
    state["m13_drive_state"] = merged

    _, check = evaluate_idle_tick(
        state,
        now=1_700_000_200,
        turn_index=2,
    )
    assert check.skip_reason == "min_interval_too_short"


def test_idle_tick_skips_when_no_structural_signal() -> None:
    state = _fully_opted_state(
        open_items=[{"id": "x", "status": "open", "next_check": "next_user_turn"}],
        temporal_state={"last_turn_at": 1_700_000_000, "last_turn_index": 1},
    )
    state, tick = evaluate_idle_tick(state, now=1_700_000_300, turn_index=2)
    assert tick.skip_reason == ""
    signals = tick.structural_signals
    assert signals is not None
    state, pre = evaluate_idle_structural_pre_filter(
        state,
        now=1_700_000_300,
        turn_index=2,
        signals=signals,
    )
    assert pre.skip_reason == "no_structural_signal"
    assert any(e.get("type") == "IdleIntrospectionSkipEvent" for e in pre.events)


def test_idle_tick_runs_stub_introspection_when_signals_present(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess"), llm=_NoLLM())  # type: ignore[name-defined]
    state = _fully_opted_state(
        open_items=[{"id": "oi", "status": "open", "title": "t", "next_check": "draft gates"}],
    )
    runtime.store.save(state)  # type: ignore[arg-type]
    runtime.set_initiative_user_opt_in(True)
    runtime.set_idle_introspection_opt_in(True)

    result = runtime.maybe_run_idle_introspection(turn_index=2, user_active=False)
    assert result["ran_introspection"] is True
    assert result["skip_reason"] == ""
    idle_result = result.get("idle_result")
    assert isinstance(idle_result, dict)
    assert idle_result.get("ran_llm") is False

    reloaded = runtime.store.load()
    initiative = reloaded["m13_drive_state"]["initiative"]
    idle = initiative["idle_introspection"]
    assert idle["reflection_count_this_session"] == 1
    assert idle["last_introspection_at"] > 0

    log_path = runtime.store.root / "conversation_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r.get("type") == "IdleIntrospectionTickEvent" for r in rows if r.get("event") == "m13_idle_audit")


def test_idle_introspection_never_emits_user_visible_message(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess2"), llm=_NoLLM())  # type: ignore[name-defined]
    state = _fully_opted_state(
        open_items=[{"id": "oi", "status": "open", "title": "t", "next_check": "later"}],
    )
    runtime.store.save(state)  # type: ignore[arg-type]
    runtime.set_initiative_user_opt_in(True)
    runtime.set_idle_introspection_opt_in(True)
    idle_result = runtime.run_idle_introspection_turn(
        now=1_700_000_400,
        turn_index=3,
        structural_signals=gather_idle_structural_signals(state, now=1_700_000_400, turn_index=3),
    )
    assert isinstance(idle_result, MVPIdleResult)
    assert idle_result.outreach_recommendation.get("should_outreach") is False
    log_path = runtime.store.root / "conversation_log.jsonl"
    if log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert not any(r.get("role") == "assistant" and r.get("text") for r in rows if "event" not in r)


def test_idle_introspection_session_cap_blocks_further_ticks() -> None:
    state = _fully_opted_state()
    from segmentum.dialogue.runtime.m13_initiative import merge_initiative_into_m13_state
    from segmentum.dialogue.runtime.m13_idle import merge_idle_introspection_into_initiative

    merged = merge_initiative_into_m13_state(state["m13_drive_state"])  # type: ignore[arg-type]
    initiative = merge_idle_introspection_into_initiative(merged.get("initiative"))
    idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
    idle["reflection_count_this_session"] = idle["max_per_session"]
    initiative["idle_introspection"] = idle
    merged["initiative"] = initiative
    state["m13_drive_state"] = merged

    _, check = evaluate_idle_tick(state, now=1_700_001_000, turn_index=5)
    assert check.skip_reason == "session_limit_reached"


def test_idle_audit_events_are_persisted(tmp_path: Path) -> None:
    runtime = MVPDialogueRuntime(store=MVPStateStore(tmp_path / "sess3"), llm=_NoLLM())  # type: ignore[name-defined]
    state = _fully_opted_state(
        open_items=[{"id": "oi", "status": "open", "title": "t", "next_check": "check later"}],
    )
    runtime.store.save(state)  # type: ignore[arg-type]
    runtime.set_initiative_user_opt_in(True)
    runtime.set_idle_introspection_opt_in(True)
    runtime.maybe_run_idle_introspection(turn_index=1)

    log_path = runtime.store.root / "conversation_log.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_types = {r.get("type") for r in rows if r.get("event") == "m13_idle_audit"}
    assert "IdleTickEvent" in audit_types


def test_disabling_opt_in_resets_state() -> None:
    state = _fully_opted_state()
    m13 = set_idle_introspection_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=False)
    from segmentum.dialogue.runtime.m13_initiative import normalize_initiative_state

    idle = normalize_idle_introspection_state(
        normalize_initiative_state(m13.get("initiative")).get("idle_introspection")
    )
    assert idle["enabled"] is False
    assert idle["last_skip_reason"] == "disabled"


def test_gather_signals_just_outreached_after_proactive() -> None:
    state = _fully_opted_state(
        open_items=[{"id": "oi", "status": "open", "next_check": "later"}],
        temporal_state={"last_turn_at": 1_700_000_000, "last_turn_index": 5},
    )
    from segmentum.dialogue.runtime.m13_initiative import merge_initiative_into_m13_state
    from segmentum.dialogue.runtime.m13_idle import merge_idle_introspection_into_initiative

    merged = merge_initiative_into_m13_state(state["m13_drive_state"])  # type: ignore[arg-type]
    initiative = merge_idle_introspection_into_initiative(merged.get("initiative"))
    initiative["last_proactive_turn_at"] = 1_700_000_050
    initiative["last_proactive_turn_index"] = 5
    merged["initiative"] = initiative
    state["m13_drive_state"] = merged

    signals = gather_idle_structural_signals(state, now=1_700_000_100, turn_index=5)
    assert signals.just_outreached_recently is True
    assert signals.should_run_llm() is False


def test_idle_tick_uses_last_user_turn_not_proactive_clock() -> None:
    state = _fully_opted_state(
        temporal_state={
            "last_turn_at": 1_700_000_200,
            "last_user_turn_at": 1_700_000_000,
            "last_turn_index": 2,
        }
    )
    _, check = evaluate_idle_tick(state, now=1_700_000_300, turn_index=2)
    assert check.skip_reason == ""


def test_expired_pending_settlement_does_not_trigger_structural_signal() -> None:
    state = _fully_opted_state()
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    reward = normalize_affective_reward_proxy_state(m13.get("affective_reward_proxy"))
    reward["pending_settlements"] = [
        {
            "pending_id": "old",
            "prior_turn_index": 0,
            "expires_after_turns": 2,
        }
    ]
    m13["affective_reward_proxy"] = reward
    state["m13_drive_state"] = m13
    signals = gather_idle_structural_signals(state, now=1_700_000_500, turn_index=10)
    assert signals.unsettled_pending_settlement_count == 0
    assert signals.should_run_llm() is False


def test_assessable_pending_settlement_triggers_structural_signal() -> None:
    state = _fully_opted_state()
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    reward = normalize_affective_reward_proxy_state(m13.get("affective_reward_proxy"))
    reward["pending_settlements"] = [
        {
            "pending_id": "due",
            "prior_turn_index": 1,
            "expires_after_turns": 5,
        }
    ]
    m13["affective_reward_proxy"] = reward
    state["m13_drive_state"] = m13
    signals = gather_idle_structural_signals(state, now=1_700_000_500, turn_index=3)
    assert signals.unsettled_pending_settlement_count == 1
    assert signals.should_run_llm() is True


def test_path_feels_stale_proxy_reads_reward_state() -> None:
    state = _fully_opted_state()
    m13 = normalize_m13_drive_state(state["m13_drive_state"])  # type: ignore[arg-type]
    reward = normalize_affective_reward_proxy_state(m13.get("affective_reward_proxy"))
    reward["path_feels_stale_proxy"] = True
    m13["affective_reward_proxy"] = reward
    state["m13_drive_state"] = m13
    signals = gather_idle_structural_signals(state, now=1_700_000_500, turn_index=2)
    assert signals.path_feels_stale_proxy is True


def test_idle_audit_throttle_skips_repeated_noisy_skip() -> None:
    idle = default_idle_introspection_state()
    idle["last_audit_logged_at"] = 1_700_000_000
    idle["last_audit_logged_skip_reason"] = "idle_time_too_short"
    assert (
        should_persist_idle_audit_events(
            idle,
            skip_reason="idle_time_too_short",
            now=1_700_000_030,
        )
        is False
    )
    assert (
        should_persist_idle_audit_events(
            idle,
            skip_reason="no_structural_signal",
            now=1_700_000_030,
        )
        is True
    )


def test_proactive_off_disables_idle_introspection() -> None:
    state = _fully_opted_state()
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=False)  # type: ignore[arg-type]
    from segmentum.dialogue.runtime.m13_initiative import normalize_initiative_state

    idle = normalize_idle_introspection_state(
        normalize_initiative_state(m13.get("initiative")).get("idle_introspection")
    )
    assert idle["last_skip_reason"] == "disabled"
