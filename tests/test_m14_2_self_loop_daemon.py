from __future__ import annotations

from datetime import datetime
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import set_initiative_user_opt_in
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    load_queued_outreach,
    set_background_continuity_opt_in,
)
from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore
from segmentum.dialogue.runtime.m14_2_scheduled_intents import ScheduledIntentStore
from segmentum.dialogue.runtime.m14_2_self_loop import M142SelfLoopDaemon
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore


def _structured_payload(text: str = "sleep on it and tell me tomorrow morning") -> dict[str, object]:
    return {
        "user_text": text,
        "scheduled_outreach_requests": [
            {
                "kind": "scheduled_outreach",
                "should_schedule": True,
                "basis": "user_explicit_request",
                "ordinary_language_intent": "Send the user-requested scheduled follow-up.",
                "due_at": "2026-05-20T09:00:00+08:00",
            }
        ],
    }


def _full_opted_state() -> dict[str, object]:
    state: dict[str, object] = {
        "open_items": [],
        "short_term_memory": [],
        "long_term_memory": [],
        "pending_expectations": [],
        "self_cognition": {"patch_history": []},
        "temporal_state": {
            "last_turn_at": 1_800_000_000,
            "last_user_turn_at": 1_800_000_000,
            "last_turn_index": 3,
            "last_reply": "ok",
        },
        "m13_drive_state": default_m13_drive_state(),
    }
    m13 = set_initiative_user_opt_in(state["m13_drive_state"], enabled=True)  # type: ignore[arg-type]
    m13 = set_idle_introspection_user_opt_in(m13, enabled=True)
    m13 = set_background_continuity_opt_in(m13, enabled=True, runner_kind="standalone_daemon")
    state["m13_drive_state"] = m13
    return state


class _DeliveryLLM:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "M14" in system_prompt or "idle_introspection" in user_prompt:
            return {
                "mode": "idle_introspection",
                "reflection_focus": {
                    "topic": "scheduled outreach",
                    "evidence_refs": ["env_due"],
                    "reflection_kind": "open_item",
                },
                "self_cognition_patch_proposal": {"apply": False},
                "memory_consolidation_proposals": [],
                "open_item_proposals": [],
                "outreach_recommendation": {
                    "should_outreach": False,
                    "reason": "scheduled_outreach_prepared_separately",
                },
            }
        if "M13" in system_prompt:
            return {"allow_delivery": True, "confidence": 0.9, "violation_codes": [], "reason_codes": []}
        return {
            "thought_type": "short",
            "llm_thinking_result": {"debug_summary": "scheduled"},
            "reply": "短跟进：这是昨晚请求的简短留言。",
            "reply_action": "answer",
            "disclosure_action": "none",
            "new_expectations": [],
            "memory_writes": [],
            "self_cognition_patch": {"apply": False},
            "open_item_writes": [],
            "habit_updates": [],
            "memory_dynamics_note": "",
        }


class _DirectOutreachLLM(_DeliveryLLM):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "M14" in system_prompt or "idle_introspection" in user_prompt:
            return {
                "mode": "idle_introspection",
                "reflection_focus": {
                    "topic": "scheduled outreach",
                    "evidence_refs": ["env_due"],
                    "reflection_kind": "open_item",
                },
                "self_cognition_patch_proposal": {"apply": False},
                "memory_consolidation_proposals": [],
                "open_item_proposals": [],
                "outreach_recommendation": {
                    "should_outreach": True,
                    "suggested_intent": "send the scheduled follow-up",
                    "evidence_refs": ["env_due"],
                },
            }
        if "M13" in system_prompt:
            return {"allow_delivery": True, "confidence": 0.9, "violation_codes": [], "reason_codes": []}
        return {
            "thought_type": "short",
            "llm_thinking_result": {"debug_summary": "direct"},
            "reply": "DIRECT_VISIBLE_REPLY_SHOULD_NOT_HAPPEN",
            "reply_action": "answer",
            "disclosure_action": "none",
            "new_expectations": [],
            "memory_writes": [],
            "self_cognition_patch": {"apply": False},
            "open_item_writes": [],
            "habit_updates": [],
            "memory_dynamics_note": "",
        }


def test_self_loop_daemon_records_clock_wake_events(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s", clock=lambda: 1_800_000_000)
    result = daemon.tick_once(record_clock_wake=True)
    events = daemon.event_store.query_events(event_types={"ClockWakeEvent"})
    assert events
    assert result["claimed_events"] >= 1


def test_user_message_event_creates_scheduled_intent_and_open_item(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s", clock=lambda: 1_800_000_000)
    event_store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: 1_800_000_000)
    event_store.append_event(
        "UserMessageCommittedEvent",
        {**_structured_payload("think about this tonight and leave me a message tomorrow morning"), "turn_index": 3},
        source="test",
        correlation_id="turn-scheduled",
    )
    daemon.tick_once(record_clock_wake=False)
    intents = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s").list_intents()
    assert len(intents) == 1
    assert any(row.get("type") == "scheduled_outreach" for row in store.load()["open_items"])


def test_due_scheduled_intent_prepares_one_outbox_entry(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s", clock=lambda: 1_800_050_000)
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_due",
        "payload": {**_structured_payload(), "turn_index": 3},
    }
    intent = intent_store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    daemon.prepare_due_intents(now=int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()))
    daemon.prepare_due_intents(now=int(datetime(2026, 5, 20, 10, 5, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()))
    rows = load_queued_outreach(tmp_path)
    assert len([row for row in rows if row.get("source_intent_id") == intent["intent_id"]]) == 1
    assert rows[0]["trigger"] == "scheduled_outreach"
    assert rows[0]["delivery_policy"]["require_m13_3_assessor"] is True
    assert intent_store.get(intent["intent_id"])["status"] == "prepared"  # type: ignore[index]


def test_outbox_drain_uses_m13_3_and_closes_intent_and_open_item(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_delivery", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    daemon.prepare_due_intents(now=due)
    result = runtime.maybe_drain_queued_outreach(turn_index=4, now=due)
    assert result["drained"] is True
    assert intent_store.get(intent["intent_id"])["status"] == "delivered"  # type: ignore[index]
    linked = [row for row in store.load()["open_items"] if row.get("intent_id") == intent["intent_id"]]
    assert linked and linked[0]["status"] == "closed"


def test_runner_never_writes_visible_assistant_text_directly(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_direct", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    daemon.prepare_due_intents(now=int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()))
    log_path = tmp_path / "conversation_log.jsonl"
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    assert '"event": "proactive_turn"' not in text
    assert "短跟进" not in text


def test_due_preparation_defers_idle_outreach_instead_of_direct_delivery(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DirectOutreachLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_direct_branch", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    daemon.prepare_due_intents(now=int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()))
    log_path = tmp_path / "conversation_log.jsonl"
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    assert '"event": "proactive_turn"' not in text
    assert "DIRECT_VISIBLE_REPLY_SHOULD_NOT_HAPPEN" not in text
    assert "ScheduledIntentPreparedEvent" in text
    rows = load_queued_outreach(tmp_path)
    assert len([row for row in rows if row.get("source_intent_id") == intent["intent_id"]]) == 1


def test_run_forever_drives_background_self_tick_even_without_due_intents(tmp_path: Path) -> None:
    """Field gap reproduction: the standalone daemon must call
    ``run_background_self_tick`` each loop iteration so ``background_ticks_today``
    increments on plain periodic wakes, not only when a ``ScheduledIntent`` is
    prepared. Without this wiring, daemons appear alive (`SelfLoopDaemonHealthEvent`)
    but `m13_drive_state.background_continuity.ticks_today` stays at 0.
    """
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")

    bg_before = store.load()["m13_drive_state"]["initiative"]["background_continuity"]
    assert int(bg_before.get("ticks_today", 0) or 0) == 0

    result = daemon._run_background_self_tick_safely()

    bg_after = store.load()["m13_drive_state"]["initiative"]["background_continuity"]
    assert int(bg_after.get("ticks_today", 0) or 0) >= 1
    assert int(bg_after.get("last_tick_at", 0) or 0) > 0
    assert result.get("ran_introspection") is False
    log_text = (tmp_path / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "BackgroundIdleTickEvent" in log_text


def test_run_forever_background_self_tick_safely_swallows_exceptions(tmp_path: Path) -> None:
    """A single faulty background self-tick must not break the M14.2 event loop;
    the daemon should audit a ``BackgroundIdleTickEvent`` with ``skip_reason='tick_error'``
    and keep running.
    """
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_DeliveryLLM())  # type: ignore[arg-type]

    def _boom(*, runner_kind: str) -> dict[str, object]:
        raise RuntimeError("simulated background tick failure")

    runtime.run_background_self_tick = _boom  # type: ignore[assignment]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")

    result = daemon._run_background_self_tick_safely()

    assert result == {"skip_reason": "tick_error", "ran_introspection": False}
    log_text = (tmp_path / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "tick_error" in log_text
    assert "simulated background tick failure" in log_text


def test_self_loop_daemon_cli_accepts_persona_session_contract() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "segmentum.dialogue.runtime.m14_2_self_loop", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "--persona" in result.stdout
    assert "--session" in result.stdout
