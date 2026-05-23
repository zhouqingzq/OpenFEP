from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state
from segmentum.dialogue.runtime.m13_idle import set_idle_introspection_user_opt_in
from segmentum.dialogue.runtime.m13_initiative import (
    ProactiveInitiativeCheckResult,
    set_initiative_user_opt_in,
)
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    enqueue_outreach_proposal,
    load_queued_outreach,
    outreach_suppression_is_transient,
    pop_next_pending_outreach,
    set_background_continuity_opt_in,
)
from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore
from segmentum.dialogue.runtime.m14_2_scheduled_intents import (
    ScheduledIntentStore,
)
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


def _full_opted_state(**overrides: object) -> dict[str, object]:
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
    state.update(overrides)
    return state


class _ScheduledLLM:
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
                "outreach_recommendation": {"should_outreach": False, "reason": "scheduled_outreach"},
            }
        if "M13" in system_prompt:
            return {"allow_delivery": True, "confidence": 0.9, "violation_codes": [], "reason_codes": []}
        return {
            "thought_type": "short",
            "llm_thinking_result": {"debug_summary": "scheduled"},
            "reply": "scheduled follow-up",
            "reply_action": "answer",
            "disclosure_action": "none",
            "new_expectations": [],
            "memory_writes": [],
            "self_cognition_patch": {"apply": False},
            "open_item_writes": [],
            "habit_updates": [],
            "memory_dynamics_note": "",
        }


def test_raw_text_semantics_do_not_create_scheduled_intent_without_llm_structured_output(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    now = datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    for index, text in enumerate(
        [
            "Tomorrow morning tell me the weather",
            "tonight mull it over; remember to say something when I come back",
            "睡一晚，醒来给我留句话",
            "think about this tomorrow",
            "remind me tomorrow about the meeting",
            "明天早上告诉我会议几点",
        ]
    ):
        event = {
            "event_id": f"env_raw_semantics_{index}",
            "event_type": "UserMessageCommittedEvent",
            "payload": {"user_text": text},
        }
        assert store.create_from_user_message_event(event, now=now) is None


def test_delivery_surface_unavailable_is_transient_suppression() -> None:
    assert outreach_suppression_is_transient("delivery_surface_unavailable") is True


def test_outbox_drain_waits_until_due_at(tmp_path: Path) -> None:
    enqueue_outreach_proposal(
        tmp_path,
        proposal={
            "proposal_id": "prop_future",
            "trigger": "scheduled_outreach",
            "source_intent_id": "intent_future",
            "ordinary_language_intent": "future",
            "proposed_topic": "scheduled outreach",
        },
        now=100,
        ttl_seconds=3600,
        due_at=200,
        source_intent_id="intent_future",
    )
    assert pop_next_pending_outreach(tmp_path, now=199) is None
    due = pop_next_pending_outreach(tmp_path, now=200)
    assert due is not None
    assert due["proposal_id"] == "prop_future"


def test_opt_out_blocks_preparation_but_preserves_audit(tmp_path: Path) -> None:
    state = _full_opted_state()
    m13 = set_background_continuity_opt_in(state["m13_drive_state"], enabled=False, runner_kind="none")  # type: ignore[arg-type]
    state["m13_drive_state"] = m13
    store = MVPStateStore(tmp_path)
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_opt", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    assert daemon.prepare_due_intents(now=due) == []
    updated = intent_store.get(intent["intent_id"])
    assert updated is not None
    assert updated["status"] == "suppressed"
    log = (tmp_path / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "ScheduledIntentSuppressedEvent" in log
    assert "user_opted_out" in log


def test_budget_exhaustion_records_scheduled_intent_suppression(tmp_path: Path) -> None:
    state = _full_opted_state()
    m13 = state["m13_drive_state"]
    assert isinstance(m13, dict)
    bg = m13["initiative"]["background_continuity"]  # type: ignore[index]
    bg["llm_calls_today"] = bg["llm_calls_budget_per_day"]
    store = MVPStateStore(tmp_path)
    store.save(state)
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_budget", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    assert daemon.prepare_due_intents(now=due) == []
    updated = intent_store.get(intent["intent_id"])
    assert updated is not None
    assert updated["status"] == "suppressed"
    assert updated.get("last_suppression_reason") == "budget_exhausted"


def test_due_intent_preparation_runs_idle_reflection(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_reflect", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    daemon.prepare_due_intents(now=due)
    log = (tmp_path / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert '"event": "m14_idle_audit"' in log
    assert "IdleIntrospectionPlanEvent" in log
    assert "ScheduledIntentPreparedEvent" in log


def test_outbox_recovery_after_crash_before_intent_prepared(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_crash", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    intent_id = str(intent["intent_id"])
    intent_store.mark_status(intent_id, "preparing", now=int(datetime(2026, 5, 20, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()))
    enqueue_outreach_proposal(
        tmp_path,
        proposal={
            "proposal_id": "prop_crash",
            "trigger": "scheduled_outreach",
            "source_intent_id": intent_id,
            "ordinary_language_intent": "test",
            "proposed_topic": "scheduled outreach",
            "evidence_refs": ["env_crash"],
            "persona_id": "p",
            "session_id": "s",
        },
        now=int(datetime(2026, 5, 20, 9, 5, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()),
        ttl_seconds=3600,
        source_intent_id=intent_id,
    )
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    result = daemon.prepare_intent(intent_store.get(intent_id) or intent, now=due)
    assert result is not None
    assert result.get("recovered") is True
    assert intent_store.get(intent_id)["status"] == "prepared"  # type: ignore[index]
    assert len(load_queued_outreach(tmp_path)) == 1


def _suppressed_check(state: dict[str, object], reason: str) -> tuple[dict[str, object], ProactiveInitiativeCheckResult]:
    return state, ProactiveInitiativeCheckResult(
        proposal=None,
        suppression_reason=reason,
        events=[],
        state_fields_read=[],
    )


def test_transient_delivery_suppression_keeps_outbox_pending(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_transient", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    daemon.prepare_due_intents(now=due)
    with patch(
        "segmentum.dialogue.runtime.mvp_loop.evaluate_proactive_initiative",
        side_effect=lambda state, **kwargs: _suppressed_check(state, "cooldown_active"),
    ):
        result = runtime.maybe_drain_queued_outreach(turn_index=4, now=due)
    assert result["drained"] is False
    assert result.get("transient") is True
    rows = load_queued_outreach(tmp_path)
    assert rows and rows[0]["status"] == "pending"
    log = (tmp_path / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "OutboxDeliveryTransientSuppressionEvent" in log


def test_hard_delivery_suppression_marks_outbox_suppressed(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s")
    intent_store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = intent_store.create_from_user_message_event(
        {"event_id": "env_hard", "payload": _structured_payload()},
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    due = int(datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    daemon.prepare_due_intents(now=due)
    with patch(
        "segmentum.dialogue.runtime.mvp_loop.evaluate_proactive_initiative",
        side_effect=lambda state, **kwargs: _suppressed_check(state, "safety_risk"),
    ):
        result = runtime.maybe_drain_queued_outreach(turn_index=4, now=due)
    assert result["drained"] is False
    rows = load_queued_outreach(tmp_path)
    assert rows and rows[0]["status"] == "suppressed"
    log = (tmp_path / "conversation_log.jsonl").read_text(encoding="utf-8")
    assert "OutboxDeliveryHardSuppressionEvent" in log


def test_ui_ping_does_not_create_scheduled_intent(tmp_path: Path) -> None:
    store = MVPStateStore(tmp_path)
    store.save(_full_opted_state())
    runtime = MVPDialogueRuntime(store=store, llm=_ScheduledLLM())  # type: ignore[arg-type]
    daemon = M142SelfLoopDaemon(runtime, persona_id="p", session_id="s", clock=lambda: 1_800_000_000)
    event_store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: 1_800_000_000)
    event_store.append_event("UIPingEvent", {"surface": "streamlit_chat"}, source="streamlit", correlation_id="ping-1")
    daemon.tick_once(record_clock_wake=False)
    assert ScheduledIntentStore(tmp_path, persona_id="p", session_id="s").list_intents() == []
    assert load_queued_outreach(tmp_path) == []


def test_background_opt_in_does_not_start_inline_runner() -> None:
    from segmentum.dialogue.runtime.chat import ChatInterface

    chat = ChatInterface.__new__(ChatInterface)
    chat._mvp_runtime = None
    chat._background_runner = None
    chat._session_id = "s"
    started = {"called": False}

    def _fake_start() -> None:
        started["called"] = True

    chat._ensure_runtime_fields = lambda: None  # type: ignore[method-assign]
    chat._maybe_enable_mvp_llm_runtime = lambda: None  # type: ignore[method-assign]
    chat._stop_background_runner = lambda: None  # type: ignore[method-assign]
    chat._start_background_runner = _fake_start  # type: ignore[method-assign]
    chat.set_background_continuity_opt_in = ChatInterface.set_background_continuity_opt_in.__get__(chat, ChatInterface)  # type: ignore[method-assign]

    class _RuntimeStub:
        def set_background_continuity_opt_in(self, enabled: bool, *, runner_kind: str) -> dict[str, object]:
            return {"user_opt_in": enabled, "runner_kind": runner_kind}

    chat._mvp_runtime = _RuntimeStub()  # type: ignore[assignment]
    bg = chat.set_background_continuity_opt_in(True)
    assert bg["runner_kind"] == "standalone_daemon"
    assert started["called"] is False


def test_user_message_event_appended_to_environment_store(tmp_path: Path) -> None:
    store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: 100)
    event_id = store.append_event(
        "UserMessageCommittedEvent",
        {"user_text": "think about this tonight and leave me a message tomorrow morning", "turn_index": 2},
        source="streamlit",
        correlation_id="turn-2",
    )
    rows = store.query_events(event_types={"UserMessageCommittedEvent"})
    assert len(rows) == 1
    assert rows[0]["event_id"] == event_id
