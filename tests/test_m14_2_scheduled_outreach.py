from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from segmentum.dialogue.runtime.m14_2_scheduled_intents import (
    ScheduledIntentStore,
    ensure_scheduled_open_item,
)


def _structured_payload(
    text: str,
    *,
    due_after_seconds: int | None = None,
    due_at: str = "",
    ordinary_language_intent: str = "Send the user-requested scheduled follow-up.",
) -> dict[str, object]:
    request: dict[str, object] = {
        "kind": "scheduled_outreach",
        "should_schedule": True,
        "basis": "user_explicit_request",
        "ordinary_language_intent": ordinary_language_intent,
    }
    if due_after_seconds is not None:
        request["due_after_seconds"] = due_after_seconds
    if due_at:
        request["due_at"] = due_at
    return {"user_text": text, "scheduled_outreach_requests": [request]}


def test_raw_english_overnight_request_does_not_create_scheduled_intent_by_keyword(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_1",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "think about this tonight and leave me a message tomorrow morning", "turn_index": 4},
    }
    assert store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 22, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    ) is None


def test_structured_english_overnight_request_creates_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_structured_1",
        "event_type": "UserMessageCommittedEvent",
        "payload": _structured_payload(
            "think about this tonight and leave me a message tomorrow morning",
            due_at="2026-05-20T09:00:00+08:00",
        ),
    }
    intent = store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 22, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    assert intent["kind"] == "scheduled_outreach"
    assert intent["status"] == "pending"
    assert intent["delivery_policy"]["require_m13_3_assessor"] is True
    assert "09:00:00" in str(intent["due_at"])


def test_raw_chinese_morning_request_does_not_create_scheduled_intent_by_keyword(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_cn",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "今晚想想，明天早上告诉我"},
    }
    assert store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    ) is None


def test_structured_chinese_morning_request_creates_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_cn_structured",
        "event_type": "UserMessageCommittedEvent",
        "payload": _structured_payload("今晚想想，明天早上告诉我", due_at="2026-05-20T09:00:00+08:00"),
    }
    intent = store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    assert "09:00:00" in str(intent["due_at"])


def test_schedule_keywords_alone_do_not_create_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_keyword_only",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "Tomorrow morning's meeting agenda has three items."},
    }
    assert store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    ) is None


def test_raw_idle_nudge_text_does_not_create_scheduled_intent_by_keyword(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    now = datetime(2026, 5, 22, 7, 57, tzinfo=ZoneInfo("Asia/Shanghai"))
    event = {
        "event_id": "env_idle_nudge",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "我有点乱，先缓一会儿。你等下如果我没继续，就拉我一把。"},
    }
    assert store.create_from_user_message_event(event, now=now) is None


def test_structured_idle_nudge_request_creates_short_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    now = datetime(2026, 5, 22, 7, 57, tzinfo=ZoneInfo("Asia/Shanghai"))
    event = {
        "event_id": "env_idle_nudge_structured",
        "event_type": "UserMessageCommittedEvent",
        "payload": _structured_payload(
            "I need a pause; come back to me if I do not continue.",
            due_after_seconds=120,
            ordinary_language_intent="Check back once after the user-requested pause.",
        ),
    }
    intent = store.create_from_user_message_event(event, now=now)
    assert intent is not None
    assert intent["kind"] == "scheduled_outreach"
    assert int(intent["due_at_epoch"]) - int(now.timestamp()) == 120
    assert intent["ordinary_language_intent"] == "Check back once after the user-requested pause."
    state: dict[str, object] = {"open_items": []}
    item = ensure_scheduled_open_item(state, intent)
    assert item["type"] == "scheduled_outreach"
    assert item["intent_id"] == intent["intent_id"]
    assert item["next_check"] == intent["due_at"]


def test_idle_nudge_fragments_alone_do_not_create_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    now = datetime(2026, 5, 22, 7, 57, tzinfo=ZoneInfo("Asia/Shanghai"))
    for index, text in enumerate(
        (
            "我有点乱，先缓一会儿。",
            "你等下提醒我一下会议。",
            "If I go quiet, it probably means I am cooking.",
        )
    ):
        event = {
            "event_id": f"env_idle_negative_{index}",
            "event_type": "UserMessageCommittedEvent",
            "payload": {"user_text": text},
        }
        assert store.create_from_user_message_event(event, now=now) is None


def test_reflection_or_reminder_phrases_do_not_become_outreach_without_structured_request(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    now = datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    for index, text in enumerate(
        [
            "think about this tomorrow",
            "keep thinking about this overnight",
            "remind me tomorrow about the meeting",
            "follow up tomorrow on the report",
        ]
    ):
        event = {
            "event_id": f"env_negative_{index}",
            "event_type": "UserMessageCommittedEvent",
            "payload": {"user_text": text},
        }
        assert store.create_from_user_message_event(event, now=now) is None


def test_structured_scheduled_intent_expires_after_due_window(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = store.create_from_user_message_event(
        {
            "event_id": "env_expire",
            "event_type": "UserMessageCommittedEvent",
            "payload": _structured_payload(
                "sleep on it and tell me tomorrow morning",
                due_at="2026-05-20T09:00:00+08:00",
            ),
        },
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    expired_at = int(datetime(2026, 5, 20, 14, 1, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    assert store.due_intents(now=expired_at) == []
    updated = store.get(str(intent["intent_id"]))
    assert updated is not None
    assert updated["status"] == "expired"


def test_scheduled_intent_creates_linked_open_item(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_oi",
        "event_type": "UserMessageCommittedEvent",
        "payload": _structured_payload("sleep on it and tell me tomorrow", due_at="2026-05-20T09:00:00+08:00"),
    }
    intent = store.create_from_user_message_event(
        event,
        now=datetime(2026, 5, 19, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    state: dict[str, object] = {"open_items": []}
    item = ensure_scheduled_open_item(state, intent)
    assert item["type"] == "scheduled_outreach"
    assert item["intent_id"] == intent["intent_id"]
    ensure_scheduled_open_item(state, intent)
    assert len(state["open_items"]) == 1  # type: ignore[arg-type]
