from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from segmentum.dialogue.runtime.m14_2_scheduled_intents import (
    ScheduledIntentStore,
    ensure_scheduled_open_item,
    resolve_due_at,
)


def test_explicit_english_overnight_request_creates_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_1",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "think about this tonight and leave me a message tomorrow morning", "turn_index": 4},
    }
    intent = store.create_from_user_message_event(event, now=datetime(2026, 5, 19, 22, 0, tzinfo=ZoneInfo("Asia/Shanghai")))
    assert intent is not None
    assert intent["kind"] == "scheduled_outreach"
    assert intent["status"] == "pending"
    assert intent["delivery_policy"]["require_m13_3_assessor"] is True


def test_explicit_chinese_morning_message_request_creates_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_cn",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "今晚想想，明天早上告诉我"},
    }
    intent = store.create_from_user_message_event(event, now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")))
    assert intent is not None
    assert "09:00:00" in str(intent["due_at"])


def test_schedule_keywords_alone_do_not_create_scheduled_intent(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_keyword_only",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "Tomorrow morning's meeting agenda has three items."},
    }
    assert store.create_from_user_message_event(event, now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))) is None


def test_chinese_immediate_tell_request_without_message_leave_is_not_scheduled_outreach(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    now = datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    event = {
        "event_id": "env_cn_fp",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "明天早上告诉我会议几点"},
    }
    assert store.create_from_user_message_event(event, now=now) is None


def test_reflection_or_reminder_phrases_do_not_become_outreach_without_message_request(tmp_path: Path) -> None:
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


def test_overdue_scheduled_intent_expires_after_due_window(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    intent = store.create_from_user_message_event(
        {
            "event_id": "env_expire",
            "event_type": "UserMessageCommittedEvent",
            "payload": {"user_text": "sleep on it and tell me tomorrow morning"},
        },
        now=datetime(2026, 5, 19, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert intent is not None
    expired_at = int(datetime(2026, 5, 20, 14, 1, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    assert store.due_intents(now=expired_at) == []
    updated = store.get(str(intent["intent_id"]))
    assert updated is not None
    assert updated["status"] == "expired"


def test_tomorrow_morning_defaults_to_0900_local_time() -> None:
    due, adjusted = resolve_due_at(
        "sleep on it and tell me tomorrow morning",
        now=datetime(2026, 5, 19, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert due.hour == 9
    assert due.day == 20
    assert adjusted == 0


def test_past_due_time_rolls_to_next_plausible_future_window() -> None:
    due, adjusted = resolve_due_at(
        "sleep on it and tell me at 8am",
        now=datetime(2026, 5, 19, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert due > datetime(2026, 5, 19, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert adjusted == 1


def test_scheduled_intent_creates_linked_open_item(tmp_path: Path) -> None:
    store = ScheduledIntentStore(tmp_path, persona_id="p", session_id="s")
    event = {
        "event_id": "env_oi",
        "event_type": "UserMessageCommittedEvent",
        "payload": {"user_text": "sleep on it and tell me tomorrow"},
    }
    intent = store.create_from_user_message_event(event, now=datetime(2026, 5, 19, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")))
    assert intent is not None
    state: dict[str, object] = {"open_items": []}
    item = ensure_scheduled_open_item(state, intent)
    assert item["type"] == "scheduled_outreach"
    assert item["intent_id"] == intent["intent_id"]
    ensure_scheduled_open_item(state, intent)
    assert len(state["open_items"]) == 1  # type: ignore[arg-type]
