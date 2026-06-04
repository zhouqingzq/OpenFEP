from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from segmentum.dialogue.runtime.m16_api import M16Gateway
from segmentum.dialogue.runtime.telegram_connector import (
    TelegramBotIdentity,
    TelegramConnector,
    normalize_telegram_update,
)
from tests.m16_1_test_helpers import _Clock, full_opted_state


class FakeTelegramApi:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []

    def get_me(self) -> dict[str, object]:
        return {"id": 777000, "username": "hutao_bot"}

    def get_updates(self, **kwargs: object) -> list[dict[str, object]]:
        return list(self.updates)

    def send_message(
        self,
        *,
        chat_id: str,
        text: str,
        message_thread_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> dict[str, object]:
        row = {
            "chat_id": chat_id,
            "text": text,
            "message_thread_id": message_thread_id,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(row)
        return {"message_id": len(self.sent_messages), **row}


def _gateway_for_connector(tmp_path: Path) -> tuple[M16Gateway, _Clock]:
    clk = _Clock()
    gateway = M16Gateway(
        clock=clk,
        session_root_resolver=lambda persona_id, session_id: tmp_path / persona_id / session_id.replace(":", "_"),
    )
    return gateway, clk


def test_normalize_private_message_routes_to_dm_session() -> None:
    bot = TelegramBotIdentity(bot_user_id="777000", username="hutao_bot", account_scope="tg_main")
    normalized = normalize_telegram_update(
        {
            "update_id": 101,
            "message": {
                "message_id": 9,
                "text": "你好，胡桃",
                "chat": {"id": 123456, "type": "private"},
                "from": {"id": 123456, "first_name": "Alice"},
            },
        },
        persona_id="hutao-prod",
        bot=bot,
    )
    assert normalized is not None
    assert normalized.session_id == "telegram:tg_main:dm:user_123456"
    assert normalized.group_turn_envelope["addressed_participant_ids"] == ["telegram:tg_main:assistant:777000"]
    assert normalized.group_turn_envelope["visible_participant_ids"] == [
        "telegram:tg_main:user:123456",
        "telegram:tg_main:assistant:777000",
    ]
    assert normalized.ingress_evidence_band == "structured_full"


def test_normalize_group_topic_message_captures_reply_and_mentions() -> None:
    bot = TelegramBotIdentity(bot_user_id="777000", username="hutao_bot", account_scope="tg_main")
    text = "@hutao_bot 你怎么看 Bob 刚才那句？"
    normalized = normalize_telegram_update(
        {
            "update_id": 202,
            "message": {
                "message_id": 9001,
                "message_thread_id": 42,
                "text": text,
                "entities": [{"type": "mention", "offset": 0, "length": 10}],
                "chat": {"id": -100987654321, "type": "supergroup"},
                "from": {"id": 123, "first_name": "Alice"},
                "reply_to_message": {
                    "message_id": 9000,
                    "from": {"id": 456, "first_name": "Bob"},
                },
            },
        },
        persona_id="hutao-prod",
        bot=bot,
    )
    assert normalized is not None
    assert normalized.session_id == "telegram:tg_main:topic:chat_-100987654321:thread_42"
    assert normalized.group_turn_envelope["speaker_participant_id"] == "telegram:tg_main:user:123"
    assert normalized.group_turn_envelope["reply_to_turn_id"].endswith(":msg:9000")
    assert normalized.group_turn_envelope["addressed_participant_ids"] == ["telegram:tg_main:assistant:777000"]
    assert "@hutao_bot" in normalized.group_turn_envelope["explicit_mentions"]
    assert "telegram:tg_main:user:456" in normalized.group_turn_envelope["visible_participant_ids"]
    assert normalized.ingress_evidence_band == "structured_partial"


def test_connector_ingest_update_sends_runtime_reply_back_to_telegram(tmp_path: Path) -> None:
    gateway, clk = _gateway_for_connector(tmp_path)
    api = FakeTelegramApi()
    connector = TelegramConnector(
        persona_id="hutao-prod",
        account_scope="tg_main",
        gateway=gateway,
        api_client=api,
        clock=clk,
    )
    handle = gateway.get_or_create_session("hutao-prod", "telegram:tg_main:dm:user_123456")
    handle.bridge.store.save(full_opted_state())
    runner = gateway.ensure_runner(handle)
    runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="收到，我在。", action="answer")  # type: ignore[method-assign]

    result = connector.ingest_update(
        {
            "update_id": 301,
            "message": {
                "message_id": 77,
                "text": "胡桃，在吗？",
                "chat": {"id": 123456, "type": "private"},
                "from": {"id": 123456, "first_name": "Alice"},
            },
        }
    )

    assert result["accepted"] is True
    assert result["session_id"] == "telegram:tg_main:dm:user_123456"
    assert len(api.sent_messages) == 1
    assert api.sent_messages[0]["chat_id"] == "123456"
    assert api.sent_messages[0]["reply_to_message_id"] == 77
    assert api.sent_messages[0]["text"] == "收到，我在。"


def test_connector_no_reply_does_not_send_telegram_message(tmp_path: Path) -> None:
    gateway, clk = _gateway_for_connector(tmp_path)
    api = FakeTelegramApi()
    connector = TelegramConnector(
        persona_id="hutao-prod",
        account_scope="tg_main",
        gateway=gateway,
        api_client=api,
        clock=clk,
    )
    handle = gateway.get_or_create_session("hutao-prod", "telegram:tg_main:group:chat_-1001")
    handle.bridge.store.save(full_opted_state())
    runner = gateway.ensure_runner(handle)
    runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="", action="no_reply")  # type: ignore[method-assign]

    result = connector.ingest_update(
        {
            "update_id": 302,
            "message": {
                "message_id": 88,
                "text": "Bob，你来答一下",
                "chat": {"id": -1001, "type": "group"},
                "from": {"id": 123456, "first_name": "Alice"},
            },
        }
    )

    assert result["accepted"] is True
    assert api.sent_messages == []


def test_poll_once_advances_offset_across_multiple_sessions(tmp_path: Path) -> None:
    gateway, clk = _gateway_for_connector(tmp_path)
    api = FakeTelegramApi()
    connector = TelegramConnector(
        persona_id="hutao-prod",
        account_scope="tg_main",
        gateway=gateway,
        api_client=api,
        clock=clk,
    )
    dm_handle = gateway.get_or_create_session("hutao-prod", "telegram:tg_main:dm:user_123456")
    dm_handle.bridge.store.save(full_opted_state())
    dm_runner = gateway.ensure_runner(dm_handle)
    dm_runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="私聊收到", action="answer")  # type: ignore[method-assign]

    group_handle = gateway.get_or_create_session("hutao-prod", "telegram:tg_main:group:chat_-1001")
    group_handle.bridge.store.save(full_opted_state())
    group_runner = gateway.ensure_runner(group_handle)
    group_runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="群里收到", action="answer")  # type: ignore[method-assign]

    api.updates = [
        {
            "update_id": 401,
            "message": {
                "message_id": 1,
                "text": "你好",
                "chat": {"id": 123456, "type": "private"},
                "from": {"id": 123456, "first_name": "Alice"},
            },
        },
        {
            "update_id": 402,
            "message": {
                "message_id": 2,
                "text": "@hutao_bot 看一下",
                "entities": [{"type": "mention", "offset": 0, "length": 10}],
                "chat": {"id": -1001, "type": "group"},
                "from": {"id": 654321, "first_name": "Bob"},
            },
        },
    ]

    result = connector.poll_once(offset=400, timeout_seconds=0)

    assert result["updates"] == 2
    assert result["next_offset"] == 403
    assert [row["session_id"] for row in result["results"]] == [
        "telegram:tg_main:dm:user_123456",
        "telegram:tg_main:group:chat_-1001",
    ]
    assert [row["text"] for row in api.sent_messages] == ["私聊收到", "群里收到"]
