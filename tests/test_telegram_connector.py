from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from segmentum.dialogue.runtime.m14_1_background_continuity import enqueue_outreach_proposal
from segmentum.dialogue.runtime.m16_api import M16Gateway
from segmentum.dialogue.runtime.telegram_connector import (
    TelegramBotIdentity,
    TelegramConnector,
    normalize_telegram_update,
    telegram_delivery_target_from_session_id,
)
from tests.m16_1_test_helpers import _Clock, full_opted_state
from tests.test_m14_2_self_loop_daemon import _DeliveryLLM
from tests.test_mvp_dialogue_runtime import FakeJSONLLM


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
                "text": "浣犲ソ锛岃儭妗?",
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
    text = "@hutao_bot 浣犳€庝箞鐪?Bob 鍒氭墠閭ｅ彞锛?"
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
    assert normalized.group_turn_envelope["addressed_participant_ids"] == [
        "telegram:tg_main:assistant:777000",
        "telegram:tg_main:user:456",
    ]
    assert normalized.group_turn_envelope["assistant_surface_label"] == "hutao_bot"
    assert "explicit_mentions" not in normalized.group_turn_envelope
    assert "telegram:tg_main:user:456" in normalized.group_turn_envelope["visible_participant_ids"]
    assert normalized.ingress_evidence_band == "structured_partial"


def test_normalize_group_plain_mentions_address_other_users_not_bot() -> None:
    bot = TelegramBotIdentity(bot_user_id="777000", username="hutao_bot", account_scope="tg_main")
    text = "@bob @carol you two decide."
    normalized = normalize_telegram_update(
        {
            "update_id": 203,
            "message": {
                "message_id": 9002,
                "text": text,
                "entities": [
                    {"type": "mention", "offset": 0, "length": 4},
                    {"type": "mention", "offset": 5, "length": 6},
                ],
                "chat": {"id": -100987654321, "type": "supergroup"},
                "from": {"id": 123, "first_name": "Alice"},
            },
        },
        persona_id="hutao-prod",
        bot=bot,
    )
    assert normalized is not None
    addressed = normalized.group_turn_envelope["addressed_participant_ids"]
    assert addressed == [
        "telegram:tg_main:username:bob",
        "telegram:tg_main:username:carol",
    ]
    assert bot.assistant_participant_id not in addressed
    assert normalized.group_turn_envelope["mentioned_participant_ids"] == addressed
    assert "explicit_mentions" not in normalized.group_turn_envelope


def test_normalize_text_mention_of_bot_uses_only_assistant_identity() -> None:
    bot = TelegramBotIdentity(bot_user_id="777000", username="hutao_bot", account_scope="tg_main")
    normalized = normalize_telegram_update(
        {
            "update_id": 204,
            "message": {
                "message_id": 9003,
                "text": "Hutao please answer.",
                "entities": [
                    {
                        "type": "text_mention",
                        "offset": 0,
                        "length": 5,
                        "user": {"id": 777000, "first_name": "Hutao"},
                    }
                ],
                "chat": {"id": -100987654321, "type": "supergroup"},
                "from": {"id": 123, "first_name": "Alice"},
            },
        },
        persona_id="hutao-prod",
        bot=bot,
    )
    assert normalized is not None
    assistant_id = "telegram:tg_main:assistant:777000"
    assert normalized.group_turn_envelope["addressed_participant_ids"] == [assistant_id]
    assert normalized.group_turn_envelope["mentioned_participant_ids"] == [assistant_id]
    assert "telegram:tg_main:user:777000" not in normalized.group_turn_envelope["visible_participant_ids"]


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
    runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="鏀跺埌锛屾垜鍦ㄣ€?", action="answer")  # type: ignore[method-assign]

    result = connector.ingest_update(
        {
            "update_id": 301,
            "message": {
                "message_id": 77,
                "text": "鑳℃锛屽湪鍚楋紵",
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
    events = handle.bridge.event_store.query_events(event_types={"ClientInputCommittedEvent"})
    assert events[-1]["payload"]["ingress_evidence_band"] == "structured_full"
    assert api.sent_messages[0]["text"] == "鏀跺埌锛屾垜鍦ㄣ€?"


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
                "text": "Bob锛屼綘鏉ョ瓟涓€涓?",
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
    dm_runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="绉佽亰鏀跺埌", action="answer")  # type: ignore[method-assign]

    group_handle = gateway.get_or_create_session("hutao-prod", "telegram:tg_main:group:chat_-1001")
    group_handle.bridge.store.save(full_opted_state())
    group_runner = gateway.ensure_runner(group_handle)
    group_runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="缇ら噷鏀跺埌", action="answer")  # type: ignore[method-assign]

    api.updates = [
        {
            "update_id": 401,
            "message": {
                "message_id": 1,
                "text": "浣犲ソ",
                "chat": {"id": 123456, "type": "private"},
                "from": {"id": 123456, "first_name": "Alice"},
            },
        },
        {
            "update_id": 402,
            "message": {
                "message_id": 2,
                "text": "@hutao_bot 鐪嬩竴涓?",
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
    assert [row["text"] for row in api.sent_messages] == ["绉佽亰鏀跺埌", "缇ら噷鏀跺埌"]


def test_telegram_delivery_target_from_session_id_supports_topic_targets() -> None:
    target = telegram_delivery_target_from_session_id(
        "telegram:tg_main:topic:chat_-100987654321:thread_42"
    )
    assert target is not None
    assert target.chat_id == "-100987654321"
    assert target.message_thread_id == 42


def test_connector_idle_drain_no_traceable_outreach_sends_nothing(tmp_path: Path) -> None:
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
    gateway.ensure_runner(handle)

    result = connector.drain_proactive_once()

    assert result["sessions_considered"] == 1
    assert result["sent_messages"] == []
    assert api.sent_messages == []


def test_connector_idle_drain_skips_traceable_proactive_messages(tmp_path: Path) -> None:
    clk = _Clock()
    gateway = M16Gateway(
        clock=clk,
        llm_factory=lambda: _DeliveryLLM(),
        session_root_resolver=lambda persona_id, session_id: tmp_path / persona_id / session_id.replace(":", "_"),
    )
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
    enqueue_outreach_proposal(
        handle.session_root,
        proposal={
            "proposal_id": "prop_tg_test",
            "trigger": "scheduled_outreach",
            "source_intent_id": "sint_tg_test",
            "evidence_refs": ["env_due"],
            "traceable_expectation_id": "sint_tg_test",
            "ordinary_language_intent": "Send the scheduled follow-up.",
            "persona_id": "hutao-prod",
            "session_id": "telegram:tg_main:dm:user_123456",
        },
        now=clk(),
        ttl_seconds=86400,
        source_intent_id="sint_tg_test",
    )
    gateway.ensure_runner(handle)

    first = connector.drain_proactive_once()
    second = connector.drain_proactive_once()

    assert first["sessions_considered"] == 1
    assert first["sent_messages"] == []
    assert first["results"] == []
    assert second["sent_messages"] == []
    assert api.sent_messages == []


def test_connector_idle_drain_blocks_when_m13_assessor_disallows_delivery(tmp_path: Path) -> None:
    class _RejectDeliveryLLM(_DeliveryLLM):
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
            if "M13" in system_prompt:
                return {
                    "allow_delivery": False,
                    "confidence": 0.9,
                    "violation_codes": ["user_active"],
                    "reason_codes": ["user_active"],
                }
            return super().complete_json(system_prompt=system_prompt, user_prompt=user_prompt)

    clk = _Clock()
    gateway = M16Gateway(
        clock=clk,
        llm_factory=lambda: _RejectDeliveryLLM(),
        session_root_resolver=lambda persona_id, session_id: tmp_path / persona_id / session_id.replace(":", "_"),
    )
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
    enqueue_outreach_proposal(
        handle.session_root,
        proposal={
            "proposal_id": "prop_tg_reject",
            "trigger": "scheduled_outreach",
            "source_intent_id": "sint_tg_reject",
            "evidence_refs": ["env_due"],
            "traceable_expectation_id": "sint_tg_reject",
            "ordinary_language_intent": "Send the scheduled follow-up.",
            "persona_id": "hutao-prod",
            "session_id": "telegram:tg_main:dm:user_123456",
        },
        now=clk(),
        ttl_seconds=86400,
        source_intent_id="sint_tg_reject",
    )
    gateway.ensure_runner(handle)

    result = connector.drain_proactive_once()

    assert result["sent_messages"] == []
    assert api.sent_messages == []


def test_connector_restart_preserves_group_thread_continuity(tmp_path: Path) -> None:
    gateway = M16Gateway(
        llm_factory=lambda: FakeJSONLLM(),
        session_root_resolver=lambda persona_id, session_id: tmp_path / persona_id / session_id.replace(":", "_"),
    )
    api = FakeTelegramApi()
    connector = TelegramConnector(
        persona_id="hutao-prod",
        account_scope="tg_main",
        gateway=gateway,
        api_client=api,
    )

    first = connector.ingest_update(
        {
            "update_id": 501,
            "message": {
                "message_id": 9001,
                "text": "@hutao_bot Bob 刚才那句不对，我纠正一下",
                "entities": [{"type": "mention", "offset": 0, "length": 10}],
                "chat": {"id": -1001, "type": "supergroup"},
                "from": {"id": 123456, "first_name": "Alice"},
                "reply_to_message": {
                    "message_id": 9000,
                    "from": {"id": 456, "first_name": "Bob"},
                },
            },
        }
    )
    assert first["accepted"] is True

    handle = gateway.get_or_create_session("hutao-prod", "telegram:tg_main:group:chat_-1001")
    preserved = handle.bridge.store.load()["temporal_state"]["group_chat_state"]["thread_policy_state"]
    assert str(preserved["last_reply_to_turn_id"]).endswith(":msg:9000")

    restarted_gateway = M16Gateway(
        llm_factory=lambda: FakeJSONLLM(),
        session_root_resolver=lambda persona_id, session_id: tmp_path / persona_id / session_id.replace(":", "_"),
    )
    restarted_api = FakeTelegramApi()
    restarted_connector = TelegramConnector(
        persona_id="hutao-prod",
        account_scope="tg_main",
        gateway=restarted_gateway,
        api_client=restarted_api,
    )
    restarted_handle = restarted_gateway.get_or_create_session("hutao-prod", "telegram:tg_main:group:chat_-1001")
    restarted_state = restarted_handle.bridge.store.load()["temporal_state"]["group_chat_state"]["thread_policy_state"]
    assert str(restarted_state["last_reply_to_turn_id"]).endswith(":msg:9000")

    second = restarted_connector.ingest_update(
        {
            "update_id": 502,
            "message": {
                "message_id": 9002,
                "text": "@hutao_bot 你先按我这个纠正来",
                "entities": [{"type": "mention", "offset": 0, "length": 10}],
                "chat": {"id": -1001, "type": "supergroup"},
                "from": {"id": 123456, "first_name": "Alice"},
                "reply_to_message": {
                    "message_id": 9000,
                    "from": {"id": 456, "first_name": "Bob"},
                },
            },
        }
    )

    assert second["accepted"] is True
    assert len(restarted_api.sent_messages) == 1
    after = restarted_handle.bridge.store.load()["temporal_state"]["group_chat_state"]["thread_policy_state"]
    assert str(after["last_reply_to_turn_id"]).endswith(":msg:9000")
