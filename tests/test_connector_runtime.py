from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from segmentum.connectors import (
    ConnectorCapabilities,
    ConnectorDeliveryReceipt,
    ConnectorDeliveryTargetStore,
    ConnectorRuntime,
    NormalizedConnectorInput,
)
from segmentum.dialogue.runtime.m16_api import M16Gateway
from tests.m16_1_test_helpers import _Clock, full_opted_state


@dataclass(frozen=True)
class FakeTarget:
    channel_id: str
    account_scope: str = "primary"

    def to_payload(self) -> dict[str, Any]:
        return {
            "platform": "fakechat",
            "account_scope": self.account_scope,
            "channel_id": self.channel_id,
        }


class FakeAdapter:
    platform = "fakechat"
    persona_id = "persona"
    account_scope = "primary"
    target_store_file = "fakechat_delivery_targets.jsonl"
    capabilities = ConnectorCapabilities(
        direct_messages=True,
        group_messages=True,
        explicit_mentions=True,
        reply_links=True,
    )

    def __init__(self) -> None:
        self.deliveries: list[dict[str, str]] = []

    def normalize_event(self, event: Mapping[str, Any]) -> NormalizedConnectorInput | None:
        text = str(event.get("text", "") or "").strip()
        if not text:
            return None
        channel_id = str(event.get("channel_id", "") or "")
        return NormalizedConnectorInput(
            platform=self.platform,
            persona_id=self.persona_id,
            session_id=f"{self.platform}:{self.account_scope}:group:{channel_id}",
            correlation_id=f"fake:{event.get('event_id', '')}",
            text=text,
            speaker_name=str(event.get("speaker_name", "") or ""),
            group_turn_envelope={
                "speaker_participant_id": f"{self.platform}:{self.account_scope}:user:1",
            },
            delivery_target=FakeTarget(channel_id=channel_id),
            ingress_evidence_band="speaker_name_only",
            platform_event_id=str(event.get("event_id", "") or ""),
            platform_message_id=str(event.get("message_id", "") or ""),
        )

    def target_from_payload(self, payload: Mapping[str, Any]) -> FakeTarget | None:
        if str(payload.get("platform", "") or "") != self.platform:
            return None
        channel_id = str(payload.get("channel_id", "") or "").strip()
        return FakeTarget(channel_id=channel_id) if channel_id else None

    def deliver(self, *, target: Any, text: str) -> ConnectorDeliveryReceipt:
        assert isinstance(target, FakeTarget)
        self.deliveries.append({"channel_id": target.channel_id, "text": text})
        return ConnectorDeliveryReceipt(
            platform=self.platform,
            platform_message_id=f"sent-{len(self.deliveries)}",
            target={"channel_id": target.channel_id},
        )


def _runtime(tmp_path: Path) -> tuple[ConnectorRuntime, FakeAdapter, M16Gateway, _Clock]:
    clock = _Clock()
    adapter = FakeAdapter()
    gateway = M16Gateway(
        clock=clock,
        session_root_resolver=lambda persona_id, session_id: tmp_path / persona_id / session_id.replace(":", "_"),
    )
    return ConnectorRuntime(adapter=adapter, gateway=gateway, clock=clock), adapter, gateway, clock


def test_generic_connector_runtime_ingests_and_delivers_without_platform_logic(tmp_path: Path) -> None:
    runtime, adapter, gateway, _ = _runtime(tmp_path)
    handle = gateway.get_or_create_session("persona", "fakechat:primary:group:room-1")
    handle.bridge.store.save(full_opted_state())
    runner = gateway.ensure_runner(handle)
    runner._inline_run_turn = lambda text, turn_index, now: SimpleNamespace(reply="shared reply", action="answer")  # type: ignore[method-assign]

    result = runtime.ingest_event(
        {"event_id": "evt-1", "message_id": "msg-1", "channel_id": "room-1", "text": "hello"}
    )

    assert result["accepted"] is True
    assert result["platform"] == "fakechat"
    assert adapter.deliveries == [{"channel_id": "room-1", "text": "shared reply"}]
    events = handle.bridge.event_store.query_events(event_types={"ClientInputCommittedEvent"})
    assert events[-1]["source"] == "fakechat_connector"


def test_generic_target_store_is_delivery_idempotent(tmp_path: Path) -> None:
    store = ConnectorDeliveryTargetStore(tmp_path, file_name="targets.jsonl")
    store.record(event_id="evt-1", correlation_id="corr-1", target=FakeTarget(channel_id="room-1"), now=1)

    assert store.load_payload("evt-1") == {
        "platform": "fakechat",
        "account_scope": "primary",
        "channel_id": "room-1",
    }

    store.mark_delivered(event_id="evt-1", platform_message_id="sent-1", now=2)
    assert store.load_payload("evt-1") is None


def test_generic_connector_rejects_normalized_input_for_another_platform(tmp_path: Path) -> None:
    runtime, _, _, _ = _runtime(tmp_path)
    normalized = NormalizedConnectorInput(
        platform="another",
        persona_id="persona",
        session_id="another:primary:group:room-1",
        correlation_id="corr-1",
        text="hello",
        speaker_name="Alice",
        group_turn_envelope={},
        delivery_target=FakeTarget(channel_id="room-1"),
        ingress_evidence_band="speaker_name_only",
    )

    with pytest.raises(ValueError, match="does not match adapter"):
        runtime.ingest_normalized(normalized)


def test_telegram_adapter_has_canonical_connector_import_path() -> None:
    from segmentum.connectors.telegram import TelegramAdapter, TelegramConnector

    assert TelegramAdapter.platform == "telegram"
    assert issubclass(TelegramConnector, ConnectorRuntime)
