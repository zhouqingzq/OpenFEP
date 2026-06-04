from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from segmentum.dialogue.runtime.m16_api import create_app
from segmentum.dialogue.runtime.m16_protocol import FORBIDDEN_ACTUATION_PAYLOAD_KEYS
from tests.m16_1_test_helpers import build_gateway


def test_ws_subscribe_returns_snapshot_without_internal_fields(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    app = create_app(gateway)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/personas/p/sessions/s/stream") as ws:
            first = ws.receive_json()
            second = ws.receive_json()
            assert first["kind"] == "Subscribed"
            assert second["kind"] == "SessionSnapshot"
            for msg in (first, second):
                payload = msg.get("payload") or {}
                for key in payload:
                    assert str(key).casefold() not in {k.casefold() for k in FORBIDDEN_ACTUATION_PAYLOAD_KEYS}


def test_ws_client_input_appends_event_without_inline_turn(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    app = create_app(gateway)
    calls: list[str] = []
    original = bridge.runtime.run_turn

    def _guard(text: str, **kwargs: object) -> object:
        calls.append(text)
        return original(text, **kwargs)

    bridge.runtime.run_turn = _guard  # type: ignore[method-assign]
    with TestClient(app) as client:
        with client.websocket_connect("/v1/personas/p/sessions/s/stream") as ws:
            ws.receive_json()
            ws.receive_json()
            ws.send_json(
                {
                    "schema_version": "m16.0",
                    "message_id": "m16c_in",
                    "persona_id": "p",
                    "session_id": "s",
                    "at": clk(),
                    "kind": "ClientInput",
                    "correlation_id": "corr_ws_in",
                    "payload": {
                        "text": "via ws",
                        "speaker_name": "zq",
                        "group_turn_envelope": {
                            "speaker_participant_id": "alice",
                            "addressed_participant_ids": ["hutao"],
                            "reply_to_turn_id": "turn_001",
                        },
                    },
                }
            )
    assert calls == []
    events = bridge.event_store.query_events(event_types={"ClientInputCommittedEvent"})
    assert any(ev["payload"]["text"] == "via ws" for ev in events)
    assert any(ev["payload"].get("speaker_name") == "zq" for ev in events)
    assert any(
        ev["payload"].get("group_turn_envelope", {}).get("speaker_participant_id") == "alice"
        for ev in events
    )


def test_ws_delivery_surface_ready_via_stream(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    app = create_app(gateway)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/personas/p/sessions/s/stream") as ws:
            ws.receive_json()
            ws.receive_json()
            ws.send_json(
                {
                    "schema_version": "m16.0",
                    "message_id": "m16c_ready",
                    "persona_id": "p",
                    "session_id": "s",
                    "at": clk(),
                    "kind": "DeliverySurfaceReady",
                    "correlation_id": "corr_ready",
                    "payload": {},
                }
            )
            ws.send_json(
                {
                    "schema_version": "m16.0",
                    "message_id": "m16c_ping",
                    "persona_id": "p",
                    "session_id": "s",
                    "at": clk(),
                    "kind": "Ping",
                    "correlation_id": "corr_ping",
                    "payload": {},
                }
            )
            assert ws.receive_json()["kind"] == "RunnerHealth"
            allowed, _ = gateway.get_or_create_session("p", "s").hub.outbox_drain_allowed(now=clk())
            assert allowed


def test_bridge_snapshot_preserves_group_turn_metadata(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    bridge.run_user_turn(
        "hello from group",
        turn_index=2,
        speaker_name="Alice",
        group_turn_envelope={
            "speaker_participant_id": "alice",
            "visible_participant_ids": ["alice", "bob", "hutao"],
            "addressed_participant_ids": ["hutao"],
            "mentioned_participant_ids": ["bob"],
            "reply_to_turn_id": "turn_001",
        },
    )

    snapshot = bridge.snapshot()
    user_rows = [row for row in snapshot["chat_tail"] if row.get("event") == "user_message"]
    assert user_rows
    assert user_rows[-1]["speaker_participant_id"] == "alice"
    assert user_rows[-1]["reply_to_turn_id"] == "turn_001"
    assert user_rows[-1]["addressed_participant_ids"] == ["hutao"]


def test_bridge_snapshot_loads_legacy_turn_row_without_claiming_native_group_ownership(tmp_path: Path) -> None:
    gateway, bridge, runner, clk = build_gateway(tmp_path)
    log_path = bridge.session_root / "conversation_log.jsonl"
    legacy_row = {
        "event": "turn",
        "user_text": "legacy hello",
        "reply": "legacy reply",
        "turn_index": 1,
        "at": clk(),
    }
    log_path.write_text(json.dumps(legacy_row, ensure_ascii=False) + "\n", encoding="utf-8")

    snapshot = bridge.snapshot()
    user_rows = [row for row in snapshot["chat_tail"] if row.get("event") == "user_message"]
    assistant_rows = [row for row in snapshot["chat_tail"] if row.get("event") == "assistant_message"]
    assert user_rows
    assert assistant_rows
    assert user_rows[-1]["text"] == "legacy hello"
    assert user_rows[-1]["speaker_participant_id"] is None
    assert user_rows[-1]["reply_to_turn_id"] is None
    assert assistant_rows[-1]["text"] == "legacy reply"
