from __future__ import annotations

import pytest

from segmentum.dialogue.runtime.m16_protocol import (
    CLOSED_ACTUATION_SUPPRESSION_REASON_CODES,
    DELIVERY_SURFACE_READY_TTL_SECONDS,
    FORBIDDEN_ACTUATION_PAYLOAD_KEYS,
    MAX_INPUT_TEXT_CHARS,
    OPENAPI_MINIMUM_ROUTES,
    SCHEMA_VERSION,
    WS_PATH_TEMPLATE,
    actuation_payload_is_safe,
    build_client_input_committed_event,
    build_ws_client_message,
    build_ws_server_message,
    delivery_surface_allows_outbox_drain,
    gateway_mutation_allowed,
    load_schema,
    openapi_paths_from_file,
    runner_control_payload_is_bounded,
    schemas_root,
    validate_actuation_event,
    validate_openapi_minimum_routes,
    validate_perception_event,
    validate_ws_client_message,
    validate_ws_server_message,
)

NOW = 1_700_000_000


def test_openapi_contains_minimum_routes() -> None:
    found = openapi_paths_from_file()
    for route in OPENAPI_MINIMUM_ROUTES:
        assert route in found, route
    assert WS_PATH_TEMPLATE in found
    assert validate_openapi_minimum_routes() == []


def test_schema_files_exist_and_parse() -> None:
    for name in (
        "ws_client_messages.schema.json",
        "ws_server_messages.schema.json",
        "perception_events.schema.json",
        "actuation_events.schema.json",
    ):
        schema = load_schema(name)
        assert isinstance(schema, dict)
        assert schema.get("$schema")


def test_ws_schema_roundtrip_subscribe_and_snapshot() -> None:
    subscribe = build_ws_client_message(
        kind="Subscribe",
        persona_id="胡桃",
        session_id="sess_a",
        payload={"resume_from_message_id": ""},
        correlation_id="corr_sub",
        now=NOW,
    )
    assert validate_ws_client_message(subscribe) == []
    assert subscribe["schema_version"] == SCHEMA_VERSION

    snapshot = build_ws_server_message(
        kind="SessionSnapshot",
        persona_id="胡桃",
        session_id="sess_a",
        payload={
            "chat_tail": [{"role": "user", "text": "hello"}],
            "runtime_hints": {"runner_alive": True},
        },
        now=NOW,
    )
    assert validate_ws_server_message(snapshot) == []


def test_client_input_maps_to_client_input_committed_event() -> None:
    event = build_client_input_committed_event(
        persona_id="胡桃",
        session_id="sess_a",
        text="你好",
        correlation_id="corr_in",
        now=NOW,
    )
    assert event["event_type"] == "ClientInputCommittedEvent"
    assert event["status"] == "pending"
    assert event["payload"]["text"] == "你好"
    assert validate_perception_event(event) == []
    assert len(event["payload"]["text"]) <= MAX_INPUT_TEXT_CHARS


def test_actuation_messages_exclude_forbidden_internal_fields() -> None:
    safe = build_ws_server_message(
        kind="AssistantMessageCommitted",
        persona_id="胡桃",
        session_id="sess_a",
        payload={"text": "好啦", "turn_index": 1},
        now=NOW,
    )
    assert actuation_payload_is_safe(safe["payload"])

    for forbidden in (
        "system_prompt",
        "conscious_markdown",
        "full_memory_dump",
        "conscious_plan",
    ):
        with pytest.raises(ValueError):
            build_ws_server_message(
                kind="AuditEvent",
                persona_id="胡桃",
                session_id="sess_a",
                payload={forbidden: "secret"},
                now=NOW,
            )

    nested = {"outer": {"memory_dynamics": {"x": 1}}}
    assert not actuation_payload_is_safe(nested)
    assert validate_ws_server_message(
        {
            "schema_version": SCHEMA_VERSION,
            "message_id": "m1",
            "persona_id": "p",
            "session_id": "s",
            "at": NOW,
            "kind": "AuditEvent",
            "payload": nested,
        }
    )


def test_protocol_version_is_explicit_on_every_ws_message() -> None:
    client = build_ws_client_message(
        kind="Ping",
        persona_id="p",
        session_id="s",
        payload={},
        correlation_id="c",
        now=NOW,
    )
    server = build_ws_server_message(
        kind="RunnerHealth",
        persona_id="p",
        session_id="s",
        payload={"alive": True},
        now=NOW,
    )
    for row in (client, server):
        assert row["schema_version"] == SCHEMA_VERSION
    bad = dict(client)
    bad["schema_version"] = "m15.9"
    assert "schema_version" in validate_ws_client_message(bad)


def test_runner_start_stop_request_schema_is_bounded() -> None:
    assert runner_control_payload_is_bounded({"command": "start", "reason": "ok"}) == []
    assert "invalid:command" in runner_control_payload_is_bounded({"command": "restart"})
    long_reason = "x" * 500
    assert "reason_too_long" in runner_control_payload_is_bounded(
        {"command": "stop", "reason": long_reason}
    )


def test_delivery_surface_ready_required_before_outbox_drain() -> None:
    ok, _ = delivery_surface_allows_outbox_drain(
        ws_subscribed=True,
        delivery_surface_ready_at=NOW,
        now=NOW,
    )
    assert ok is True

    blocked_no_sub, code_no_sub = delivery_surface_allows_outbox_drain(
        ws_subscribed=False,
        delivery_surface_ready_at=NOW,
        now=NOW,
    )
    assert blocked_no_sub is False
    assert code_no_sub == "delivery_surface_unavailable"

    blocked_no_ready, code_no_ready = delivery_surface_allows_outbox_drain(
        ws_subscribed=True,
        delivery_surface_ready_at=None,
        now=NOW,
    )
    assert blocked_no_ready is False
    assert code_no_ready == "delivery_surface_not_ready"

    stale, code_stale = delivery_surface_allows_outbox_drain(
        ws_subscribed=True,
        delivery_surface_ready_at=NOW - DELIVERY_SURFACE_READY_TTL_SECONDS - 1,
        now=NOW,
    )
    assert stale is False
    assert code_stale == "delivery_surface_not_ready"


def test_gateway_mutation_requires_localhost_or_dev_token() -> None:
    ok_local, _ = gateway_mutation_allowed(client_host="127.0.0.1")
    assert ok_local is True

    ok_v6, _ = gateway_mutation_allowed(client_host="::1")
    assert ok_v6 is True

    blocked, code = gateway_mutation_allowed(client_host="203.0.113.10")
    assert blocked is False
    assert code == "gateway_mutation_forbidden"

    ok_token, _ = gateway_mutation_allowed(
        client_host="203.0.113.10",
        authorization_header="Bearer dev-secret",
        configured_dev_token="dev-secret",
    )
    assert ok_token is True


def test_actuation_suppression_reason_codes_are_closed_reuse_set() -> None:
    event = {
        "type": "ProactiveDeliverySuppressedEvent",
        "at": NOW,
        "persona_id": "p",
        "session_id": "s",
        "reason_code": "diagnostic_expectation_pool_noisy",
        "payload": {"reason_code": "diagnostic_expectation_pool_noisy"},
    }
    assert validate_actuation_event(event) == []
    bad = dict(event)
    bad["reason_code"] = "made_up_keyword_cue"
    assert "invalid:reason_code" in validate_actuation_event(bad)


def test_perception_and_actuation_audit_maps_are_non_empty() -> None:
    from segmentum.dialogue.runtime import m16_protocol

    assert m16_protocol.PERCEPTION_EVENT_AUDIT_MAP
    assert m16_protocol.ACTUATION_EVENT_AUDIT_MAP
    assert "ProactiveMessageDeliveredEvent" in m16_protocol.ACTUATION_EVENT_AUDIT_MAP


def test_forbidden_key_set_covers_prompt_and_memory_surfaces() -> None:
    assert "system_prompt" in FORBIDDEN_ACTUATION_PAYLOAD_KEYS
    assert "conscious_markdown" in FORBIDDEN_ACTUATION_PAYLOAD_KEYS
    assert "full_memory_dump" in FORBIDDEN_ACTUATION_PAYLOAD_KEYS


def test_schemas_root_points_at_repo_schemas_m16() -> None:
    root = schemas_root()
    assert root.is_dir()
    assert (root / "http.openapi.yaml").is_file()


def test_client_input_event_id_is_stable_when_provided() -> None:
    event = build_client_input_committed_event(
        persona_id="p",
        session_id="s",
        text="hi",
        correlation_id="c",
        event_id="evt_fixed",
        now=NOW,
    )
    assert event["event_id"] == "evt_fixed"


def test_closed_suppression_codes_include_m13_delivery_surface_codes() -> None:
    assert "delivery_surface_unavailable" in CLOSED_ACTUATION_SUPPRESSION_REASON_CODES
    assert "delivery_surface_not_ready" in CLOSED_ACTUATION_SUPPRESSION_REASON_CODES


def test_m16_perception_types_registered_in_m14_2_event_bus(tmp_path) -> None:
    from pathlib import Path

    from segmentum.dialogue.runtime.m14_2_event_bus import ENVIRONMENT_EVENT_TYPES, EnvironmentEventStore
    from segmentum.dialogue.runtime.m16_protocol import M16_PERCEPTION_EVENT_TYPES

    assert M16_PERCEPTION_EVENT_TYPES.issubset(ENVIRONMENT_EVENT_TYPES)
    store = EnvironmentEventStore(
        Path(tmp_path),
        persona_id="p",
        session_id="s",
    )
    event_id = store.append_event(
        "ClientInputCommittedEvent",
        {"text": "ping", "char_count": 4},
        source="m16_gateway",
        correlation_id="corr_1",
    )
    assert event_id
