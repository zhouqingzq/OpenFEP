"""M16.0 consciousness runner wire protocol helpers.

This module defines message kinds, schema paths, validation helpers, and
boundary rules for the Path B perception/actuation gateway. It does not run
cognition, start servers, or interpret user semantics.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

ENGINEERING_PROXY_LABEL = "mvp_local_consciousness_runner"
SCHEMA_VERSION = "m16.0"

MAX_INPUT_TEXT_CHARS = 8000
MAX_CORRELATION_ID_CHARS = 120
MAX_MESSAGE_ID_CHARS = 120
MAX_SNAPSHOT_CHAT_ROWS = 40
DELIVERY_SURFACE_READY_TTL_SECONDS = 45
RUNNER_CONTROL_MAX_REASON_CHARS = 160

WS_CLIENT_MESSAGE_KINDS = frozenset(
    {
        "Subscribe",
        "Ping",
        "ClientInput",
        "DeliverySurfaceReady",
        "DeliveryAck",
        "Unsubscribe",
    }
)

WS_SERVER_MESSAGE_KINDS = frozenset(
    {
        "Subscribed",
        "SessionSnapshot",
        "UserMessageAccepted",
        "AssistantMessageCommitted",
        "ProactiveMessageCommitted",
        "AuditEvent",
        "RunnerHealth",
        "RunnerSuppression",
        "Error",
    }
)

M16_PERCEPTION_EVENT_TYPES = frozenset(
    {
        "ClientInputCommittedEvent",
        "DeliverySurfaceConnectedEvent",
        "DeliverySurfaceReadyEvent",
        "DeliverySurfaceDisconnectedEvent",
        "RunnerControlCommandEvent",
    }
)

M16_ACTUATION_EVENT_TYPES = frozenset(
    {
        "AssistantMessagePreparedEvent",
        "AssistantMessageDeliveredEvent",
        "ProactiveMessagePreparedEvent",
        "ProactiveMessageDeliveredEvent",
        "ProactiveDeliverySuppressedEvent",
        "RunnerHealthEvent",
    }
)

# Keys that must never appear on default client actuation payloads.
FORBIDDEN_ACTUATION_PAYLOAD_KEYS = frozenset(
    {
        "system_prompt",
        "user_prompt",
        "raw_prompt",
        "raw_prompt_text",
        "full_prompt",
        "prompt_text",
        "conscious_markdown",
        "full_conscious_markdown",
        "conscious_plan",
        "memory_dynamics",
        "full_memory_dump",
        "memory_dump",
        "short_term_memory",
        "long_term_memory",
        "llm_thinking_result",
        "diagnostics",
        "internal_patch",
        "patch_payload",
        "m13_drive_state",
        "meta_control_intents",
    }
)

OPENAPI_MINIMUM_ROUTES = (
    "/health",
    "/v1/personas/{persona_id}/sessions",
    "/v1/personas/{persona_id}/sessions/{session_id}",
    "/v1/personas/{persona_id}/sessions/{session_id}/input",
    "/v1/personas/{persona_id}/sessions/{session_id}/snapshot",
    "/v1/personas/{persona_id}/sessions/{session_id}/runner/start",
    "/v1/personas/{persona_id}/sessions/{session_id}/runner/stop",
    "/v1/personas/{persona_id}/sessions/{session_id}/runner/status",
)

WS_PATH_TEMPLATE = "/v1/personas/{persona_id}/sessions/{session_id}/stream"

# M16 perception rows append into the M14.2 store; map to audit vocabulary.
PERCEPTION_EVENT_AUDIT_MAP: dict[str, str] = {
    "ClientInputCommittedEvent": "m16_perception_audit",
    "DeliverySurfaceConnectedEvent": "m16_perception_audit",
    "DeliverySurfaceReadyEvent": "m16_perception_audit",
    "DeliverySurfaceDisconnectedEvent": "m16_perception_audit",
    "RunnerControlCommandEvent": "m16_perception_audit",
}

ACTUATION_EVENT_AUDIT_MAP: dict[str, str] = {
    "AssistantMessagePreparedEvent": "m16_actuation_audit",
    "AssistantMessageDeliveredEvent": "conversation_log",
    "ProactiveMessagePreparedEvent": "m16_actuation_audit",
    "ProactiveMessageDeliveredEvent": "m13_proactive_audit",
    "ProactiveDeliverySuppressedEvent": "m13_proactive_audit",
    "RunnerHealthEvent": "m14_2_audit",
}

# Closed reuse set for actuation suppression stream (subset of M13/M14 codes).
CLOSED_ACTUATION_SUPPRESSION_REASON_CODES = frozenset(
    {
        "initiative_disabled",
        "implicit_idle_disabled",
        "delivery_channel_unavailable",
        "idle_time_too_short",
        "user_active",
        "user_typing",
        "cooldown_active",
        "session_limit_reached",
        "safety_risk",
        "no_traceable_proactive_target",
        "no_high_value_target",
        "diagnostic_expectation_pool_noisy",
        "meta_control_trigger_suppressed",
        "budget_exhausted",
        "llm_unavailable",
        "delivery_surface_unavailable",
        "delivery_surface_not_ready",
    }
)

LOCALHOST_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def schemas_root() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / "m16"


def load_schema(name: str) -> dict[str, Any]:
    path = schemas_root() / name
    return json.loads(path.read_text(encoding="utf-8"))


def _new_id(prefix: str = "m16") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _epoch(now: int | None = None) -> int:
    return int(now if now is not None else time.time())


def _errors_for_envelope(
    row: Mapping[str, Any],
    *,
    allowed_kinds: frozenset[str],
    require_correlation: bool,
) -> list[str]:
    errors: list[str] = []
    if str(row.get("schema_version", "") or "") != SCHEMA_VERSION:
        errors.append("schema_version")
    for key in ("message_id", "session_id", "persona_id", "at", "kind", "payload"):
        if key not in row:
            errors.append(f"missing:{key}")
    if require_correlation and not str(row.get("correlation_id", "") or "").strip():
        errors.append("missing:correlation_id")
    kind = str(row.get("kind", "") or "")
    if kind and kind not in allowed_kinds:
        errors.append("invalid:kind")
    if len(str(row.get("message_id", "") or "")) > MAX_MESSAGE_ID_CHARS:
        errors.append("message_id_too_long")
    if len(str(row.get("correlation_id", "") or "")) > MAX_CORRELATION_ID_CHARS:
        errors.append("correlation_id_too_long")
    if not isinstance(row.get("payload"), Mapping):
        errors.append("payload_not_object")
    return errors


def validate_ws_client_message(row: Mapping[str, Any]) -> list[str]:
    return _errors_for_envelope(
        row,
        allowed_kinds=WS_CLIENT_MESSAGE_KINDS,
        require_correlation=True,
    )


def validate_ws_server_message(row: Mapping[str, Any]) -> list[str]:
    errors = _errors_for_envelope(
        row,
        allowed_kinds=WS_SERVER_MESSAGE_KINDS,
        require_correlation=False,
    )
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        errors.extend(_forbidden_payload_errors(payload))
    return errors


def validate_perception_event(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    event_type = str(row.get("event_type", "") or "")
    if event_type not in M16_PERCEPTION_EVENT_TYPES:
        errors.append("invalid:event_type")
    for key in ("event_id", "at", "persona_id", "session_id", "source", "payload"):
        if key not in row:
            errors.append(f"missing:{key}")
    if not isinstance(row.get("payload"), Mapping):
        errors.append("payload_not_object")
    return errors


def validate_actuation_event(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    event_type = str(row.get("type", "") or row.get("event_type", "") or "")
    if event_type not in M16_ACTUATION_EVENT_TYPES:
        errors.append("invalid:type")
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        errors.append("payload_not_object")
    else:
        errors.extend(_forbidden_payload_errors(payload))
    reason_code = str(row.get("reason_code", "") or "")
    if event_type == "ProactiveDeliverySuppressedEvent" and reason_code:
        if reason_code not in CLOSED_ACTUATION_SUPPRESSION_REASON_CODES:
            errors.append("invalid:reason_code")
    return errors


def _forbidden_payload_errors(payload: Mapping[str, Any], *, prefix: str = "") -> list[str]:
    errors: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        lowered = str(key).casefold()
        if lowered in FORBIDDEN_ACTUATION_PAYLOAD_KEYS or lowered in {
            item.casefold() for item in FORBIDDEN_ACTUATION_PAYLOAD_KEYS
        }:
            errors.append(f"forbidden:{path}")
        if isinstance(value, Mapping):
            errors.extend(_forbidden_payload_errors(value, prefix=path))
        elif isinstance(value, list) and value and isinstance(value[0], Mapping):
            for index, item in enumerate(value[:3]):
                if isinstance(item, Mapping):
                    errors.extend(_forbidden_payload_errors(item, prefix=f"{path}[{index}]"))
    return errors


def actuation_payload_is_safe(payload: Mapping[str, Any]) -> bool:
    return not _forbidden_payload_errors(payload)


def build_ws_client_message(
    *,
    kind: str,
    persona_id: str,
    session_id: str,
    payload: Mapping[str, Any],
    correlation_id: str,
    now: int | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    if kind not in WS_CLIENT_MESSAGE_KINDS:
        raise ValueError(f"unsupported client kind: {kind}")
    return {
        "schema_version": SCHEMA_VERSION,
        "message_id": (message_id or _new_id("m16c"))[:MAX_MESSAGE_ID_CHARS],
        "persona_id": persona_id,
        "session_id": session_id,
        "at": _epoch(now),
        "kind": kind,
        "correlation_id": str(correlation_id or _new_id("corr"))[:MAX_CORRELATION_ID_CHARS],
        "payload": dict(payload),
    }


def build_ws_server_message(
    *,
    kind: str,
    persona_id: str,
    session_id: str,
    payload: Mapping[str, Any],
    now: int | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    if kind not in WS_SERVER_MESSAGE_KINDS:
        raise ValueError(f"unsupported server kind: {kind}")
    msg = {
        "schema_version": SCHEMA_VERSION,
        "message_id": (message_id or _new_id("m16s"))[:MAX_MESSAGE_ID_CHARS],
        "persona_id": persona_id,
        "session_id": session_id,
        "at": _epoch(now),
        "kind": kind,
        "payload": dict(payload),
    }
    if not actuation_payload_is_safe(msg["payload"]):
        raise ValueError("actuation payload contains forbidden internal fields")
    return msg


def build_client_input_committed_event(
    *,
    persona_id: str,
    session_id: str,
    text: str,
    correlation_id: str,
    source: str = "m16_gateway",
    now: int | None = None,
    event_id: str | None = None,
    speaker_name: str = "",
) -> dict[str, Any]:
    bounded_text = str(text or "")[:MAX_INPUT_TEXT_CHARS]
    payload: dict[str, Any] = {
        "text": bounded_text,
        "char_count": len(bounded_text),
    }
    bounded_speaker = str(speaker_name or "").strip()[:64]
    if bounded_speaker:
        payload["speaker_name"] = bounded_speaker
    event = {
        "event_id": event_id or _new_id("m16evt"),
        "event_type": "ClientInputCommittedEvent",
        "source": source,
        "at": _epoch(now),
        "persona_id": persona_id,
        "session_id": session_id,
        "correlation_id": str(correlation_id or _new_id("corr"))[:MAX_CORRELATION_ID_CHARS],
        "payload": payload,
        "status": "pending",
    }
    errors = validate_perception_event(event)
    if errors:
        raise ValueError(f"invalid perception event: {errors}")
    return event


def runner_control_payload_is_bounded(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    command = str(payload.get("command", "") or "")
    if command not in {"start", "stop", "status"}:
        errors.append("invalid:command")
    reason = str(payload.get("reason", "") or "")
    if len(reason) > RUNNER_CONTROL_MAX_REASON_CHARS:
        errors.append("reason_too_long")
    return errors


def delivery_surface_allows_outbox_drain(
    *,
    ws_subscribed: bool,
    delivery_surface_ready_at: int | None,
    now: int | None = None,
) -> tuple[bool, str]:
    """Both an active WS subscription and a fresh DeliverySurfaceReady are required."""
    if not ws_subscribed:
        return False, "delivery_surface_unavailable"
    ready_at = int(delivery_surface_ready_at or 0)
    if ready_at <= 0:
        return False, "delivery_surface_not_ready"
    age = _epoch(now) - ready_at
    if age > DELIVERY_SURFACE_READY_TTL_SECONDS:
        return False, "delivery_surface_not_ready"
    return True, ""


def gateway_mutation_allowed(
    *,
    client_host: str,
    authorization_header: str | None = None,
    configured_dev_token: str | None = None,
) -> tuple[bool, str]:
    host = str(client_host or "").strip().lower()
    if host in LOCALHOST_HOSTS:
        return True, ""
    token = ""
    if authorization_header:
        parts = authorization_header.strip().split(None, 1)
        if len(parts) == 2 and parts[0].casefold() == "bearer":
            token = parts[1].strip()
    expected = str(configured_dev_token or "").strip()
    if expected and token and token == expected:
        return True, ""
    return False, "gateway_mutation_forbidden"


def openapi_paths_from_file() -> set[str]:
    path = schemas_root() / "http.openapi.yaml"
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^\s{2}(/[^\s:]+):\s*$", text, flags=re.MULTILINE))


def validate_openapi_minimum_routes() -> list[str]:
    found = openapi_paths_from_file()
    missing = [route for route in OPENAPI_MINIMUM_ROUTES if route not in found]
    errors: list[str] = []
    if missing:
        errors.extend(f"missing_route:{route}" for route in missing)
    if WS_PATH_TEMPLATE not in text_if_ws_documented():
        errors.append("missing_ws_path")
    return errors


def text_if_ws_documented() -> str:
    path = schemas_root() / "http.openapi.yaml"
    return path.read_text(encoding="utf-8")


def bounded_snapshot_shape(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    chat_tail = payload.get("chat_tail")
    if chat_tail is not None:
        if not isinstance(chat_tail, list):
            errors.append("chat_tail_not_list")
        elif len(chat_tail) > MAX_SNAPSHOT_CHAT_ROWS:
            errors.append("chat_tail_too_long")
    hints = payload.get("runtime_hints")
    if hints is not None and not isinstance(hints, Mapping):
        errors.append("runtime_hints_not_object")
    return errors
