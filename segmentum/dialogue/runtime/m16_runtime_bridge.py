"""Thin Path B adapters for the M16 consciousness runner and gateway."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore
from segmentum.dialogue.runtime.m16_protocol import (
    ACTUATION_EVENT_AUDIT_MAP,
    ENGINEERING_PROXY_LABEL,
    build_client_input_committed_event,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore, _mapping


RUNNER_KIND = "m16_gateway_runner"
M16_PROCESSED_EVENTS = "m16_processed_events.jsonl"
M16_ACTUATION_LOG = "m16_actuation.jsonl"


def _now(clock: Callable[[], int] | None = None) -> int:
    if clock is None:
        return int(time.time())
    return int(clock())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


@dataclass
class M16SessionBridge:
    persona_id: str
    session_id: str
    session_root: Path
    runtime: MVPDialogueRuntime
    consumer_id: str = ""
    clock: Callable[[], int] | None = None

    def __post_init__(self) -> None:
        self.session_root = Path(self.session_root)
        self.store = self.runtime.store
        if not self.consumer_id:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.session_id)
            self.consumer_id = f"m16_runner_{safe[:48]}"
        self.event_store = EnvironmentEventStore(
            self.session_root,
            persona_id=self.persona_id,
            session_id=self.session_id,
            clock=self.clock,
        )
        self._processed_path = self.session_root / M16_PROCESSED_EVENTS
        self._actuation_path = self.session_root / M16_ACTUATION_LOG

    def append_client_input(self, *, text: str, correlation_id: str, source: str = "m16_gateway") -> str:
        row = build_client_input_committed_event(
            persona_id=self.persona_id,
            session_id=self.session_id,
            text=text,
            correlation_id=correlation_id,
            source=source,
            now=_now(self.clock),
        )
        return self.event_store.append_event(
            "ClientInputCommittedEvent",
            dict(row["payload"]),
            source=source,
            correlation_id=correlation_id,
        )

    def append_perception_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None,
        *,
        source: str,
        correlation_id: str,
    ) -> str:
        return self.event_store.append_event(
            event_type,
            dict(payload or {}),
            source=source,
            correlation_id=correlation_id,
        )

    def claim_events(
        self,
        *,
        limit: int = 16,
        event_types: set[str] | frozenset[str] | None = None,
        lease_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        return self.event_store.claim_events(
            self.consumer_id,
            limit=limit,
            event_types=event_types,
            lease_seconds=lease_seconds,
        )

    def ack_event(self, event_id: str, *, result: Mapping[str, Any] | None = None) -> None:
        self.event_store.ack_event(event_id, self.consumer_id, result=result)

    def fail_event(self, event_id: str, reason: str, *, retryable: bool = True) -> None:
        self.event_store.fail_event(event_id, self.consumer_id, reason, retryable=retryable)

    def turn_index(self) -> int:
        state = self.store.load()
        temporal = _mapping(state.get("temporal_state"))
        return int(temporal.get("last_turn_index", 0) or 0)

    def next_user_turn_index(self) -> int:
        return self.turn_index() + 1

    def run_user_turn(self, text: str, *, turn_index: int | None = None, speaker_name: str = "") -> Any:
        idx = int(turn_index if turn_index is not None else self.next_user_turn_index())
        return self.runtime.run_turn(
            text,
            turn_index=idx,
            speaker_name=speaker_name or "default_user",
            bus_messages=[{"type": "M16UserInputEvent", "source": "m16_runner", "turn_index": idx}],
            now=_now(self.clock),
        )

    def run_idle_cognitive_tick(self, *, idle_seconds: float = 0.0, now: int | None = None) -> dict[str, Any]:
        ts = int(now if now is not None else _now(self.clock))
        return self.runtime.run_idle_cognitive_tick(
            turn_index=self.turn_index(),
            idle_seconds=idle_seconds,
            now=ts,
        )

    def run_background_self_tick(self) -> dict[str, Any]:
        return dict(self.runtime.run_background_self_tick(runner_kind=RUNNER_KIND))

    def drain_queued_outreach(self, *, now: int | None = None) -> dict[str, Any]:
        ts = int(now if now is not None else _now(self.clock))
        result = dict(self.runtime.maybe_drain_queued_outreach(turn_index=self.turn_index(), now=ts))
        if result.get("drained") and not str(result.get("proposal_id", "") or "").strip():
            from segmentum.dialogue.runtime.m14_1_background_continuity import load_queued_outreach

            for row in reversed(load_queued_outreach(self.session_root)):
                if str(row.get("status", "") or "") == "delivered":
                    result["proposal_id"] = str(row.get("proposal_id", "") or "")
                    break
        return result

    def is_event_processed(self, event_id: str) -> bool:
        needle = str(event_id or "").strip()
        if not needle:
            return False
        return any(str(row.get("event_id", "")) == needle for row in _read_jsonl(self._processed_path))

    def mark_event_processed(self, event_id: str, *, now: int | None = None) -> None:
        _append_jsonl(
            self._processed_path,
            {
                "event_id": str(event_id),
                "at": int(now if now is not None else _now(self.clock)),
                "consumer_id": self.consumer_id,
            },
        )

    def was_actuation_delivered(self, delivery_id: str) -> bool:
        needle = str(delivery_id or "").strip()
        if not needle:
            return False
        return any(str(row.get("delivery_id", "")) == needle for row in _read_jsonl(self._actuation_path))

    def record_actuation(
        self,
        *,
        delivery_id: str,
        actuation_type: str,
        payload: Mapping[str, Any],
        correlation_id: str = "",
        now: int | None = None,
    ) -> bool:
        if self.was_actuation_delivered(delivery_id):
            return False
        ts = int(now if now is not None else _now(self.clock))
        row = {
            "delivery_id": delivery_id,
            "type": actuation_type,
            "at": ts,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "correlation_id": str(correlation_id or "")[:120],
            "payload": dict(payload),
            "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
            "runner_kind": RUNNER_KIND,
        }
        _append_jsonl(self._actuation_path, row)
        channel = ACTUATION_EVENT_AUDIT_MAP.get(actuation_type, "m16_actuation_audit")
        self.runtime.store.append_log({"event": channel, **row})
        return True

    def append_runner_audit(
        self,
        *,
        typ: str,
        correlation_id: str = "",
        now: int | None = None,
        **fields: Any,
    ) -> None:
        ts = int(now if now is not None else _now(self.clock))
        self.runtime.store.append_log(
            {
                "event": "m16_runner_audit",
                "type": typ,
                "at": ts,
                "persona_id": self.persona_id,
                "session_id": self.session_id,
                "correlation_id": str(correlation_id or "")[:120],
                "engineering_proxy_label": ENGINEERING_PROXY_LABEL,
                "runner_kind": RUNNER_KIND,
                **fields,
            }
        )

    def snapshot(self, *, chat_limit: int = 20) -> dict[str, Any]:
        state = self.store.load()
        temporal = _mapping(state.get("temporal_state"))
        m13 = _mapping(state.get("m13_drive_state"))
        initiative = _mapping(m13.get("initiative"))
        chat_tail: list[dict[str, Any]] = []
        log_path = self.session_root / "conversation_log.jsonl"
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines()[-chat_limit:]:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("event") in {"user_message", "assistant_message", "proactive_turn"}:
                    chat_tail.append(
                        {
                            "event": row.get("event"),
                            "text": str(row.get("text", row.get("reply", "")) or "")[:500],
                            "turn_index": row.get("turn_index"),
                            "at": row.get("at"),
                        }
                    )
        return {
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "chat_tail": chat_tail[-chat_limit:],
            "runtime_hints": {
                "last_turn_index": int(temporal.get("last_turn_index", 0) or 0),
                "last_user_turn_at": int(temporal.get("last_user_turn_at", 0) or 0),
                "initiative_enabled": bool(initiative.get("enabled")),
                "runner_kind": RUNNER_KIND,
            },
        }
