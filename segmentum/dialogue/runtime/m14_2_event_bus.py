"""M14.2 durable environment event store.

The store is deliberately narrow: append environment observations, claim them
with leases, and record acknowledgements.  It does not interpret intent, call an
LLM, or mutate MVP state.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

M14_2_ENGINEERING_PROXY_LABEL = "mvp_local_decoupled_self_loop"

_BASE_ENVIRONMENT_EVENT_TYPES = frozenset(
    {
        "UserMessageCommittedEvent",
        "UIPingEvent",
        "UISessionClosedEvent",
        "RunnerStartedEvent",
        "RunnerStoppedEvent",
        "ClockWakeEvent",
        "ScheduledIntentDueEvent",
        "OutboxDeliverySurfaceAvailableEvent",
    }
)

try:
    from segmentum.dialogue.runtime.m16_protocol import M16_PERCEPTION_EVENT_TYPES
except ImportError:  # pragma: no cover - pre-M16.0 checkout
    M16_PERCEPTION_EVENT_TYPES = frozenset()

ENVIRONMENT_EVENT_TYPES = _BASE_ENVIRONMENT_EVENT_TYPES | M16_PERCEPTION_EVENT_TYPES

TERMINAL_EVENT_STATUSES = frozenset({"acked", "expired"})
DEFAULT_IDEMPOTENCY_WINDOW_SECONDS = 24 * 3600


def _now_int(clock: Any | None = None) -> int:
    if clock is None:
        return int(time.time())
    value = clock() if callable(clock) else clock
    return int(value)


@contextmanager
def event_bus_file_lock(root: Path, *, timeout: float = 5.0) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "environment_event_store.lock"
    deadline = time.monotonic() + timeout
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.01)
    if not acquired:
        raise TimeoutError(f"could not acquire event bus lock: {lock_path}")
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class EnvironmentEventStore:
    root: Path
    persona_id: str
    session_id: str
    clock: Any | None = None
    idempotency_window_seconds: int = DEFAULT_IDEMPOTENCY_WINDOW_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def events_path(self) -> Path:
        return self.root / "environment_events.jsonl"

    @property
    def claims_path(self) -> Path:
        return self.root / "environment_event_claims.jsonl"

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None,
        *,
        source: str,
        correlation_id: str | None = None,
    ) -> str:
        if event_type not in ENVIRONMENT_EVENT_TYPES:
            raise ValueError(f"unsupported environment event type: {event_type}")
        now = _now_int(self.clock)
        corr = str(correlation_id or "").strip()
        with event_bus_file_lock(self.root):
            if corr:
                for event in reversed(self._current_events_unlocked(now=now)):
                    if (
                        event.get("event_type") == event_type
                        and event.get("correlation_id") == corr
                        and now - int(event.get("at", 0) or 0) <= self.idempotency_window_seconds
                    ):
                        return str(event.get("event_id"))
            event_id = f"env_{uuid.uuid4().hex}"
            row = {
                "event_id": event_id,
                "event_type": event_type,
                "source": str(source or "unknown")[:64],
                "at": now,
                "persona_id": self.persona_id,
                "session_id": self.session_id,
                "correlation_id": corr,
                "payload": dict(payload or {}),
                "status": "pending",
                "claimed_by": "",
                "claimed_at": 0,
                "acked_at": 0,
                "failure_reason": "",
                "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
            }
            _append_jsonl(self.events_path, row)
            _append_jsonl(
                self.claims_path,
                {
                    "record_type": "audit",
                    "type": "EnvironmentEventAppendedEvent",
                    "at": now,
                    "event_id": event_id,
                    "event_type": event_type,
                    "persona_id": self.persona_id,
                    "session_id": self.session_id,
                    "correlation_id": corr,
                    "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
                },
            )
            return event_id

    def claim_events(
        self,
        consumer_id: str,
        *,
        limit: int,
        event_types: list[str] | tuple[str, ...] | set[str] | None = None,
        lease_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        now = _now_int(self.clock)
        wanted = {str(t) for t in event_types} if event_types else None
        claimed: list[dict[str, Any]] = []
        with event_bus_file_lock(self.root):
            current = self._current_events_unlocked(now=now, lease_seconds=lease_seconds)
            for event in current:
                if len(claimed) >= max(1, int(limit)):
                    break
                if wanted is not None and event.get("event_type") not in wanted:
                    continue
                status = str(event.get("status", "pending"))
                if status in TERMINAL_EVENT_STATUSES:
                    continue
                if status == "failed" and not bool(event.get("retryable", True)):
                    continue
                if status == "claimed" and now - int(event.get("claimed_at", 0) or 0) <= lease_seconds:
                    continue
                claim_row = {
                    "record_type": "claim",
                    "type": "EnvironmentEventClaimedEvent",
                    "at": now,
                    "event_id": event.get("event_id"),
                    "consumer_id": str(consumer_id),
                    "lease_seconds": max(1, int(lease_seconds)),
                    "persona_id": self.persona_id,
                    "session_id": self.session_id,
                    "correlation_id": event.get("correlation_id", ""),
                    "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
                }
                _append_jsonl(self.claims_path, claim_row)
                merged = dict(event)
                merged["status"] = "claimed"
                merged["claimed_by"] = str(consumer_id)
                merged["claimed_at"] = now
                claimed.append(merged)
        return claimed

    def ack_event(self, event_id: str, consumer_id: str, result: Mapping[str, Any] | None = None) -> None:
        now = _now_int(self.clock)
        with event_bus_file_lock(self.root):
            event = self._event_by_id_unlocked(event_id, now=now)
            if not event or str(event.get("claimed_by", "")) != str(consumer_id):
                raise ValueError("event is not claimed by this consumer")
            _append_jsonl(
                self.claims_path,
                {
                    "record_type": "ack",
                    "type": "EnvironmentEventAckedEvent",
                    "at": now,
                    "event_id": event_id,
                    "consumer_id": str(consumer_id),
                    "result": dict(result or {}),
                    "persona_id": self.persona_id,
                    "session_id": self.session_id,
                    "correlation_id": event.get("correlation_id", ""),
                    "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
                },
            )

    def fail_event(self, event_id: str, consumer_id: str, reason: str, *, retryable: bool = True) -> None:
        now = _now_int(self.clock)
        with event_bus_file_lock(self.root):
            event = self._event_by_id_unlocked(event_id, now=now)
            if not event or str(event.get("claimed_by", "")) != str(consumer_id):
                raise ValueError("event is not claimed by this consumer")
            _append_jsonl(
                self.claims_path,
                {
                    "record_type": "fail",
                    "type": "EnvironmentEventFailedEvent",
                    "at": now,
                    "event_id": event_id,
                    "consumer_id": str(consumer_id),
                    "reason": str(reason)[:160],
                    "retryable": bool(retryable),
                    "persona_id": self.persona_id,
                    "session_id": self.session_id,
                    "correlation_id": event.get("correlation_id", ""),
                    "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
                },
            )

    def query_events(
        self,
        *,
        event_types: list[str] | tuple[str, ...] | set[str] | None = None,
        statuses: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        now = _now_int(self.clock)
        wanted_types = {str(t) for t in event_types} if event_types else None
        wanted_statuses = {str(s) for s in statuses} if statuses else None
        rows = self._current_events_unlocked(now=now)
        out: list[dict[str, Any]] = []
        for event in rows:
            if wanted_types is not None and event.get("event_type") not in wanted_types:
                continue
            if wanted_statuses is not None and event.get("status") not in wanted_statuses:
                continue
            out.append(event)
        if limit is not None:
            return out[-max(1, int(limit)) :]
        return out

    def _event_by_id_unlocked(self, event_id: str, *, now: int) -> dict[str, Any] | None:
        for event in self._current_events_unlocked(now=now):
            if str(event.get("event_id")) == str(event_id):
                return event
        return None

    def _current_events_unlocked(self, *, now: int, lease_seconds: int = 60) -> list[dict[str, Any]]:
        events = [dict(row) for row in _read_jsonl(self.events_path)]
        by_id = {str(row.get("event_id")): row for row in events if row.get("event_id")}
        for row in _read_jsonl(self.claims_path):
            event_id = str(row.get("event_id", ""))
            if not event_id or event_id not in by_id:
                continue
            event = by_id[event_id]
            rec_type = str(row.get("record_type", ""))
            if rec_type == "claim":
                event["status"] = "claimed"
                event["claimed_by"] = str(row.get("consumer_id", ""))
                event["claimed_at"] = int(row.get("at", 0) or 0)
            elif rec_type == "ack":
                event["status"] = "acked"
                event["acked_at"] = int(row.get("at", 0) or 0)
            elif rec_type == "fail":
                event["status"] = "failed"
                event["failure_reason"] = str(row.get("reason", ""))
                if bool(row.get("retryable", True)):
                    event["retryable"] = True
                else:
                    event["retryable"] = False
        for event in events:
            if str(event.get("status")) == "claimed":
                claimed_at = int(event.get("claimed_at", 0) or 0)
                if now - claimed_at > max(1, int(lease_seconds)):
                    event["status"] = "pending"
                    event["claimed_by"] = ""
        return events


__all__ = [
    "ENVIRONMENT_EVENT_TYPES",
    "EnvironmentEventStore",
    "M14_2_ENGINEERING_PROXY_LABEL",
    "event_bus_file_lock",
]
