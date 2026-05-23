"""M14.2 deterministic scheduled outreach intent extraction and storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from segmentum.dialogue.runtime.m14_2_event_bus import (
    M14_2_ENGINEERING_PROXY_LABEL,
    event_bus_file_lock,
)

DEFAULT_DUE_WINDOW_SECONDS = 4 * 3600


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
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


def _now_dt(now: datetime | int | float | None, tz: ZoneInfo) -> datetime:
    if isinstance(now, datetime):
        return now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    if now is None:
        return datetime.now(tz)
    return datetime.fromtimestamp(float(now), tz)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _epoch(text: str) -> int:
    return int(datetime.fromisoformat(text).timestamp())


def _structured_scheduled_outreach_request(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_candidates: list[Any] = []
    many = payload.get("scheduled_outreach_requests")
    if isinstance(many, list):
        raw_candidates.extend(many)
    one = payload.get("scheduled_outreach_request")
    if one is not None:
        raw_candidates.append(one)
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            continue
        kind = str(candidate.get("kind", "scheduled_outreach") or "scheduled_outreach")
        if kind != "scheduled_outreach":
            continue
        if candidate.get("should_schedule") is False:
            continue
        basis = str(candidate.get("basis", "") or "")
        if basis and basis != "user_explicit_request":
            continue
        has_due_after = candidate.get("due_after_seconds") is not None
        has_due_at = bool(str(candidate.get("due_at", "") or "").strip())
        if not (has_due_after or has_due_at):
            continue
        return dict(candidate)
    return None


def _resolve_structured_due_at(
    request: Mapping[str, Any],
    *,
    now: datetime | int | float | None,
    timezone_name: str,
) -> datetime | None:
    tz = ZoneInfo(timezone_name)
    base = _now_dt(now, tz)
    due_at_text = str(request.get("due_at", "") or "").strip()
    if due_at_text:
        try:
            due = datetime.fromisoformat(due_at_text)
        except ValueError:
            return None
        return due.astimezone(tz) if due.tzinfo else due.replace(tzinfo=tz)
    try:
        seconds = int(request.get("due_after_seconds"))
    except (TypeError, ValueError):
        return None
    seconds = max(30, min(seconds, 24 * 3600))
    return base + timedelta(seconds=seconds)


@dataclass(frozen=True)
class ScheduledIntentStore:
    root: Path
    persona_id: str
    session_id: str
    timezone_name: str = "Asia/Shanghai"

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.root / "scheduled_intents.jsonl"

    def create_from_user_message_event(
        self,
        event: Mapping[str, Any],
        *,
        now: datetime | int | float | None = None,
    ) -> dict[str, Any] | None:
        payload = event.get("payload", {}) if isinstance(event.get("payload"), Mapping) else {}
        text = str(payload.get("user_text", payload.get("text", "")) or "")
        structured_request = _structured_scheduled_outreach_request(payload)
        if structured_request is None:
            return None
        source_event_id = str(event.get("event_id", ""))
        existing = self.intent_for_source_event(source_event_id)
        if existing is not None:
            return existing
        due = _resolve_structured_due_at(structured_request, now=now, timezone_name=self.timezone_name)
        if due is None:
            return None
        adjusted = 0
        ordinary_language_intent = str(
            structured_request.get("ordinary_language_intent", "") or _ordinary_language_intent(text)
        )[:240]
        created = _now_dt(now, ZoneInfo(self.timezone_name))
        intent_id = f"sint_{uuid.uuid4().hex}"
        row = {
            "record_type": "intent",
            "intent_id": intent_id,
            "kind": "scheduled_outreach",
            "created_at": _iso(created),
            "created_at_epoch": int(created.timestamp()),
            "due_at": _iso(due),
            "due_at_epoch": int(due.timestamp()),
            "due_window_seconds": DEFAULT_DUE_WINDOW_SECONDS,
            "source_event_id": source_event_id,
            "source_turn_id": str(payload.get("turn_id", payload.get("turn_index", "")) or ""),
            "user_request_excerpt": text[:240],
            "ordinary_language_intent": ordinary_language_intent,
            "status": "pending",
            "delivery_policy": {
                "require_m13_3_assessor": True,
                "max_visible_messages": 1,
                "no_direct_generation": True,
            },
            "evidence_refs": [source_event_id] if source_event_id else [],
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "time_adjusted_to_future": bool(adjusted),
            "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
        }
        with event_bus_file_lock(self.root):
            _append_jsonl(self.path, row)
            _append_jsonl(
                self.path,
                {
                    "record_type": "audit",
                    "type": "ScheduledIntentCreatedEvent",
                    "at": int(created.timestamp()),
                    "intent_id": intent_id,
                    "source_event_id": source_event_id,
                    "due_at": row["due_at"],
                    "persona_id": self.persona_id,
                    "session_id": self.session_id,
                    "correlation_id": str(event.get("correlation_id", "")),
                    "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
                },
            )
        return row

    def list_intents(self, *, statuses: set[str] | None = None) -> list[dict[str, Any]]:
        rows = self._current_intents()
        if statuses is None:
            return rows
        return [row for row in rows if str(row.get("status", "")) in statuses]

    def due_intents(self, *, now: datetime | int | float | None = None) -> list[dict[str, Any]]:
        ts = int(_now_dt(now, ZoneInfo(self.timezone_name)).timestamp())
        self.expire_overdue_intents(now=ts)
        out: list[dict[str, Any]] = []
        for row in self.list_intents(statuses={"pending", "preparing"}):
            due = int(row.get("due_at_epoch", 0) or _epoch(str(row.get("due_at"))))
            if due <= ts:
                out.append(row)
        return out

    def expire_overdue_intents(self, *, now: datetime | int | float | None = None) -> list[dict[str, Any]]:
        ts = int(_now_dt(now, ZoneInfo(self.timezone_name)).timestamp())
        expired: list[dict[str, Any]] = []
        for row in self.list_intents(statuses={"pending", "preparing"}):
            due = int(row.get("due_at_epoch", 0) or _epoch(str(row.get("due_at"))))
            window = max(1, int(row.get("due_window_seconds", DEFAULT_DUE_WINDOW_SECONDS) or DEFAULT_DUE_WINDOW_SECONDS))
            if due and ts > due + window:
                updated = self.mark_status(
                    str(row.get("intent_id", "")),
                    "expired",
                    now=ts,
                    reason="expired",
                )
                if updated is not None:
                    expired.append(updated)
        return expired

    def intent_for_source_event(self, source_event_id: str) -> dict[str, Any] | None:
        if not source_event_id:
            return None
        for row in self._current_intents():
            if str(row.get("source_event_id", "")) == str(source_event_id):
                return row
        return None

    def get(self, intent_id: str) -> dict[str, Any] | None:
        for row in self._current_intents():
            if str(row.get("intent_id", "")) == str(intent_id):
                return row
        return None

    def mark_status(
        self,
        intent_id: str,
        status: str,
        *,
        now: datetime | int | float | None = None,
        reason: str = "",
        proposal_id: str = "",
    ) -> dict[str, Any] | None:
        ts_dt = _now_dt(now, ZoneInfo(self.timezone_name))
        current = self.get(intent_id)
        if current is None:
            return None
        update = {
            "record_type": "update",
            "type": "ScheduledIntentUpdatedEvent",
            "at": int(ts_dt.timestamp()),
            "intent_id": intent_id,
            "status": status,
            "reason": str(reason)[:160],
            "proposal_id": str(proposal_id),
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
        }
        event_type_by_status = {
            "preparing": "ScheduledIntentPreparationStartedEvent",
            "prepared": "ScheduledIntentPreparedEvent",
            "suppressed": "ScheduledIntentSuppressedEvent",
            "expired": "ScheduledIntentExpiredEvent",
            "delivered": "ScheduledIntentUpdatedEvent",
        }
        update["type"] = event_type_by_status.get(status, "ScheduledIntentUpdatedEvent")
        with event_bus_file_lock(self.root):
            _append_jsonl(self.path, update)
        merged = dict(current)
        merged["status"] = status
        if reason:
            merged["last_suppression_reason"] = reason
        if proposal_id:
            merged["proposal_id"] = proposal_id
        return merged

    def _current_intents(self) -> list[dict[str, Any]]:
        intents: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in _read_jsonl(self.path):
            record_type = str(row.get("record_type", "intent"))
            intent_id = str(row.get("intent_id", ""))
            if not intent_id:
                continue
            if record_type == "intent":
                if intent_id not in intents:
                    order.append(intent_id)
                intents[intent_id] = dict(row)
            elif record_type == "update" and intent_id in intents:
                intents[intent_id]["status"] = str(row.get("status", intents[intent_id].get("status", "")))
                if row.get("reason"):
                    intents[intent_id]["last_suppression_reason"] = str(row.get("reason"))
                if row.get("proposal_id"):
                    intents[intent_id]["proposal_id"] = str(row.get("proposal_id"))
        return [intents[i] for i in order if i in intents]


def ensure_scheduled_open_item(state: dict[str, Any], intent: Mapping[str, Any]) -> dict[str, Any]:
    items = state.setdefault("open_items", [])
    if not isinstance(items, list):
        items = []
        state["open_items"] = items
    intent_id = str(intent.get("intent_id", ""))
    existing = next(
        (
            row
            for row in items
            if isinstance(row, dict)
            and str(row.get("type", "")) == "scheduled_outreach"
            and str(row.get("intent_id", "")) == intent_id
        ),
        None,
    )
    due_at = str(intent.get("due_at", ""))
    if existing is not None:
        existing["status"] = existing.get("status") or "open"
        existing["next_check"] = due_at
        return existing
    item = {
        "id": f"open_scheduled_{intent_id}",
        "type": "scheduled_outreach",
        "status": "open",
        "title": "scheduled outreach intent",
        "summary": str(intent.get("ordinary_language_intent", ""))[:160],
        "next_check": due_at,
        "intent_id": intent_id,
        "source_event_id": str(intent.get("source_event_id", "")),
        "created_at": int(intent.get("created_at_epoch", time.time()) or time.time()),
        "evidence_refs": list(intent.get("evidence_refs", []) or [])[:8],
        "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
    }
    items.append(item)
    return item


def close_scheduled_open_item(state: dict[str, Any], intent_id: str, *, status: str = "closed") -> bool:
    changed = False
    for row in state.get("open_items", []) or []:
        if isinstance(row, dict) and str(row.get("intent_id", "")) == str(intent_id):
            row["status"] = status
            changed = True
    return changed


def _ordinary_language_intent(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) > 180:
        compact = compact[:177] + "..."
    return f"Prepare one short approved follow-up for the user's scheduled request: {compact}"


__all__ = [
    "ScheduledIntentStore",
    "close_scheduled_open_item",
    "ensure_scheduled_open_item",
]
