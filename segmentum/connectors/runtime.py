"""Shared ingestion and delivery runtime for external platform connectors."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from segmentum.dialogue.runtime.m16_api import M16Gateway, M16SessionHandle

from .contracts import (
    CONNECTOR_CONTRACT_VERSION,
    ConnectorAdapter,
    ConnectorCapabilities,
    ConnectorDeliveryTarget,
    NormalizedConnectorInput,
)


DEFAULT_TARGET_STORE_FILE = "connector_delivery_targets.jsonl"


def connector_now(clock: Any | None = None) -> int:
    if clock is None:
        return int(time.time())
    value = clock() if callable(clock) else clock
    return int(value)


def _mapping(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}


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
class ConnectorDeliveryTargetStore:
    """Append-only, platform-neutral delivery target ledger."""

    root: Path
    file_name: str = DEFAULT_TARGET_STORE_FILE

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_name = str(self.file_name or DEFAULT_TARGET_STORE_FILE).strip() or DEFAULT_TARGET_STORE_FILE

    @property
    def path(self) -> Path:
        return self.root / self.file_name

    def record(
        self,
        *,
        event_id: str,
        correlation_id: str,
        target: ConnectorDeliveryTarget,
        now: int | None = None,
    ) -> None:
        _append_jsonl(
            self.path,
            {
                "record_type": "target",
                "event_id": str(event_id),
                "correlation_id": str(correlation_id or ""),
                "status": "pending",
                "at": int(now if now is not None else time.time()),
                "target": target.to_payload(),
            },
        )

    def load_payload(self, event_id: str) -> dict[str, Any] | None:
        needle = str(event_id or "").strip()
        if not needle:
            return None
        latest_target: dict[str, Any] | None = None
        for row in reversed(_read_jsonl(self.path)):
            if str(row.get("event_id", "") or "") != needle:
                continue
            if str(row.get("record_type", "") or "") == "delivered":
                return None
            if str(row.get("record_type", "") or "") == "target" and latest_target is None:
                latest_target = _mapping(row.get("target"))
        return latest_target

    def mark_delivered(
        self,
        *,
        event_id: str,
        platform_message_id: str = "",
        now: int | None = None,
    ) -> None:
        _append_jsonl(
            self.path,
            {
                "record_type": "delivered",
                "event_id": str(event_id),
                "platform_message_id": str(platform_message_id or ""),
                "at": int(now if now is not None else time.time()),
            },
        )


class ConnectorRuntime:
    """Routes normalized platform events through the shared dialogue runtime."""

    def __init__(
        self,
        *,
        adapter: ConnectorAdapter,
        gateway: M16Gateway | None = None,
        clock: Any | None = None,
    ) -> None:
        self.adapter = adapter
        self.gateway = gateway or M16Gateway(clock=clock)
        self.clock = clock

    @property
    def platform(self) -> str:
        return str(self.adapter.platform or "").strip()

    @property
    def persona_id(self) -> str:
        return str(self.adapter.persona_id or "").strip() or "default"

    @property
    def account_scope(self) -> str:
        return str(self.adapter.account_scope or "").strip() or "default"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return self.adapter.capabilities

    def target_store(self, session_root: Path) -> ConnectorDeliveryTargetStore:
        return ConnectorDeliveryTargetStore(
            session_root,
            file_name=str(self.adapter.target_store_file or DEFAULT_TARGET_STORE_FILE),
        )

    def ingest_event(self, event: Mapping[str, Any], *, max_cycles: int = 4) -> dict[str, Any]:
        normalized = self.adapter.normalize_event(event)
        if normalized is None:
            return {"accepted": False, "ignored": "unsupported_event", "platform": self.platform}
        return self.ingest_normalized(normalized, max_cycles=max_cycles)

    def ingest_normalized(
        self,
        normalized: NormalizedConnectorInput,
        *,
        max_cycles: int = 4,
    ) -> dict[str, Any]:
        if normalized.platform != self.platform:
            raise ValueError(
                f"normalized platform {normalized.platform!r} does not match adapter {self.platform!r}"
            )
        if normalized.contract_version != CONNECTOR_CONTRACT_VERSION:
            raise ValueError(
                f"connector contract {normalized.contract_version!r} is not supported; "
                f"expected {CONNECTOR_CONTRACT_VERSION!r}"
            )
        handle = self.gateway.get_or_create_session(normalized.persona_id, normalized.session_id)
        target_store = self.target_store(handle.session_root)
        event_id = handle.bridge.append_client_input(
            text=normalized.text,
            correlation_id=normalized.correlation_id,
            source=f"{self.platform}_connector",
            speaker_name=normalized.speaker_name,
            group_turn_envelope=normalized.group_turn_envelope,
            ingress_evidence_band=normalized.ingress_evidence_band,
        )
        target_store.record(
            event_id=event_id,
            correlation_id=normalized.correlation_id,
            target=normalized.delivery_target,
            now=connector_now(self.clock),
        )
        runner = self.gateway.ensure_runner(handle)
        processed_rows: list[dict[str, Any]] = []
        sent_messages: list[dict[str, Any]] = []
        for _ in range(max(1, int(max_cycles))):
            step = runner.run_once(now=connector_now(self.clock), max_steps=1)
            processed_rows.extend(dict(row) for row in step.processed if isinstance(row, Mapping))
            sent_messages.extend(self.deliver_processed_rows(handle.session_root, step.processed))
            if handle.bridge.is_event_processed(event_id):
                break
        return {
            "accepted": True,
            "platform": self.platform,
            "event_id": event_id,
            "persona_id": normalized.persona_id,
            "session_id": normalized.session_id,
            "correlation_id": normalized.correlation_id,
            "ingress_evidence_band": normalized.ingress_evidence_band,
            "processed": processed_rows,
            "sent_messages": sent_messages,
        }

    def deliver_processed_rows(
        self,
        session_root: Path,
        processed_rows: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        target_store = self.target_store(session_root)
        sent: list[dict[str, Any]] = []
        for row in processed_rows:
            event_id = str(row.get("event_id", "") or "").strip()
            reply = str(row.get("reply", "") or "").strip()
            if not event_id or not reply:
                continue
            target_payload = target_store.load_payload(event_id)
            if target_payload is None:
                continue
            target = self.adapter.target_from_payload(target_payload)
            if target is None:
                continue
            receipt = self.adapter.deliver(target=target, text=reply)
            target_store.mark_delivered(
                event_id=event_id,
                platform_message_id=receipt.platform_message_id,
                now=connector_now(self.clock),
            )
            sent.append({"event_id": event_id, **receipt.to_payload()})
        return sent

    def platform_handles(self) -> list[M16SessionHandle]:
        prefix = f"{self.platform}:{self.account_scope}:"
        return [
            handle
            for handle in self.gateway.sessions.values()
            if handle.persona_id == self.persona_id and handle.session_id.startswith(prefix)
        ]

    def drain_proactive_once(self, *, max_sessions: int = 8) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "sessions_considered": min(len(self.platform_handles()), max(0, int(max_sessions))),
            "sent_messages": [],
            "results": [],
        }
