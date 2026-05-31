"""In-process WebSocket fan-out and delivery-surface tracking for M16 gateway."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from segmentum.dialogue.runtime.m16_protocol import (
    SCHEMA_VERSION,
    build_ws_server_message,
    delivery_surface_allows_outbox_drain,
    validate_ws_server_message,
)


@dataclass
class DeliverySurfaceState:
    ws_subscribed: bool = False
    subscriber_count: int = 0
    delivery_surface_ready_at: int = 0
    last_subscribe_at: int = 0


@dataclass
class M16WsHub:
    persona_id: str
    session_id: str
    clock: Callable[[], int] | None = None
    _subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    _delivery: DeliverySurfaceState = field(default_factory=DeliverySurfaceState)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    def _now(self) -> int:
        if self.clock is None:
            return int(time.time())
        return int(self.clock())

    def register_subscriber(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        with self._lock:
            self._subscribers.append(queue)
            self._delivery.ws_subscribed = True
            self._delivery.subscriber_count = len(self._subscribers)
            self._delivery.last_subscribe_at = self._now()
        return queue

    def unregister_subscriber(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
            self._delivery.subscriber_count = len(self._subscribers)
            self._delivery.ws_subscribed = bool(self._subscribers)

    def mark_delivery_surface_ready(self, *, correlation_id: str = "") -> None:
        with self._lock:
            self._delivery.delivery_surface_ready_at = self._now()

    def clear_delivery_surface_ready(self) -> None:
        with self._lock:
            self._delivery.delivery_surface_ready_at = 0

    def delivery_state(self) -> DeliverySurfaceState:
        with self._lock:
            return DeliverySurfaceState(
                ws_subscribed=self._delivery.ws_subscribed,
                subscriber_count=self._delivery.subscriber_count,
                delivery_surface_ready_at=self._delivery.delivery_surface_ready_at,
                last_subscribe_at=self._delivery.last_subscribe_at,
            )

    def outbox_drain_allowed(self, *, now: int | None = None) -> tuple[bool, str]:
        state = self.delivery_state()
        return delivery_surface_allows_outbox_drain(
            ws_subscribed=state.ws_subscribed,
            delivery_surface_ready_at=state.delivery_surface_ready_at,
            now=now if now is not None else self._now(),
        )

    def _put_to_subscribers(self, message: dict[str, Any], subscribers: list[asyncio.Queue[dict[str, Any]]]) -> None:
        payload = dict(message)
        for queue in subscribers:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                continue

    def publish(self, message: dict[str, Any]) -> None:
        errors = validate_ws_server_message(message)
        if errors:
            raise ValueError(f"invalid ws server message: {errors}")
        with self._lock:
            subscribers = list(self._subscribers)
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                if asyncio.get_running_loop() is loop:
                    self._put_to_subscribers(message, subscribers)
                    return
            except RuntimeError:
                pass
            loop.call_soon_threadsafe(lambda: self._put_to_subscribers(message, subscribers))
            return
        self._put_to_subscribers(message, subscribers)

    def build_and_publish(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        now: int | None = None,
    ) -> dict[str, Any]:
        message = build_ws_server_message(
            kind=kind,
            persona_id=self.persona_id,
            session_id=self.session_id,
            payload=payload,
            now=now if now is not None else self._now(),
        )
        self.publish(message)
        return message

    def subscribed_snapshot_message(self, *, snapshot: dict[str, Any], now: int | None = None) -> dict[str, Any]:
        subscribed = self.build_and_publish(
            kind="Subscribed",
            payload={"schema_version": SCHEMA_VERSION},
            now=now,
        )
        self.build_and_publish(kind="SessionSnapshot", payload=snapshot, now=now)
        return subscribed
