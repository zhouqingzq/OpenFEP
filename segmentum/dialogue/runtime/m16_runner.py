"""M16.1 consciousness runner: owns Path B scheduling for one persona+session."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from segmentum.dialogue.runtime.m14_1_background_continuity import (
    read_runner_lock,
    release_runner_lock,
    runner_lock_is_alive,
    try_acquire_runner_lock,
)
from segmentum.dialogue.runtime.m16_protocol import build_ws_server_message
from segmentum.dialogue.runtime.m16_runtime_bridge import M16SessionBridge, RUNNER_KIND
from segmentum.dialogue.runtime.m16_ws_hub import M16WsHub
from segmentum.dialogue.runtime.mvp_loop import _mapping


DEFAULT_TICK_INTERVAL_SECONDS = 2
DEFAULT_SILENCE_TICK_SECONDS = 5
CLAIM_EVENT_TYPES = frozenset(
    {
        "ClientInputCommittedEvent",
        "DeliverySurfaceReadyEvent",
        "RunnerControlCommandEvent",
    }
)


@dataclass
class RunnerStatus:
    running: bool
    pid: int
    runner_kind: str
    last_health_at: int = 0
    last_tick_at: int = 0
    steps_total: int = 0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "pid": self.pid,
            "runner_kind": self.runner_kind,
            "last_health_at": self.last_health_at,
            "last_tick_at": self.last_tick_at,
            "steps_total": self.steps_total,
            "last_error": self.last_error,
        }


@dataclass
class RunnerStepResult:
    claimed_events: int
    processed: list[dict[str, Any]]
    actuation_messages: list[dict[str, Any]]
    health: dict[str, Any]
    suppression_reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claimed_events": self.claimed_events,
            "processed": self.processed,
            "actuation_messages": self.actuation_messages,
            "health": self.health,
            "suppression_reason_code": self.suppression_reason_code,
        }


@dataclass
class ConsciousnessRunner:
    bridge: M16SessionBridge
    hub: M16WsHub
    tick_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS
    silence_tick_seconds: int = DEFAULT_SILENCE_TICK_SECONDS
    clock: Callable[[], int] | None = None
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _nudge: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock_held: bool = field(default=False, init=False, repr=False)
    _status: RunnerStatus = field(default_factory=lambda: RunnerStatus(False, 0, RUNNER_KIND), init=False)
    _last_silence_tick_at: int = field(default=0, init=False)
    _inline_run_turn: Callable[..., Any] | None = field(default=None, init=False, repr=False)

    def _now(self) -> int:
        if self.clock is None:
            return int(time.time())
        return int(self.clock())

    def start(self) -> RunnerStatus:
        if self._status.running:
            return self.status()
        ok, info = try_acquire_runner_lock(self.bridge.session_root, runner_kind=RUNNER_KIND, now=self._now())
        if not ok:
            self._status.last_error = "runner_collision"
            return RunnerStatus(
                running=False,
                pid=int(getattr(info, "pid", 0) or 0),
                runner_kind=str(getattr(info, "runner_kind", "") or ""),
                last_error="runner_collision",
            )
        self._lock_held = True
        self._stop.clear()
        self.bridge.append_runner_audit(typ="ConsciousnessRunnerStartEvent", now=self._now())
        self._thread = threading.Thread(target=self._loop, name="m16-consciousness-runner", daemon=True)
        self._thread.start()
        self._status = RunnerStatus(running=True, pid=info.pid if info else 0, runner_kind=RUNNER_KIND)
        return self.status()

    def stop(self, *, graceful_seconds: int = 15) -> RunnerStatus:
        del graceful_seconds  # bounded join below
        if not self._status.running:
            return self.status()
        self._stop.set()
        self._nudge.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._lock_held:
            release_runner_lock(self.bridge.session_root)
            self._lock_held = False
        self.bridge.append_runner_audit(typ="ConsciousnessRunnerStopEvent", now=self._now())
        self._status.running = False
        return self.status()

    def status(self) -> RunnerStatus:
        lock = read_runner_lock(self.bridge.session_root)
        alive = runner_lock_is_alive(lock) if lock is not None else False
        if self._status.running and not alive and lock is not None:
            self._status.running = False
        if lock is not None and alive:
            self._status.pid = lock.pid
            self._status.runner_kind = lock.runner_kind
        return self._status

    def nudge(self) -> None:
        self._nudge.set()

    def run_once_for_tests(self, *, now: int | None = None, max_steps: int = 1) -> RunnerStepResult:
        ts = int(now if now is not None else self._now())
        return self._run_cycle(ts, max_event_steps=max(1, int(max_steps)))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ts = self._now()
                self._run_cycle(ts, max_event_steps=16)
            except Exception as exc:
                self._status.last_error = type(exc).__name__
            self._status.last_tick_at = self._now()
            triggered = self._nudge.wait(timeout=max(0.1, float(self.tick_interval_seconds)))
            if triggered:
                self._nudge.clear()

    def _run_cycle(self, ts: int, *, max_event_steps: int) -> RunnerStepResult:
        processed: list[dict[str, Any]] = []
        actuation_messages: list[dict[str, Any]] = []
        claimed = self.bridge.claim_events(limit=max_event_steps, event_types=CLAIM_EVENT_TYPES)
        for event in claimed:
            try:
                row = self._handle_event(event, now=ts)
                processed.append(row)
                self.bridge.ack_event(str(event.get("event_id", "")), result=row)
                actuation_messages.extend(list(row.get("actuation_messages", []) or []))
            except Exception as exc:
                self.bridge.fail_event(str(event.get("event_id", "")), type(exc).__name__, retryable=True)
                processed.append({"event_id": event.get("event_id"), "error": type(exc).__name__})

        if self._silence_tick_due(ts):
            idle = self.bridge.run_idle_cognitive_tick(
                idle_seconds=max(0, ts - int(_mapping(self.bridge.store.load().get("temporal_state")).get("last_user_turn_at", ts) or ts)),
                now=ts,
            )
            processed.append({"phase": "idle_cognitive_tick", **idle})
            self._last_silence_tick_at = ts
            bg = self.bridge.run_background_self_tick()
            processed.append({"phase": "background_self_tick", **bg})

        allowed, suppress_code = self.hub.outbox_drain_allowed(now=ts)
        if allowed:
            drain = self.bridge.drain_queued_outreach(now=ts)
            processed.append({"phase": "outbox_drain", **drain})
            if drain.get("drained"):
                actuation_messages.extend(self._actuation_for_drain(drain, now=ts))
        elif suppress_code:
            self._publish_suppression(suppress_code, now=ts)

        health = self._emit_health(ts, processed_count=len(processed))
        self.bridge.append_runner_audit(
            typ="ConsciousnessRunnerTickEvent",
            now=ts,
            claimed_events=len(claimed),
            processed_count=len(processed),
        )
        self._status.steps_total += 1
        self._status.last_health_at = ts
        return RunnerStepResult(
            claimed_events=len(claimed),
            processed=processed,
            actuation_messages=actuation_messages,
            health=health,
            suppression_reason_code="" if allowed else suppress_code,
        )

    def _silence_tick_due(self, now: int) -> bool:
        if self._last_silence_tick_at <= 0:
            self._last_silence_tick_at = now
            return True
        return now - self._last_silence_tick_at >= self.silence_tick_seconds

    def _handle_event(self, event: Mapping[str, Any], *, now: int) -> dict[str, Any]:
        event_type = str(event.get("event_type", "") or "")
        event_id = str(event.get("event_id", "") or "")
        correlation_id = str(event.get("correlation_id", "") or "")
        if event_type == "ClientInputCommittedEvent":
            return self._handle_client_input(event, now=now)
        if event_type == "DeliverySurfaceReadyEvent":
            self.hub.mark_delivery_surface_ready(correlation_id=correlation_id)
            return {"delivery_surface_ready": True}
        if event_type == "RunnerControlCommandEvent":
            command = str(_mapping(event.get("payload")).get("command", "") or "")
            if command == "stop":
                self._stop.set()
            return {"command": command}
        return {"ignored": event_type, "event_id": event_id}

    def _handle_client_input(self, event: Mapping[str, Any], *, now: int) -> dict[str, Any]:
        event_id = str(event.get("event_id", "") or "")
        correlation_id = str(event.get("correlation_id", "") or "")
        if self.bridge.is_event_processed(event_id):
            return {"skipped": "already_processed", "event_id": event_id, "actuation_messages": []}
        payload = _mapping(event.get("payload"))
        text = str(payload.get("text", "") or "")
        turn_index = self.bridge.next_user_turn_index()
        if self._inline_run_turn is not None:
            result = self._inline_run_turn(text, turn_index=turn_index, now=now)
        else:
            result = self.bridge.run_user_turn(text, turn_index=turn_index)
        self.bridge.mark_event_processed(event_id, now=now)
        delivery_id = f"assistant:{event_id}"
        actuation_messages: list[dict[str, Any]] = []
        if self.bridge.record_actuation(
            delivery_id=delivery_id,
            actuation_type="AssistantMessageDeliveredEvent",
            payload={"text": str(getattr(result, "reply", "") or ""), "turn_index": turn_index},
            correlation_id=correlation_id,
            now=now,
        ):
            accepted = self.hub.build_and_publish(
                kind="UserMessageAccepted",
                payload={"event_id": event_id, "turn_index": turn_index},
                now=now,
            )
            committed = self.hub.build_and_publish(
                kind="AssistantMessageCommitted",
                payload={
                    "text": str(getattr(result, "reply", "") or ""),
                    "turn_index": turn_index,
                    "delivery_id": delivery_id,
                },
                now=now,
            )
            actuation_messages.extend([accepted, committed])
            self.bridge.append_runner_audit(
                typ="GatewayActuationPublishedEvent",
                correlation_id=correlation_id,
                now=now,
                delivery_id=delivery_id,
            )
        return {
            "event_id": event_id,
            "turn_index": turn_index,
            "reply": str(getattr(result, "reply", "") or ""),
            "actuation_messages": actuation_messages,
        }

    def _actuation_for_drain(self, drain: Mapping[str, Any], *, now: int) -> list[dict[str, Any]]:
        proposal_id = str(drain.get("proposal_id", "") or "")
        reply = str(drain.get("reply", "") or "")
        if not proposal_id or not reply:
            return []
        delivery_id = f"proactive:{proposal_id}"
        if not self.bridge.record_actuation(
            delivery_id=delivery_id,
            actuation_type="ProactiveMessageDeliveredEvent",
            payload={"text": reply, "proposal_id": proposal_id},
            now=now,
        ):
            return []
        msg = self.hub.build_and_publish(
            kind="ProactiveMessageCommitted",
            payload={"text": reply, "proposal_id": proposal_id, "delivery_id": delivery_id},
            now=now,
        )
        self.bridge.append_runner_audit(
            typ="GatewayActuationPublishedEvent",
            now=now,
            delivery_id=delivery_id,
            proposal_id=proposal_id,
        )
        return [msg]

    def _publish_suppression(self, reason_code: str, *, now: int) -> None:
        self.hub.build_and_publish(
            kind="RunnerSuppression",
            payload={"reason_code": reason_code},
            now=now,
        )
        self.bridge.record_actuation(
            delivery_id=f"suppression:{now}:{reason_code}",
            actuation_type="ProactiveDeliverySuppressedEvent",
            payload={"reason_code": reason_code},
            now=now,
        )

    def _emit_health(self, now: int, *, processed_count: int) -> dict[str, Any]:
        allowed, reason = self.hub.outbox_drain_allowed(now=now)
        health = {
            "at": now,
            "processed_count": processed_count,
            "delivery_surface_ready": allowed,
            "delivery_surface_reason": reason,
            "steps_total": self._status.steps_total,
        }
        self.bridge.append_runner_audit(typ="ConsciousnessRunnerHealthEvent", now=now, **health)
        self.hub.build_and_publish(kind="RunnerHealth", payload=health, now=now)
        return health
