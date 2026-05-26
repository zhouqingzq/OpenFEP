"""M14.2 standalone MVP self-loop daemon.

The daemon consumes environment events and due scheduled intents.  It prepares
durable outbox proposals only; delivery still goes through M13.3.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_idle import (
    gather_idle_structural_signals,
    normalize_idle_introspection_state,
)
from segmentum.dialogue.runtime.m13_initiative import build_reflection_outreach_proposal
from segmentum.dialogue.runtime.m14_1_background_continuity import (
    BackgroundBudgetExhausted,
    BackgroundLLMMeter,
    check_background_budgets,
    enqueue_outreach_proposal,
    load_queued_outreach,
    maybe_rollover_daily_counters,
    merge_background_continuity_into_initiative,
    normalize_background_continuity_state,
    read_runner_lock,
    release_runner_lock,
    record_background_tick,
    session_file_lock,
    try_acquire_runner_lock,
)
from segmentum.dialogue.runtime.m14_2_event_bus import (
    M14_2_ENGINEERING_PROXY_LABEL,
    EnvironmentEventStore,
)
from segmentum.dialogue.runtime.m14_2_scheduled_intents import (
    ScheduledIntentStore,
    ensure_scheduled_open_item,
)
from segmentum.dialogue.runtime.mvp_loop import MVPDialogueRuntime, MVPStateStore
from segmentum.dialogue.runtime.m13_drive import _mapping, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_initiative import normalize_initiative_state


def _safe_component(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw or ""))
    return safe.strip("_") or "default"


def default_session_root(persona_id: str, session_id: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return (
        project_root
        / "artifacts"
        / "mvp_personas"
        / _safe_component(persona_id)
        / "sessions"
        / _safe_component(session_id)
    )


@dataclass
class M142SelfLoopDaemon:
    runtime: MVPDialogueRuntime
    persona_id: str
    session_id: str
    tick_interval_seconds: int = 90
    timezone_name: str = "Asia/Shanghai"
    runner_kind: str = "standalone_daemon"
    consumer_id: str = ""
    clock: Any | None = None

    def __post_init__(self) -> None:
        if not self.consumer_id:
            self.consumer_id = f"m14_2_{_safe_component(self.persona_id)}_{_safe_component(self.session_id)}"
        self.event_store = EnvironmentEventStore(
            self.runtime.store.root,
            persona_id=self.persona_id,
            session_id=self.session_id,
            clock=self.clock,
        )
        self.intent_store = ScheduledIntentStore(
            self.runtime.store.root,
            persona_id=self.persona_id,
            session_id=self.session_id,
            timezone_name=self.timezone_name,
        )
        self._stop = False

    def start_once(self) -> bool:
        now = self._now()
        ok, info = try_acquire_runner_lock(self.runtime.store.root, runner_kind=self.runner_kind, now=now)
        if not ok:
            self._audit(
                {
                    "type": "SelfLoopDaemonStopEvent",
                    "at": now,
                    "reason": "runner_collision",
                    "existing_pid": getattr(info, "pid", 0),
                }
            )
            return False
        self.event_store.append_event(
            "RunnerStartedEvent",
            {"runner_kind": self.runner_kind, "pid": getattr(info, "pid", 0) if info else 0},
            source="cli_runner",
            correlation_id=f"runner-start:{self.runner_kind}:{now}",
        )
        self._audit({"type": "SelfLoopDaemonStartEvent", "at": now})
        return True

    def stop(self, *, reason: str = "stop_requested") -> None:
        if self._stop:
            return
        self._stop = True
        now = self._now()
        self.event_store.append_event(
            "RunnerStoppedEvent",
            {"runner_kind": self.runner_kind, "reason": reason},
            source="cli_runner",
            correlation_id=f"runner-stop:{self.runner_kind}:{now}",
        )
        self._audit({"type": "SelfLoopDaemonStopEvent", "at": now, "reason": reason})
        release_runner_lock(self.runtime.store.root)

    def run_forever(self) -> None:
        if not self.start_once():
            return
        try:
            while not self._stop:
                self.tick_once(record_clock_wake=True)
                self._run_background_self_tick_safely()
                time.sleep(max(1, int(self.tick_interval_seconds)))
        finally:
            self.stop(reason="loop_exit")

    def _run_background_self_tick_safely(self) -> dict[str, Any]:
        """Drive the M14.1 periodic background self/reflection tick alongside the
        M14.2 event-driven loop. Without this, ``background_ticks_today`` only
        increments when a scheduled intent comes due, so a daemon with no due
        intents looks alive (``SelfLoopDaemonHealthEvent``) but the M14.1
        counters stay at zero. Exceptions are caught so a single tick failure
        cannot kill the event loop.
        """
        try:
            return dict(self.runtime.run_background_self_tick(runner_kind=self.runner_kind))
        except Exception as exc:
            self._audit(
                {
                    "type": "BackgroundIdleTickEvent",
                    "at": self._now(),
                    "skip_reason": "tick_error",
                    "detail": str(exc)[:240],
                }
            )
            return {"skip_reason": "tick_error", "ran_introspection": False}

    def tick_once(self, *, record_clock_wake: bool = True) -> dict[str, Any]:
        now = self._now()
        if record_clock_wake:
            self.event_store.append_event(
                "ClockWakeEvent",
                {"runner_kind": self.runner_kind},
                source="clock",
                correlation_id=f"clock:{now}",
            )
        claimed = self.event_store.claim_events(
            self.consumer_id,
            limit=16,
            event_types={
                "UserMessageCommittedEvent",
                "ClockWakeEvent",
                "ScheduledIntentDueEvent",
                "UIPingEvent",
                "OutboxDeliverySurfaceAvailableEvent",
                "UISessionClosedEvent",
            },
            lease_seconds=60,
        )
        event_results: list[dict[str, Any]] = []
        for event in claimed:
            try:
                result = self._handle_event(event, now=now)
                self.event_store.ack_event(str(event.get("event_id")), self.consumer_id, result=result)
                event_results.append({"event_id": event.get("event_id"), "result": result})
            except Exception as exc:
                self.event_store.fail_event(str(event.get("event_id")), self.consumer_id, str(exc), retryable=True)
                event_results.append({"event_id": event.get("event_id"), "error": type(exc).__name__})
        prepared = self.prepare_due_intents(now=now)
        all_events = self.event_store.query_events(limit=400)
        event_status_counts = {
            "acked_count": sum(1 for row in all_events if str(row.get("status", "")) == "acked"),
            "pending_count": sum(1 for row in all_events if str(row.get("status", "")) == "pending"),
            "claimed_count": sum(1 for row in all_events if str(row.get("status", "")) == "claimed"),
            "failed_count": sum(1 for row in all_events if str(row.get("status", "")) == "failed"),
            "expired_count": sum(1 for row in all_events if str(row.get("status", "")) == "expired"),
        }
        terminal = event_status_counts["acked_count"] + event_status_counts["expired_count"]
        pending_like = event_status_counts["pending_count"] + event_status_counts["claimed_count"] + event_status_counts["failed_count"]
        terminal_ratio = round(terminal / max(1, terminal + pending_like), 4)
        state = self.runtime.store.load()
        bg = _mapping(_mapping(_mapping(state.get("m13_drive_state")).get("initiative")).get("background_continuity"))
        llm_available = self.runtime.llm is not None
        self._audit(
            {
                "type": "SelfLoopDaemonHealthEvent",
                "at": now,
                "claimed_events": len(claimed),
                "prepared_due_intents": len(prepared),
                "environment_events_pending_acked_ratio": terminal_ratio,
                "environment_events_terminal_ratio": terminal_ratio,
                "environment_event_status_counts": dict(event_status_counts),
                "llm_available": llm_available,
                "llm_unavailable_reason": "" if llm_available else "llm_unavailable",
                "background_ran_llm": bool(bg.get("last_background_ran_llm", False)),
            }
        )
        return {
            "claimed_events": len(claimed),
            "event_results": event_results,
            "prepared_intents": prepared,
            "environment_events_pending_acked_ratio": terminal_ratio,
            "environment_events_terminal_ratio": terminal_ratio,
            "environment_event_status_counts": dict(event_status_counts),
            "llm_available": llm_available,
            "llm_unavailable_reason": "" if llm_available else "llm_unavailable",
            "background_ran_llm": bool(bg.get("last_background_ran_llm", False)),
        }

    def prepare_due_intents(self, *, now: int | None = None) -> list[dict[str, Any]]:
        ts = int(now if now is not None else self._now())
        prepared: list[dict[str, Any]] = []
        for intent in self.intent_store.due_intents(now=ts):
            try:
                result = self.prepare_intent(intent, now=ts)
                if result:
                    prepared.append(result)
            except Exception as exc:
                self.intent_store.mark_status(
                    str(intent.get("intent_id", "")),
                    "suppressed",
                    now=ts,
                    reason=type(exc).__name__,
                )
                self._audit(
                    {
                        "type": "ScheduledIntentSuppressedEvent",
                        "at": ts,
                        "reason": type(exc).__name__,
                        "detail": str(exc)[:160],
                        "source_intent_id": str(intent.get("intent_id", "")),
                    }
                )
        return prepared

    def prepare_intent(self, intent: Mapping[str, Any], *, now: int | None = None) -> dict[str, Any] | None:
        ts = int(now if now is not None else self._now())
        intent_id = str(intent.get("intent_id", ""))
        if not intent_id:
            return None

        for row in load_queued_outreach(self.runtime.store.root):
            if str(row.get("source_intent_id", "")) == intent_id:
                self.intent_store.mark_status(
                    intent_id,
                    "prepared",
                    now=ts,
                    proposal_id=str(row.get("proposal_id", "")),
                )
                return {
                    "intent_id": intent_id,
                    "proposal_id": str(row.get("proposal_id", "")),
                    "recovered": True,
                }

        turn_index = 0
        with session_file_lock(self.runtime.store.root):
            state = self.runtime.store.load()
            turn_index = self._turn_index(state)
            m13_state = normalize_m13_drive_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            idle = normalize_idle_introspection_state(initiative.get("idle_introspection"))
            bg, rollover = maybe_rollover_daily_counters(bg, now=ts)
            if rollover:
                self._audit({"runner_kind": self.runner_kind, **rollover})
            if not bool(bg.get("user_opt_in")):
                self.intent_store.mark_status(intent_id, "suppressed", now=ts, reason="user_opted_out")
                self._audit_suppression(intent, reason="user_opted_out", now=ts)
                return None
            if not bool(initiative.get("user_opt_in")) or not bool(idle.get("enabled")):
                self.intent_store.mark_status(
                    intent_id,
                    "suppressed",
                    now=ts,
                    reason="idle_introspection_disabled",
                )
                self._audit_suppression(intent, reason="idle_introspection_disabled", now=ts)
                return None
            block = check_background_budgets(bg)
            if block:
                bg["last_budget_block_reason"] = block
                initiative["background_continuity"] = bg
                m13_state["initiative"] = initiative
                state["m13_drive_state"] = m13_state
                self.runtime.store.save(state)
                self.intent_store.mark_status(intent_id, "suppressed", now=ts, reason="budget_exhausted")
                self._audit_suppression(intent, reason="budget_exhausted", now=ts)
                return None
            self.intent_store.mark_status(intent_id, "preparing", now=ts)
            ensure_scheduled_open_item(state, intent)
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.runtime.store.save(state)

        self._audit(
            {
                "type": "ScheduledIntentPreparationStartedEvent",
                "at": ts,
                "source_intent_id": intent_id,
            }
        )

        wall_start = time.monotonic()
        idle_result = None
        meter: BackgroundLLMMeter | None = None
        original_llm = self.runtime.llm
        try:
            state = self.runtime.store.load()
            signals = gather_idle_structural_signals(state, now=ts, turn_index=turn_index)
            m13_state = normalize_m13_drive_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            meter = BackgroundLLMMeter(self.runtime.llm, bg)
            self.runtime.llm = meter  # type: ignore[assignment]
            idle_result = self.runtime.run_idle_introspection_turn(
                now=ts,
                turn_index=turn_index,
                structural_signals=signals,
                queue_outreach=False,
                allow_direct_outreach=False,
                background_runner_kind=self.runner_kind,
            )
        except BackgroundBudgetExhausted as exc:
            if meter is not None:
                self.runtime._persist_background_meter(meter, now=ts, block_reason=exc.reason)
            self.intent_store.mark_status(intent_id, "suppressed", now=ts, reason="budget_exhausted")
            self._audit_suppression(intent, reason="budget_exhausted", now=ts)
            return None
        finally:
            self.runtime.llm = original_llm
            if meter is not None:
                self.runtime._persist_background_meter(meter, now=ts)

        if idle_result is None:
            return None

        focus = _mapping(idle_result.reflection_focus)
        topic = str(focus.get("topic", "scheduled outreach") or "scheduled outreach")[:120]
        evidence = list(intent.get("evidence_refs", []) or [])[:8]
        focus_refs = [str(r) for r in focus.get("evidence_refs", []) or [] if r]
        if focus_refs:
            evidence = (evidence + focus_refs)[:8]
        if not evidence:
            evidence = [f"open_scheduled_{intent_id}"]

        state = self.runtime.store.load()
        m13_state = normalize_m13_drive_state(state.get("m13_drive_state", {}))
        initiative = merge_background_continuity_into_initiative(
            normalize_initiative_state(m13_state.get("initiative"))
        )
        proposal = build_reflection_outreach_proposal(
            suggested_intent=str(intent.get("ordinary_language_intent", "")),
            evidence_refs=evidence,
            proposed_topic=topic,
            now=ts,
            initiative=initiative,
        )
        proposal_payload = proposal.to_dict()
        proposal_payload["trigger"] = "scheduled_outreach"
        proposal_payload["source_intent_id"] = intent_id
        proposal_payload["due_at"] = int(intent.get("due_at_epoch", ts) or ts)
        proposal_payload["persona_id"] = self.persona_id
        proposal_payload["session_id"] = self.session_id
        sig_dict = idle_result.diagnostics.get("structural_signals", {})
        entry = enqueue_outreach_proposal(
            self.runtime.store.root,
            proposal=proposal_payload,
            now=ts,
            ttl_seconds=max(3600, int(intent.get("due_window_seconds", 4 * 3600) or 4 * 3600)),
            drive_snapshot={
                "runner_kind": self.runner_kind,
                "scheduled_intent_status": str(intent.get("status", "")),
                "boredom_band": _mapping(sig_dict).get("boredom_band", ""),
                "ran_llm": bool(idle_result.ran_llm),
            },
            due_at=int(intent.get("due_at_epoch", ts) or ts),
            source_intent_id=intent_id,
        )
        self.intent_store.mark_status(
            intent_id,
            "prepared",
            now=ts,
            proposal_id=str(entry.get("proposal_id", "")),
        )

        with session_file_lock(self.runtime.store.root):
            state = self.runtime.store.load()
            m13_state = normalize_m13_drive_state(state.get("m13_drive_state", {}))
            initiative = merge_background_continuity_into_initiative(
                normalize_initiative_state(m13_state.get("initiative"))
            )
            bg = normalize_background_continuity_state(initiative.get("background_continuity"))
            ran_introspection = bool(idle_result.ran_llm or idle_result.reflection_focus)
            bg = record_background_tick(
                bg,
                wallclock_seconds=time.monotonic() - wall_start,
                ran_introspection=ran_introspection,
            )
            bg["last_tick_at"] = ts
            bg["last_budget_block_reason"] = ""
            initiative["background_continuity"] = bg
            m13_state["initiative"] = initiative
            state["m13_drive_state"] = m13_state
            self.runtime.store.save(state)

        self._audit(
            {
                "type": "ScheduledIntentPreparedEvent",
                "at": ts,
                "source_intent_id": intent_id,
                "proposal_id": str(entry.get("proposal_id", "")),
                "ran_llm": bool(idle_result.ran_llm),
            }
        )
        self._audit(
            {
                "type": "OutboxEntryCreatedEvent",
                "at": ts,
                "source_intent_id": intent_id,
                "proposal_id": str(entry.get("proposal_id", "")),
                "ordinary_language_intent": str(intent.get("ordinary_language_intent", ""))[:240],
            }
        )
        return {"intent_id": intent_id, "proposal_id": str(entry.get("proposal_id", ""))}

    def _handle_event(self, event: Mapping[str, Any], *, now: int) -> dict[str, Any]:
        event_type = str(event.get("event_type", ""))
        if event_type == "UserMessageCommittedEvent":
            intent = self.intent_store.create_from_user_message_event(event, now=now)
            if intent is None:
                return {"scheduled_intent_created": False}
            with session_file_lock(self.runtime.store.root):
                state = self.runtime.store.load()
                item = ensure_scheduled_open_item(state, intent)
                self.runtime.store.save(state)
            self._audit(
                {
                    "type": "ScheduledIntentCreatedEvent",
                    "at": now,
                    "source_event_id": event.get("event_id", ""),
                    "source_intent_id": intent.get("intent_id", ""),
                    "open_item_id": item.get("id", ""),
                }
            )
            return {"scheduled_intent_created": True, "intent_id": intent.get("intent_id", "")}
        if event_type == "ClockWakeEvent":
            due = self.intent_store.due_intents(now=now)
            for intent in due:
                self.event_store.append_event(
                    "ScheduledIntentDueEvent",
                    {"intent_id": intent.get("intent_id", ""), "due_at": intent.get("due_at", "")},
                    source="cli_runner",
                    correlation_id=f"due:{intent.get('intent_id', '')}",
                )
            return {"due_intents": len(due)}
        if event_type == "ScheduledIntentDueEvent":
            intent_id = str(_mapping(event.get("payload")).get("intent_id", ""))
            intent = self.intent_store.get(intent_id)
            if intent:
                prepared = self.prepare_intent(intent, now=now)
                return {"prepared": bool(prepared), "intent_id": intent_id}
            return {"introspection": "skipped", "reason": "intent_not_found", "intent_id": intent_id}
        if event_type in {"UIPingEvent", "OutboxDeliverySurfaceAvailableEvent", "UISessionClosedEvent"}:
            return {"introspection": "skipped", "reason": "environment_audit_event"}
        return {"ignored": True}

    def _turn_index(self, state: Mapping[str, Any]) -> int:
        temporal = _mapping(state.get("temporal_state"))
        return int(temporal.get("last_turn_index", 0) or 0)

    def _now(self) -> int:
        if self.clock is None:
            return int(time.time())
        return int(self.clock() if callable(self.clock) else self.clock)

    def _audit_suppression(self, intent: Mapping[str, Any], *, reason: str, now: int) -> None:
        self._audit(
            {
                "type": "ScheduledIntentSuppressedEvent",
                "at": now,
                "reason": reason,
                "source_intent_id": str(intent.get("intent_id", "")),
            }
        )

    def _audit(self, row: Mapping[str, Any]) -> None:
        payload = {
            "event": "m14_2_audit",
            "runner_kind": self.runner_kind,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "correlation_id": str(row.get("correlation_id", "")),
            "engineering_proxy_label": M14_2_ENGINEERING_PROXY_LABEL,
            **dict(row),
        }
        self.runtime.store.append_log(payload)


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M14.2 MVP self-loop daemon")
    parser.add_argument("--session-root", default="", help="MVP session directory (test/debug override)")
    parser.add_argument("--persona", default="", help="Persona id")
    parser.add_argument("--session", default="", help="Session id")
    parser.add_argument("--tick-interval", type=int, default=90)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args(argv)
    if args.session_root:
        session_root = Path(args.session_root).resolve()
        persona_id = args.persona or "default"
        session_id = args.session or session_root.name
    else:
        if not args.persona or not args.session:
            parser.error("either --session-root or both --persona and --session are required")
        persona_id = args.persona
        session_id = args.session
        session_root = default_session_root(persona_id, session_id)

    from segmentum.dialogue.runtime.mvp_loop import OpenRouterJSONClient

    store = MVPStateStore(session_root)
    llm = OpenRouterJSONClient.available() and OpenRouterJSONClient() or None
    runtime = MVPDialogueRuntime(store=store, llm=llm)
    daemon = M142SelfLoopDaemon(
        runtime,
        persona_id=persona_id,
        session_id=session_id,
        tick_interval_seconds=args.tick_interval,
        timezone_name=args.timezone,
    )

    def _handle_signal(*_args: object) -> None:
        daemon.stop(reason="signal")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    daemon.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
