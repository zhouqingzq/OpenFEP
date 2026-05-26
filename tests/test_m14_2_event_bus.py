from __future__ import annotations

import json
from pathlib import Path

from segmentum.dialogue.runtime.m14_2_event_bus import EnvironmentEventStore


def test_event_bus_append_claim_ack_roundtrip(tmp_path: Path) -> None:
    now = 1_800_000_000
    store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: now)
    event_id = store.append_event(
        "UserMessageCommittedEvent",
        {"user_text": "think about this tonight and leave me a message tomorrow morning"},
        source="test",
        correlation_id="turn-1",
    )
    claimed = store.claim_events("worker", limit=1)
    assert claimed[0]["event_id"] == event_id
    assert claimed[0]["status"] == "claimed"
    store.ack_event(event_id, "worker", {"ok": True})
    events = store.query_events(statuses={"acked"})
    assert len(events) == 1
    assert events[0]["acked_at"] == now
    raw_events = [
        json.loads(line)
        for line in store.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert raw_events[0]["status"] == "pending"


def test_event_bus_claim_lease_recovers_after_crash(tmp_path: Path) -> None:
    clock = {"now": 100}
    store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: clock["now"])
    store.append_event("ClockWakeEvent", {}, source="clock", correlation_id="wake-1")
    first = store.claim_events("worker-a", limit=1, lease_seconds=10)
    assert first and first[0]["claimed_by"] == "worker-a"
    assert not store.claim_events("worker-b", limit=1, lease_seconds=10)
    clock["now"] = 111
    recovered = store.claim_events("worker-b", limit=1, lease_seconds=10)
    assert recovered and recovered[0]["claimed_by"] == "worker-b"


def test_event_bus_non_retryable_failure_is_not_reclaimed(tmp_path: Path) -> None:
    clock = {"now": 100}
    store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: clock["now"])
    event_id = store.append_event("UserMessageCommittedEvent", {"user_text": "x"}, source="test", correlation_id="x")
    claimed = store.claim_events("worker-a", limit=1, lease_seconds=1)
    assert claimed and claimed[0]["event_id"] == event_id
    store.fail_event(event_id, "worker-a", "fatal", retryable=False)
    clock["now"] = 102
    assert store.query_events()[0]["retryable"] is False
    assert store.claim_events("worker-b", limit=1, lease_seconds=1) == []


def test_event_bus_idempotent_correlation_id(tmp_path: Path) -> None:
    store = EnvironmentEventStore(tmp_path, persona_id="p", session_id="s", clock=lambda: 100)
    first = store.append_event("UIPingEvent", {}, source="test", correlation_id="same")
    second = store.append_event("UIPingEvent", {"rerun": True}, source="test", correlation_id="same")
    assert second == first
    assert len(store.query_events()) == 1
