from __future__ import annotations

import json

from segmentum.agent import SegmentAgent
from segmentum.cognitive_events import (
    COGNITIVE_EVENT_CONSUMERS,
    COGNITIVE_EVENT_TYPES,
    CognitiveEvent,
    CognitiveEventBus,
    make_cognitive_event,
    make_cognitive_event_id,
)
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions


def _event(
    event_type: str,
    *,
    source: str = "test",
    persona_id: str = "persona-a",
    salience: float = 0.5,
    sequence_index: int = 0,
) -> CognitiveEvent:
    return make_cognitive_event(
        event_type=event_type,
        turn_id="turn_0001",
        cycle=3,
        session_id="session-a",
        persona_id=persona_id,
        source=source,
        sequence_index=sequence_index,
        payload={"nested": {"ok": True}},
        salience=salience,
        priority=0.4,
        ttl=2,
        timestamp="2026-05-01T00:00:00Z",
    )


def test_cognitive_event_json_safe_round_trip() -> None:
    event = _event("ObservationEvent")

    encoded = json.dumps(event.to_dict(), sort_keys=True)
    decoded = CognitiveEvent.from_dict(json.loads(encoded))

    assert decoded == event
    assert decoded.payload["persona_id"] == "persona-a"


def test_deterministic_event_id_helper_ignores_clock() -> None:
    kwargs = {
        "session_id": "session-a",
        "turn_id": "turn_0001",
        "cycle": 7,
        "event_type": "DecisionEvent",
        "source": "SegmentAgent.decision_cycle_from_dict",
        "sequence_index": 2,
    }

    assert make_cognitive_event_id(**kwargs) == make_cognitive_event_id(**kwargs)
    assert make_cognitive_event_id(**kwargs).startswith(
        "cognitive-session-a-turn-0001-000007-decisionevent"
    )

    persona_scoped = make_cognitive_event_id(**kwargs, persona_id="persona-a")
    other_persona = make_cognitive_event_id(**kwargs, persona_id="persona-b")
    assert persona_scoped != other_persona
    assert "persona-a" in persona_scoped


def test_event_bus_filters_by_type_source_persona_and_salience() -> None:
    bus = CognitiveEventBus()
    bus.publish(_event("ObservationEvent", source="observer", salience=0.9))
    bus.publish(_event("DecisionEvent", source="agent", salience=0.4))
    bus.publish(
        _event(
            "DecisionEvent",
            source="agent",
            persona_id="persona-b",
            salience=0.8,
            sequence_index=2,
        )
    )

    assert len(bus.filter(event_type="DecisionEvent")) == 2
    assert len(bus.filter(source="observer")) == 1
    assert [event.persona_id for event in bus.filter(persona_id="persona-b")] == [
        "persona-b"
    ]
    assert {event.event_type for event in bus.filter(min_salience=0.75)} == {
        "ObservationEvent",
        "DecisionEvent",
    }


def test_event_bus_clear_expired_and_round_trip() -> None:
    live = _event("GenerationEvent", sequence_index=1)
    expired = make_cognitive_event(
        event_type="OutcomeEvent",
        turn_id="turn_0001",
        cycle=3,
        session_id="session-a",
        persona_id="persona-a",
        source="outcome",
        sequence_index=2,
        payload={},
        ttl=0,
        timestamp="2026-05-01T00:00:00Z",
    )
    bus = CognitiveEventBus((live, expired))

    bus.clear_expired()
    restored = CognitiveEventBus.from_dict(bus.to_dict())

    assert restored.events() == (live,)


def test_each_mvp_event_type_has_planned_consumer() -> None:
    assert set(COGNITIVE_EVENT_CONSUMERS) == set(COGNITIVE_EVENT_TYPES)
    for event_type in COGNITIVE_EVENT_TYPES:
        assert COGNITIVE_EVENT_CONSUMERS[event_type]


def test_run_conversation_event_layer_has_no_behavior_side_effects() -> None:
    lines = [
        "hello, can we check this?",
        "thanks, that makes sense.",
    ]
    observer = DialogueObserver()

    agent_without_events = SegmentAgent()
    register_dialogue_actions(agent_without_events.action_registry)
    turns_without_events = run_conversation(
        agent_without_events,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=17,
        partner_uid=2,
        session_id="m61-side-effect",
    )

    agent_with_events = SegmentAgent()
    register_dialogue_actions(agent_with_events.action_registry)
    bus = CognitiveEventBus()
    turns_with_events = run_conversation(
        agent_with_events,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=17,
        partner_uid=2,
        session_id="m61-side-effect",
        persona_id="persona-a",
        cognitive_event_bus=bus,
    )

    assert [turn.action for turn in turns_with_events] == [
        turn.action for turn in turns_without_events
    ]
    assert [turn.text for turn in turns_with_events] == [
        turn.text for turn in turns_without_events
    ]
    assert {event.event_type for event in bus.events()} == set(COGNITIVE_EVENT_TYPES)
    assert all(event.persona_id == "persona-a" for event in bus.events())
