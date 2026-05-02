from __future__ import annotations

from segmentum.agent import SegmentAgent
from segmentum.cognitive_events import CognitiveEvent, CognitiveEventBus, make_cognitive_event
from segmentum.cognitive_state import CognitiveStateMVP, default_cognitive_state
from segmentum.cognition import AttentionGate, AttentionGateConfig, CognitiveLoop
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions


def _event(
    event_type: str = "PathSelectionEvent",
    *,
    turn_id: str = "turn_0001",
    persona_id: str = "persona-a",
    salience: float = 0.5,
    priority: float = 0.5,
    ttl: int = 1,
    sequence_index: int = 0,
    payload: dict[str, object] | None = None,
) -> CognitiveEvent:
    return make_cognitive_event(
        event_type=event_type,
        turn_id=turn_id,
        cycle=1,
        session_id="session-a",
        persona_id=persona_id,
        source="test",
        sequence_index=sequence_index,
        payload=payload or {"selected_action": f"action-{sequence_index}"},
        salience=salience,
        priority=priority,
        ttl=ttl,
        timestamp="2026-05-01T00:00:00Z",
    )


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


def test_attention_gate_filters_events_by_salience_priority_ttl_and_budget() -> None:
    high_salience = _event(salience=0.9, priority=0.1, sequence_index=1)
    high_priority = _event(salience=0.1, priority=0.9, sequence_index=2)
    high_both = _event(salience=0.8, priority=0.8, sequence_index=3)
    low_value = _event(salience=0.1, priority=0.1, sequence_index=4)
    expired = _event(salience=1.0, priority=1.0, ttl=0, sequence_index=5)

    result = AttentionGate(
        AttentionGateConfig(min_salience=0.5, min_priority=0.7, event_budget=2)
    ).select([high_salience, high_priority, high_both, low_value, expired])

    assert result.selected_events == (high_priority, high_both)
    assert high_salience in result.trace_only_events
    assert low_value in result.trace_only_events
    assert expired in result.trace_only_events
    assert result.dropped_reasons[high_salience.event_id] == "over_budget"
    assert result.dropped_reasons[low_value.event_id] == "low_salience_priority"
    assert result.dropped_reasons[expired.event_id] == "expired"


def test_expired_events_are_not_consumed() -> None:
    bus = CognitiveEventBus()
    expired = bus.publish(_event(ttl=0, salience=1.0, priority=1.0, sequence_index=1))
    live = bus.publish(_event(salience=0.8, priority=0.8, sequence_index=2))
    selected_event_ids: list[str] = []

    def capture_updater(
        previous: CognitiveStateMVP | None,
        **kwargs: object,
    ) -> CognitiveStateMVP:
        selected_event_ids.extend(
            event.event_id
            for event in kwargs.get("events", ())
            if isinstance(event, CognitiveEvent)
        )
        return default_cognitive_state()

    CognitiveLoop(bus, state_updater=capture_updater).consume_and_update(
        None,
        turn_id="turn_0001",
        persona_id="persona-a",
        diagnostics=None,
        observation={},
    )

    assert expired.event_id not in bus.consumed_event_ids()
    assert live.event_id in bus.consumed_event_ids()
    assert selected_event_ids == [live.event_id]


def test_low_salience_events_remain_trace_only() -> None:
    bus = CognitiveEventBus()
    low_value = bus.publish(_event(salience=0.1, priority=0.1, sequence_index=1))
    selected_event_ids: list[str] = []

    def capture_updater(
        previous: CognitiveStateMVP | None,
        **kwargs: object,
    ) -> CognitiveStateMVP:
        selected_event_ids.extend(
            event.event_id
            for event in kwargs.get("events", ())
            if isinstance(event, CognitiveEvent)
        )
        return default_cognitive_state()

    result = CognitiveLoop(bus, state_updater=capture_updater).consume_and_update(
        None,
        turn_id="turn_0001",
        persona_id="persona-a",
        diagnostics=None,
        observation={},
    )

    assert low_value.event_id in bus.consumed_event_ids()
    assert low_value in bus.events()
    assert low_value in result.trace_only_events
    assert selected_event_ids == []


def test_message_bus_events_are_consumed_by_cognitive_loop() -> None:
    bus = CognitiveEventBus()
    selected = bus.publish(
        _event(
            salience=0.9,
            priority=0.9,
            sequence_index=1,
            payload={"selected_action": "ask_question"},
        )
    )

    result = CognitiveLoop(bus).consume_and_update(
        None,
        turn_id="turn_0001",
        persona_id="persona-a",
        diagnostics=None,
        observation={},
    )

    assert selected.event_id in bus.consumed_event_ids()
    assert result.consumed_events == (selected,)
    assert result.selected_events == (selected,)
    assert result.state.task.explicit_request == "ask_question"


def test_cognitive_loop_updates_agent_latest_cognitive_state() -> None:
    bus = CognitiveEventBus()
    agent = _agent()

    turns = run_conversation(
        agent,
        ["hello, can we check this?"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=681,
        partner_uid=2,
        session_id="m6x-loop",
        persona_id="persona-a",
        cognitive_event_bus=bus,
    )

    consumed_ids = set(bus.consumed_event_ids())
    consumed_events = [event for event in bus.events() if event.event_id in consumed_ids]

    assert turns[0].cognitive_state is agent.latest_cognitive_state
    assert agent.latest_cognitive_state is not None
    assert consumed_events
    assert {event.turn_id for event in consumed_events} == {"turn_0000"}
    assert "PathSelectionEvent" in {event.event_type for event in consumed_events}
