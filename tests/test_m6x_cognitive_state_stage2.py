from __future__ import annotations

import json

from segmentum.cognitive_events import CognitiveEventBus, make_cognitive_event
from segmentum.cognitive_state import CognitiveStateMVP, update_cognitive_state
from segmentum.cognition import CognitiveLoop
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(action: str, score: float, efe: float) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "test"}},
        policy_score=score,
        expected_free_energy=efe,
        predicted_error=0.2,
        action_ambiguity=0.2,
        risk=0.1,
        preferred_probability=0.5,
        memory_bias=0.0,
        pattern_bias=0.0,
        policy_bias=0.0,
        epistemic_bonus=0.0,
        workspace_bias=0.0,
        social_bias=0.0,
        commitment_bias=0.0,
        identity_bias=0.0,
        ledger_bias=0.0,
        subject_bias=0.0,
        goal_alignment=0.0,
        value_score=0.0,
        predicted_outcome="stable",
        predicted_effects={},
        dominant_component="expected_free_energy",
        cost=0.0,
    )


def _diagnostics(
    *,
    first_score: float = 1.0,
    second_score: float = 0.5,
    first_efe: float = 0.2,
    second_efe: float = 0.7,
    prediction_error: float = 0.2,
) -> DecisionDiagnostics:
    ranked = [
        _option("ask_question", first_score, first_efe),
        _option("reflect", second_score, second_efe),
    ]
    return DecisionDiagnostics(
        chosen=ranked[0],
        ranked_options=ranked,
        prediction_error=prediction_error,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in ranked},
        explanation="stage2 fixture",
        active_goal="dialogue_turn",
    )


def _event(
    event_type: str,
    *,
    payload: dict[str, object],
    sequence_index: int,
) -> object:
    return make_cognitive_event(
        event_type=event_type,
        turn_id="turn_0001",
        cycle=1,
        session_id="session-a",
        persona_id="persona-a",
        source="test",
        sequence_index=sequence_index,
        payload=payload,
        salience=0.9,
        priority=0.8,
        ttl=1,
        timestamp="2026-05-01T00:00:00Z",
    )


def test_cognitive_loop_updates_resource_user_world_and_path_state() -> None:
    bus = CognitiveEventBus()
    bus.publish(
        _event(
            "ObservationEvent",
            payload={"current_turn": "I am confused and missing context"},
            sequence_index=1,
        )
    )
    bus.publish(
        _event(
            "PathSelectionEvent",
            payload={"selected_action": "ask_question"},
            sequence_index=2,
        )
    )

    result = CognitiveLoop(bus).consume_and_update(
        None,
        turn_id="turn_0001",
        persona_id="persona-a",
        diagnostics=_diagnostics(first_score=1.0, second_score=0.96),
        observation={
            "missing_context": 0.75,
            "hidden_intent": 0.74,
            "relationship_depth": 0.2,
            "stress": 0.7,
        },
    )
    state = result.state

    assert state.resource.selected_event_count == 2
    assert state.resource.pressure_sources == ["stress"]
    assert state.user.inferred_intent == "ambiguous"
    assert state.user.ambiguity == 0.75
    assert "missing_context" in state.world.salient_conditions
    assert state.world.uncertainty == 0.75
    assert state.candidate_paths.selected_action == "ask_question"
    assert state.candidate_paths.candidate_count == 2
    assert state.candidate_paths.low_margin


def test_gap_detector_returns_structured_blocking_soft_latent_gaps() -> None:
    state = update_cognitive_state(
        None,
        events=[
            _event(
                "PathSelectionEvent",
                payload={"selected_action": "ask_question"},
                sequence_index=1,
            )
        ],
        diagnostics=_diagnostics(
            first_score=1.0,
            second_score=0.97,
            first_efe=0.3,
            second_efe=0.32,
            prediction_error=0.4,
        ),
        observation={"missing_context": 0.7},
        previous_outcome="failed",
    )

    statuses = {gap.status for gap in state.gaps.structured_gaps}
    descriptions = {gap.description for gap in state.gaps.structured_gaps}

    assert {"blocking", "soft", "latent"}.issubset(statuses)
    assert "prior failure should be repaired before escalation" in descriptions
    assert state.gaps.blocking_gaps
    assert state.gaps.contextual_gaps


def test_cognitive_state_backward_compatible_with_existing_fields() -> None:
    legacy_payload = {
        "task": {
            "explicit_request": "ask_question",
            "inferred_need": "respond",
            "current_goal": "dialogue_turn",
            "task_phase": "act",
            "success_criteria": ["address selected action"],
            "urgency": 0.25,
        },
        "memory": {
            "activated_memories": [],
            "reusable_patterns": [],
            "memory_conflicts": [],
            "abstraction_candidates": [],
            "memory_helpfulness": 0.0,
        },
        "gaps": {
            "epistemic_gaps": [],
            "contextual_gaps": ["missing context for confident response"],
            "instrumental_gaps": [],
            "resource_gaps": [],
            "social_gaps": [],
            "blocking_gaps": [],
        },
        "affect": {
            "mood_valence": 0.5,
            "arousal": 0.25,
            "social_safety": 0.75,
            "irritation": 0.0,
            "warmth": 0.5,
            "fatigue_pressure": 0.0,
            "repair_need": 0.0,
            "decay_rate": 0.18,
            "affective_notes": [],
        },
        "meta_control": {
            "lambda_energy": 0.25,
            "lambda_attention": 0.35,
            "lambda_memory": 0.25,
            "lambda_control": 0.35,
            "beta_efe": 0.5,
            "exploration_temperature": 0.35,
            "control_gain": 0.35,
            "memory_retrieval_gain": 0.25,
            "abstraction_gain": 0.2,
        },
    }

    restored = CognitiveStateMVP.from_dict(legacy_payload)
    round_tripped = CognitiveStateMVP.from_dict(
        json.loads(json.dumps(restored.to_dict()))
    )
    legacy_view = restored.to_legacy_dict()

    assert round_tripped == restored
    assert set(legacy_view) == {"task", "memory", "gaps", "affect", "meta_control"}
    assert "structured_gaps" not in legacy_view["gaps"]
    assert restored.resource.attention_budget == 8
    assert restored.user.inferred_intent == "unknown"
    assert restored.world.observable_channels == {}
    assert restored.candidate_paths.candidate_count == 0
