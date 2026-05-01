from __future__ import annotations

import json
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.cognitive_events import make_cognitive_event
from segmentum.cognitive_state import (
    CognitiveStateMVP,
    default_cognitive_state,
    update_cognitive_state,
)
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.tracing import JsonlTraceWriter
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(action: str, score: float, efe: float) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "explore"}},
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
    memory_hit: bool = False,
) -> DecisionDiagnostics:
    ranked = [
        _option("ask_question", first_score, first_efe),
        _option("reflect", second_score, second_efe),
    ]
    return DecisionDiagnostics(
        chosen=ranked[0],
        ranked_options=ranked,
        prediction_error=prediction_error,
        retrieved_memories=[
            {"episode_id": "ep-1", "summary": "ask before advising"}
        ]
        if memory_hit
        else [],
        policy_scores={item.choice: item.policy_score for item in ranked},
        explanation="bounded test diagnostics",
        active_goal="dialogue_turn",
        memory_hit=memory_hit,
        retrieved_episode_ids=["ep-1"] if memory_hit else [],
        memory_context_summary="prior clarification helped",
        prediction_delta={"confusion": 0.25} if memory_hit else {},
    )


def _event(selected_action: str = "ask_question"):
    return make_cognitive_event(
        event_type="PathSelectionEvent",
        turn_id="turn_0001",
        cycle=1,
        session_id="session",
        persona_id="persona",
        source="test",
        sequence_index=0,
        payload={"selected_action": selected_action},
        timestamp="2026-05-01T00:00:00Z",
    )


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


def test_default_state_creation() -> None:
    state = default_cognitive_state()

    assert state.task.task_phase == "observe"
    assert state.affect.social_safety > state.affect.irritation
    assert state.to_dict()["meta_control"]["beta_efe"] == 0.5


def test_update_is_deterministic_for_same_inputs() -> None:
    kwargs = {
        "events": [_event()],
        "diagnostics": _diagnostics(memory_hit=True),
        "observation": {"emotional_tone": 0.55, "conflict_tension": 0.1},
        "previous_outcome": "neutral",
        "self_prior_summary": {"reusable_patterns": ["clarify gently"]},
    }

    first = update_cognitive_state(None, **kwargs)
    second = update_cognitive_state(None, **kwargs)

    assert first == second


def test_json_round_trip() -> None:
    state = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(memory_hit=True),
        observation={"emotional_tone": 0.6},
        self_prior_summary="stable careful helper",
    )

    restored = CognitiveStateMVP.from_dict(json.loads(json.dumps(state.to_dict())))

    assert restored == state


def test_low_margin_creates_epistemic_or_contextual_gap() -> None:
    state = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(
            first_score=1.0,
            second_score=0.95,
            first_efe=0.4,
            second_efe=0.45,
        ),
        observation={"missing_context": 0.6},
    )

    assert state.gaps.epistemic_gaps or state.gaps.contextual_gaps
    assert state.task.task_phase == "clarify"


def test_high_conflict_creates_social_gap_and_affect_pressure() -> None:
    calm = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(),
        observation={"emotional_tone": 0.55, "conflict_tension": 0.0},
    )
    tense = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(),
        observation={"emotional_tone": 0.3, "conflict_tension": 0.9},
    )

    assert tense.gaps.social_gaps
    assert tense.affect.social_safety < calm.affect.social_safety
    assert tense.affect.repair_need > calm.affect.repair_need


def test_positive_or_repaired_prior_outcome_recovers_warmth_and_social_safety() -> None:
    strained = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(),
        observation={"emotional_tone": 0.25, "conflict_tension": 0.9},
        previous_outcome="failed",
    )
    repaired = update_cognitive_state(
        strained,
        events=[_event()],
        diagnostics=_diagnostics(),
        observation={"emotional_tone": 0.7, "conflict_tension": 0.1},
        previous_outcome="repaired",
    )

    assert repaired.affect.warmth > strained.affect.warmth
    assert repaired.affect.social_safety > strained.affect.social_safety


def test_prior_negative_outcome_affects_meta_control_state() -> None:
    neutral = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(),
        observation={"emotional_tone": 0.5},
        previous_outcome="neutral",
    )
    failed = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(),
        observation={"emotional_tone": 0.5},
        previous_outcome="failed",
    )

    assert failed.meta_control.lambda_control > neutral.meta_control.lambda_control
    assert failed.gaps.blocking_gaps


def test_self_prior_summary_is_consumed_without_full_file_access(tmp_path: Path) -> None:
    self_prior = tmp_path / "Self-consciousness.md"
    self_prior.write_text("FULL SELF PRIOR SHOULD NOT BE READ", encoding="utf-8")
    before = self_prior.read_text(encoding="utf-8")

    state = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=_diagnostics(memory_hit=True),
        observation={"emotional_tone": 0.5},
        self_prior_summary={"summary": "compressed stable helper prior"},
    )

    assert "compressed stable helper prior" in state.memory.reusable_patterns
    assert self_prior.read_text(encoding="utf-8") == before
    assert "FULL SELF PRIOR SHOULD NOT BE READ" not in json.dumps(
        state.to_dict(),
        ensure_ascii=False,
    )


def test_update_does_not_change_chosen_action_or_ranking() -> None:
    diagnostics = _diagnostics(first_score=1.0, second_score=0.99)
    before = [item.choice for item in diagnostics.ranked_options]
    chosen_before = diagnostics.chosen.choice

    update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=diagnostics,
        observation={"conflict_tension": 0.8},
        previous_outcome="failed",
    )

    assert diagnostics.chosen.choice == chosen_before
    assert [item.choice for item in diagnostics.ranked_options] == before


def test_run_conversation_attaches_state_without_changing_actions() -> None:
    lines = ["hello, can we check this?", "thanks, that makes sense."]
    observer = DialogueObserver()

    without_state = run_conversation(
        _agent(),
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=630,
        partner_uid=2,
        session_id="m63-side-effect",
    )
    with_state_agent = _agent()
    with_state = run_conversation(
        with_state_agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=630,
        partner_uid=2,
        session_id="m63-side-effect",
        session_context_extra={"self_prior_summary": {"summary": "compressed"}},
    )

    assert [turn.action for turn in with_state] == [
        turn.action for turn in without_state
    ]
    assert all(turn.cognitive_state is not None for turn in with_state)
    assert with_state_agent.latest_cognitive_state is not None


def test_turn_trace_contains_compact_cognitive_state(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"

    run_conversation(
        _agent(),
        ["I am unsure what you mean"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=631,
        session_id="m63-trace",
        trace_writer=JsonlTraceWriter(trace_path),
        session_context_extra={"self_prior_summary": "compressed prior only"},
    )
    row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

    assert "cognitive_state" in row
    assert set(row["cognitive_state"]) == {
        "task",
        "memory",
        "gaps",
        "affect",
        "meta_control",
    }
    assert "compressed prior only" in row["cognitive_state"]["memory"]["reusable_patterns"]
