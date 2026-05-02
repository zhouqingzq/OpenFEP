from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from segmentum.agent import SegmentAgent
from segmentum.cognitive_paths import (
    cognitive_paths_from_diagnostics,
    path_competition_summary,
)
from segmentum.cognitive_state import default_cognitive_state, update_cognitive_state
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.dialogue.turn_trace import ConsciousMarkdownWriter
from segmentum.meta_control_guidance import (
    MetaControlGuidance,
    generate_meta_control_guidance,
)
from segmentum.tracing import JsonlTraceWriter
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(
    action: str,
    *,
    policy_score: float,
    expected_free_energy: float,
    risk: float = 0.1,
) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "test"}},
        policy_score=policy_score,
        expected_free_energy=expected_free_energy,
        predicted_error=0.2,
        action_ambiguity=0.1,
        risk=risk,
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
        predicted_outcome=f"{action}_outcome",
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
        _option(
            "ask_question",
            policy_score=first_score,
            expected_free_energy=first_efe,
        ),
        _option(
            "reflect",
            policy_score=second_score,
            expected_free_energy=second_efe,
        ),
    ]
    return DecisionDiagnostics(
        chosen=ranked[0],
        ranked_options=ranked,
        prediction_error=prediction_error,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in ranked},
        explanation="m65 fixture",
    )


def _state(
    diagnostics: DecisionDiagnostics | None = None,
    observation: dict[str, float] | None = None,
    *,
    previous_outcome: str = "",
) -> object:
    return update_cognitive_state(
        None,
        events=[],
        diagnostics=diagnostics or _diagnostics(),
        observation=observation or {"emotional_tone": 0.5},
        previous_outcome=previous_outcome,
    )


def _path_summary(diagnostics: DecisionDiagnostics) -> dict[str, object]:
    return path_competition_summary(cognitive_paths_from_diagnostics(diagnostics))


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


def test_guidance_is_deterministic_for_same_state_and_diagnostics() -> None:
    diagnostics = _diagnostics(first_score=1.0, second_score=0.96)
    state = _state(diagnostics, {"missing_context": 0.7})
    kwargs = {
        "diagnostics": diagnostics,
        "path_summary": _path_summary(diagnostics),
        "previous_outcome": "neutral",
        "prompt_budget": {"used_ratio": 0.4},
    }

    first = generate_meta_control_guidance(state, **kwargs)
    second = generate_meta_control_guidance(state, **kwargs)

    assert first == second
    assert MetaControlGuidance.from_dict(first.to_dict()) == first


def test_low_margin_sets_clarification_and_lower_assertion_guidance() -> None:
    diagnostics = _diagnostics(
        first_score=1.0,
        second_score=0.97,
        first_efe=0.3,
        second_efe=0.32,
    )

    guidance = generate_meta_control_guidance(
        _state(diagnostics, {"missing_context": 0.55}),
        diagnostics=diagnostics,
        path_summary=_path_summary(diagnostics),
    )

    assert guidance.ask_clarifying_question
    assert guidance.lower_assertiveness
    assert "low decision margin" in guidance.trigger_reasons


def test_high_conflict_sets_repair_and_control_guidance() -> None:
    diagnostics = _diagnostics()
    state = _state(
        diagnostics,
        {"emotional_tone": 0.25, "conflict_tension": 0.95},
    )

    guidance = generate_meta_control_guidance(
        state,
        diagnostics=diagnostics,
        path_summary=_path_summary(diagnostics),
    )

    assert guidance.prefer_repair_strategy
    assert guidance.increase_control_gain
    assert guidance.deescalate_affect


def test_low_social_safety_preserves_warmth_or_deescalates() -> None:
    state = default_cognitive_state()
    state = replace(
        state,
        affect=replace(state.affect, social_safety=0.35, repair_need=0.3),
    )

    guidance = generate_meta_control_guidance(state, diagnostics=_diagnostics())

    assert guidance.deescalate_affect or guidance.preserve_warmth


def test_high_irritation_reduces_intensity_without_accusatory_language() -> None:
    state = default_cognitive_state()
    state = replace(
        state,
        affect=replace(state.affect, irritation=0.8, arousal=0.75),
    )

    guidance = generate_meta_control_guidance(state, diagnostics=_diagnostics())
    text = json.dumps(guidance.to_dict(), ensure_ascii=False).lower()

    assert guidance.reduce_intensity
    for forbidden in ("suspicious", "paranoia", "deception", "manipulation", "accuse"):
        assert forbidden not in text


def test_high_hidden_intent_avoids_overinterpretation_not_paranoia() -> None:
    diagnostics = _diagnostics()
    state = _state(
        diagnostics,
        {"hidden_intent": 0.9, "relationship_depth": 0.0, "emotional_tone": 0.5},
    )

    guidance = generate_meta_control_guidance(
        state,
        diagnostics=diagnostics,
        path_summary=_path_summary(diagnostics),
    )
    text = json.dumps(guidance.to_dict(), ensure_ascii=False).lower()

    assert guidance.avoid_overinterpreting_hidden_intent
    assert "suspicious" not in text
    assert "paranoia" not in text


def test_prior_failure_increases_repair_or_caution_guidance() -> None:
    diagnostics = _diagnostics()
    guidance = generate_meta_control_guidance(
        _state(diagnostics, {"emotional_tone": 0.45}, previous_outcome="failed"),
        diagnostics=diagnostics,
        path_summary=_path_summary(diagnostics),
        previous_outcome="failed",
    )

    assert guidance.prefer_repair_strategy or guidance.increase_caution
    assert "previous negative outcome" in guidance.trigger_reasons


def test_guidance_generation_does_not_change_selected_action_or_scores() -> None:
    diagnostics = _diagnostics(first_score=1.0, second_score=0.99)
    state = _state(diagnostics, {"conflict_tension": 0.9})
    before_chosen = diagnostics.chosen
    before_ranked = list(diagnostics.ranked_options)
    before_scores = [item.policy_score for item in diagnostics.ranked_options]

    generate_meta_control_guidance(
        state,
        diagnostics=diagnostics,
        path_summary=_path_summary(diagnostics),
        previous_outcome="failed",
        prompt_budget={"used_ratio": 0.96},
    )

    assert diagnostics.chosen is before_chosen
    assert diagnostics.ranked_options == before_ranked
    assert [item.policy_score for item in diagnostics.ranked_options] == before_scores


def test_conversation_trace_generation_diagnostics_and_conscious_include_guidance(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    conscious_writer = ConsciousMarkdownWriter(tmp_path / "conscious")

    turns = run_conversation(
        _agent(),
        ["I feel tense and I am not sure what you mean"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=650,
        session_id="m65-session",
        persona_id="m65-persona",
        trace_writer=JsonlTraceWriter(trace_path),
        conscious_writer=conscious_writer,
    )
    row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    conscious_path = (
        tmp_path
        / "conscious"
        / "personas"
        / "m65-persona"
        / "sessions"
        / "m65-session"
        / "Conscious.md"
    )
    markdown = conscious_path.read_text(encoding="utf-8")

    assert turns[0].meta_control_guidance is not None
    assert row["meta_control_guidance"]
    assert "affective_maintenance_summary" in row
    assert row["generation_diagnostics"]["meta_control_guidance"]
    assert row["fep_prompt_capsule"]["meta_control_guidance"]
    assert "Meta-control Guidance" in markdown
    assert "不是认知状态的来源" in markdown


def test_run_conversation_guidance_does_not_change_actions() -> None:
    lines = ["hello, can we check this?", "thanks, that makes sense."]
    observer = DialogueObserver()

    baseline = run_conversation(
        _agent(),
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=651,
        partner_uid=2,
        session_id="m65-side-effect",
    )
    with_guidance = run_conversation(
        _agent(),
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=651,
        partner_uid=2,
        session_id="m65-side-effect",
        session_context_extra={"prompt_budget": {"used_ratio": 0.95}},
    )

    assert [turn.action for turn in with_guidance] == [turn.action for turn in baseline]
    assert [turn.text for turn in with_guidance] == [turn.text for turn in baseline]
