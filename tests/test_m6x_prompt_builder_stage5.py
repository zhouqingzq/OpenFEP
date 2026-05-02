from __future__ import annotations

from types import SimpleNamespace

from segmentum.agent import SegmentAgent
from segmentum.dialogue.cognitive_guidance import (
    build_compressed_cognitive_guidance,
    render_compressed_cognitive_guidance,
)
from segmentum.dialogue.fep_prompt import build_fep_prompt_capsule
from segmentum.dialogue.generator import _format_compact_capsule_guidance
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.dialogue.runtime.prompts import PromptBuilder


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


def _diagnostics(policy_margin: float = 0.02) -> SimpleNamespace:
    top_policy = 0.80
    options = [
        SimpleNamespace(
            choice="empathize",
            expected_free_energy=0.20,
            policy_score=top_policy,
            risk=0.8,
            predicted_outcome="dialogue_reward",
            dominant_component="expected_free_energy",
        ),
        SimpleNamespace(
            choice="ask_question",
            expected_free_energy=0.215,
            policy_score=top_policy - policy_margin,
            risk=1.0,
            predicted_outcome="dialogue_epistemic_gain",
            dominant_component="epistemic_bonus",
        ),
    ]
    return SimpleNamespace(
        chosen=options[0],
        ranked_options=options,
        prediction_error=0.18,
        workspace_broadcast_channels=["hidden_intent"],
        workspace_suppressed_channels=[],
    )


def _capsule() -> dict[str, object]:
    capsule = build_fep_prompt_capsule(
        _diagnostics(),
        {"hidden_intent": 0.78},
        previous_outcome="social_threat",
        self_prior_summary={
            "summary": "compact prior only",
            "raw_markdown": "Self-consciousness.md FULL PRIOR SHOULD NOT LEAK",
        },
        meta_control_guidance={
            "lower_assertiveness": True,
            "avoid_overinterpreting_hidden_intent": True,
            "raw_diagnostics": "FULL DIAGNOSTICS SHOULD NOT LEAK",
        },
        affective_guidance={"actions": ["deescalate"], "summary": "keep stance gentle"},
        omitted_signals=["raw_events"],
    ).to_dict()
    capsule["raw_events"] = [
        {
            "event_id": "raw-1",
            "payload": {"secret": "RAW EVENT PAYLOAD SHOULD NOT LEAK"},
        }
    ]
    capsule["event_dump"] = "RAW EVENT DUMP SHOULD NOT LEAK"
    capsule["active_gaps"] = {
        "blocking_gaps": ["needs_current_user_goal"],
        "soft_gaps": ["SHOULD NOT APPEAR: unsupported stage key"],
    }
    capsule["selected_path_summary"] = {
        "proposed_action": "empathize",
        "expected_outcome": "dialogue_reward",
        "total_cost": 0.31,
        "posterior_weight": 0.58,
    }
    capsule["path_competition_summary"] = {"selection_margin": 0.03}
    capsule["memory_use_guidance"] = {
        "reduce_memory_reliance": True,
        "memory_conflict_count": 2,
        "activated_memory_count": 3,
    }
    return capsule


def test_prompt_builder_uses_compressed_cognitive_guidance_not_raw_events() -> None:
    capsule = _capsule()
    prompt = PromptBuilder(persona_name="test").build_system_prompt(
        _agent(),
        "empathize",
        0.35,
        0.62,
        current_turn="do you really remember me?",
        fep_capsule=capsule,
    )

    assert "Current task: follow chosen action empathize" in prompt
    assert "Current goal: aim for dialogue_reward" in prompt
    assert "Selected path: empathize" in prompt
    assert "Missing gaps: blocking: needs_current_user_goal" in prompt
    assert "Compact self-prior for stance only: compact prior only" in prompt
    assert "Affective guidance is about response stance" in prompt
    assert "Memory use: treat recalled context as tentative" in prompt
    assert "raw_events" not in prompt
    assert "RAW EVENT PAYLOAD SHOULD NOT LEAK" not in prompt
    assert "FULL PRIOR SHOULD NOT LEAK" not in prompt


def test_low_margin_selection_reduces_assertiveness() -> None:
    capsule = _capsule()
    guidance = build_compressed_cognitive_guidance(capsule)
    text = render_compressed_cognitive_guidance(capsule)

    assert guidance["assertiveness"] == "lower"
    assert "Assertiveness: lower" in text
    assert "avoid overclaiming" in text


def test_prompt_does_not_include_raw_event_dump() -> None:
    text = _format_compact_capsule_guidance(_capsule())

    assert "raw_events" not in text
    assert "event_id" not in text
    assert "payload" not in text
    assert "RAW EVENT DUMP SHOULD NOT LEAK" not in text
    assert "RAW EVENT PAYLOAD SHOULD NOT LEAK" not in text


def test_prompt_does_not_claim_consciousness() -> None:
    text = render_compressed_cognitive_guidance(_capsule()).lower()

    assert "i am conscious" not in text
    assert "my consciousness" not in text
    assert "inner monologue" not in text
    assert "self-consciousness.md" not in text
    assert "conscious.md" not in text
