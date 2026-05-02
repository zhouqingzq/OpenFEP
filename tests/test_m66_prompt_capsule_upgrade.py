from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from segmentum.agent import SegmentAgent
from segmentum.cognitive_events import CognitiveEventBus
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.fep_prompt import build_fep_prompt_capsule
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.dialogue.runtime.prompts import PromptBuilder
from segmentum.tracing import JsonlTraceWriter


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


def _diagnostics() -> SimpleNamespace:
    options = [
        SimpleNamespace(
            choice="empathize",
            expected_free_energy=0.20,
            policy_score=0.90,
            risk=0.8,
            predicted_outcome="dialogue_reward",
            dominant_component="expected_free_energy",
        ),
        SimpleNamespace(
            choice="ask_question",
            expected_free_energy=0.215,
            policy_score=0.87,
            risk=1.2,
            predicted_outcome="dialogue_epistemic_gain",
            dominant_component="epistemic_bonus",
        ),
    ]
    return SimpleNamespace(
        chosen=options[0],
        ranked_options=options,
        prediction_error=0.42,
        workspace_broadcast_channels=["hidden_intent", "conflict_tension"],
        workspace_suppressed_channels=["topic_novelty"],
        retrieved_memories=[
            {
                "episode_id": "ep-1",
                "summary": "compact useful memory",
                "raw_content": "FULL MEMORY DUMP SHOULD NOT LEAK",
            }
        ],
        retrieved_episode_ids=["ep-1"],
        memory_context_summary="compact memory context",
    )


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def test_old_builder_call_still_serializes_old_fields() -> None:
    capsule = build_fep_prompt_capsule(
        _diagnostics(),
        {"hidden_intent": 0.76, "conflict_tension": 0.52},
        previous_outcome="SOCIAL_THREAT",
    ).to_dict()

    for key in (
        "chosen_action",
        "top_alternatives",
        "policy_margin",
        "efe_margin",
        "decision_uncertainty",
        "hidden_intent_label",
        "cognitive_paths",
        "path_competition",
    ):
        assert key in capsule
    assert capsule["chosen_action"] == "empathize"
    assert capsule["hidden_intent_label"] == "clear_subtext"
    assert capsule["self_prior_summary"] is None
    assert capsule["prompt_budget_summary"] is None


def test_new_fields_serialize_with_compression_and_redaction() -> None:
    self_prior = {
        "summary": "stable stance summary",
        "stable_patterns": ["ask before asserting"],
        "raw_markdown": "# Self-consciousness.md\nFULL MARKDOWN BODY SHOULD NOT LEAK",
        "content": "Conscious.md FULL CONTENT SHOULD NOT LEAK",
    }
    capsule = build_fep_prompt_capsule(
        _diagnostics(),
        {"hidden_intent": 0.82, "conflict_tension": 0.62},
        previous_outcome="social_threat",
        self_prior_summary=self_prior,
        meta_control_guidance={
            "lower_assertiveness": True,
            "guidance_notes": ["Use provisional wording."],
            "raw_diagnostics": "FULL DIAGNOSTICS SHOULD NOT LEAK",
        },
        affective_state={
            "mood_valence": 0.4,
            "repair_need": 0.7,
            "affective_notes": ["RAW AFFECTIVE NOTE SHOULD NOT LEAK"],
        },
        affective_guidance={"actions": ["deescalate"], "summary": "keep the reply warm"},
        prompt_budget={"used_ratio": 0.91, "omitted_signals": ["raw_events"]},
        omitted_signals=["raw_events", "full_diagnostics"],
        persona_id="persona-a",
        session_id="session-a",
    ).to_dict()
    text = _json_text(capsule)

    assert capsule["persona_id"] == "persona-a"
    assert capsule["session_id"] == "session-a"
    assert capsule["self_prior_summary"]["summary"] == "stable stance summary"
    assert capsule["selected_path_summary"]["proposed_action"] == "empathize"
    assert capsule["path_competition_summary"]["path_count"] == 2
    assert capsule["affective_state_summary"]["repair_need"] == 0.7
    assert capsule["affective_guidance"]["actions"] == ["deescalate"]
    assert capsule["prompt_budget_summary"]["used_ratio"] == 0.91
    assert capsule["omitted_signals"] == ["raw_events", "full_diagnostics"]
    assert "FULL MARKDOWN BODY SHOULD NOT LEAK" not in text
    assert "FULL CONTENT SHOULD NOT LEAK" not in text
    assert "FULL DIAGNOSTICS SHOULD NOT LEAK" not in text
    assert "RAW AFFECTIVE NOTE SHOULD NOT LEAK" not in text
    assert "FULL MEMORY DUMP SHOULD NOT LEAK" not in text


def test_guidance_changes_capsule_contents() -> None:
    first = build_fep_prompt_capsule(
        _diagnostics(),
        {"hidden_intent": 0.2},
        meta_control_guidance={"lower_assertiveness": True},
        affective_guidance={"actions": ["deescalate"]},
    ).to_dict()
    second = build_fep_prompt_capsule(
        _diagnostics(),
        {"hidden_intent": 0.2},
        meta_control_guidance={"ask_clarifying_question": True},
        affective_guidance={"actions": ["preserve_warmth"]},
    ).to_dict()

    assert first["meta_control_guidance"] != second["meta_control_guidance"]
    assert first["affective_guidance"] != second["affective_guidance"]


def test_prompt_builder_uses_compressed_guidance_without_raw_markdown() -> None:
    capsule = build_fep_prompt_capsule(
        _diagnostics(),
        {"hidden_intent": 0.9, "conflict_tension": 0.6},
        self_prior_summary={
            "summary": "compact prior only",
            "raw_markdown": "Self-consciousness.md FULL PRIOR SHOULD NOT LEAK",
        },
        meta_control_guidance={"avoid_overinterpreting_hidden_intent": True},
        affective_guidance={"actions": ["deescalate"], "summary": "keep stance gentle"},
        omitted_signals=["raw_events"],
    ).to_dict()
    prompt = PromptBuilder(persona_name="test").build_system_prompt(
        _agent(),
        "empathize",
        0.35,
        0.62,
        current_turn="do you really remember me?",
        fep_capsule=capsule,
    )

    assert "Compact self-prior for stance only" in prompt
    assert "compact prior only" in prompt
    assert "Affective guidance is about response stance" in prompt
    assert "low-confidence observable signals" in prompt
    assert "FULL PRIOR SHOULD NOT LEAK" not in prompt
    assert "Self-consciousness.md" not in prompt
    assert "raw_events" not in prompt


def test_runtime_prompt_assembly_event_trace_and_diagnostics_are_real_chain(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    bus = CognitiveEventBus()
    turns = run_conversation(
        _agent(),
        ["I am not sure you understood me"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=660,
        session_id="m66-session",
        persona_id="m66-persona",
        session_context_extra={
            "self_prior_summary": {
                "summary": "prefer careful repair",
                "raw_markdown": "Conscious.md FULL CONSCIOUS SHOULD NOT LEAK",
            },
            "prompt_budget": {"used_ratio": 0.9, "omitted_signals": ["raw_events"]},
        },
        cognitive_event_bus=bus,
        trace_writer=JsonlTraceWriter(trace_path),
    )
    row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    prompt_events = bus.filter(event_type="PromptAssemblyEvent")
    payload = prompt_events[0].payload
    capsule = turns[0].generation_diagnostics["fep_prompt_capsule"]
    text = _json_text({"row": row, "payload": payload, "capsule": capsule})

    assert prompt_events
    assert "included_signals" in payload
    assert "omitted_signals" in payload
    assert "prompt_budget_summary" in payload
    assert payload["redaction_status"]["raw_events_included"] is False
    assert "fep_prompt_capsule" in turns[0].generation_diagnostics
    assert capsule["self_prior_summary"]["summary"] == "prefer careful repair"
    assert capsule["prompt_budget_summary"]["used_ratio"] == 0.9
    assert capsule["omitted_signals"] == ["raw_events"]
    assert row["fep_prompt_capsule"]["self_prior_summary"]["summary"] == "prefer careful repair"
    assert row["generation_diagnostics"]["prompt_capsule_guidance"]
    assert "FULL CONSCIOUS SHOULD NOT LEAK" not in text
    assert "Conscious.md" not in text
    assert "raw_events" in payload["omitted_signals"]
