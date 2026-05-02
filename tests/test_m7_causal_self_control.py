from __future__ import annotations

import json

import pytest

from segmentum.cognitive_control import (
    CognitiveControlAdapter,
    CognitiveControlSignal,
    MetaControlPolicy,
)
from segmentum.cognitive_paths import (
    cognitive_path_candidates_from_diagnostics,
    cognitive_paths_from_diagnostics,
    path_competition_summary,
    select_cognitive_path_candidate,
)
from segmentum.cognitive_state import update_cognitive_state
from segmentum.dialogue.fep_prompt import build_fep_prompt_capsule
from segmentum.dialogue.m67_scenarios import evaluate_m67_scenario, m67_scenarios
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(
    action: str,
    *,
    policy_score: float = 1.0,
    expected_free_energy: float = 0.2,
    memory_bias: float = 0.0,
    pattern_bias: float = 0.0,
    dominant_component: str = "expected_free_energy",
    action_ambiguity: float = 0.1,
    risk: float = 0.1,
) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "m7"}},
        policy_score=policy_score,
        expected_free_energy=expected_free_energy,
        predicted_error=0.2,
        action_ambiguity=action_ambiguity,
        risk=risk,
        preferred_probability=0.5,
        memory_bias=memory_bias,
        pattern_bias=pattern_bias,
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
        dominant_component=dominant_component,
        cost=0.0,
    )


def _diagnostics(options: list[InterventionScore], *, prediction_error: float = 0.35) -> DecisionDiagnostics:
    return DecisionDiagnostics(
        chosen=options[0],
        ranked_options=options,
        prediction_error=prediction_error,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in options},
        explanation="m7 fixture",
    )


def _summary(diagnostics: DecisionDiagnostics) -> dict[str, object]:
    return path_competition_summary(cognitive_paths_from_diagnostics(diagnostics))


def test_cognitive_control_signal_created_from_high_severity_gap() -> None:
    diagnostics = _diagnostics(
        [
            _option("explain", policy_score=1.0, expected_free_energy=0.25),
            _option("ask_question", policy_score=0.96, expected_free_energy=0.27),
        ]
    )
    state = update_cognitive_state(
        None,
        events=[],
        diagnostics=diagnostics,
        observation={"missing_context": 0.82, "hidden_intent": 0.8},
    )

    signal = MetaControlPolicy().derive(state, diagnostics, _summary(diagnostics))

    assert signal.clarification_bias > 0.0
    assert signal.assertion_strength < 1.0
    assert "blocking_or_high_severity_gap" in signal.reason


def test_cognitive_control_signal_reduces_assertion_on_low_margin() -> None:
    diagnostics = _diagnostics(
        [
            _option("explain", policy_score=1.0, expected_free_energy=0.30),
            _option("ask_question", policy_score=0.99, expected_free_energy=0.31),
        ]
    )
    state = update_cognitive_state(
        None,
        events=[],
        diagnostics=diagnostics,
        observation={"emotional_tone": 0.5},
    )

    signal = MetaControlPolicy().derive(state, diagnostics, _summary(diagnostics))

    assert signal.assertion_strength < 0.75
    assert signal.clarification_bias > 0.0
    assert "low_selection_margin" in signal.reason


def test_memory_overdominance_reduces_memory_retrieval_gain() -> None:
    diagnostics = _diagnostics(
        [
            _option(
                "memory_dominant",
                memory_bias=0.9,
                dominant_component="memory_bias",
            )
        ]
    )
    state = update_cognitive_state(
        None,
        events=[],
        diagnostics=diagnostics,
        observation={"emotional_tone": 0.5},
    )

    signal = MetaControlPolicy().derive(state, diagnostics, _summary(diagnostics))

    assert signal.memory_retrieval_gain < 1.0
    assert CognitiveControlAdapter.to_meta_control_signal(
        signal
    ).memory_retrieval_gain_multiplier < 1.0


def test_resource_overload_adjusts_effective_temperature_or_candidate_budget() -> None:
    diagnostics = _diagnostics([_option("explain"), _option("ask_question")])
    state = update_cognitive_state(
        None,
        events=[],
        diagnostics=diagnostics,
        observation={"fatigue": 0.95, "stress": 0.9},
    )

    signal = MetaControlPolicy().derive(state, diagnostics, _summary(diagnostics))

    assert signal.effective_temperature_delta > 0.0 or signal.candidate_budget_delta < 0
    assert "resource_overload" in signal.reason


def test_path_scoring_changes_when_cognitive_control_enabled() -> None:
    diagnostics = _diagnostics(
        [
            _option("explain", expected_free_energy=0.22),
            _option("ask_question", expected_free_energy=0.24),
        ]
    )
    baseline = cognitive_path_candidates_from_diagnostics(diagnostics)
    controlled = cognitive_path_candidates_from_diagnostics(
        diagnostics,
        cognitive_control=CognitiveControlSignal(
            clarification_bias=0.25,
            effective_temperature_delta=0.1,
            reason="test",
        ),
    )

    assert [item.total_cost for item in controlled] != [
        item.total_cost for item in baseline
    ]
    assert [item.posterior_weight for item in controlled] != [
        item.posterior_weight for item in baseline
    ]


def test_clarification_bias_reduces_clarify_path_cost() -> None:
    diagnostics = _diagnostics(
        [
            _option("ask_question", expected_free_energy=0.25),
            _option("explain", expected_free_energy=0.25),
        ]
    )

    baseline = cognitive_path_candidates_from_diagnostics(diagnostics)[0]
    controlled = cognitive_path_candidates_from_diagnostics(
        diagnostics,
        cognitive_control=CognitiveControlSignal(
            clarification_bias=0.3,
            reason="test",
        ),
    )[0]

    assert controlled.proposed_action == "ask_question"
    assert controlled.total_cost < baseline.total_cost
    assert controlled.cognitive_control_adjustments["clarification_bias"] == pytest.approx(0.3)


def test_low_confidence_gap_can_shift_selected_path_to_clarify_or_marks_action_shift_candidate() -> None:
    diagnostics = _diagnostics(
        [
            _option("explain", policy_score=1.0, expected_free_energy=0.20),
            _option("ask_question", policy_score=0.99, expected_free_energy=0.22),
        ]
    )
    baseline = select_cognitive_path_candidate(
        cognitive_path_candidates_from_diagnostics(diagnostics)
    )
    controlled = select_cognitive_path_candidate(
        cognitive_path_candidates_from_diagnostics(
            diagnostics,
            cognitive_control=CognitiveControlSignal(
                clarification_bias=0.3,
                assertion_strength=0.6,
                reason="blocking_or_high_severity_gap;low_selection_margin",
            ),
        )
    )

    assert baseline.selected_path is not None
    assert controlled.selected_path is not None
    assert baseline.selected_path.proposed_action == "explain"
    assert controlled.selected_path.proposed_action == "ask_question"


def test_self_agenda_persists_unresolved_gap_across_turns() -> None:
    diagnostics = _diagnostics(
        [
            _option("explain", policy_score=1.0, expected_free_energy=0.25),
            _option("ask_question", policy_score=0.96, expected_free_energy=0.27),
        ]
    )
    first = update_cognitive_state(
        None,
        events=[],
        diagnostics=diagnostics,
        observation={"missing_context": 0.82},
    )
    second = update_cognitive_state(
        first,
        events=[],
        diagnostics=diagnostics,
        observation={"missing_context": 0.0, "emotional_tone": 0.5},
    )

    assert first.self_agenda.unresolved_gaps
    assert first.self_agenda.unresolved_gaps[0] in second.self_agenda.unresolved_gaps


def test_prompt_receives_compressed_control_guidance_not_raw_events() -> None:
    diagnostics = _diagnostics([_option("explain"), _option("ask_question")])
    capsule = build_fep_prompt_capsule(
        diagnostics,
        {"missing_context": 0.8},
        cognitive_control_guidance={
            **CognitiveControlAdapter.compact_prompt_guidance(
                CognitiveControlSignal(
                    clarification_bias=0.3,
                    assertion_strength=0.6,
                    reason="blocking_or_high_severity_gap",
                )
            ),
            "raw_events": [{"payload": "SHOULD NOT LEAK"}],
        },
    ).to_dict()
    text = json.dumps(capsule, ensure_ascii=False)

    assert capsule["cognitive_control_guidance"]["clarification_bias"] == 0.3
    assert "SHOULD NOT LEAK" not in text
    assert "raw_events" not in text


def test_existing_m6_behavior_still_passes(tmp_path) -> None:
    scenario = next(item for item in m67_scenarios() if item.scenario_id == "low_margin_ambiguity")
    result = evaluate_m67_scenario(scenario, trace_dir=tmp_path / "traces")

    assert result["passed"] is True
    assert result["enabled"]["selected_action"] == result["ablated"]["selected_action"]
