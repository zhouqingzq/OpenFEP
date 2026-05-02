from __future__ import annotations

import pytest

from segmentum.cognitive_events import make_cognitive_event
from segmentum.cognitive_paths import (
    cognitive_path_candidates_from_diagnostics,
    select_cognitive_path_candidate,
)
from segmentum.cognitive_state import update_cognitive_state
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(
    action: str,
    *,
    policy_score: float,
    expected_free_energy: float,
    cost: float = 0.0,
    workspace_bias: float = 0.0,
    memory_bias: float = 0.0,
    pattern_bias: float = 0.0,
    action_ambiguity: float = 0.0,
    risk: float = 0.0,
    social_bias: float = 0.0,
    commitment_bias: float = 0.0,
    goal_alignment: float = 0.0,
    value_score: float = 0.0,
) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "stage3"}},
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
        workspace_bias=workspace_bias,
        social_bias=social_bias,
        commitment_bias=commitment_bias,
        identity_bias=0.0,
        ledger_bias=0.0,
        subject_bias=0.0,
        goal_alignment=goal_alignment,
        value_score=value_score,
        predicted_outcome=f"{action}_outcome",
        predicted_effects={},
        dominant_component="expected_free_energy",
        cost=cost,
    )


def _diagnostics(options: list[InterventionScore]) -> DecisionDiagnostics:
    return DecisionDiagnostics(
        chosen=options[0],
        ranked_options=options,
        prediction_error=0.4,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in options},
        explanation="stage3 fixture",
    )


def _event() -> object:
    return make_cognitive_event(
        event_type="PathSelectionEvent",
        turn_id="turn_0001",
        cycle=1,
        session_id="session-a",
        persona_id="persona-a",
        source="test",
        sequence_index=1,
        payload={"selected_action": "ask_question"},
        salience=0.9,
        priority=0.8,
        ttl=1,
        timestamp="2026-05-01T00:00:00Z",
    )


def test_candidate_paths_include_cost_components() -> None:
    diagnostics = _diagnostics(
        [
            _option(
                "ask_question",
                policy_score=1.0,
                expected_free_energy=0.2,
                cost=0.1,
                workspace_bias=-0.2,
                memory_bias=0.05,
                pattern_bias=-0.03,
                action_ambiguity=0.4,
                risk=0.5,
                social_bias=0.1,
                commitment_bias=0.1,
                goal_alignment=0.2,
                value_score=0.1,
            )
        ]
    )

    candidate = cognitive_path_candidates_from_diagnostics(diagnostics)[0]

    assert candidate.cost_components == {
        "current_free_energy": pytest.approx(0.4),
        "expected_free_energy": pytest.approx(0.2),
        "energy_cost": pytest.approx(0.1),
        "attention_cost": pytest.approx(0.2),
        "memory_cost": pytest.approx(0.08),
        "control_cost": pytest.approx(0.4),
        "social_risk": pytest.approx(0.3),
        "long_term_value": pytest.approx(0.3),
    }
    assert candidate.total_cost == pytest.approx(0.455)


def test_path_scoring_uses_meta_control_lambdas() -> None:
    diagnostics = _diagnostics(
        [
            _option(
                "cheap",
                policy_score=1.0,
                expected_free_energy=0.2,
                cost=0.1,
                workspace_bias=-0.1,
                memory_bias=0.2,
                action_ambiguity=0.2,
                risk=0.1,
            )
        ]
    )

    baseline = cognitive_path_candidates_from_diagnostics(diagnostics)[0]
    weighted = cognitive_path_candidates_from_diagnostics(
        diagnostics,
        meta_control={
            "lambda_energy": 1.0,
            "lambda_attention": 1.0,
            "lambda_memory": 1.0,
            "lambda_control": 1.0,
            "beta_efe": 1.0,
            "exploration_temperature": 0.35,
        },
    )[0]

    assert weighted.total_cost > baseline.total_cost
    assert weighted.scoring_lambdas["lambda_memory"] == 1.0
    assert weighted.scoring_lambdas["beta_efe"] == 1.0


def test_path_selection_uses_total_cost_temperature_and_margin() -> None:
    diagnostics = _diagnostics(
        [
            _option("policy_first", policy_score=2.0, expected_free_energy=0.8, risk=0.7),
            _option("cost_best", policy_score=1.0, expected_free_energy=0.1, risk=0.1),
        ]
    )

    candidates = cognitive_path_candidates_from_diagnostics(
        diagnostics,
        meta_control={"exploration_temperature": 0.2},
    )
    selection = select_cognitive_path_candidate(candidates)

    assert selection.selected_path is not None
    assert selection.runner_up_path is not None
    assert selection.selected_path.proposed_action == "cost_best"
    assert selection.selection_margin > 0.0
    assert selection.uncertainty < 1.0
    assert sum(candidate.posterior_weight for candidate in candidates) == pytest.approx(1.0)


def test_low_margin_selection_sets_uncertainty() -> None:
    diagnostics = _diagnostics(
        [
            _option("ask_question", policy_score=1.0, expected_free_energy=0.2, risk=0.1),
            _option("reflect", policy_score=0.99, expected_free_energy=0.2, risk=0.1),
        ]
    )

    candidates = cognitive_path_candidates_from_diagnostics(
        diagnostics,
        meta_control={"exploration_temperature": 1.0},
    )
    selection = select_cognitive_path_candidate(candidates, low_margin_threshold=0.2)

    assert selection.selection_margin < 0.2
    assert selection.uncertainty > 0.8
    assert selection.low_confidence_reason == "low_selection_margin"


def test_cognitive_state_records_alternative_selection_without_changing_chosen_action() -> None:
    diagnostics = _diagnostics(
        [
            _option("policy_first", policy_score=2.0, expected_free_energy=0.8, risk=0.7),
            _option("cost_best", policy_score=1.0, expected_free_energy=0.1, risk=0.1),
        ]
    )
    chosen_before = diagnostics.chosen.choice

    state = update_cognitive_state(
        None,
        events=[_event()],
        diagnostics=diagnostics,
        observation={"missing_context": 0.1},
    )

    assert diagnostics.chosen.choice == chosen_before
    assert state.candidate_paths.selected_action == "policy_first"
    assert state.candidate_paths.alternative_selection == "cost_best"
    assert state.candidate_paths.selection_margin > 0.0
    assert state.candidate_paths.top_candidates[0]["cost_components"]
