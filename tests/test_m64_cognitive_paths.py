import json
from pathlib import Path

import pytest

from segmentum.cognitive_paths import (
    PROXY_COST_FIELDS,
    cognitive_paths_from_diagnostics,
    path_competition_summary,
)
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.fep_prompt import build_fep_prompt_capsule
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.tracing import JsonlTraceWriter
from segmentum.agent import SegmentAgent
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
        action_descriptor={"name": action, "params": {"strategy": "test"}},
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


def _diagnostics(
    *,
    first_score: float = 1.5,
    second_score: float = 1.1,
    third_score: float = 0.2,
) -> DecisionDiagnostics:
    ranked = [
        _option(
            "ask_question",
            policy_score=first_score,
            expected_free_energy=0.20,
            cost=0.10,
            workspace_bias=-0.05,
            memory_bias=0.03,
            pattern_bias=-0.02,
            action_ambiguity=0.04,
            risk=0.50,
            social_bias=0.10,
            commitment_bias=0.05,
            goal_alignment=0.12,
            value_score=0.08,
        ),
        _option(
            "reflect",
            policy_score=second_score,
            expected_free_energy=0.35,
            cost=0.05,
            workspace_bias=0.02,
            memory_bias=-0.04,
            pattern_bias=0.01,
            action_ambiguity=0.08,
            risk=0.40,
            social_bias=0.00,
            commitment_bias=0.02,
            goal_alignment=0.05,
            value_score=0.03,
        ),
        _option(
            "disagree",
            policy_score=third_score,
            expected_free_energy=0.80,
            cost=0.12,
            workspace_bias=-0.01,
            memory_bias=0.00,
            pattern_bias=0.00,
            action_ambiguity=0.20,
            risk=0.90,
            social_bias=0.10,
            commitment_bias=0.00,
            goal_alignment=-0.10,
            value_score=0.00,
        ),
    ]
    return DecisionDiagnostics(
        chosen=ranked[0],
        ranked_options=ranked,
        prediction_error=0.27,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in ranked},
        explanation="m64 fixture",
    )


def _agent() -> SegmentAgent:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    return agent


def test_path_adapter_preserves_ranked_order_and_chosen_path() -> None:
    diagnostics = _diagnostics()

    paths = cognitive_paths_from_diagnostics(diagnostics)

    assert [path.proposed_action for path in paths] == [
        item.choice for item in diagnostics.ranked_options
    ]
    assert paths[0].proposed_action == diagnostics.chosen.choice
    assert paths[0].path_id == "path_0_ask_question"
    assert paths[0].source_policy_score == diagnostics.chosen.policy_score


def test_path_field_mapping_and_proxy_labels() -> None:
    diagnostics = _diagnostics()
    path = cognitive_paths_from_diagnostics(diagnostics)[0]

    assert path.current_free_energy == pytest.approx(diagnostics.prediction_error)
    assert path.expected_free_energy == pytest.approx(0.20)
    assert path.energy_cost == pytest.approx(0.10)
    assert path.attention_cost == pytest.approx(0.05)
    assert path.memory_cost == pytest.approx(0.05)
    assert path.control_cost == pytest.approx(0.04)
    assert path.social_risk == pytest.approx(0.35)
    assert path.long_term_value == pytest.approx(0.20)
    assert path.total_cost == pytest.approx(0.59)
    assert set(PROXY_COST_FIELDS) <= set(path.proxy_fields)


def test_posterior_weights_are_stable_and_sum_to_one() -> None:
    diagnostics = _diagnostics(
        first_score=1000.0,
        second_score=999.0,
        third_score=-1000.0,
    )

    paths = cognitive_paths_from_diagnostics(diagnostics)
    total = sum(path.posterior_weight for path in paths)

    assert total == pytest.approx(1.0)
    assert paths[0].posterior_weight > paths[1].posterior_weight > paths[2].posterior_weight
    assert all(0.0 <= path.posterior_weight <= 1.0 for path in paths)


def test_path_competition_summary_identifies_runner_up_and_margins() -> None:
    diagnostics = _diagnostics()
    paths = cognitive_paths_from_diagnostics(diagnostics)

    summary = path_competition_summary(paths)

    assert summary["chosen_path"]["action"] == "ask_question"
    assert summary["runner_up_path"]["action"] == "reflect"
    assert summary["policy_margin"] == pytest.approx(0.4)
    assert summary["efe_margin"] == pytest.approx(0.15)
    assert summary["posterior_margin"] > 0.0
    assert "derived approximations" in summary["proxy_notice"]
    assert set(PROXY_COST_FIELDS) <= set(summary["proxy_fields"])


def test_adapter_does_not_mutate_diagnostics_or_policy_scores() -> None:
    diagnostics = _diagnostics()
    before_objects = list(diagnostics.ranked_options)
    before_order = [item.choice for item in diagnostics.ranked_options]
    before_scores = [item.policy_score for item in diagnostics.ranked_options]
    before_chosen = diagnostics.chosen

    cognitive_paths_from_diagnostics(diagnostics)

    assert diagnostics.chosen is before_chosen
    assert diagnostics.ranked_options == before_objects
    assert [item.choice for item in diagnostics.ranked_options] == before_order
    assert [item.policy_score for item in diagnostics.ranked_options] == before_scores


def test_fep_prompt_capsule_contains_path_view_without_changing_choice() -> None:
    diagnostics = _diagnostics()

    capsule = build_fep_prompt_capsule(
        diagnostics,
        {"hidden_intent": 0.25},
    ).to_dict()

    assert capsule["chosen_action"] == diagnostics.chosen.choice
    assert capsule["cognitive_paths"][0]["proposed_action"] == diagnostics.chosen.choice
    assert capsule["path_competition"]["runner_up_path"]["action"] == "reflect"
    assert diagnostics.chosen.choice == "ask_question"


def test_turn_trace_contains_cognitive_paths_and_path_competition(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"

    turns = run_conversation(
        _agent(),
        ["I am unsure what you mean"],
        observer=DialogueObserver(),
        generator=RuleBasedGenerator(),
        master_seed=640,
        session_id="m64-trace",
        trace_writer=JsonlTraceWriter(trace_path),
    )
    row = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

    assert "cognitive_paths" in row
    assert "path_competition" in row
    assert row["cognitive_paths"]
    assert row["cognitive_paths"][0]["proposed_action"] == turns[0].action
    assert row["path_competition"]["chosen_path"]["action"] == turns[0].action
    assert row["fep_prompt_capsule"]["cognitive_paths"]
