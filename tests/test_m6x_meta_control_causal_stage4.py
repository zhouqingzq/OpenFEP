from __future__ import annotations

from types import SimpleNamespace

import pytest

from segmentum.agent import SegmentAgent
from segmentum.cognitive_paths import cognitive_path_candidates_from_diagnostics
from segmentum.meta_control import (
    MetaControlSignal,
    adjust_memory_retrieval,
    adjust_path_scoring_meta_control,
    derive_meta_control_signal,
)
from segmentum.memory_retrieval import RetrievalQuery
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(
    action: str,
    *,
    policy_score: float = 1.0,
    expected_free_energy: float = 0.2,
    memory_bias: float = 0.0,
    pattern_bias: float = 0.0,
    dominant_component: str = "expected_free_energy",
) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action, "params": {"strategy": "stage4"}},
        policy_score=policy_score,
        expected_free_energy=expected_free_energy,
        predicted_error=0.2,
        action_ambiguity=0.2,
        risk=0.1,
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


def _diagnostics(options: list[InterventionScore]) -> DecisionDiagnostics:
    return DecisionDiagnostics(
        chosen=options[0],
        ranked_options=options,
        prediction_error=0.3,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in options},
        explanation="stage4 fixture",
    )


def test_meta_control_changes_memory_retrieval_gain() -> None:
    signal = MetaControlSignal(
        signal_id="test",
        memory_retrieval_gain_multiplier=0.5,
        retrieval_k_delta=-1,
        reasons=("memory_conflict",),
    )

    adjustment = adjust_memory_retrieval(
        k=4,
        memory_retrieval_gain=0.8,
        signal=signal,
    )

    assert adjustment.adjusted["k"] == 3
    assert adjustment.adjusted["memory_retrieval_gain"] == pytest.approx(0.4)
    assert adjustment.rollback == {"k": 4, "memory_retrieval_gain": 0.8}


def test_agent_retrieval_uses_meta_control_adjusted_k() -> None:
    agent = SegmentAgent()
    captured: dict[str, int] = {}

    agent.sync_memory_awareness_to_long_term_memory = lambda: None  # type: ignore[method-assign]
    agent._decision_retrieval_query = lambda *args, **kwargs: RetrievalQuery()  # type: ignore[method-assign]

    def fake_retrieve_for_decision(*args: object, **kwargs: object) -> object:
        captured["k"] = int(kwargs["k"])
        return SimpleNamespace(
            candidates=[],
            to_dict=lambda: {
                "candidates": [],
                "recall_hypothesis": {},
                "reconstruction_trace": {},
            },
        )

    agent.retrieve_for_decision = fake_retrieve_for_decision  # type: ignore[method-assign]
    agent.active_meta_control_signal = MetaControlSignal(
        signal_id="test",
        memory_retrieval_gain_multiplier=0.5,
        retrieval_k_delta=-1,
    )

    agent._retrieve_decision_memories(
        observed={},
        baseline_prediction={},
        baseline_errors={},
        current_state_snapshot={},
        k=3,
    )

    assert captured["k"] == 2
    assert agent.last_retrieval_result["meta_control_adjustment"]["adjusted"]["k"] == 2
    assert (
        agent.last_retrieval_result["meta_control_adjustment"]["rollback"]["k"]
        == 3
    )


def test_meta_control_changes_path_scoring_lambdas() -> None:
    diagnostics = _diagnostics(
        [
            _option("first", expected_free_energy=0.2),
            _option("second", expected_free_energy=0.4),
        ]
    )
    base_meta = {
        "lambda_memory": 0.4,
        "lambda_control": 0.4,
        "beta_efe": 0.4,
        "exploration_temperature": 0.3,
    }
    signal = MetaControlSignal(
        signal_id="test",
        lambda_memory_multiplier=0.5,
        lambda_control_multiplier=1.25,
        beta_efe_multiplier=1.2,
        effective_temperature_delta=0.2,
    )

    adjustment = adjust_path_scoring_meta_control(base_meta, signal)
    candidates = cognitive_path_candidates_from_diagnostics(
        diagnostics,
        meta_control=adjustment.adjusted,
    )

    assert adjustment.adjusted["lambda_memory"] == pytest.approx(0.2)
    assert adjustment.adjusted["lambda_control"] == pytest.approx(0.5)
    assert adjustment.adjusted["exploration_temperature"] == pytest.approx(0.5)
    assert candidates[0].effective_temperature == pytest.approx(0.5)


def test_memory_bias_overdominance_triggers_control_signal() -> None:
    diagnostics = _diagnostics(
        [
            _option(
                "memory_dominant",
                memory_bias=0.9,
                dominant_component="memory_bias",
            )
        ]
    )

    signal = derive_meta_control_signal(diagnostics=diagnostics)

    assert signal.memory_retrieval_gain_multiplier < 1.0
    assert signal.retrieval_k_delta < 0
    assert "memory_overdominance_or_conflict" in signal.reasons


def test_resource_overload_increases_effective_temperature_or_compresses_candidates() -> None:
    signal = derive_meta_control_signal(
        state={"resource": {"overload": True, "cognitive_load": 0.95}},
    )
    adjustment = adjust_path_scoring_meta_control(
        {
            "lambda_energy": 0.25,
            "lambda_attention": 0.35,
            "lambda_memory": 0.25,
            "lambda_control": 0.35,
            "beta_efe": 0.5,
            "exploration_temperature": 0.35,
        },
        signal,
    )

    assert signal.effective_temperature_delta > 0.0
    assert signal.candidate_limit == 3
    assert adjustment.adjusted["exploration_temperature"] > adjustment.original[
        "exploration_temperature"
    ]
    assert adjustment.adjusted["candidate_limit"] == 3
