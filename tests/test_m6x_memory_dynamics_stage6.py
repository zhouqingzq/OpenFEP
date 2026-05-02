from __future__ import annotations

from types import SimpleNamespace

from segmentum.agent import SegmentAgent
from segmentum.cognitive_state import update_cognitive_state
from segmentum.dialogue.prediction_bridge import register_dialogue_actions
from segmentum.memory_dynamics import (
    consolidate_successful_path_pattern,
    detect_memory_interference,
)
from segmentum.meta_control import (
    adjust_memory_retrieval,
    derive_meta_control_signal,
    memory_overdominance_detected,
)
from segmentum.meta_control_guidance import generate_meta_control_guidance
from segmentum.types import DecisionDiagnostics, InterventionScore


def _option(
    action: str,
    *,
    policy_score: float = 0.9,
    expected_free_energy: float = 0.2,
    predicted_outcome: str = "dialogue_reward",
) -> InterventionScore:
    return InterventionScore(
        choice=action,
        action_descriptor={"name": action},
        policy_score=policy_score,
        expected_free_energy=expected_free_energy,
        predicted_error=0.1,
        action_ambiguity=0.1,
        risk=0.2,
        preferred_probability=0.6,
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
        predicted_outcome=predicted_outcome,
        predicted_effects={},
        dominant_component="expected_free_energy",
        cost=0.05,
    )


def _diagnostics() -> DecisionDiagnostics:
    ranked = [
        _option("empathize", policy_score=0.91, expected_free_energy=0.2),
        _option("ask_question", policy_score=0.72, expected_free_energy=0.42),
    ]
    return DecisionDiagnostics(
        chosen=ranked[0],
        ranked_options=ranked,
        prediction_error=0.25,
        retrieved_memories=[],
        policy_scores={item.choice: item.policy_score for item in ranked},
        explanation="stage6 diagnostics",
        memory_hit=False,
    )


def test_successful_path_encoded_as_reusable_pattern() -> None:
    patterns, updated = consolidate_successful_path_pattern(
        [],
        diagnostics=_diagnostics(),
        outcome_label="social_reward",
        cycle=4,
    )

    assert updated is not None
    assert updated["action"] == "empathize"
    assert updated["expected_outcome"] == "social_reward"
    assert updated["support_count"] == 1
    assert patterns == [updated]


def test_outcome_driven_consolidation_updates_reusable_pattern() -> None:
    diagnostics = _diagnostics()
    patterns, first = consolidate_successful_path_pattern(
        [],
        diagnostics=diagnostics,
        outcome_label="dialogue_reward",
        cycle=1,
    )
    patterns, second = consolidate_successful_path_pattern(
        patterns,
        diagnostics=diagnostics,
        outcome_label="dialogue_reward",
        cycle=5,
    )

    assert first is not None
    assert second is not None
    assert len(patterns) == 1
    assert patterns[0]["support_count"] == 2
    assert patterns[0]["success_count"] == 2
    assert patterns[0]["last_seen_cycle"] == 5
    assert patterns[0]["confidence"] == 1.0


def test_successful_free_energy_outcome_does_not_encode_neutral_pattern() -> None:
    diagnostics = _diagnostics()
    diagnostics.chosen.predicted_outcome = "neutral"

    patterns, updated = consolidate_successful_path_pattern(
        [],
        diagnostics=diagnostics,
        outcome_label="neutral",
        cycle=3,
        outcome={"free_energy_drop": 0.12},
    )

    assert updated is not None
    assert patterns[0]["expected_outcome"] == "free_energy_reduction"
    assert patterns[0]["pattern_id"] == "path:empathize:free_energy_reduction"


def test_agent_integrate_outcome_consolidates_successful_path() -> None:
    agent = SegmentAgent()
    register_dialogue_actions(agent.action_registry)
    agent.last_decision_diagnostics = _diagnostics()

    agent.integrate_outcome(
        "empathize",
        {"social": 0.7, "danger": 0.1},
        {"social": 0.4, "danger": 0.1},
        {"social": 0.3},
        free_energy_before=1.0,
        free_energy_after=0.4,
    )

    assert agent.long_term_memory.reusable_cognitive_paths
    assert agent.latest_memory_consolidation["updated"] is True
    assert agent.latest_memory_consolidation["pattern"]["action"] == "empathize"


def test_memory_interference_detected() -> None:
    signal = detect_memory_interference(
        retrieved_memories=[
            {
                "episode_id": "ep-good",
                "action": "empathize",
                "predicted_outcome": "dialogue_reward",
            },
            {
                "episode_id": "ep-bad",
                "action": "empathize",
                "predicted_outcome": "dialogue_threat",
            },
        ],
        prediction_delta={"relationship_depth": 0.56},
    )

    assert signal.detected is True
    assert signal.kind == "memory_interference"
    assert signal.severity >= 0.55
    assert "same_action_conflicting_memory" in signal.reasons
    assert signal.conflicting_episode_ids == ("ep-good", "ep-bad")


def test_memory_overdominance_detected_from_interference_signal() -> None:
    diagnostics = SimpleNamespace(
        chosen=SimpleNamespace(dominant_component="expected_free_energy"),
        ranked_options=[],
        memory_interference={"detected": True, "severity": 0.62},
    )

    assert memory_overdominance_detected(diagnostics) is True


def test_memory_conflict_reduces_memory_gain() -> None:
    diagnostics = _diagnostics()
    diagnostics.memory_hit = True
    diagnostics.retrieved_memories = [
        {
            "episode_id": "ep-good",
            "action": "empathize",
            "predicted_outcome": "dialogue_reward",
        },
        {
            "episode_id": "ep-bad",
            "action": "empathize",
            "predicted_outcome": "dialogue_threat",
        },
    ]
    diagnostics.prediction_delta = {"relationship_depth": 0.56}
    state = update_cognitive_state(
        None,
        events=(),
        diagnostics=diagnostics,
        observation={"relationship_depth": 0.4, "conflict_tension": 0.4},
    )
    guidance = generate_meta_control_guidance(state, diagnostics=diagnostics)
    signal = derive_meta_control_signal(
        state=state,
        guidance=guidance,
        diagnostics=diagnostics,
    )
    adjustment = adjust_memory_retrieval(
        k=4,
        memory_retrieval_gain=state.meta_control.memory_retrieval_gain,
        signal=signal,
    )

    assert state.memory.memory_conflicts
    assert not hasattr(diagnostics, "memory_interference")
    assert guidance.reduce_memory_reliance is True
    assert adjustment.adjusted["memory_retrieval_gain"] < adjustment.original["memory_retrieval_gain"]
    assert adjustment.adjusted["k"] < adjustment.original["k"]
