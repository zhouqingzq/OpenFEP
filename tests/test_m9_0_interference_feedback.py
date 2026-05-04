"""M9.0 R4: Memory Interference and Overdominance Feedback tests."""

import pytest

from segmentum.memory_dynamics import (
    InterferenceFeedback,
    MemoryInterferenceSignal,
    apply_interference_to_evidence_contract,
    derive_interference_feedback,
    detect_memory_interference,
    memory_overdominance_detected_from_retrieval,
)


# ── R4.1: Interference detection (existing M6, verified working) ─────────

def test_memory_interference_detected_with_conflicting_outcomes():
    """Retrieved memories with both positive and negative outcomes trigger interference."""
    memories = [
        {"episode_id": "ep1", "predicted_outcome": "dialogue_reward"},
        {"episode_id": "ep2", "predicted_outcome": "dialogue_threat"},
    ]
    signal = detect_memory_interference(retrieved_memories=memories)
    assert signal.detected
    assert signal.kind == "memory_interference"
    assert signal.severity >= 0.5


def test_no_interference_detected_without_conflict():
    """No conflict in outcomes → no interference."""
    memories = [
        {"episode_id": "ep1", "predicted_outcome": "dialogue_reward"},
        {"episode_id": "ep2", "predicted_outcome": "dialogue_reward"},
    ]
    signal = detect_memory_interference(retrieved_memories=memories)
    assert not signal.detected


def test_overdominance_winner_take_most():
    """Winner-take-most weights above threshold trigger overdominance."""
    retrieval = {
        "recall_hypothesis": {
            "winner_take_most_weights": {"ep1": 0.85, "ep2": 0.1, "ep3": 0.05},
        },
    }
    assert memory_overdominance_detected_from_retrieval(retrieval)


def test_overdominance_single_candidate():
    """Single candidate in retrieval results triggers overdominance."""
    retrieval = {
        "candidates": [{"memory_id": "only_one"}],
    }
    assert memory_overdominance_detected_from_retrieval(retrieval)


def test_no_overdominance_with_normal_distribution():
    """Well-distributed retrieval weights do not trigger overdominance."""
    retrieval = {
        "recall_hypothesis": {
            "winner_take_most_weights": {"ep1": 0.4, "ep2": 0.35, "ep3": 0.25},
        },
        "candidates": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    }
    assert not memory_overdominance_detected_from_retrieval(retrieval)


# ── R4.2: Interference feedback derivation ──────────────────────────────

def test_memory_conflict_reduces_assertiveness_and_memory_reliance():
    """Memory interference should reduce assertiveness and memory reliance in generation."""
    interference = MemoryInterferenceSignal(
        detected=True,
        kind="memory_interference",
        severity=0.6,
        conflicting_episode_ids=("ep1", "ep2"),
        reasons=("retrieved_outcome_conflict",),
    )
    feedback = derive_interference_feedback(interference)
    assert feedback.reduce_assertiveness
    assert feedback.reduce_memory_reliance
    assert feedback.increase_caution
    assert feedback.severity == 0.6


def test_interference_feedback_severity_thresholds():
    """Lower severity may increase caution but not reduce assertiveness."""
    mild = MemoryInterferenceSignal(
        detected=True, kind="memory_interference", severity=0.3,
        reasons=("memory_prediction_conflict",),
    )
    feedback = derive_interference_feedback(mild)
    assert feedback.increase_caution  # triggered at 0.3
    assert not feedback.reduce_assertiveness  # requires 0.35+
    assert not feedback.reduce_memory_reliance  # requires 0.45+


def test_overdominance_alone_triggers_memory_reduction():
    """Overdominance even without explicit interference triggers memory reduction."""
    feedback = derive_interference_feedback(overdominance_detected=True)
    assert feedback.reduce_memory_reliance
    assert feedback.severity >= 0.5


def test_no_interference_has_no_effect():
    """No interference and no overdominance → no feedback effects."""
    feedback = derive_interference_feedback(
        interference=None, overdominance_detected=False,
    )
    assert not feedback.reduce_assertiveness
    assert not feedback.reduce_memory_reliance
    assert not feedback.increase_caution
    assert not feedback.increase_clarification_bias


def test_interference_feedback_to_dict():
    """to_dict() serializes all control flags."""
    interference = MemoryInterferenceSignal(
        detected=True, kind="memory_interference", severity=0.7,
    )
    feedback = derive_interference_feedback(interference)
    d = feedback.to_dict()
    assert d["reduce_assertiveness"] is True
    assert d["severity"] == 0.7
    assert "reason" in d


# ── R4.3: Application to control surfaces ───────────────────────────────

def test_apply_interference_to_evidence_contract_reduces_assertiveness():
    """Interference feedback translates into reduced control parameters."""
    feedback = InterferenceFeedback(
        reduce_assertiveness=True,
        reduce_memory_reliance=True,
        increase_caution=True,
        severity=0.6,
        reason="test interference",
    )
    params = apply_interference_to_evidence_contract(
        feedback,
        current_caution_level=0.5,
        current_assertiveness=0.5,
        memory_retrieval_gain=0.5,
    )
    assert params["caution_level"] > 0.5
    assert params["assertiveness"] < 0.5
    assert params["memory_retrieval_gain"] < 0.5
    assert params["interference_severity"] == 0.6


def test_apply_interference_no_effect_when_clean():
    """No-interference feedback leaves parameters unchanged."""
    feedback = InterferenceFeedback(
        reduce_assertiveness=False,
        reduce_memory_reliance=False,
        increase_caution=False,
        severity=0.0,
    )
    params = apply_interference_to_evidence_contract(
        feedback,
        current_caution_level=0.5,
        current_assertiveness=0.5,
        memory_retrieval_gain=0.5,
    )
    assert params["caution_level"] == 0.5
    assert params["assertiveness"] == 0.5
    assert params["memory_retrieval_gain"] == 0.5


def test_apply_interference_respects_bounds():
    """Control parameters stay within [0.1, 1.0] bounds."""
    feedback = InterferenceFeedback(
        reduce_assertiveness=True,
        reduce_memory_reliance=True,
        increase_caution=True,
        severity=0.99,
    )
    params = apply_interference_to_evidence_contract(
        feedback,
        current_caution_level=0.95,
        current_assertiveness=0.15,
        memory_retrieval_gain=0.15,
    )
    assert 0.1 <= params["caution_level"] <= 1.0
    assert 0.1 <= params["assertiveness"] <= 1.0
    assert 0.1 <= params["memory_retrieval_gain"] <= 1.0


def test_interference_does_not_silently_steer():
    """Memory overdominance feedback includes a traceable reason.

    The reason field ensures overdominance does not silently steer the reply.
    """
    interference = MemoryInterferenceSignal(
        detected=True, kind="memory_interference", severity=0.65,
    )
    feedback = derive_interference_feedback(
        interference, overdominance_detected=True,
    )
    assert "overdominance" in feedback.reason
    assert feedback.reason, "Feedback must have a traceable reason"
