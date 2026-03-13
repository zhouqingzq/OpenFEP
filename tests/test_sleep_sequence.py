from __future__ import annotations

from segmentum.predicate_eval import evaluate
from segmentum.sleep_consolidator import SleepConsolidator
from segmentum.types import SequenceCondition, SequenceStep, SleepRule


def _episode(tick: int, *, action: str = "forage", outcome: str = "resource_loss") -> dict[str, object]:
    return {
        "cycle": tick,
        "timestamp": tick,
        "cluster_id": 0,
        "action_taken": action,
        "predicted_outcome": outcome,
        "prediction_error": 0.45,
        "total_surprise": 1.25,
    }


def test_sequence_detection_with_repeated_episodes() -> None:
    consolidator = SleepConsolidator(surprise_threshold=0.4, minimum_support=3)
    episodes = [_episode(tick) for tick in range(1, 6)]
    result = consolidator.consolidate(
        sleep_cycle_id=1,
        current_cycle=5,
        episodes=episodes,
        transition_statistics={},
        outcome_distributions={},
    )
    seq_rules = [rule for rule in result.rules if rule.rule_type == "sequence_pattern"]
    assert seq_rules
    assert seq_rules[0].sequence_condition is not None
    assert seq_rules[0].sequence_condition.min_occurrences >= 3


def test_sequence_rule_stronger_penalty() -> None:
    consolidator = SleepConsolidator(surprise_threshold=0.4, minimum_support=3)
    single_rule = SleepRule(
        rule_id="single",
        type="risk_pattern",
        cluster=0,
        action="forage",
        observed_outcome="resource_loss",
        confidence=0.8,
        support=3,
        average_surprise=1.0,
        average_prediction_error=0.4,
        timestamp=5,
    )
    seq_rule = SleepRule(
        rule_id="seq",
        type="sequence_pattern",
        cluster=0,
        action="forage",
        observed_outcome="resource_loss",
        confidence=0.8,
        support=5,
        average_surprise=1.0,
        average_prediction_error=0.4,
        timestamp=5,
        sequence_condition=SequenceCondition(
            steps=[SequenceStep(action_name="forage", outcome="resource_loss")],
            window_ticks=10,
            min_occurrences=5,
        ),
    )
    updates = consolidator._model_updates([single_rule, seq_rule])
    single_bias = next(
        update.delta
        for update in updates
        if update.rule_id == "single" and update.update_type == "preference_penalty"
    )
    seq_bias = next(
        update.delta
        for update in updates
        if update.rule_id == "seq" and update.update_type == "preference_penalty"
    )
    assert abs(seq_bias) > abs(single_bias)


def test_sequence_condition_serialization() -> None:
    rule = SleepRule(
        rule_id="seq",
        type="sequence_pattern",
        cluster=1,
        action="api_call",
        observed_outcome="failure",
        confidence=0.9,
        support=4,
        average_surprise=1.0,
        average_prediction_error=0.3,
        timestamp=10,
        sequence_condition=SequenceCondition(
            steps=[SequenceStep(action_name="api_call", outcome="failure")],
            window_ticks=10,
            min_occurrences=3,
        ),
    )
    restored = SleepRule.from_dict(rule.to_dict())
    assert restored.sequence_condition is not None
    assert restored.sequence_condition.min_occurrences == 3
    assert restored.rule_type == "sequence_pattern"


def test_no_sequence_with_insufficient_episodes() -> None:
    consolidator = SleepConsolidator(surprise_threshold=0.4, minimum_support=3)
    result = consolidator.consolidate(
        sleep_cycle_id=1,
        current_cycle=2,
        episodes=[_episode(1), _episode(2)],
        transition_statistics={},
        outcome_distributions={},
    )
    assert not [rule for rule in result.rules if rule.rule_type == "sequence_pattern"]


def test_predicate_evaluator() -> None:
    assert evaluate(0.2, "< 0.3") is True
    assert evaluate(0.5, "< 0.3") is False
    assert evaluate(429, "== 429") is True
    assert evaluate(0.0, ">= 0") is True
    assert evaluate(0.0, "!= 0") is False
