from __future__ import annotations

import random

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.self_model import RuntimeFailureEvent, build_default_self_model
from tests.test_counterfactual import (
    HARMFUL_BODY_STATE,
    HARMFUL_OUTCOME,
    OBS_DANGEROUS_DICT,
    PREDICTION,
    _errors,
)


def _seed_dangerous_memory(agent: SegmentAgent, *, count: int = 5) -> None:
    for cycle in range(1, count + 1):
        decision = agent.long_term_memory.maybe_store_episode(
            cycle=cycle,
            observation=OBS_DANGEROUS_DICT,
            prediction=PREDICTION,
            errors=_errors(),
            action="forage",
            outcome=HARMFUL_OUTCOME,
            body_state=HARMFUL_BODY_STATE,
        )
        if not decision.episode_created:
            agent.long_term_memory.store_episode(
                cycle=cycle,
                observation=OBS_DANGEROUS_DICT,
                prediction=PREDICTION,
                errors=_errors(),
                action="forage",
                outcome=HARMFUL_OUTCOME,
                body_state=HARMFUL_BODY_STATE,
            )


def test_memory_context_changes_prediction_surface() -> None:
    observation = Observation(**OBS_DANGEROUS_DICT)

    baseline = SegmentAgent(rng=random.Random(17))
    trained = SegmentAgent(rng=random.Random(17))
    _seed_dangerous_memory(trained)

    baseline_diag = baseline.decision_cycle(observation)["diagnostics"]
    trained_diag = trained.decision_cycle(observation)["diagnostics"]

    assert not baseline_diag.memory_hit
    assert trained_diag.memory_hit
    assert trained_diag.memory_context_summary
    assert trained_diag.prediction_before_memory
    assert trained_diag.prediction_after_memory
    assert trained_diag.prediction_delta
    assert trained_diag.prediction_before_memory != trained_diag.prediction_after_memory
    assert trained_diag.prediction_after_memory != baseline_diag.prediction_after_memory
    assert "chronic_threat=" in trained_diag.memory_context_summary
    assert trained_diag.prediction_after_memory["danger"] >= trained_diag.prediction_before_memory["danger"]


def test_runtime_failure_event_preserves_self_vs_world_evidence() -> None:
    model = build_default_self_model()

    self_failure = RuntimeFailureEvent(
        name="TokenLimitExceeded",
        stage="tool_runtime",
        category="context_budget",
        origin_hint="self",
        details={"message": "context budget exhausted"},
        resource_state={"tokens_remaining": 0, "cpu_budget": 0.8, "memory_free": 512.0},
    )
    world_failure = RuntimeFailureEvent(
        name="HTTPTimeout",
        stage="tool_runtime",
        category="timeout",
        origin_hint="world",
        details={"message": "external service timed out"},
        resource_state={"tokens_remaining": 200, "cpu_budget": 0.8, "memory_free": 512.0},
    )

    self_result = model.inspect_event(self_failure)
    world_result = model.inspect_event(world_failure)

    assert self_result.classification == "self_error"
    assert self_result.attribution == "self"
    assert self_result.evidence["category"] == "context_budget"
    assert world_result.classification == "world_error"
    assert world_result.attribution == "world"
    assert world_result.evidence["category"] == "timeout"
