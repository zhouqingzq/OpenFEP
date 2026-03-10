from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation

SEED = 29
ACTION = "forage"
EXPECTED_ARTIFACT = {
    "scenario": "M2.3 prediction flattening",
    "prediction_error_before_sleep": 0.634562,
    "prediction_error_after_sleep": 0.372936,
    "cluster": 0,
    "action": ACTION,
    "rule_ids": ["sleep-1-0-forage-survival_threat"],
    "seed": SEED,
}


def anomaly_observation() -> dict[str, float]:
    return {
        "food": 0.38,
        "danger": 0.58,
        "novelty": 0.22,
        "shelter": 0.18,
        "temperature": 0.46,
        "social": 0.18,
    }


def anomaly_prediction() -> dict[str, float]:
    return {
        "food": 0.72,
        "danger": 0.18,
        "novelty": 0.42,
        "shelter": 0.42,
        "temperature": 0.50,
        "social": 0.30,
    }


def anomaly_errors() -> dict[str, float]:
    observation = anomaly_observation()
    prediction = anomaly_prediction()
    return {
        key: observation[key] - prediction[key]
        for key in observation
    }


def harmful_outcome() -> dict[str, float]:
    return {
        "energy_delta": -0.08,
        "stress_delta": 0.24,
        "fatigue_delta": 0.16,
        "temperature_delta": 0.02,
        "free_energy_drop": -0.42,
    }


def baseline_body_state() -> dict[str, float]:
    return {
        "energy": 0.22,
        "stress": 0.30,
        "fatigue": 0.18,
        "temperature": 0.46,
    }


def harmful_body_state() -> dict[str, float]:
    return {
        "energy": 0.18,
        "stress": 0.82,
        "fatigue": 0.32,
        "temperature": 0.46,
    }


def build_agent() -> SegmentAgent:
    agent = SegmentAgent(rng=random.Random(SEED))
    body_state = baseline_body_state()
    agent.energy = body_state["energy"]
    agent.stress = body_state["stress"]
    agent.fatigue = body_state["fatigue"]
    agent.temperature = body_state["temperature"]
    agent.cycle = 6
    agent.long_term_memory.sleep_minimum_support = 3
    return agent


def populate_repeated_anomaly(agent: SegmentAgent) -> None:
    agent.long_term_memory.maybe_store_episode(
        cycle=1,
        observation=anomaly_observation(),
        prediction=anomaly_prediction(),
        errors=anomaly_errors(),
        action=ACTION,
        outcome=harmful_outcome(),
        body_state=harmful_body_state(),
    )
    for cycle in range(2, 6):
        agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=anomaly_observation(),
            prediction=anomaly_prediction(),
            errors=anomaly_errors(),
            action=ACTION,
            outcome=harmful_outcome(),
            body_state=harmful_body_state(),
        )


def restore_same_situation(agent: SegmentAgent) -> None:
    body_state = baseline_body_state()
    agent.energy = body_state["energy"]
    agent.stress = body_state["stress"]
    agent.fatigue = body_state["fatigue"]
    agent.temperature = body_state["temperature"]


def build_prediction_flattening_artifact() -> dict[str, object]:
    agent = build_agent()
    populate_repeated_anomaly(agent)
    agent.long_term_memory.assign_clusters()
    observation = Observation(**anomaly_observation())
    prediction_error_before_sleep, _ = agent.estimate_action_prediction_error(
        observation,
        ACTION,
    )

    summary = agent.sleep()
    restore_same_situation(agent)
    prediction_error_after_sleep, cluster = agent.estimate_action_prediction_error(
        observation,
        ACTION,
    )

    return {
        "scenario": "M2.3 prediction flattening",
        "prediction_error_before_sleep": round(prediction_error_before_sleep, 6),
        "prediction_error_after_sleep": round(prediction_error_after_sleep, 6),
        "cluster": cluster,
        "action": ACTION,
        "rule_ids": list(summary.rule_ids),
        "seed": SEED,
    }


class SleepPredictionFlatteningTests(unittest.TestCase):
    def test_sleep_flattens_prediction_error_for_same_action(self) -> None:
        artifact = build_prediction_flattening_artifact()

        self.assertLess(
            artifact["prediction_error_after_sleep"],
            artifact["prediction_error_before_sleep"],
        )
        self.assertEqual(artifact["action"], ACTION)
        self.assertIsInstance(artifact["cluster"], int)
        self.assertTrue(artifact["rule_ids"])

    def test_estimator_includes_slow_weight_confidence_when_requested(self) -> None:
        agent = build_agent()
        populate_repeated_anomaly(agent)
        agent.long_term_memory.assign_clusters()
        agent.world_model.set_outcome_distribution(0, ACTION, {"survival_threat": 1.0})
        observation = Observation(**anomaly_observation())

        plain_error, cluster = agent.estimate_action_prediction_error(
            observation,
            ACTION,
            include_slow_weights=False,
        )
        slow_weighted_error, slow_cluster = agent.estimate_action_prediction_error(
            observation,
            ACTION,
            include_slow_weights=True,
        )

        self.assertEqual(cluster, 0)
        self.assertEqual(slow_cluster, 0)
        self.assertLess(slow_weighted_error, plain_error)

    def test_prediction_flattening_artifact_matches_fixture(self) -> None:
        actual = build_prediction_flattening_artifact()

        self.assertEqual(EXPECTED_ARTIFACT, actual)


if __name__ == "__main__":
    unittest.main()
