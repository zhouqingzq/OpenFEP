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


def secondary_observation() -> dict[str, float]:
    return {
        "food": 0.12,
        "danger": 0.88,
        "novelty": 0.61,
        "shelter": 0.74,
        "temperature": 0.31,
        "social": 0.09,
    }


def secondary_prediction() -> dict[str, float]:
    return {
        "food": 0.54,
        "danger": 0.36,
        "novelty": 0.29,
        "shelter": 0.22,
        "temperature": 0.48,
        "social": 0.41,
    }


def secondary_errors() -> dict[str, float]:
    observation = secondary_observation()
    prediction = secondary_prediction()
    return {
        key: observation[key] - prediction[key]
        for key in observation
    }


def secondary_body_state() -> dict[str, float]:
    return {
        "energy": 0.34,
        "stress": 0.74,
        "fatigue": 0.51,
        "temperature": 0.31,
    }


def drift_observation() -> dict[str, float]:
    return {
        "food": 0.91,
        "danger": 0.07,
        "novelty": 0.94,
        "shelter": 0.86,
        "temperature": 0.79,
        "social": 0.82,
    }


def drift_prediction() -> dict[str, float]:
    return {
        "food": 0.28,
        "danger": 0.41,
        "novelty": 0.22,
        "shelter": 0.34,
        "temperature": 0.51,
        "social": 0.37,
    }


def drift_errors() -> dict[str, float]:
    observation = drift_observation()
    prediction = drift_prediction()
    return {
        key: observation[key] - prediction[key]
        for key in observation
    }


def drift_body_state() -> dict[str, float]:
    return {
        "energy": 0.81,
        "stress": 0.11,
        "fatigue": 0.14,
        "temperature": 0.79,
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


def populate_secondary_rule_pattern(agent: SegmentAgent) -> None:
    agent.long_term_memory.maybe_store_episode(
        cycle=1,
        observation=secondary_observation(),
        prediction=secondary_prediction(),
        errors=secondary_errors(),
        action="hide",
        outcome=harmful_outcome(),
        body_state=secondary_body_state(),
    )
    for cycle in range(2, 4):
        agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=secondary_observation(),
            prediction=secondary_prediction(),
            errors=secondary_errors(),
            action="hide",
            outcome=harmful_outcome(),
            body_state=secondary_body_state(),
        )


def populate_unruled_drift(agent: SegmentAgent) -> None:
    agent.long_term_memory.maybe_store_episode(
        cycle=1,
        observation=drift_observation(),
        prediction=drift_prediction(),
        errors=drift_errors(),
        action="rest",
        outcome=harmful_outcome(),
        body_state=drift_body_state(),
    )
    agent.long_term_memory.store_episode(
        cycle=2,
        observation=drift_observation(),
        prediction=drift_prediction(),
        errors=drift_errors(),
        action="rest",
        outcome=harmful_outcome(),
        body_state=drift_body_state(),
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

    cm = summary.consolidation_metrics
    artifact = {
        "scenario": "M2.3 prediction flattening",
        "prediction_error_before_sleep": round(prediction_error_before_sleep, 6),
        "prediction_error_after_sleep": round(prediction_error_after_sleep, 6),
        "cluster": cluster,
        "action": ACTION,
        "rule_ids": list(summary.rule_ids),
        "seed": SEED,
    }
    if cm is not None:
        artifact["consolidation_metrics"] = cm.to_dict()
    return artifact


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

        for key in EXPECTED_ARTIFACT:
            self.assertEqual(EXPECTED_ARTIFACT[key], actual[key], f"mismatch on {key}")

    def test_conditioned_metrics_present_and_consistent(self) -> None:
        artifact = build_prediction_flattening_artifact()
        cm = artifact.get("consolidation_metrics")
        self.assertIsNotNone(cm, "consolidation_metrics should be present")

        # The conditioned PE for the rule-targeted cluster should show flattening
        ruled = [c for c in cm["cluster_pe"] if c["has_rule"]]
        self.assertTrue(ruled, "at least one cluster should have a rule")
        for c in ruled:
            self.assertEqual(c["action"], ACTION)
            self.assertLess(
                c["pe_after"],
                c["pe_before"],
                f"conditioned PE for rule cluster {c['cluster_id']}:{c['action']} "
                f"should decrease after sleep",
            )

        # Conditioned aggregates should show improvement
        self.assertLess(
            cm["conditioned_pe_after"],
            cm["conditioned_pe_before"],
            "aggregate conditioned PE should decrease",
        )

        # Novelty baseline should be positive
        self.assertGreater(cm["novelty_baseline"], 0.0)

        # Raw PE from consolidation should be in a reasonable range
        self.assertGreater(cm["raw_pe_before"], 0.0)
        self.assertGreater(cm["raw_pe_after"], 0.0)

    def test_normalised_pe_decreases_for_ruled_clusters(self) -> None:
        artifact = build_prediction_flattening_artifact()
        cm = artifact.get("consolidation_metrics")
        self.assertIsNotNone(cm)
        # Normalised PE should decrease when conditioned PE decreases
        # and novelty baseline is stable
        self.assertLessEqual(
            cm["normalised_pe_after"],
            cm["normalised_pe_before"],
            "normalised PE should not increase",
        )

    def test_normalised_pe_uses_conditioned_metric_not_raw_global_pe(self) -> None:
        agent = build_agent()
        populate_repeated_anomaly(agent)
        populate_unruled_drift(agent)
        agent.long_term_memory.assign_clusters()

        summary = agent.sleep()
        cm = summary.consolidation_metrics
        self.assertIsNotNone(cm)

        cluster_pe = cm.to_dict()["cluster_pe"]
        ruled = [c for c in cluster_pe if c["has_rule"]]
        non_ruled = [c for c in cluster_pe if not c["has_rule"]]

        self.assertTrue(ruled)
        self.assertTrue(non_ruled, "test needs an unrelated drift cluster")
        self.assertNotAlmostEqual(
            cm.raw_pe_before,
            cm.conditioned_pe_before,
            places=6,
            msg="raw PE must differ from conditioned PE for this regression test",
        )

        expected_before = cm.conditioned_pe_before / cm.novelty_baseline
        expected_after = cm.conditioned_pe_after / cm.novelty_baseline
        self.assertAlmostEqual(cm.normalised_pe_before, expected_before, places=6)
        self.assertAlmostEqual(cm.normalised_pe_after, expected_after, places=6)

    def test_conditioned_pe_is_weighted_by_episode_count(self) -> None:
        agent = build_agent()
        populate_repeated_anomaly(agent)
        populate_secondary_rule_pattern(agent)
        agent.long_term_memory.assign_clusters()

        summary = agent.sleep()
        cm = summary.consolidation_metrics
        self.assertIsNotNone(cm)

        ruled = [c for c in cm.cluster_pe if c.has_rule]
        self.assertGreaterEqual(len(ruled), 2, "test needs multiple ruled groups")

        total_count = sum(c.episode_count for c in ruled)
        expected_before = sum(c.pe_before * c.episode_count for c in ruled) / total_count
        expected_after = sum(c.pe_after * c.episode_count for c in ruled) / total_count

        self.assertAlmostEqual(cm.conditioned_pe_before, expected_before, places=6)
        self.assertAlmostEqual(cm.conditioned_pe_after, expected_after, places=6)


if __name__ == "__main__":
    unittest.main()
