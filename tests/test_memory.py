from __future__ import annotations

import unittest

from segmentum.memory import (
    LongTermMemory,
    RISK_WEIGHT,
    ValueHierarchy,
    compute_prediction_error,
)


def baseline_observation() -> dict[str, float]:
    return {
        "food": 0.40,
        "danger": 0.95,
        "novelty": 0.30,
        "shelter": 0.20,
        "temperature": 0.45,
        "social": 0.25,
    }


def baseline_prediction() -> dict[str, float]:
    return {
        "food": 0.70,
        "danger": 0.10,
        "novelty": 0.45,
        "shelter": 0.50,
        "temperature": 0.50,
        "social": 0.35,
    }


def baseline_errors() -> dict[str, float]:
    return {
        "food": -0.30,
        "danger": 0.85,
        "novelty": -0.15,
        "shelter": -0.30,
        "temperature": -0.05,
        "social": -0.10,
    }


def baseline_body_state() -> dict[str, float]:
    return {
        "energy": 0.10,
        "stress": 0.75,
        "fatigue": 0.25,
        "temperature": 0.45,
    }


class EpisodicMemoryTests(unittest.TestCase):
    def test_value_hierarchy(self) -> None:
        hierarchy = ValueHierarchy()

        self.assertEqual(hierarchy.score("survival_threat"), -1000.0)
        self.assertEqual(hierarchy.score("integrity_loss"), -100.0)
        self.assertEqual(hierarchy.score("resource_loss"), -10.0)
        self.assertEqual(hierarchy.score("neutral"), 0.0)
        self.assertEqual(hierarchy.score("resource_gain"), 5.0)
        self.assertAlmostEqual(
            sum(hierarchy.probability_distribution.values()),
            1.0,
            places=12,
        )
        self.assertLess(
            hierarchy.risk("resource_gain"),
            hierarchy.risk("neutral"),
        )
        self.assertAlmostEqual(hierarchy.evaluate(
            state_snapshot={
                "observation": baseline_observation(),
                "prediction": baseline_prediction(),
                "errors": baseline_errors(),
                "body_state": baseline_body_state(),
            },
            outcome={
                "energy_delta": -0.05,
                "stress_delta": 0.10,
                "free_energy_drop": -0.40,
            },
        ), -1.0)

    def test_prediction_error_is_normalized(self) -> None:
        prediction_error = compute_prediction_error(
            baseline_observation(),
            baseline_prediction(),
        )

        self.assertGreater(prediction_error, 0.0)
        self.assertLessEqual(prediction_error, 1.0)

    def test_surprise_trigger(self) -> None:
        memory = LongTermMemory(surprise_threshold=1.0)

        decision = memory.maybe_store_episode(
            cycle=1,
            observation=baseline_observation(),
            prediction=baseline_prediction(),
            errors=baseline_errors(),
            action="hide",
            outcome={
                "energy_delta": -0.05,
                "stress_delta": 0.10,
                "free_energy_drop": -0.40,
            },
            body_state=baseline_body_state(),
        )

        self.assertEqual(decision.predicted_outcome, "survival_threat")
        self.assertEqual(decision.value_label, "survival_threat")
        self.assertAlmostEqual(decision.value_score, -1.0)
        self.assertGreater(decision.risk, 0.0)
        self.assertGreater(decision.prediction_error, 0.0)
        self.assertGreater(decision.total_surprise, memory.surprise_threshold)
        self.assertTrue(decision.episode_created)
        self.assertEqual(len(memory.episodes), 1)

    def test_default_surprise_gate_skips_low_value_low_error_ticks(self) -> None:
        memory = LongTermMemory()

        decision = memory.maybe_store_episode(
            cycle=1,
            observation={
                "food": 0.50,
                "danger": 0.10,
                "novelty": 0.30,
                "shelter": 0.40,
                "temperature": 0.50,
                "social": 0.25,
            },
            prediction={
                "food": 0.49,
                "danger": 0.11,
                "novelty": 0.29,
                "shelter": 0.39,
                "temperature": 0.50,
                "social": 0.24,
            },
            errors={
                "food": 0.01,
                "danger": -0.01,
                "novelty": 0.01,
                "shelter": 0.01,
                "temperature": 0.0,
                "social": 0.01,
            },
            action="rest",
            outcome={
                "energy_delta": 0.0,
                "stress_delta": 0.0,
                "free_energy_drop": 0.01,
            },
            body_state={
                "energy": 0.80,
                "stress": 0.10,
                "fatigue": 0.10,
                "temperature": 0.50,
            },
        )

        self.assertEqual(decision.predicted_outcome, "resource_gain")
        self.assertAlmostEqual(decision.value_score, hierarchy := ValueHierarchy().normalized_score("resource_gain"))
        self.assertLess(decision.total_surprise, memory.surprise_threshold)
        self.assertFalse(decision.episode_created)
        self.assertEqual(len(memory.episodes), 0)

    def test_duplicate_filter(self) -> None:
        memory = LongTermMemory(
            surprise_threshold=1.0,
            duplicate_similarity_threshold=0.999,
        )
        kwargs = {
            "cycle": 2,
            "observation": baseline_observation(),
            "prediction": baseline_prediction(),
            "errors": baseline_errors(),
            "action": "hide",
            "outcome": {
                "energy_delta": -0.05,
                "stress_delta": 0.10,
                "free_energy_drop": -0.40,
            },
            "body_state": baseline_body_state(),
        }

        first = memory.maybe_store_episode(**kwargs)
        second = memory.maybe_store_episode(**kwargs)

        self.assertTrue(first.episode_created)
        self.assertFalse(second.episode_created)
        self.assertEqual(len(memory.episodes), 1)

    def test_high_prediction_error_triggers_episode_storage_for_neutral_outcome(self) -> None:
        memory = LongTermMemory(surprise_threshold=4.0)

        decision = memory.maybe_store_episode(
            cycle=3,
            observation={
                "food": 0.05,
                "danger": 0.10,
                "novelty": 0.05,
                "shelter": 0.05,
                "temperature": 0.10,
                "social": 0.05,
            },
            prediction={
                "food": 0.95,
                "danger": 0.95,
                "novelty": 0.95,
                "shelter": 0.95,
                "temperature": 0.95,
                "social": 0.95,
            },
            errors={
                "food": -0.90,
                "danger": -0.85,
                "novelty": -0.90,
                "shelter": -0.90,
                "temperature": -0.85,
                "social": -0.90,
            },
            action="scan",
            outcome={
                "energy_delta": 0.0,
                "stress_delta": 0.0,
                "free_energy_drop": 0.0,
            },
            body_state={
                "energy": 0.80,
                "stress": 0.10,
                "fatigue": 0.10,
                "temperature": 0.50,
            },
        )

        self.assertEqual(decision.predicted_outcome, "neutral")
        self.assertGreater(decision.prediction_error, 0.80)
        self.assertAlmostEqual(
            decision.total_surprise,
            decision.prediction_error + (RISK_WEIGHT * decision.risk),
        )
        self.assertGreater(decision.total_surprise, memory.surprise_threshold)
        self.assertTrue(decision.episode_created)
        self.assertEqual(len(memory.episodes), 1)

    def test_high_risk_outcome_triggers_episode_storage(self) -> None:
        memory = LongTermMemory(surprise_threshold=10.0)

        decision = memory.maybe_store_episode(
            cycle=4,
            observation={
                "food": 0.60,
                "danger": 0.10,
                "novelty": 0.40,
                "shelter": 0.50,
                "temperature": 0.50,
                "social": 0.40,
            },
            prediction={
                "food": 0.59,
                "danger": 0.11,
                "novelty": 0.39,
                "shelter": 0.49,
                "temperature": 0.50,
                "social": 0.39,
            },
            errors={
                "food": 0.01,
                "danger": -0.01,
                "novelty": 0.01,
                "shelter": 0.01,
                "temperature": 0.0,
                "social": 0.01,
            },
            action="hide",
            outcome={
                "energy_delta": -0.05,
                "stress_delta": 0.02,
                "free_energy_drop": -0.05,
            },
            body_state={
                "energy": 0.75,
                "stress": 0.15,
                "fatigue": 0.10,
                "temperature": 0.50,
            },
        )

        self.assertEqual(decision.predicted_outcome, "resource_loss")
        self.assertLess(decision.preferred_probability, 1e-4)
        self.assertGreater(decision.risk, 10.0)
        self.assertAlmostEqual(
            decision.total_surprise,
            decision.prediction_error + (RISK_WEIGHT * decision.risk),
        )
        self.assertGreater(decision.total_surprise, memory.surprise_threshold)
        self.assertTrue(decision.episode_created)
        self.assertEqual(len(memory.episodes), 1)


if __name__ == "__main__":
    unittest.main()
