from __future__ import annotations

from dataclasses import replace
import random
import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.memory import (
    LongTermMemory,
    RISK_WEIGHT,
    ValueHierarchy,
    compute_prediction_error,
)
from segmentum.sleep_consolidator import SleepConsolidator, SleepLLMExtractor
from segmentum.types import SleepRule


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


class SleepConsolidationTests(unittest.TestCase):
    def _build_rule(
        self,
        *,
        rule_id: str,
        confidence: float,
    ) -> SleepRule:
        return SleepRule(
            rule_id=rule_id,
            type="risk_pattern",
            cluster=0,
            action="forage",
            observed_outcome="survival_threat",
            confidence=confidence,
            support=3,
            average_surprise=10.0,
            average_prediction_error=0.3,
            timestamp=1,
        )

    def _store_episode(
        self,
        memory: LongTermMemory,
        *,
        cycle: int,
        action: str = "forage",
        observation: dict[str, float] | None = None,
        prediction: dict[str, float] | None = None,
        errors: dict[str, float] | None = None,
        outcome: dict[str, float] | None = None,
        body_state: dict[str, float] | None = None,
    ) -> None:
        observation = observation or baseline_observation()
        prediction = prediction or baseline_prediction()
        errors = errors or baseline_errors()
        outcome = outcome or {
            "energy_delta": -0.05,
            "stress_delta": 0.10,
            "free_energy_drop": -0.40,
        }
        body_state = body_state or baseline_body_state()
        memory.store_episode(
            cycle=cycle,
            observation=observation,
            prediction=prediction,
            errors=errors,
            action=action,
            outcome=outcome,
            body_state=body_state,
        )

    def test_sleep_trigger_uses_interval_or_memory_threshold(self) -> None:
        memory = LongTermMemory(sleep_interval=200, memory_threshold=3)

        self.assertFalse(memory.should_sleep(199))
        self.assertTrue(memory.should_sleep(200))

        for cycle in range(1, 5):
            self._store_episode(memory, cycle=cycle)
        self.assertTrue(memory.should_sleep(5))

    def test_sleep_rule_extraction_requires_support_and_dominance(self) -> None:
        consolidator = SleepConsolidator(
            surprise_threshold=1.0,
            minimum_support=3,
        )
        episodes = [
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.30,
                "total_surprise": 10.0,
            },
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.28,
                "total_surprise": 11.0,
            },
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.27,
                "total_surprise": 12.0,
            },
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "neutral",
                "prediction_error": 0.05,
                "total_surprise": 2.0,
            },
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "integrity_loss",
                "prediction_error": 0.22,
                "total_surprise": 9.0,
            },
        ]

        result = consolidator.consolidate(
            sleep_cycle_id=1,
            current_cycle=10,
            episodes=episodes,
            transition_statistics={"0:forage": {"0": 1.0}},
            outcome_distributions={"0:forage": {"survival_threat": 0.6}},
        )

        self.assertEqual(len(result.rules), 1)
        self.assertEqual(result.rules[0].observed_outcome, "survival_threat")
        self.assertEqual(result.rules[0].support, 3)
        self.assertEqual(len(result.semantic_memory_entries), 1)
        self.assertEqual(len(result.model_updates), 2)

    def test_sleep_rule_extraction_applies_optional_llm_hook(self) -> None:
        calls: list[tuple[int, int]] = []

        def llm_extractor(rules, episodes):  # noqa: ANN001
            calls.append((len(rules), len(episodes)))
            return [replace(rules[0], confidence=0.95)]

        consolidator = SleepConsolidator(
            surprise_threshold=1.0,
            minimum_support=3,
            llm_extractor=llm_extractor,
        )
        episodes = [
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.30,
                "total_surprise": 10.0,
            },
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.28,
                "total_surprise": 11.0,
            },
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.27,
                "total_surprise": 12.0,
            },
        ]

        result = consolidator.consolidate(
            sleep_cycle_id=2,
            current_cycle=12,
            episodes=episodes,
            transition_statistics={"0:forage": {"0": 1.0}},
            outcome_distributions={"0:forage": {"survival_threat": 1.0}},
        )

        self.assertEqual(calls, [(1, 3)])
        self.assertEqual(result.rules[0].confidence, 0.95)
        self.assertEqual(result.semantic_memory_entries[0].confidence, 0.95)

    def test_sleep_llm_extractor_is_a_safe_no_op_without_llm(self) -> None:
        extractor = SleepLLMExtractor()
        rules = [
            result_rule := self._build_rule(
                rule_id="sleep-1-0-forage-survival_threat",
                confidence=0.72,
            )
        ]

        refined = extractor(rules, [{"cluster_id": 0, "action_taken": "forage"}])

        self.assertEqual(refined, [result_rule])

    def test_sleep_llm_extractor_can_refine_rules_when_llm_is_available(self) -> None:
        captured: list[tuple[int, int]] = []

        def llm(*, summary, rules, episodes):  # noqa: ANN001
            captured.append((summary["rule_count"], summary["episode_count"]))
            return [replace(rules[0], confidence=0.99)]

        extractor = SleepLLMExtractor(llm=llm)
        rules = [
            self._build_rule(
                rule_id="sleep-2-0-forage-survival_threat",
                confidence=0.72,
            )
        ]

        refined = extractor(rules, [{"cluster_id": 0, "action_taken": "forage"}])

        self.assertEqual(captured, [(1, 1)])
        self.assertEqual(refined[0].confidence, 0.99)

    def test_sleep_replay_updates_transition_model(self) -> None:
        agent = SegmentAgent(rng=random.Random(7))
        agent.long_term_memory.minimum_support = 2
        for cycle in range(1, 7):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="hide" if cycle % 2 == 0 else "forage",
                observation={
                    "food": 0.30 + (0.02 * cycle),
                    "danger": 0.80 - (0.03 * cycle),
                    "novelty": 0.20,
                    "shelter": 0.30,
                    "temperature": 0.45,
                    "social": 0.20,
                },
                prediction=baseline_prediction(),
                errors=baseline_errors(),
            )

        summary = agent.sleep()

        self.assertGreater(summary.episodes_sampled, 0)
        self.assertGreater(summary.clusters_created, 0)
        self.assertGreater(summary.world_model_updates, 0)
        self.assertTrue(agent.world_model.transition_model)
        self.assertTrue(agent.world_model.transition_counts)

    def test_high_risk_patterns_reduce_policy_bias(self) -> None:
        agent = SegmentAgent(rng=random.Random(11))
        agent.long_term_memory.minimum_support = 2
        risky_observation = {
            "food": 0.20,
            "danger": 0.92,
            "novelty": 0.15,
            "shelter": 0.18,
            "temperature": 0.45,
            "social": 0.10,
        }
        risky_prediction = {
            "food": 0.70,
            "danger": 0.20,
            "novelty": 0.45,
            "shelter": 0.50,
            "temperature": 0.50,
            "social": 0.35,
        }
        risky_errors = {
            key: risky_observation[key] - risky_prediction[key]
            for key in risky_observation
        }
        for cycle in range(1, 7):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="forage",
                observation=risky_observation,
                prediction=risky_prediction,
                errors=risky_errors,
                outcome={
                    "energy_delta": -0.08,
                    "stress_delta": 0.25,
                    "free_energy_drop": -0.45,
                },
                body_state={
                    "energy": 0.12,
                    "stress": 0.82,
                    "fatigue": 0.35,
                    "temperature": 0.45,
                },
            )

        agent.sleep()
        diagnostics = agent.decision_cycle(Observation(**risky_observation))["diagnostics"]
        forage_option = next(
            option for option in diagnostics.ranked_options if option.choice == "forage"
        )

        self.assertLess(forage_option.policy_bias, 0.0)
        self.assertEqual(agent.sleep_history[-1].policy_bias_updates, 1)
        self.assertTrue(agent.semantic_memory)
        self.assertEqual(agent.semantic_memory[-1].rule_type, "risk_pattern")

    def test_high_variance_outcomes_trigger_epistemic_bonus(self) -> None:
        agent = SegmentAgent(rng=random.Random(19))
        agent.long_term_memory.minimum_support = 5
        ambiguous_observation = {
            "food": 0.55,
            "danger": 0.45,
            "novelty": 0.70,
            "shelter": 0.40,
            "temperature": 0.50,
            "social": 0.35,
        }
        ambiguous_prediction = {
            "food": 0.52,
            "danger": 0.42,
            "novelty": 0.45,
            "shelter": 0.42,
            "temperature": 0.50,
            "social": 0.35,
        }
        ambiguous_errors = {
            key: ambiguous_observation[key] - ambiguous_prediction[key]
            for key in ambiguous_observation
        }

        for cycle in range(1, 4):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="scan",
                observation=ambiguous_observation,
                prediction=ambiguous_prediction,
                errors=ambiguous_errors,
                outcome={
                    "energy_delta": 0.08,
                    "stress_delta": -0.04,
                    "free_energy_drop": 0.12,
                },
                body_state={
                    "energy": 0.82,
                    "stress": 0.18,
                    "fatigue": 0.20,
                    "temperature": 0.50,
                },
            )
        for cycle in range(4, 7):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="scan",
                observation=ambiguous_observation,
                prediction=ambiguous_prediction,
                errors=ambiguous_errors,
                outcome={
                    "energy_delta": -0.12,
                    "stress_delta": 0.30,
                    "free_energy_drop": -0.40,
                },
                body_state={
                    "energy": 0.14,
                    "stress": 0.80,
                    "fatigue": 0.32,
                    "temperature": 0.50,
                },
            )

        agent.sleep()
        diagnostics = agent.decision_cycle(Observation(**ambiguous_observation))["diagnostics"]
        scan_option = next(
            option for option in diagnostics.ranked_options if option.choice == "scan"
        )

        self.assertGreater(scan_option.epistemic_bonus, 0.0)
        self.assertEqual(agent.sleep_history[-1].epistemic_bonus_updates, 1)

    def test_online_clustering_spawns_new_clusters_for_distant_states(self) -> None:
        memory = LongTermMemory(cluster_distance_threshold=0.15)
        self._store_episode(
            memory,
            cycle=1,
            observation={
                "food": 0.92,
                "danger": 0.08,
                "novelty": 0.15,
                "shelter": 0.90,
                "temperature": 0.50,
                "social": 0.15,
            },
            prediction={
                "food": 0.85,
                "danger": 0.10,
                "novelty": 0.20,
                "shelter": 0.85,
                "temperature": 0.50,
                "social": 0.20,
            },
            errors={
                "food": 0.07,
                "danger": -0.02,
                "novelty": -0.05,
                "shelter": 0.05,
                "temperature": 0.0,
                "social": -0.05,
            },
        )
        self._store_episode(
            memory,
            cycle=2,
            observation={
                "food": 0.05,
                "danger": 0.95,
                "novelty": 0.95,
                "shelter": 0.10,
                "temperature": 0.30,
                "social": 0.85,
            },
            prediction={
                "food": 0.10,
                "danger": 0.85,
                "novelty": 0.80,
                "shelter": 0.15,
                "temperature": 0.35,
                "social": 0.75,
            },
            errors={
                "food": -0.05,
                "danger": 0.10,
                "novelty": 0.15,
                "shelter": -0.05,
                "temperature": -0.05,
                "social": 0.10,
            },
        )

        clusters_created = memory.assign_clusters()

        self.assertEqual(clusters_created, 2)
        self.assertEqual(len(memory.cluster_centroids), 2)
        self.assertNotEqual(memory.episodes[0]["cluster_id"], memory.episodes[1]["cluster_id"])

    def test_surprise_based_forgetting_removes_predictable_episodes(self) -> None:
        agent = SegmentAgent(rng=random.Random(23))
        agent.long_term_memory.minimum_support = 2
        agent.long_term_memory.surprise_threshold = 0.20
        for cycle in range(1, 5):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="scan",
                observation={
                    "food": 0.85,
                    "danger": 0.10,
                    "novelty": 0.55,
                    "shelter": 0.75,
                    "temperature": 0.50,
                    "social": 0.40,
                },
                prediction={
                    "food": 0.84,
                    "danger": 0.11,
                    "novelty": 0.52,
                    "shelter": 0.74,
                    "temperature": 0.50,
                    "social": 0.39,
                },
                errors={
                    "food": 0.01,
                    "danger": -0.01,
                    "novelty": 0.03,
                    "shelter": 0.01,
                    "temperature": 0.0,
                    "social": 0.01,
                },
                outcome={
                    "energy_delta": 0.10,
                    "stress_delta": -0.02,
                    "free_energy_drop": 0.15,
                },
                body_state={
                    "energy": 0.88,
                    "stress": 0.12,
                    "fatigue": 0.16,
                    "temperature": 0.50,
                },
            )

        before = len(agent.long_term_memory.episodes)
        summary = agent.sleep()

        self.assertLess(len(agent.long_term_memory.episodes), before)
        self.assertGreater(summary.episodes_deleted, 0)

    def test_sleep_compresses_episodes_without_erasing_semantic_memory(self) -> None:
        agent = SegmentAgent(rng=random.Random(29))
        agent.long_term_memory.sleep_minimum_support = 3
        for cycle in range(1, 6):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="forage",
                observation={
                    "food": 0.38,
                    "danger": 0.58,
                    "novelty": 0.22,
                    "shelter": 0.18,
                    "temperature": 0.46,
                    "social": 0.18,
                },
                prediction={
                    "food": 0.72,
                    "danger": 0.18,
                    "novelty": 0.42,
                    "shelter": 0.42,
                    "temperature": 0.50,
                    "social": 0.30,
                },
                errors={
                    "food": -0.34,
                    "danger": 0.40,
                    "novelty": -0.20,
                    "shelter": -0.24,
                    "temperature": -0.04,
                    "social": -0.12,
                },
                outcome={
                    "energy_delta": -0.08,
                    "stress_delta": 0.24,
                    "fatigue_delta": 0.16,
                    "temperature_delta": 0.02,
                    "free_energy_drop": -0.42,
                },
                body_state={
                    "energy": 0.18,
                    "stress": 0.82,
                    "fatigue": 0.32,
                    "temperature": 0.46,
                },
            )

        summary = agent.sleep()

        self.assertGreater(summary.compression_removed, 0)
        self.assertTrue(agent.semantic_memory)
        self.assertGreaterEqual(summary.semantic_entries_written, 1)
        self.assertGreaterEqual(summary.rules_extracted, 1)
        self.assertEqual(len(agent.semantic_memory), summary.semantic_entries_written)

    def test_sleep_pipeline_records_episode_compression_result(self) -> None:
        agent = SegmentAgent(rng=random.Random(31))
        self._store_episode(agent.long_term_memory, cycle=1)
        calls: list[str] = []

        def fake_compress() -> int:
            calls.append("compress")
            return 2

        agent.long_term_memory.compress_episodes = fake_compress  # type: ignore[method-assign]

        summary = agent.sleep()

        self.assertEqual(calls, ["compress"])
        self.assertEqual(summary.compression_removed, 2)
        self.assertGreaterEqual(summary.memory_compressed, 2)

    def test_prediction_error_decreases_over_multiple_sleep_cycles(self) -> None:
        agent = SegmentAgent(rng=random.Random(13))
        target_observation = {
            "food": 0.95,
            "danger": 0.05,
            "novelty": 0.55,
            "shelter": 0.65,
            "temperature": 0.48,
            "social": 0.35,
        }
        target_prediction = dict(agent.world_model.beliefs)
        target_errors = {
            key: target_observation[key] - target_prediction[key]
            for key in target_observation
        }
        for cycle in range(1, 9):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="exploit_shelter",
                observation=target_observation,
                prediction=target_prediction,
                errors=target_errors,
                outcome={
                    "energy_delta": 0.02,
                    "stress_delta": -0.05,
                    "free_energy_drop": 0.10,
                },
                body_state={
                    "energy": 0.75,
                    "stress": 0.20,
                    "fatigue": 0.25,
                    "temperature": 0.48,
                },
            )

        first_summary = agent.sleep()
        first_after = first_summary.prediction_error_after

        for cycle in range(9, 17):
            self._store_episode(
                agent.long_term_memory,
                cycle=cycle,
                action="exploit_shelter",
                observation=target_observation,
                prediction=target_prediction,
                errors=target_errors,
                outcome={
                    "energy_delta": 0.02,
                    "stress_delta": -0.05,
                    "free_energy_drop": 0.10,
                },
                body_state={
                    "energy": 0.75,
                    "stress": 0.20,
                    "fatigue": 0.25,
                    "temperature": 0.48,
                },
            )

        second_summary = agent.sleep()

        self.assertLess(first_summary.prediction_error_after, first_summary.prediction_error_before)
        self.assertLess(second_summary.prediction_error_after, second_summary.prediction_error_before)
        self.assertLessEqual(second_summary.prediction_error_after, first_after)


if __name__ == "__main__":
    unittest.main()
