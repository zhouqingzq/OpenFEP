"""Tests that validate LLM-enhanced sleep consolidation as a first-class path.

A deterministic ``MockSleepLLMExtractor`` simulates what a real LLM would do:
boost confidence on high-support risk rules, leave others alone.  The tests
verify that the LLM path is exercised end-to-end and that its output
propagates into semantic memory and slow weights.
"""
from __future__ import annotations

import random
import unittest
from dataclasses import replace

from segmentum.agent import SegmentAgent
from segmentum.sleep_consolidator import (
    HeuristicSleepExtractor,
    SleepConsolidator,
    SleepLLMExtractor,
)
from segmentum.types import SleepRule

# ---------------------------------------------------------------------------
# Deterministic mock 鈥?Task 2
# ---------------------------------------------------------------------------

CONFIDENCE_BOOST = 0.05


class MockSleepLLMExtractor:
    """Deterministic stand-in for a real LLM rule refiner.

    Behaviour:
    * Raises confidence of every rule by ``CONFIDENCE_BOOST``, clamped to 0.99.
    * Records each invocation so tests can assert the path was taken.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.last_rules: list[SleepRule] = []
        self.last_episodes: list[dict[str, object]] = []

    def __call__(
        self,
        rules: list[SleepRule],
        episodes: list[dict[str, object]],
    ) -> list[SleepRule]:
        self.call_count += 1
        self.last_rules = list(rules)
        self.last_episodes = list(episodes)
        return [
            replace(rule, confidence=min(0.99, rule.confidence + CONFIDENCE_BOOST))
            for rule in rules
        ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

OBSERVATION = {
    "food": 0.38,
    "danger": 0.58,
    "novelty": 0.22,
    "shelter": 0.18,
    "temperature": 0.46,
    "social": 0.18,
}
PREDICTION = {
    "food": 0.72,
    "danger": 0.18,
    "novelty": 0.42,
    "shelter": 0.42,
    "temperature": 0.50,
    "social": 0.30,
}
ERRORS = {key: OBSERVATION[key] - PREDICTION[key] for key in OBSERVATION}
HARMFUL_OUTCOME = {
    "energy_delta": -0.08,
    "stress_delta": 0.24,
    "fatigue_delta": 0.16,
    "temperature_delta": 0.02,
    "free_energy_drop": -0.42,
}
HARMFUL_BODY_STATE = {
    "energy": 0.18,
    "stress": 0.82,
    "fatigue": 0.32,
    "temperature": 0.46,
}


def _populate_anomaly_episodes(agent: SegmentAgent, count: int = 3) -> None:
    """Store ``count`` identical high-surprise episodes."""
    for cycle in range(1, count + 1):
        agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=OBSERVATION,
            prediction=PREDICTION,
            errors=ERRORS,
            action="forage",
            outcome=HARMFUL_OUTCOME,
            body_state=HARMFUL_BODY_STATE,
        )


# ---------------------------------------------------------------------------
# Tests 鈥?Task 3
# ---------------------------------------------------------------------------


class SleepLLMConsolidationTests(unittest.TestCase):
    """End-to-end tests for the LLM-enhanced sleep consolidation path."""

    def test_llm_extractor_is_invoked_during_sleep(self) -> None:
        """The mock LLM extractor must actually be called during ``agent.sleep()``."""
        mock = MockSleepLLMExtractor()
        agent = SegmentAgent(rng=random.Random(29), sleep_llm_extractor=mock)
        agent.energy = 0.22
        agent.stress = 0.30
        agent.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(agent)

        summary = agent.sleep()

        self.assertGreater(mock.call_count, 0, "LLM extractor was never called")
        self.assertTrue(summary.llm_used, "SleepSummary.llm_used should be True")

    def test_llm_confidence_boost_propagates_to_semantic_memory(self) -> None:
        """Confidence adjusted by the LLM must appear in semantic memory entries."""
        mock = MockSleepLLMExtractor()
        agent = SegmentAgent(rng=random.Random(29), sleep_llm_extractor=mock)
        agent.energy = 0.22
        agent.stress = 0.30
        agent.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(agent)

        # Capture heuristic confidence by running without LLM first
        baseline_agent = SegmentAgent(rng=random.Random(29))
        baseline_agent.energy = 0.22
        baseline_agent.stress = 0.30
        baseline_agent.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(baseline_agent)
        baseline_summary = baseline_agent.sleep()
        heuristic_confidences = {
            entry.rule_id: entry.confidence
            for entry in baseline_agent.semantic_memory
        }

        # Now run with LLM
        summary = agent.sleep()

        self.assertTrue(agent.semantic_memory, "No semantic memory produced")
        for entry in agent.semantic_memory:
            heuristic_conf = heuristic_confidences.get(entry.rule_id)
            if heuristic_conf is not None:
                expected = min(0.99, heuristic_conf + CONFIDENCE_BOOST)
                self.assertAlmostEqual(
                    entry.confidence,
                    expected,
                    places=4,
                    msg=f"Rule {entry.rule_id}: expected LLM-boosted confidence "
                    f"{expected}, got {entry.confidence}",
                )

    def test_llm_boosted_confidence_affects_slow_weights(self) -> None:
        """Higher confidence from LLM must produce larger slow-weight deltas."""
        mock = MockSleepLLMExtractor()
        agent = SegmentAgent(rng=random.Random(29), sleep_llm_extractor=mock)
        agent.energy = 0.22
        agent.stress = 0.30
        agent.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(agent)

        baseline = SegmentAgent(rng=random.Random(29))
        baseline.energy = 0.22
        baseline.stress = 0.30
        baseline.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(baseline)
        baseline.sleep()

        agent.sleep()

        # threat_priors are derived from confidence * support; higher confidence
        # should yield a strictly larger slow-weight update in this fixture.
        for cluster_key in agent.world_model.threat_priors:
            llm_threat = agent.world_model.threat_priors[cluster_key]
            base_threat = baseline.world_model.threat_priors.get(cluster_key, 0.0)
            self.assertGreater(
                llm_threat,
                base_threat,
                f"Cluster {cluster_key}: LLM threat prior {llm_threat} "
                f"should be > baseline {base_threat}",
            )
            llm_penalty = agent.world_model.preference_penalties.get(cluster_key, {}).get("forage", 0.0)
            base_penalty = baseline.world_model.preference_penalties.get(cluster_key, {}).get("forage", 0.0)
            self.assertLess(
                llm_penalty,
                base_penalty,
                f"Cluster {cluster_key}: LLM preference penalty {llm_penalty} "
                f"should be more negative than baseline {base_penalty}",
            )

    def test_sleep_summary_records_llm_participation(self) -> None:
        """SleepSummary must expose ``llm_used=True`` and valid rule ids."""
        mock = MockSleepLLMExtractor()
        agent = SegmentAgent(rng=random.Random(29), sleep_llm_extractor=mock)
        agent.energy = 0.22
        agent.stress = 0.30
        agent.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(agent)

        summary = agent.sleep()

        self.assertTrue(summary.llm_used)
        self.assertGreater(summary.rules_extracted, 0)
        self.assertTrue(summary.rule_ids)

    def test_no_llm_extractor_produces_llm_used_false(self) -> None:
        """Without an LLM extractor, ``llm_used`` must be ``False``."""
        agent = SegmentAgent(rng=random.Random(29))
        agent.energy = 0.22
        agent.stress = 0.30
        agent.long_term_memory.minimum_support = 1
        _populate_anomaly_episodes(agent)

        summary = agent.sleep()

        self.assertFalse(summary.llm_used)

    def test_heuristic_extractor_is_always_present_as_default(self) -> None:
        """SleepConsolidator always fills the extraction slot, even without LLM."""
        consolidator = SleepConsolidator(surprise_threshold=1.0)
        self.assertIsInstance(consolidator.llm_extractor, HeuristicSleepExtractor)

    def test_extraction_stage_runs_in_heuristic_mode(self) -> None:
        """The extraction stage must execute even in heuristic-only mode.

        Proves the pipeline is ``heuristic -> extraction stage -> semantic``
        rather than ``heuristic -> (skip) -> semantic``.
        """
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
            }
        ] * 3
        result = consolidator.consolidate(
            sleep_cycle_id=1,
            current_cycle=6,
            episodes=episodes,
            transition_statistics={},
            outcome_distributions={},
        )
        # Even without an LLM, rules must come out of the extraction stage.
        self.assertFalse(result.llm_used)
        self.assertGreater(len(result.rules), 0)
        self.assertGreater(len(result.rules_before_llm), 0)
        # The heuristic extractor clamps confidence, so rules survive intact.
        for rule in result.rules:
            self.assertGreaterEqual(rule.confidence, 0.05)
            self.assertLessEqual(rule.confidence, 0.99)

    def test_consolidator_tracks_rules_before_llm(self) -> None:
        """SleepConsolidationResult should contain the pre-LLM heuristic rules."""
        mock = MockSleepLLMExtractor()
        consolidator = SleepConsolidator(
            surprise_threshold=1.0,
            minimum_support=3,
            llm_extractor=mock,
        )
        episodes = [
            {
                "cluster_id": 0,
                "action_taken": "forage",
                "predicted_outcome": "survival_threat",
                "prediction_error": 0.30,
                "total_surprise": 10.0,
            }
        ] * 3
        result = consolidator.consolidate(
            sleep_cycle_id=1,
            current_cycle=6,
            episodes=episodes,
            transition_statistics={},
            outcome_distributions={},
        )
        self.assertTrue(result.llm_used)
        self.assertTrue(result.rules_before_llm)
        # The pre-LLM rules should have lower confidence than post-LLM rules.
        for before, after in zip(result.rules_before_llm, result.rules):
            self.assertLess(before.confidence, after.confidence)


if __name__ == "__main__":
    unittest.main()



