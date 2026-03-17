from __future__ import annotations

import unittest

from segmentum.agent import SegmentAgent
from segmentum.m220_benchmarks import (
    _rollout_initialized,
    narrative_initialization_scenarios,
    run_m220_determinism_probe,
    run_m220_stress_probe,
)
from segmentum.narrative_initialization import NarrativeInitializer


class TestM220NarrativeInitialization(unittest.TestCase):
    def test_initializer_is_deterministic_for_same_seed_and_text(self) -> None:
        scenario = narrative_initialization_scenarios()["social_trusting"]
        probe = run_m220_determinism_probe(seed=901)

        self.assertTrue(probe["passed"])
        self.assertEqual(probe["first"]["metrics"], probe["second"]["metrics"])
        self.assertEqual(
            probe["first"]["initialization"]["policy_distribution"],
            probe["second"]["initialization"]["policy_distribution"],
        )
        self.assertEqual(scenario.expected_action_metric, "seek_contact_rate")

    def test_social_narrative_changes_attention_and_behavior_against_ablation(self) -> None:
        scenario = narrative_initialization_scenarios()["social_trusting"]
        initialized = _rollout_initialized(
            scenario=scenario,
            seed=220,
            cycles=24,
            apply_initialization=True,
        )
        ablated = _rollout_initialized(
            scenario=scenario,
            seed=220,
            cycles=24,
            apply_initialization=False,
        )

        self.assertGreater(
            initialized["metrics"]["social_attention_rate"],
            ablated["metrics"]["social_attention_rate"],
        )
        self.assertGreater(
            initialized["metrics"]["seek_contact_rate"],
            ablated["metrics"]["seek_contact_rate"],
        )

    def test_stress_probe_degrades_gracefully_for_malformed_text(self) -> None:
        payload = run_m220_stress_probe(seed=777)
        self.assertTrue(payload["passed"])
        self.assertAlmostEqual(payload["distribution_sum"], 1.0, places=6)

    def test_initializer_writes_identity_commitment_and_preferences(self) -> None:
        scenario = narrative_initialization_scenarios()["exploratory_adaptive"]
        agent = SegmentAgent()
        result = NarrativeInitializer().initialize_agent(
            agent=agent,
            episodes=list(scenario.episodes),
            apply_policy_seed=True,
        )

        self.assertTrue(result.learned_preferences)
        self.assertTrue(result.identity_commitments)
        self.assertIn(
            result.learned_preferences[0],
            result.policy_distribution,
        )


if __name__ == "__main__":
    unittest.main()
