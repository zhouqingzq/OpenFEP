from __future__ import annotations

import random
import unittest

from segmentum.agent import SegmentAgent
from tests.test_counterfactual import _populate_dangerous_episodes

SEED = 42
EXPECTED_ARTIFACT = {
    "scenario": "M2.4 counterfactual learning",
    "seed": SEED,
    "cluster": 0,
    "episodes_before_sleep": 5,
    "counterfactual_episodes_evaluated": 3,
    "counterfactual_insights_generated": 7,
    "counterfactual_insights_absorbed": 1,
    "counterfactual_energy_spent": 0.095,
    "forage_bias_pre_sleep": 0.0,
    "forage_bias_pre_counterfactual": -0.45,
    "forage_bias_after": -0.75,
    "hide_bias_pre_sleep": 0.0,
    "hide_bias_pre_counterfactual": 0.0,
    "hide_bias_after": 0.3,
    "policy_delta": 0.15,
    "sandbox_label": "虚拟沙盒推演",
    "absorbed_counterfactual_action": "hide",
}


def build_counterfactual_artifact() -> dict[str, object]:
    agent = SegmentAgent(rng=random.Random(SEED))
    agent.energy = 0.70
    agent.stress = 0.40
    agent.fatigue = 0.25
    agent.temperature = 0.46
    agent.long_term_memory.minimum_support = 1
    agent.long_term_memory.sleep_minimum_support = 1
    _populate_dangerous_episodes(agent, count=5)
    agent.long_term_memory.assign_clusters()

    cluster_id = int(agent.long_term_memory.episodes[0]["cluster_id"])
    forage_bias_before = float(agent.world_model.get_policy_bias(cluster_id, "forage"))
    hide_bias_before = float(agent.world_model.get_policy_bias(cluster_id, "hide"))
    episodes_before_sleep = len(agent.long_term_memory.episodes)

    agent.cycle = 20
    summary = agent.sleep()
    sandbox_entry = next(
        (
            entry
            for entry in summary.counterfactual_log
            if entry.get("type") == "virtual_sandbox_reasoning"
        ),
        {},
    )
    absorption_entry = next(
        (
            entry
            for entry in summary.counterfactual_log
            if entry.get("type") == "absorption"
        ),
        {},
    )

    policy_delta = float(absorption_entry.get("policy_delta", 0.0))
    forage_bias_after = float(agent.world_model.get_policy_bias(cluster_id, "forage"))
    hide_bias_after = float(agent.world_model.get_policy_bias(cluster_id, "hide"))
    forage_bias_pre_counterfactual = float(absorption_entry.get("new_orig_bias", forage_bias_after)) + policy_delta
    hide_bias_pre_counterfactual = float(absorption_entry.get("new_cf_bias", hide_bias_after)) - policy_delta

    return {
        "scenario": "M2.4 counterfactual learning",
        "seed": SEED,
        "cluster": cluster_id,
        "episodes_before_sleep": episodes_before_sleep,
        "counterfactual_episodes_evaluated": summary.counterfactual_episodes_evaluated,
        "counterfactual_insights_generated": summary.counterfactual_insights_generated,
        "counterfactual_insights_absorbed": summary.counterfactual_insights_absorbed,
        "counterfactual_energy_spent": round(summary.counterfactual_energy_spent, 3),
        "forage_bias_pre_sleep": round(forage_bias_before, 3),
        "forage_bias_pre_counterfactual": round(forage_bias_pre_counterfactual, 3),
        "forage_bias_after": round(forage_bias_after, 3),
        "hide_bias_pre_sleep": round(hide_bias_before, 3),
        "hide_bias_pre_counterfactual": round(hide_bias_pre_counterfactual, 3),
        "hide_bias_after": round(hide_bias_after, 3),
        "policy_delta": round(policy_delta, 3),
        "sandbox_label": sandbox_entry.get("label", ""),
        "absorbed_counterfactual_action": absorption_entry.get("counterfactual_action", ""),
    }


class CounterfactualArtifactTests(unittest.TestCase):
    def test_counterfactual_artifact_matches_fixture(self) -> None:
        self.assertEqual(EXPECTED_ARTIFACT, build_counterfactual_artifact())

    def test_counterfactual_artifact_shows_policy_shift(self) -> None:
        artifact = build_counterfactual_artifact()
        self.assertLess(artifact["forage_bias_after"], artifact["forage_bias_pre_counterfactual"])
        self.assertGreater(artifact["hide_bias_after"], artifact["hide_bias_pre_counterfactual"])
        self.assertEqual(artifact["sandbox_label"], "虚拟沙盒推演")


if __name__ == "__main__":
    unittest.main()

