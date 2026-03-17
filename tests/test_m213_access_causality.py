from __future__ import annotations

import unittest

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation


class M213AccessCausalityTests(unittest.TestCase):
    def test_workspace_changes_policy_ranking_under_same_observation(self) -> None:
        observation = Observation(
            food=0.2,
            danger=0.05,
            novelty=1.0,
            shelter=0.1,
            temperature=0.5,
            social=0.1,
        )

        baseline = SegmentAgent()
        baseline.configure_global_workspace(enabled=False)
        baseline_result = baseline.decision_cycle(observation)["diagnostics"]

        workspace_agent = SegmentAgent()
        workspace_agent.configure_global_workspace(
            enabled=True,
            capacity=1,
            action_bias_gain=3.0,
            memory_gate_gain=0.08,
        )
        workspace_result = workspace_agent.decision_cycle(observation)["diagnostics"]

        self.assertEqual(baseline_result.chosen.choice, "hide")
        self.assertEqual(workspace_result.chosen.choice, "scan")
        self.assertGreater(
            workspace_result.policy_scores["scan"],
            baseline_result.policy_scores["scan"],
        )
        self.assertEqual(workspace_result.workspace_broadcast_channels, ["novelty"])

    def test_workspace_focus_appears_in_explanation_and_suppressed_channels_do_not(self) -> None:
        agent = SegmentAgent()
        agent.configure_global_workspace(
            enabled=True,
            capacity=1,
            action_bias_gain=3.0,
            memory_gate_gain=0.08,
        )
        diagnostics = agent.decision_cycle(
            Observation(
                food=0.2,
                danger=0.05,
                novelty=1.0,
                shelter=0.1,
                temperature=0.5,
                social=0.1,
            )
        )["diagnostics"]

        explanation = diagnostics.explanation
        self.assertIn("novelty", explanation)
        self.assertNotIn("danger, novelty", explanation)
        self.assertEqual(diagnostics.structured_explanation["workspace_focus"], ["novelty"])

    def test_workspace_lowers_memory_gate_threshold_during_storage(self) -> None:
        thresholds: list[float] = []
        agent = SegmentAgent()
        agent.configure_global_workspace(
            enabled=True,
            capacity=1,
            action_bias_gain=3.0,
            memory_gate_gain=0.12,
        )
        agent.decision_cycle(
            Observation(
                food=0.2,
                danger=0.05,
                novelty=1.0,
                shelter=0.1,
                temperature=0.5,
                social=0.1,
            )
        )
        original = agent.long_term_memory.maybe_store_episode

        def wrapper(*args, **kwargs):  # noqa: ANN002,ANN003
            thresholds.append(agent.long_term_memory.surprise_threshold)
            return original(*args, **kwargs)

        agent.long_term_memory.maybe_store_episode = wrapper  # type: ignore[method-assign]
        agent.integrate_outcome(
            choice="rest",
            observed={
                "food": 0.45,
                "danger": 0.15,
                "novelty": 0.55,
                "shelter": 0.40,
                "temperature": 0.50,
                "social": 0.20,
            },
            prediction={
                "food": 0.44,
                "danger": 0.16,
                "novelty": 0.54,
                "shelter": 0.39,
                "temperature": 0.50,
                "social": 0.22,
            },
            errors={
                "food": 0.01,
                "danger": -0.01,
                "novelty": 0.01,
                "shelter": 0.01,
                "temperature": 0.0,
                "social": -0.02,
            },
            free_energy_before=0.21,
            free_energy_after=0.24,
        )

        self.assertEqual(len(thresholds), 1)
        self.assertLess(thresholds[0], 0.40)


if __name__ == "__main__":
    unittest.main()
