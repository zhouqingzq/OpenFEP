from __future__ import annotations

import unittest

from segmentum.m43_modeling import run_m43_single_task_suite


class TestM43Baselines(unittest.TestCase):
    def test_agent_beats_at_least_one_relevant_baseline_on_heldout_likelihood(self) -> None:
        payload = run_m43_single_task_suite(seed=43)
        agent = payload["heldout"]["agent"]["metrics"]["heldout_likelihood"]
        statistical = payload["heldout"]["statistical"]["metrics"]["heldout_likelihood"]
        self.assertGreater(agent, payload["heldout"]["signal_detection"]["metrics"]["heldout_likelihood"])
        self.assertGreaterEqual(payload["evidence"]["agent_vs_statistical"]["mean"], agent - statistical - 0.01)
        self.assertIn("statistical_cv_metrics", payload["evidence"])


if __name__ == "__main__":
    unittest.main()
