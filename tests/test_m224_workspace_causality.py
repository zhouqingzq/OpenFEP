from __future__ import annotations

import unittest

from segmentum.m224_benchmarks import SEED_SET, run_m224_workspace_benchmark


class TestM224WorkspaceCausality(unittest.TestCase):
    def test_workspace_removal_degrades_multiple_downstream_functions(self) -> None:
        payload = run_m224_workspace_benchmark(seed_set=list(SEED_SET))
        report = payload["acceptance_report"]
        breakdown = report["downstream_causality_breakdown"]
        self.assertGreaterEqual(breakdown["policy_causality_gain"]["mean_delta"], 0.08)
        self.assertGreaterEqual(breakdown["memory_priority_gain"]["mean_delta"], 0.10)
        self.assertGreaterEqual(breakdown["maintenance_priority_gain"]["mean_delta"], 0.08)
        self.assertGreaterEqual(breakdown["metacognitive_review_gain"]["mean_delta"], 0.08)


if __name__ == "__main__":
    unittest.main()
