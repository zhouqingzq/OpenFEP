from __future__ import annotations

import unittest

from segmentum.m224_benchmarks import SEED_SET, run_m224_workspace_benchmark


class TestM224CapacityPressure(unittest.TestCase):
    def test_capacity_changes_alter_downstream_outcomes(self) -> None:
        payload = run_m224_workspace_benchmark(seed_set=list(SEED_SET))
        capacity = payload["acceptance_report"]["capacity_breakdown"]
        self.assertGreaterEqual(len(capacity["monotonic_metrics"]), 3)
        self.assertGreaterEqual(capacity["workspace_capacity_effect_size"], 0.06)
        self.assertGreater(
            capacity["default"]["policy_causality_gain"]["mean"],
            capacity["low"]["policy_causality_gain"]["mean"],
        )
        self.assertGreater(
            capacity["default"]["memory_priority_gain"]["mean"],
            capacity["low"]["memory_priority_gain"]["mean"],
        )


if __name__ == "__main__":
    unittest.main()
