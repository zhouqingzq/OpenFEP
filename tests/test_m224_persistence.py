from __future__ import annotations

import unittest

from segmentum.m224_benchmarks import SEED_SET, run_m224_workspace_benchmark


class TestM224Persistence(unittest.TestCase):
    def test_persistence_improves_cross_tick_coordination(self) -> None:
        payload = run_m224_workspace_benchmark(seed_set=list(SEED_SET))
        breakdown = payload["acceptance_report"]["persistence_breakdown"]
        self.assertGreaterEqual(
            breakdown["full_workspace"]["workspace_persistence_gain"]["mean"],
            0.10,
        )
        self.assertLessEqual(
            breakdown["full_workspace"]["broadcast_to_action_latency"]["mean"],
            2.0,
        )
        self.assertGreaterEqual(
            breakdown["full_workspace"]["broadcast_to_memory_alignment"]["mean"],
            0.80,
        )
        evidence = payload["artifacts"]["persistence_trace"]["variant_breakdown"]["full_workspace"]
        self.assertTrue(any(item["carry_over_tick_count"] > 0 for item in evidence))
        self.assertTrue(
            any(
                tick["carry_over_contents"]
                for item in evidence
                for tick in item["trace"]
            )
        )


if __name__ == "__main__":
    unittest.main()
