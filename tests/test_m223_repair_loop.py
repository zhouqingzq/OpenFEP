from __future__ import annotations

import unittest

from segmentum.m223_benchmarks import run_m223_self_consistency_benchmark


class TestM223RepairLoop(unittest.TestCase):
    def test_repair_improves_alignment(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=[223, 242])
        full = payload["variant_metrics"]["with_repair"]
        no_repair = payload["variant_metrics"]["no_repair"]
        self.assertGreater(full["repair_success_rate"], 0.0)
        self.assertGreater(full["commitment_alignment_rate"], no_repair["commitment_alignment_rate"])
        self.assertGreater(full["post_repair_alignment_gain"], no_repair["post_repair_alignment_gain"])


if __name__ == "__main__":
    unittest.main()
