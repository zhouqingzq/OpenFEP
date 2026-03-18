from __future__ import annotations

import unittest

from segmentum.m223_benchmarks import run_m223_self_consistency_benchmark


class TestM223BoundedIdentityUpdate(unittest.TestCase):
    def test_identity_updates_remain_bounded(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=[223, 242])
        breakdown = payload["bounded_update_breakdown"]
        self.assertGreaterEqual(
            float(breakdown["commitment_persistence_score"]["mean"]),
            0.75,
        )
        self.assertLessEqual(
            float(breakdown["core_commitment_flip_rate"]["mean"]),
            0.10,
        )
        self.assertLessEqual(
            float(breakdown["identity_rewrite_ratio"]["mean"]),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
