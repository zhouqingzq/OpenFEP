from __future__ import annotations

import unittest

from segmentum.m44_cross_task import compare_shared_vs_independent


class TestM44SharedParameters(unittest.TestCase):
    def test_shared_parameters_retain_cross_task_core(self) -> None:
        payload = compare_shared_vs_independent(seed=44)
        self.assertGreaterEqual(payload["stability_analysis"]["shared_parameter_count"], 4)
        self.assertLessEqual(payload["stability_analysis"]["parameter_distance_mean"], 0.10)
        self.assertGreaterEqual(payload["shared"]["heldout"]["igt"]["policy_alignment_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
