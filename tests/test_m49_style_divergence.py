from __future__ import annotations

import unittest

from segmentum.m49_longitudinal import run_longitudinal_style_suite


class TestM49StyleDivergence(unittest.TestCase):
    def test_between_profile_divergence_exceeds_within_profile_drift(self) -> None:
        payload = run_longitudinal_style_suite()
        self.assertTrue(payload["summary"]["style_divergence_reproducible"])
        self.assertGreater(payload["summary"]["within_profile_cross_seed_distance_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
