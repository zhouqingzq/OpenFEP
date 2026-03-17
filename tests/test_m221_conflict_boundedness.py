from __future__ import annotations

import unittest

from segmentum.m221_benchmarks import run_m221_open_narrative_benchmark


class TestM221ConflictBoundedness(unittest.TestCase):
    def test_conflicting_narratives_produce_bounded_uncertainty(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        summary = payload["variant_breakdown"]["conflicting_narrative"]
        self.assertGreaterEqual(summary["conflict_uncertainty_score"]["mean"], 0.20)
        self.assertLessEqual(summary["extreme_single_commitment_ratio"]["mean"], 0.70)
        self.assertLessEqual(summary["policy_distribution_l1_distance"]["mean"], 0.35)

    def test_low_signal_text_degrades_gracefully(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        degradation = payload["variant_breakdown"]["low_signal_degradation"]["malformed_text_degradation_ratio"]["mean"]
        self.assertGreaterEqual(degradation, 0.40)
        self.assertLessEqual(degradation, 0.85)


if __name__ == "__main__":
    unittest.main()
