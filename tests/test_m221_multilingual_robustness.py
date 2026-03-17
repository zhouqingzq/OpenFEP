from __future__ import annotations

import unittest

from segmentum.m221_benchmarks import run_m221_open_narrative_benchmark


class TestM221MultilingualRobustness(unittest.TestCase):
    def test_multilingual_mixed_text_remains_stable(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        summary = payload["variant_breakdown"]["multilingual_robustness"]
        self.assertGreaterEqual(summary["identity_commitment_jaccard"]["mean"], 0.75)
        self.assertLessEqual(summary["policy_distribution_l1_distance"]["mean"], 0.25)
        self.assertLessEqual(summary["narrative_prior_l1_distance"]["mean"], 0.25)

    def test_noisy_text_does_not_drift_far(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        summary = payload["variant_breakdown"]["canonical_vs_noisy"]
        self.assertLessEqual(summary["policy_distribution_l1_distance"]["mean"], 0.25)
        self.assertGreaterEqual(summary["attention_target_consistency"]["mean"], 0.80)
        self.assertGreaterEqual(summary["action_metric_retention"]["mean"], 0.85)


if __name__ == "__main__":
    unittest.main()
