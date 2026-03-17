from __future__ import annotations

import unittest

from segmentum.m221_benchmarks import run_m221_determinism_probe, run_m221_open_narrative_benchmark


class TestM221ParaphraseStability(unittest.TestCase):
    def test_same_seed_same_text_is_deterministic(self) -> None:
        payload = run_m221_determinism_probe(seed=221, cycles=24)
        self.assertTrue(payload["passed"])

    def test_paraphrase_preserves_identity_direction(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        summary = payload["variant_breakdown"]["canonical_vs_paraphrase"]
        self.assertGreaterEqual(summary["identity_commitment_jaccard"]["mean"], 0.80)
        self.assertLessEqual(summary["policy_distribution_l1_distance"]["mean"], 0.18)
        self.assertLessEqual(summary["narrative_prior_l1_distance"]["mean"], 0.20)
        self.assertLessEqual(summary["personality_profile_l1_distance"]["mean"], 0.15)


if __name__ == "__main__":
    unittest.main()
