from __future__ import annotations

import unittest

from segmentum.m221_benchmarks import run_m221_open_narrative_benchmark


class TestM221AdversarialSurface(unittest.TestCase):
    def test_adversarial_keyword_stuffing_does_not_hijack_identity(self) -> None:
        payload = run_m221_open_narrative_benchmark(cycles=24)
        summary = payload["variant_breakdown"]["canonical_vs_adversarial_surface"]
        self.assertLessEqual(summary["wrong_direction_initialization_rate"]["mean"], 0.15)
        self.assertLessEqual(summary["identity_commitment_flip_rate"]["mean"], 0.10)
        self.assertLessEqual(summary["attention_channel_replacement_rate"]["mean"], 0.15)


if __name__ == "__main__":
    unittest.main()
