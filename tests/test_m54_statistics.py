"""Directional paired tests for M5.4 validation statistics."""

from __future__ import annotations

import unittest

from segmentum.dialogue.validation.statistics import paired_comparison


class TestM54DirectionalStats(unittest.TestCase):
    def test_all_negative_diffs_not_significant_better(self) -> None:
        p_val, sig, mean_diff, better = paired_comparison(
            [0.4, 0.5, 0.45],
            [0.7, 0.8, 0.75],
            test="t_test",
            alpha=0.05,
        )
        self.assertGreaterEqual(p_val, 0.0)
        self.assertFalse(better)
        self.assertFalse(sig)
        self.assertLess(mean_diff, 0.0)

    def test_clearly_positive_worse_baseline_t_test(self) -> None:
        p_val, sig, mean_diff, better = paired_comparison(
            [0.9, 0.91, 0.92],
            [0.2, 0.21, 0.22],
            test="t_test",
            alpha=0.05,
        )
        self.assertTrue(better)
        self.assertLess(p_val, 0.05)
        self.assertTrue(sig)


if __name__ == "__main__":
    unittest.main()
