"""Directional paired tests for M5.4 validation statistics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from segmentum.dialogue.validation.statistics import paired_comparison

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - exercised on minimal dev envs
    wilcoxon = None


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

    def test_wilcoxon_matches_scipy_greater(self) -> None:
        if wilcoxon is None:
            self.skipTest("scipy is required for exact Wilcoxon comparison")
        personality = [0.7, 0.8, 0.75, 0.81, 0.77]
        baseline = [0.6, 0.7, 0.72, 0.79, 0.70]
        diffs = [p - b for p, b in zip(personality, baseline)]
        p_val, _, _, _ = paired_comparison(personality, baseline, test="wilcoxon")
        expected = float(wilcoxon(diffs, alternative="greater", zero_method="wilcox", mode="auto").pvalue)
        self.assertAlmostEqual(p_val, round(expected, 6))

    def test_wilcoxon_missing_scipy_does_not_fabricate_significance(self) -> None:
        with patch(
            "segmentum.dialogue.validation.statistics._wilcoxon_greater_p",
            side_effect=ImportError("scipy unavailable"),
        ):
            p_val, sig, mean_diff, better = paired_comparison(
                [0.9, 0.91, 0.92],
                [0.2, 0.21, 0.22],
                test="wilcoxon",
            )
        self.assertTrue(better)
        self.assertGreater(mean_diff, 0.0)
        self.assertEqual(p_val, 1.0)
        self.assertFalse(sig)


if __name__ == "__main__":
    unittest.main()
