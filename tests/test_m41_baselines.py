from __future__ import annotations

import unittest

from segmentum.m41_external_validation import compare_models_on_cross_source_holdout


class TestM41Baselines(unittest.TestCase):
    def test_style_inference_beats_simpler_baselines_on_holdout(self) -> None:
        payload = compare_models_on_cross_source_holdout()
        self.assertEqual(payload["analysis_type"], "baseline_comparison")
        ranked = payload["models"]
        self.assertGreaterEqual(len(ranked), 5)
        self.assertEqual(ranked[0]["sessions"][0]["model_label"], "style_inference_model")
        best = ranked[0]
        worst = ranked[-1]
        self.assertGreaterEqual(best["metrics"]["classification_accuracy"], worst["metrics"]["classification_accuracy"])
        self.assertLessEqual(
            best["metrics"]["parameter_recovery_stability"]["mean_mae"],
            worst["metrics"]["parameter_recovery_stability"]["mean_mae"],
        )


if __name__ == "__main__":
    unittest.main()
