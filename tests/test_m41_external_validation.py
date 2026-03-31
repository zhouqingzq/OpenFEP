from __future__ import annotations

import unittest

from segmentum.m41_external_validation import run_cross_source_holdout_validation


class TestM41ExternalValidation(unittest.TestCase):
    def test_cross_source_holdout_reports_accuracy_stability_and_failures(self) -> None:
        payload = run_cross_source_holdout_validation()
        self.assertEqual(payload["analysis_type"], "cross_source_holdout_validation")
        self.assertEqual(payload["training_design"]["source"], "internal_scenario_library")
        self.assertIn("classification_accuracy", payload["metrics"])
        self.assertIn("calibration_error", payload["metrics"])
        self.assertIn("stability_report", payload["metrics"])
        self.assertGreaterEqual(payload["metrics"]["classification_accuracy"], 0.66)
        self.assertLessEqual(payload["metrics"]["calibration_error"], 0.35)
        self.assertGreaterEqual(payload["metrics"]["stability_report"]["subjects_with_repeat_sessions"], 3)
        self.assertIn("failure_samples", payload)


if __name__ == "__main__":
    unittest.main()
