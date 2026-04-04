from __future__ import annotations

import unittest

from segmentum.m4_cognitive_style import PARAMETER_REFERENCE
from segmentum.m41_identifiability import (
    build_identifiability_report,
    cross_generator_identifiability_report,
    synthetic_family_recoverability_report,
    synthetic_recoverability_summary,
)


class TestM41Identifiability(unittest.TestCase):
    def test_cross_generator_identifiability_recovers_most_parameters(self) -> None:
        payload = synthetic_family_recoverability_report()
        self.assertEqual(payload["analysis_type"], "same_framework_synthetic_recoverability")
        self.assertEqual(payload["legacy_analysis_type"], "cross_generator_identifiability")
        self.assertEqual(payload["benchmark_scope"], "same-framework synthetic recoverability sidecar")
        self.assertEqual(payload["claim_envelope"], "sidecar_synthetic_diagnostic")
        self.assertEqual(payload["legacy_status"], "m42_plus_preresearch_sidecar")
        self.assertEqual(payload["validation_type"], "synthetic_holdout_same_framework")
        self.assertEqual(payload["interpretation"], "within synthetic family recoverability only")
        self.assertTrue(payload["inference_path_blinded"])
        self.assertTrue(all(check["inference_path_blinded"] for check in payload["blindness_checks"]))
        self.assertEqual(payload["primary_recovery_model"]["model_type"], "per_parameter_weighted_linear_regression")
        self.assertEqual(set(payload["candidate_bank_baseline"]), set(PARAMETER_REFERENCE))
        self.assertEqual(set(payload["primary_vs_baseline_delta"]), set(PARAMETER_REFERENCE))
        recoverable = [name for name, row in payload["parameter_recovery"].items() if row["mae"] < 0.25]
        self.assertGreaterEqual(len(recoverable), 5)
        self.assertEqual(payload["train_test_seed_overlap"], 0)
        self.assertTrue(any(row["primary_better_or_equal"] for row in payload["primary_vs_baseline_delta"].values()))
        self.assertIn("does not count as M4.1 acceptance", payload["validation_limits"][-1])

    def test_legacy_report_shape_remains_available(self) -> None:
        payload = synthetic_recoverability_summary()
        self.assertEqual(payload["analysis_type"], "same_framework_recoverability_summary")
        self.assertEqual(payload["legacy_analysis_type"], "parameter_identifiability_report")
        self.assertEqual(payload["legacy_status"], "m42_plus_preresearch_sidecar")
        self.assertEqual(payload["interpretation"], "within synthetic family recoverability only")
        self.assertEqual(set(payload["parameter_recovery"]), set(PARAMETER_REFERENCE))
        self.assertEqual(set(payload["parameter_coupling"]), set(PARAMETER_REFERENCE))
        self.assertIn("primary_recovery_model", payload)
        self.assertIn("candidate_bank_baseline", payload)
        self.assertIn("validation_limits", payload)

    def test_legacy_identifiability_entrypoints_remain_aliases(self) -> None:
        self.assertEqual(cross_generator_identifiability_report(), synthetic_family_recoverability_report())
        self.assertEqual(build_identifiability_report(), synthetic_recoverability_summary())


if __name__ == "__main__":
    unittest.main()
