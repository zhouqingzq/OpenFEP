from __future__ import annotations

from statistics import mean, pvariance
import unittest

from segmentum.m4_cognitive_style import PARAMETER_REFERENCE
from segmentum.m41_falsification import run_parameter_falsification_suite, run_same_framework_sensitivity_suite


def _cohens_d(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    left_var = pvariance(left) if len(left) > 1 else 0.0
    right_var = pvariance(right) if len(right) > 1 else 0.0
    pooled = (((len(left) - 1) * left_var) + ((len(right) - 1) * right_var)) / max(1, (len(left) + len(right) - 2))
    if pooled <= 1e-9:
        return 0.0 if abs(left_mean - right_mean) < 1e-9 else 1.0
    return abs((left_mean - right_mean) / (pooled ** 0.5))


class TestM41Falsification(unittest.TestCase):
    def test_falsification_suite_is_structural_and_allows_failure(self) -> None:
        payload = run_same_framework_sensitivity_suite()
        self.assertEqual(payload["analysis_type"], "same_framework_sensitivity_suite")
        self.assertEqual(payload["benchmark_scope"], "same-framework parameter sensitivity sidecar")
        self.assertEqual(payload["claim_envelope"], "sidecar_synthetic_diagnostic")
        self.assertEqual(payload["legacy_status"], "m42_plus_preresearch_sidecar")
        self.assertEqual(payload["validation_type"], "synthetic_holdout_same_framework")
        self.assertEqual(payload["interpretation"], "internal generator sensitivity only; not external falsification or M4 acceptance evidence")
        self.assertEqual(set(payload["experiments"]), set(PARAMETER_REFERENCE))
        passed = 0
        for parameter_name, experiment in payload["experiments"].items():
            with self.subTest(parameter=parameter_name):
                self.assertEqual(experiment["analysis_type"], "parameter_sensitivity_probe")
                self.assertIn(experiment["supported"], [True, False])
                self.assertIn("cohens_d", experiment)
                self.assertIn("alternative_explanation_test", experiment)
                passed += int(bool(experiment["supported"]))
        self.assertGreaterEqual(passed, 6)
        control_metric = payload["control_metric"]
        self.assertEqual(control_metric["metric"], "session_length")
        self.assertIn("high_condition_series", control_metric)
        self.assertIn("low_condition_series", control_metric)
        self.assertEqual(control_metric["control_strength"], "weak")
        self.assertIn("near-constant by construction", control_metric["control_limitation"])
        self.assertEqual(
            control_metric["cohens_d"],
            round(_cohens_d(control_metric["high_condition_series"], control_metric["low_condition_series"]), 6),
        )
        self.assertEqual(control_metric["high_condition_summary"]["count"], len(control_metric["high_condition_series"]))
        self.assertEqual(control_metric["low_condition_summary"]["count"], len(control_metric["low_condition_series"]))
        self.assertLess(control_metric["cohens_d"], 0.2)

    def test_legacy_falsification_entrypoint_remains_alias(self) -> None:
        self.assertEqual(run_parameter_falsification_suite(), run_same_framework_sensitivity_suite())


if __name__ == "__main__":
    unittest.main()
