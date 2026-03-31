from __future__ import annotations

import unittest

from segmentum.m43_modeling import run_fitted_confidence_agent, run_m43_single_task_suite


class TestM43SingleTaskFit(unittest.TestCase):
    def test_fitted_agent_is_deterministic(self) -> None:
        first = run_fitted_confidence_agent(seed=43)
        second = run_fitted_confidence_agent(seed=43)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["predictions"], second["predictions"])

    def test_suite_produces_failure_analysis_and_heldout_runs(self) -> None:
        payload = run_m43_single_task_suite(seed=43)
        self.assertIn("largest_calibration_gap_trial", payload["failure_analysis"])
        self.assertGreaterEqual(payload["heldout"]["agent"]["trial_count"], 1)
        self.assertIn("agent_vs_signal_detection", payload["evidence"])
        self.assertIn("agent_vs_statistical", payload["evidence"])

    def test_small_sample_forces_weak_claim_envelope(self) -> None:
        payload = run_m43_single_task_suite(seed=43)
        self.assertEqual(payload["recommendation"], "NOT_READY")
        self.assertFalse(payload["evidence"]["sample_size_sufficient_for_claim"])

    def test_statistical_baseline_must_be_checked(self) -> None:
        payload = run_m43_single_task_suite(seed=43)
        self.assertIn("agent_beats_statistical_baseline_on_heldout_likelihood", payload["failure_analysis"])


if __name__ == "__main__":
    unittest.main()
