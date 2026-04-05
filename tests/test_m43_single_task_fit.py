from __future__ import annotations

import unittest
from unittest.mock import patch

from segmentum.m43_modeling import (
    _choose_parameter_activity,
    _metric_noise_floor,
    _simulate_confidence_trials,
    _simulate_igt_trials,
    _summarize_confidence_shift,
    _summarize_igt_shift,
    run_fitted_confidence_agent,
    run_fitted_igt_agent,
    run_m43_single_task_suite,
    run_parameter_sensitivity_analysis,
)
from segmentum.m4_benchmarks import BenchmarkTrial, IowaGamblingTaskAdapter, default_acceptance_benchmark_root
from segmentum.m4_cognitive_style import CognitiveStyleParameters


SMALL_EXTERNAL_LIMITS = {
    "confidence_train_max_trials": 1200,
    "confidence_validation_max_trials": 1200,
    "confidence_heldout_max_trials": 1500,
    "sensitivity_confidence_max_trials": 600,
}


class TestM43SingleTaskFit(unittest.TestCase):
    def test_blocked_without_external_bundle_when_smoke_is_disallowed(self) -> None:
        with patch("segmentum.m43_modeling.default_acceptance_benchmark_root", return_value=None):
            payload = run_m43_single_task_suite(benchmark_root=None, allow_smoke_test=False)
        self.assertTrue(payload["blocked"])
        self.assertEqual(payload["acceptance_state"], "blocked_missing_external_bundle")

    def test_smoke_path_is_deterministic(self) -> None:
        first = run_fitted_confidence_agent(seed=43, allow_smoke_test=True, sample_limits={"confidence_heldout_max_trials": 4})
        second = run_fitted_confidence_agent(seed=43, allow_smoke_test=True, sample_limits={"confidence_heldout_max_trials": 4})
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["predictions"], second["predictions"])

    def test_confidence_simulation_heldout_likelihood_varies_with_probability_outputs(self) -> None:
        trials = [
            BenchmarkTrial("t1", "s1", "sess1", 0.15, "right", "right", 0.55, 500, "validation", "demo", "a.csv"),
            BenchmarkTrial("t2", "s1", "sess1", 0.10, "right", "right", 0.53, 510, "validation", "demo", "a.csv"),
            BenchmarkTrial("t3", "s1", "sess1", -0.12, "left", "left", 0.52, 520, "validation", "demo", "a.csv"),
            BenchmarkTrial("t4", "s1", "sess1", -0.18, "left", "left", 0.56, 530, "validation", "demo", "a.csv"),
            BenchmarkTrial("t5", "s1", "sess1", 0.05, "right", "right", 0.51, 540, "validation", "demo", "a.csv"),
            BenchmarkTrial("t6", "s1", "sess1", -0.04, "left", "left", 0.50, 550, "validation", "demo", "a.csv"),
        ]
        low_parameters = CognitiveStyleParameters(
            uncertainty_sensitivity=1.0,
            error_aversion=0.0,
            exploration_bias=1.0,
            attention_selectivity=0.0,
            confidence_gain=0.0,
            update_rigidity=0.0,
            resource_pressure_sensitivity=0.0,
            virtual_prediction_error_gain=0.0,
        )
        high_parameters = CognitiveStyleParameters(
            uncertainty_sensitivity=0.0,
            error_aversion=1.0,
            exploration_bias=0.0,
            attention_selectivity=1.0,
            confidence_gain=1.0,
            update_rigidity=1.0,
            resource_pressure_sensitivity=1.0,
            virtual_prediction_error_gain=1.0,
        )

        low_payload = _simulate_confidence_trials(trials, low_parameters, seed=19)
        high_payload = _simulate_confidence_trials(trials, high_parameters, seed=19)
        low_probabilities = [float(row["predicted_probability_right"]) for row in low_payload["predictions"]]
        high_probabilities = [float(row["predicted_probability_right"]) for row in high_payload["predictions"]]

        self.assertNotEqual(low_payload["metrics"]["heldout_likelihood"], high_payload["metrics"]["heldout_likelihood"])
        self.assertNotEqual(low_probabilities, high_probabilities)

    def test_smoke_parameter_sensitivity_reports_nonzero_confidence_likelihood_effects(self) -> None:
        payload = run_parameter_sensitivity_analysis(
            seed=45,
            allow_smoke_test=True,
            sample_limits={"sensitivity_confidence_max_trials": 120},
        )
        heldout_deltas = [
            float(row["measured_effects"]["confidence_heldout_likelihood"])
            for row in payload["parameters"]
        ]

        self.assertTrue(any(delta > 0.0 for delta in heldout_deltas))

    def test_same_seed_same_params_shift_summaries_are_zero(self) -> None:
        confidence_trials = [
            BenchmarkTrial("t1", "s1", "sess1", 0.15, "right", "right", 0.55, 500, "validation", "demo", "a.csv"),
            BenchmarkTrial("t2", "s1", "sess1", -0.10, "left", "left", 0.53, 510, "validation", "demo", "a.csv"),
            BenchmarkTrial("t3", "s1", "sess1", 0.05, "right", "right", 0.51, 540, "validation", "demo", "a.csv"),
        ]
        parameters = CognitiveStyleParameters()
        first_confidence = _simulate_confidence_trials(confidence_trials, parameters, seed=19, include_predictions=False)
        second_confidence = _simulate_confidence_trials(confidence_trials, parameters, seed=19, include_predictions=False)
        confidence_shift = _summarize_confidence_shift(first_confidence, second_confidence)
        self.assertTrue(all(float(value) == 0.0 for value in confidence_shift.values()))

        igt_trials = IowaGamblingTaskAdapter().load_trials(allow_smoke_test=True, protocol_mode="smoke_flexible")
        first_igt = _simulate_igt_trials(igt_trials, parameters, seed=44, include_predictions=False)
        second_igt = _simulate_igt_trials(igt_trials, parameters, seed=44, include_predictions=False)
        igt_shift = _summarize_igt_shift(first_igt, second_igt)
        self.assertTrue(all(float(value) == 0.0 for value in igt_shift.values()))

    def test_same_params_different_seeds_are_measured_as_noise_floor(self) -> None:
        trials = [
            BenchmarkTrial("t1", "s1", "sess1", 0.15, "right", "right", 0.55, 500, "validation", "demo", "a.csv"),
            BenchmarkTrial("t2", "s1", "sess1", 0.10, "right", "right", 0.53, 510, "validation", "demo", "a.csv"),
            BenchmarkTrial("t3", "s1", "sess1", -0.12, "left", "left", 0.52, 520, "validation", "demo", "a.csv"),
            BenchmarkTrial("t4", "s1", "sess1", -0.18, "left", "left", 0.56, 530, "validation", "demo", "a.csv"),
        ]
        parameters = CognitiveStyleParameters()
        base_payload = _simulate_confidence_trials(trials, parameters, seed=19, include_predictions=False)
        alt_payload = _simulate_confidence_trials(trials, parameters, seed=21, include_predictions=False)
        noise_floor = _metric_noise_floor(base_payload, [alt_payload], task="confidence")

        self.assertGreater(float(noise_floor["confidence_mean_abs_probability_shift"]), 0.0)

    def test_parameter_activity_requires_real_behavior_change(self) -> None:
        decision = _choose_parameter_activity(
            "uncertainty_sensitivity",
            [
                {
                    "step": 0.10,
                    "direction": "high",
                    "shift_summary": {
                        "confidence_ambiguous_confidence": 0.040,
                        "confidence_confidence_separation": 0.030,
                        "confidence_mean_abs_probability_shift": 0.0,
                        "confidence_choice_flip_rate": 0.0,
                    },
                    "noise_floor": {
                        "confidence_ambiguous_confidence": 0.001,
                        "confidence_confidence_separation": 0.001,
                    },
                }
            ],
        )

        self.assertFalse(decision["active"])
        self.assertFalse(decision["behavior_change_ok"])

    def test_smoke_parameter_sensitivity_reports_explanations_and_legacy_keys(self) -> None:
        payload = run_parameter_sensitivity_analysis(
            seed=45,
            allow_smoke_test=True,
            sample_limits={"sensitivity_confidence_max_trials": 120},
        )

        self.assertEqual(payload["analysis_protocol"]["confidence_split"], "heldout")
        self.assertEqual(payload["analysis_protocol"]["igt_split"], "heldout")
        self.assertIn("confidence", payload["baseline_parameters"])
        self.assertIn("igt", payload["baseline_parameters"])
        self.assertEqual(len(payload["parameters"]), 8)
        for row in payload["parameters"]:
            self.assertIn("confidence_heldout_likelihood", row["measured_effects"])
            self.assertIn("confidence_brier_score", row["measured_effects"])
            self.assertIn("confidence_alignment", row["measured_effects"])
            self.assertIn("igt_deck_match_rate", row["measured_effects"])
            self.assertIn("igt_advantageous_ratio", row["measured_effects"])
            self.assertIn("igt_learning_curve_slope", row["measured_effects"])
            self.assertIn("signal_to_noise", row)
            self.assertIn("noise_floor", row)
            self.assertIn("classification_reason", row)
            self.assertIn("relevant_metrics", row)
            if row["active"]:
                self.assertTrue(row["behavior_change_ok"])
                self.assertGreaterEqual(float(row["winning_effect"]), float(row["activity_threshold"]))

    def test_smoke_igt_default_policy_does_not_collapse_to_always_advantageous(self) -> None:
        trials = IowaGamblingTaskAdapter().load_trials(allow_smoke_test=True, protocol_mode="smoke_flexible")
        payload = _simulate_igt_trials(trials, CognitiveStyleParameters(), seed=44, include_predictions=False)

        self.assertLess(float(payload["metrics"]["advantageous_choice_rate"]), 0.95)

    def test_smoke_igt_behavior_changes_when_parameters_change(self) -> None:
        trials = IowaGamblingTaskAdapter().load_trials(allow_smoke_test=True, protocol_mode="smoke_flexible")
        exploratory = CognitiveStyleParameters(
            uncertainty_sensitivity=0.15,
            error_aversion=0.10,
            exploration_bias=1.0,
            attention_selectivity=0.25,
            confidence_gain=0.20,
            update_rigidity=0.05,
            resource_pressure_sensitivity=0.10,
            virtual_prediction_error_gain=0.05,
        )
        cautious = CognitiveStyleParameters(
            uncertainty_sensitivity=1.0,
            error_aversion=1.0,
            exploration_bias=0.0,
            attention_selectivity=0.95,
            confidence_gain=1.0,
            update_rigidity=1.0,
            resource_pressure_sensitivity=1.0,
            virtual_prediction_error_gain=1.0,
        )

        exploratory_payload = _simulate_igt_trials(trials, exploratory, seed=44, include_predictions=False)
        cautious_payload = _simulate_igt_trials(trials, cautious, seed=44, include_predictions=False)
        exploratory_metrics = exploratory_payload["metrics"]
        cautious_metrics = cautious_payload["metrics"]
        exploratory_decks = [row["chosen_deck"] for row in exploratory_payload["trial_trace"]]
        cautious_decks = [row["chosen_deck"] for row in cautious_payload["trial_trace"]]

        self.assertTrue(
            exploratory_decks != cautious_decks
            or float(exploratory_metrics["advantageous_choice_rate"]) != float(cautious_metrics["advantageous_choice_rate"])
            or float(exploratory_metrics["late_advantageous_rate"]) != float(cautious_metrics["late_advantageous_rate"])
            or float(exploratory_metrics["learning_curve_slope"]) != float(cautious_metrics["learning_curve_slope"])
        )

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_external_confidence_fit_uses_real_bundle_and_baseline_ladder(self) -> None:
        payload = run_fitted_confidence_agent(
            seed=43,
            benchmark_root=default_acceptance_benchmark_root(),
            sample_limits=SMALL_EXTERNAL_LIMITS,
        )
        self.assertEqual(payload["source_type"], "external_bundle")
        self.assertEqual(payload["claim_envelope"], "benchmark_eval")
        self.assertGreaterEqual(payload["trial_count"], 1000)
        self.assertIn("random", payload["baselines"])
        self.assertIn("statistical_logistic", payload["baselines"])
        self.assertIn("human_match_ceiling", payload["baselines"])
        self.assertIn("lower_baselines_beaten", payload["baseline_ladder"])
        self.assertTrue(payload["leakage_check"]["subject"]["ok"])

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_external_igt_fit_uses_real_bundle_and_reports_failure_modes(self) -> None:
        payload = run_fitted_igt_agent(
            seed=44,
            benchmark_root=default_acceptance_benchmark_root(),
            sample_limits=SMALL_EXTERNAL_LIMITS,
        )
        self.assertEqual(payload["source_type"], "external_bundle")
        self.assertEqual(payload["claim_envelope"], "benchmark_eval")
        self.assertEqual(payload["protocol_mode"], "standard_100")
        self.assertGreaterEqual(payload["subject_summary"]["subject_count"], 3)
        self.assertIn("frequency_matching", payload["baselines"])
        self.assertIn("human_behavior", payload["baselines"])
        self.assertLess(float(payload["metrics"]["advantageous_choice_rate"]), 0.95)
        self.assertTrue(payload["failure_modes"])

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_external_parameter_sensitivity_classifies_expected_real_bundle_parameters(self) -> None:
        payload = run_parameter_sensitivity_analysis(
            seed=45,
            benchmark_root=default_acceptance_benchmark_root(),
            sample_limits=SMALL_EXTERNAL_LIMITS,
        )
        by_parameter = {str(row["parameter"]): bool(row["active"]) for row in payload["parameters"]}

        self.assertEqual(
            by_parameter,
            {
                "uncertainty_sensitivity": True,
                "error_aversion": True,
                "exploration_bias": True,
                "attention_selectivity": True,
                "confidence_gain": True,
                "update_rigidity": True,
                "resource_pressure_sensitivity": False,
                "virtual_prediction_error_gain": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
