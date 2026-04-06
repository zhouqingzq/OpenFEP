from __future__ import annotations

import unittest
from unittest.mock import patch

from segmentum.m44_cross_task import (
    _task_fit_with_fixed_parameter,
    classify_parameter_stability,
    compute_degradation_matrix,
    fit_joint_parameters,
    run_m44_cross_task_suite,
)
from segmentum.m4_cognitive_style import CognitiveStyleParameters, PARAMETER_REFERENCE
from segmentum.m4_benchmarks import default_acceptance_benchmark_root


SMALL_EXTERNAL_LIMITS = {
    "confidence_train_max_trials": 1200,
    "confidence_validation_max_trials": 1200,
    "confidence_heldout_max_trials": 1500,
    "sensitivity_confidence_max_trials": 600,
    "igt_train_max_subjects": 3,
    "igt_validation_max_subjects": 3,
    "igt_heldout_max_subjects": 3,
    "architecture_candidate_count": 2,
}


class TestM44CrossTask(unittest.TestCase):
    def _sensitivity_payload(self, *active_names: str) -> dict[str, object]:
        active = set(active_names)
        return {
            "source_type": "external_bundle",
            "claim_envelope": "benchmark_eval",
            "parameters": [{"parameter": name, "active": name in active} for name in PARAMETER_REFERENCE],
        }

    def test_blocked_without_external_bundle_when_smoke_is_disallowed(self) -> None:
        with patch("segmentum.m44_cross_task._resolve_benchmark_root", return_value=None):
            payload = run_m44_cross_task_suite(benchmark_root=None, allow_smoke_test=False)
        self.assertTrue(payload["blocked"])
        self.assertEqual(payload["acceptance_state"], "blocked_missing_external_bundle")

    def test_degradation_computation_correctness(self) -> None:
        payload = compute_degradation_matrix(
            confidence_specific_cell={"metrics": {"heldout_likelihood": -0.40}},
            igt_specific_cell={"metrics": {"igt_behavioral_similarity": 0.50}},
            joint_cell_confidence={"metrics": {"heldout_likelihood": -0.44}},
            joint_cell_igt={"metrics": {"igt_behavioral_similarity": 0.44}},
        )

        self.assertAlmostEqual(payload["confidence_joint_vs_specific"]["relative_degradation"], 0.10, places=6)
        self.assertAlmostEqual(payload["igt_joint_vs_specific"]["relative_degradation"], 0.12, places=6)
        self.assertTrue(payload["igt_joint_vs_specific"]["meaningful_degradation"])

    def test_smoke_joint_fit_produces_shared_parameter_vector(self) -> None:
        payload = fit_joint_parameters(
            seed=44,
            allow_smoke_test=True,
            sample_limits={"confidence_train_max_trials": 48, "confidence_validation_max_trials": 48, "confidence_heldout_max_trials": 48},
        )

        self.assertEqual(payload["claim_envelope"], "synthetic_diagnostic")
        self.assertIn("selected_parameters", payload)
        self.assertEqual(set(payload["selected_parameters"]), {"schema_version", "uncertainty_sensitivity", "error_aversion", "exploration_bias", "attention_selectivity", "confidence_gain", "update_rigidity", "resource_pressure_sensitivity", "virtual_prediction_error_gain"})
        self.assertIn("igt_behavioral_similarity", payload["heldout_metrics"]["igt"])

    def test_smoke_suite_reports_cross_task_outputs(self) -> None:
        payload = run_m44_cross_task_suite(
            seed=44,
            allow_smoke_test=True,
            sample_limits={"confidence_train_max_trials": 48, "confidence_validation_max_trials": 48, "confidence_heldout_max_trials": 48, "architecture_candidate_count": 4},
        )

        self.assertFalse(payload["blocked"])
        self.assertEqual(set(payload["degradation"]["cross_application_matrix"]), {"confidence_specific", "igt_specific", "joint"})
        self.assertEqual(len(payload["parameter_stability"]["parameters"]), 8)
        self.assertIn("igt_behavioral_similarity", payload["degradation"]["cross_application_matrix"]["joint"]["igt"]["metrics"])
        self.assertEqual(set(payload["weight_sensitivity"]["fits"]), {"default", "igt_heavy", "confidence_heavy"})
        self.assertEqual(
            next(row for row in payload["parameter_stability"]["parameters"] if row["parameter"] == "resource_pressure_sensitivity")["classification"],
            "inert",
        )

    def test_classification_uses_ablation_signal_not_raw_gap_alone(self) -> None:
        confidence_specific = CognitiveStyleParameters(uncertainty_sensitivity=0.0)
        igt_specific = CognitiveStyleParameters(uncertainty_sensitivity=0.65)
        joint_parameters = CognitiveStyleParameters(uncertainty_sensitivity=0.30)

        def fake_fixed_fit(**kwargs: object) -> dict[str, object]:
            selected = CognitiveStyleParameters.from_dict(
                {**kwargs["start_parameters"].to_dict(), **kwargs["fixed_values"]}  # type: ignore[index]
            )
            return {"parameters": selected, "selected_parameters": selected.to_dict()}

        def fake_confidence_payload(
            trials: list[object],
            parameters: CognitiveStyleParameters,
            *,
            seed: int,
            include_predictions: bool = False,
        ) -> dict[str, object]:
            metric = -0.400 if parameters.uncertainty_sensitivity < 0.05 else -0.398
            return {"metrics": {"heldout_likelihood": metric}}

        def fake_igt_payload(
            trials: list[object],
            parameters: CognitiveStyleParameters,
            *,
            seed: int,
            include_predictions: bool = False,
        ) -> dict[str, object]:
            metric = 0.500 if parameters.uncertainty_sensitivity > 0.60 else 0.490
            return {"metrics": {"igt_behavioral_similarity": metric}}

        with (
            patch("segmentum.m44_cross_task._task_fit_with_fixed_parameter", side_effect=fake_fixed_fit),
            patch("segmentum.m44_cross_task._confidence_payload", side_effect=fake_confidence_payload),
            patch("segmentum.m44_cross_task._igt_payload", side_effect=fake_igt_payload),
        ):
            payload = classify_parameter_stability(
                confidence_train=[object()],
                confidence_validation=[object()],
                confidence_heldout=[object()],
                igt_train=[object()],
                igt_validation=[object()],
                igt_heldout=[object()],
                confidence_specific=confidence_specific,
                igt_specific=igt_specific,
                joint_parameters=joint_parameters,
                sensitivity_payload=self._sensitivity_payload("uncertainty_sensitivity"),
                seed=44,
            )

        target = next(row for row in payload["parameters"] if row["parameter"] == "uncertainty_sensitivity")
        self.assertEqual(target["classification"], "stable")
        self.assertEqual(target["evidence"]["gap"], 0.65)
        self.assertEqual(target["evidence"]["rationale"], "two_task_tolerance")

    def test_classification_matches_heldout_seeds_between_anchor_and_ablation(self) -> None:
        confidence_specific = CognitiveStyleParameters(uncertainty_sensitivity=0.0)
        igt_specific = CognitiveStyleParameters(uncertainty_sensitivity=0.65)
        joint_parameters = CognitiveStyleParameters(uncertainty_sensitivity=0.30)
        confidence_seeds: list[int] = []
        igt_seeds: list[int] = []

        def fake_fixed_fit(**kwargs: object) -> dict[str, object]:
            selected = CognitiveStyleParameters.from_dict(
                {**kwargs["start_parameters"].to_dict(), **kwargs["fixed_values"]}  # type: ignore[index]
            )
            return {"parameters": selected, "selected_parameters": selected.to_dict()}

        def fake_confidence_payload(
            trials: list[object],
            parameters: CognitiveStyleParameters,
            *,
            seed: int,
            include_predictions: bool = False,
        ) -> dict[str, object]:
            confidence_seeds.append(seed)
            return {"metrics": {"heldout_likelihood": -0.40}}

        def fake_igt_payload(
            trials: list[object],
            parameters: CognitiveStyleParameters,
            *,
            seed: int,
            include_predictions: bool = False,
        ) -> dict[str, object]:
            igt_seeds.append(seed)
            return {"metrics": {"igt_behavioral_similarity": 0.50}}

        with (
            patch("segmentum.m44_cross_task._task_fit_with_fixed_parameter", side_effect=fake_fixed_fit),
            patch("segmentum.m44_cross_task._confidence_payload", side_effect=fake_confidence_payload),
            patch("segmentum.m44_cross_task._igt_payload", side_effect=fake_igt_payload),
        ):
            classify_parameter_stability(
                confidence_train=[object()],
                confidence_validation=[object()],
                confidence_heldout=[object()],
                igt_train=[object()],
                igt_validation=[object()],
                igt_heldout=[object()],
                confidence_specific=confidence_specific,
                igt_specific=igt_specific,
                joint_parameters=joint_parameters,
                sensitivity_payload=self._sensitivity_payload("uncertainty_sensitivity"),
                seed=44,
            )

        self.assertEqual(confidence_seeds, [245, 245])
        self.assertEqual(igt_seeds, [345, 345])

    def test_fixed_parameter_refit_uses_same_training_seed_for_current_and_candidates(self) -> None:
        seen_calls: list[tuple[tuple[object, ...], int]] = []

        def fake_confidence_payload(
            trials: list[object],
            parameters: CognitiveStyleParameters,
            *,
            seed: int,
            include_predictions: bool = False,
        ) -> dict[str, object]:
            seen_calls.append((tuple(trials), seed))
            return {"metrics": {"heldout_likelihood": -0.40}}

        with (
            patch("segmentum.m44_cross_task._confidence_payload", side_effect=fake_confidence_payload),
            patch("segmentum.m44_cross_task._score_confidence_metrics", return_value=0.0),
        ):
            _task_fit_with_fixed_parameter(
                task_id="confidence",
                training_trials=["train"],
                validation_trials=["validation"],
                seed=44,
                fixed_values={},
                start_parameters=CognitiveStyleParameters(),
            )

        training_seeds = {seed for trials, seed in seen_calls if trials == ("train",)}
        validation_seeds = {seed for trials, seed in seen_calls if trials == ("validation",)}
        self.assertEqual(training_seeds, {44})
        self.assertEqual(validation_seeds, {244})

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_external_suite_uses_real_bundle_and_reports_required_artifacts(self) -> None:
        payload = fit_joint_parameters(
            seed=44,
            benchmark_root=default_acceptance_benchmark_root(),
            sample_limits=SMALL_EXTERNAL_LIMITS,
        )

        self.assertEqual(payload["source_type"], "external_bundle")
        self.assertEqual(payload["claim_envelope"], "benchmark_eval")
        self.assertGreaterEqual(payload["training_trial_count"]["confidence"], 1000)
        self.assertGreaterEqual(payload["heldout_trial_count"]["igt"], 300)
        self.assertIn("igt_behavioral_similarity", payload["heldout_metrics"]["igt"])


if __name__ == "__main__":
    unittest.main()
