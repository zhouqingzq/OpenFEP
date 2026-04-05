from __future__ import annotations

import math
import unittest

from segmentum.m4_benchmarks import (
    DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
    EXTERNAL_BENCHMARK_ROOT,
    ConfidenceDatabaseAdapter,
    IowaGamblingTaskAdapter,
    STANDARD_IGT_TRIAL_COUNT,
    TwoArmedBanditAdapter,
    preprocess_iowa_gambling_task,
    task_adapter_registry,
    preprocess_confidence_database,
    run_iowa_gambling_benchmark,
    run_task_adapter,
)
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM42BenchmarkAdapter(unittest.TestCase):
    def test_registry_exposes_third_task_without_core_special_case(self) -> None:
        registry = task_adapter_registry()
        self.assertIn("two_armed_bandit", registry)
        result = run_task_adapter("two_armed_bandit", seed=12, trial_count=12, include_predictions=False, include_subject_summary=False)
        self.assertEqual(result["trial_count"], 12)
        self.assertEqual(result["claim_envelope"], "smoke_only")
        self.assertEqual(result["bundle_mode"], "synthetic_protocol")
        self.assertEqual(result["benchmark_status"]["benchmark_state"], "smoke_only")
        self.assertFalse(result["benchmark_status"]["acceptance_ready"])
        self.assertTrue(result["trial_export_validation"]["ok"])

    def test_confidence_schema_exposes_human_aligned_export(self) -> None:
        schema = ConfidenceDatabaseAdapter().schema()
        self.assertEqual(schema["benchmark_id"], "confidence_database")
        self.assertIn("trial_export_schema", schema)
        self.assertIn("agent_choice", schema["trial_export_schema"]["required"])

    def test_igt_schema_declares_standard_protocol(self) -> None:
        schema = IowaGamblingTaskAdapter().schema()
        self.assertEqual(schema["benchmark_id"], "iowa_gambling_task")
        self.assertEqual(schema["standard_protocol"]["trial_count"], STANDARD_IGT_TRIAL_COUNT)

    def test_bandit_schema_is_consistently_smoke_only(self) -> None:
        schema = TwoArmedBanditAdapter().schema()
        self.assertEqual(schema["benchmark_id"], "two_armed_bandit")
        self.assertEqual(schema["status"], "smoke_only")
        self.assertEqual(schema["benchmark_state"], "smoke_only")
        self.assertEqual(schema["source_type"], "synthetic_protocol")
        self.assertEqual(schema["bundle_mode"], "synthetic_protocol")
        self.assertEqual(schema["claim_envelope"], "smoke_only")
        self.assertTrue(schema["smoke_test_only"])
        self.assertTrue(schema["is_synthetic"])

    def test_smoke_igt_fixture_is_rejected_as_standard_protocol(self) -> None:
        with self.assertRaises(ValueError):
            run_iowa_gambling_benchmark(seed=44, allow_smoke_test=True, include_predictions=False, include_subject_summary=False)

    def test_confidence_smoke_predictions_use_decision_probabilities_not_stimulus_scaffold(self) -> None:
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
        low_payload = run_task_adapter(
            "confidence_database",
            parameters=low_parameters,
            seed=43,
            allow_smoke_test=True,
            max_trials=12,
        )
        high_payload = run_task_adapter(
            "confidence_database",
            parameters=high_parameters,
            seed=43,
            allow_smoke_test=True,
            max_trials=12,
        )
        low_probabilities = [float(row["predicted_probability_right"]) for row in low_payload["predictions"]]
        high_probabilities = [float(row["predicted_probability_right"]) for row in high_payload["predictions"]]
        scaffold_probabilities = [
            round(1.0 / (1.0 + math.exp(-float(row["stimulus_strength"]) * 4.0)), 6)
            for row in low_payload["trial_trace"]
        ]

        self.assertNotEqual(low_probabilities, scaffold_probabilities)
        self.assertTrue(
            any(abs(low_value - high_value) > 1e-6 for low_value, high_value in zip(low_probabilities, high_probabilities))
        )

    @unittest.skipUnless((EXTERNAL_BENCHMARK_ROOT / "confidence_database" / "manifest.json").exists(), "external bundle required")
    def test_confidence_external_dataset_preprocesses_as_acceptance_grade(self) -> None:
        payload = preprocess_confidence_database(
            benchmark_root=EXTERNAL_BENCHMARK_ROOT,
            selected_source_dataset=DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
        )
        self.assertEqual(payload["bundle_mode"], "external_bundle")
        self.assertEqual(payload["claim_envelope"], "benchmark_eval")
        self.assertGreater(payload["trial_count"], 1000)
        self.assertTrue(payload["leakage_check"]["ok"])
        self.assertTrue(payload["leakage_check"]["subject"]["ok"])
        self.assertTrue(payload["leakage_check"]["session"]["ok"])

    @unittest.skipUnless((EXTERNAL_BENCHMARK_ROOT / "iowa_gambling_task" / "manifest.json").exists(), "external bundle required")
    def test_igt_external_dataset_exposes_leakage_check(self) -> None:
        payload = preprocess_iowa_gambling_task(
            benchmark_root=EXTERNAL_BENCHMARK_ROOT,
        )
        self.assertEqual(payload["bundle_mode"], "external_bundle")
        self.assertEqual(payload["claim_envelope"], "benchmark_eval")
        self.assertIn("leakage_check", payload)
        self.assertTrue(payload["leakage_check"]["ok"])
        self.assertTrue(payload["leakage_check"]["subject"]["ok"])


if __name__ == "__main__":
    unittest.main()
