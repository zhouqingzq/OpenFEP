from __future__ import annotations

import unittest

from segmentum.m4_benchmarks import (
    DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
    EXTERNAL_BENCHMARK_ROOT,
    ConfidenceDatabaseAdapter,
    IowaGamblingTaskAdapter,
    STANDARD_IGT_TRIAL_COUNT,
    task_adapter_registry,
    preprocess_confidence_database,
    run_iowa_gambling_benchmark,
    run_task_adapter,
)


class TestM42BenchmarkAdapter(unittest.TestCase):
    def test_registry_exposes_third_task_without_core_special_case(self) -> None:
        registry = task_adapter_registry()
        self.assertIn("two_armed_bandit", registry)
        result = run_task_adapter("two_armed_bandit", seed=12, trial_count=12, include_predictions=False, include_subject_summary=False)
        self.assertEqual(result["trial_count"], 12)
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

    def test_smoke_igt_fixture_is_rejected_as_standard_protocol(self) -> None:
        with self.assertRaises(ValueError):
            run_iowa_gambling_benchmark(seed=44, allow_smoke_test=True, include_predictions=False, include_subject_summary=False)

    @unittest.skipUnless((EXTERNAL_BENCHMARK_ROOT / "confidence_database" / "manifest.json").exists(), "external bundle required")
    def test_confidence_external_dataset_preprocesses_as_acceptance_grade(self) -> None:
        payload = preprocess_confidence_database(
            benchmark_root=EXTERNAL_BENCHMARK_ROOT,
            selected_source_dataset=DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
        )
        self.assertEqual(payload["bundle_mode"], "external_bundle")
        self.assertEqual(payload["claim_envelope"], "benchmark_eval")
        self.assertGreater(payload["trial_count"], 1000)


if __name__ == "__main__":
    unittest.main()
