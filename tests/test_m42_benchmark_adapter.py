from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from segmentum.benchmark_registry import benchmark_status, import_benchmark_bundle, list_benchmark_bundles, load_benchmark_bundle, validate_benchmark_bundle
from segmentum.m4_benchmarks import ConfidenceDatabaseAdapter, IowaGamblingTaskAdapter, detect_subject_leakage, preprocess_confidence_database, preprocess_iowa_gambling_task, run_iowa_gambling_benchmark
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM42BenchmarkAdapter(unittest.TestCase):
    def test_preprocess_produces_all_required_splits(self) -> None:
        payload = preprocess_confidence_database(allow_smoke_test=True)
        splits = {row["split"] for row in payload["trials"]}
        self.assertEqual(payload["manifest"]["benchmark_id"], "confidence_database")
        self.assertEqual(splits, {"train", "validation", "heldout"})
        self.assertEqual(payload["claim_envelope"], "smoke_only")
        self.assertTrue(payload["leakage_check"]["subject"]["ok"])

    def test_protocol_schemas_exist_for_confidence_and_igt(self) -> None:
        confidence_schema = ConfidenceDatabaseAdapter().schema()
        igt_schema = IowaGamblingTaskAdapter().schema()
        self.assertEqual(confidence_schema["benchmark_id"], "confidence_database")
        self.assertEqual(igt_schema["benchmark_id"], "iowa_gambling_task")
        self.assertEqual(igt_schema["status"], "repo_smoke_fixture")
        self.assertTrue(confidence_schema["external_bundle_preferred"])
        self.assertEqual(confidence_schema["benchmark_state"], "blocked_missing_external_bundle")
        self.assertEqual(igt_schema["benchmark_state"], "blocked_missing_external_bundle")

    def test_benchmark_bundle_registry_exposes_provenance(self) -> None:
        confidence_bundle = load_benchmark_bundle("confidence_database")
        all_ids = {bundle.benchmark_id for bundle in list_benchmark_bundles()}
        self.assertEqual(confidence_bundle.source_type, "repo_curated_sample")
        self.assertTrue(confidence_bundle.smoke_test_only)
        self.assertIn("iowa_gambling_task", all_ids)

    def test_validation_reports_clean_repo_bundle(self) -> None:
        validation = validate_benchmark_bundle("confidence_database")
        self.assertTrue(validation.ok)
        self.assertEqual(validation.record_count_declared, validation.record_count_observed)
        self.assertTrue(validation.smoke_test_only)
        self.assertEqual(validation.benchmark_state, "blocked_missing_external_bundle")
        self.assertTrue(validation.blockers)

    def test_smoke_manifest_is_recognized_as_non_acceptance_ready(self) -> None:
        status = benchmark_status("confidence_database")
        self.assertEqual(status.benchmark_state, "blocked_missing_external_bundle")
        self.assertIn("smoke_only", status.available_states)
        self.assertFalse(status.acceptance_ready)
        self.assertTrue(status.blockers[0].strip())

    def test_import_external_bundle_requires_valid_record_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "incoming_confidence"
            source_dir.mkdir(parents=True)
            manifest = {
                "benchmark_id": "confidence_database",
                "version": "9.9.9",
                "record_count": 3,
                "data_file": "bundle.jsonl",
                "source_type": "external_bundle",
                "source_label": "temp_fixture",
                "grouping_fields": ["session_id", "subject_id"],
                "default_split_unit": "session_id",
                "fields": sorted(
                    [
                        "trial_id",
                        "subject_id",
                        "session_id",
                        "stimulus_strength",
                        "correct_choice",
                        "human_choice",
                        "human_confidence",
                        "rt_ms",
                    ]
                ),
            }
            (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source_dir / "bundle.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "ext_001",
                        "subject_id": "s_ext",
                        "session_id": "sess_a",
                        "stimulus_strength": 0.5,
                        "correct_choice": "right",
                        "human_choice": "right",
                        "human_confidence": 0.7,
                        "rt_ms": 800,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                import_benchmark_bundle(source_dir, destination_root=Path(tmp) / "registry")

    def test_import_external_bundle_succeeds_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "incoming_confidence"
            source_dir.mkdir(parents=True)
            manifest = {
                "benchmark_id": "confidence_database",
                "version": "1.2.3",
                "record_count": 3,
                "data_file": "bundle.jsonl",
                "source_type": "external_bundle",
                "source_label": "temp_fixture",
                "grouping_fields": ["session_id", "subject_id"],
                "default_split_unit": "session_id",
                "fields": sorted(
                    [
                        "trial_id",
                        "subject_id",
                        "session_id",
                        "stimulus_strength",
                        "correct_choice",
                        "human_choice",
                        "human_confidence",
                        "rt_ms",
                    ]
                ),
            }
            rows = [
                {
                    "trial_id": "ext_001",
                    "subject_id": "s_ext",
                    "session_id": "sess_a",
                    "stimulus_strength": 0.5,
                    "correct_choice": "right",
                    "human_choice": "right",
                    "human_confidence": 0.7,
                    "rt_ms": 800,
                },
                {
                    "trial_id": "ext_002",
                    "subject_id": "s_ext",
                    "session_id": "sess_b",
                    "stimulus_strength": -0.4,
                    "correct_choice": "left",
                    "human_choice": "left",
                    "human_confidence": 0.68,
                    "rt_ms": 820,
                },
                {
                    "trial_id": "ext_003",
                    "subject_id": "s_other",
                    "session_id": "sess_c",
                    "stimulus_strength": 0.3,
                    "correct_choice": "right",
                    "human_choice": "right",
                    "human_confidence": 0.63,
                    "rt_ms": 780,
                },
            ]
            (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source_dir / "bundle.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            imported = import_benchmark_bundle(source_dir, destination_root=Path(tmp) / "registry")
            self.assertEqual(imported.benchmark_id, "confidence_database")
            self.assertEqual(imported.source_type, "external_bundle")
            validation = validate_benchmark_bundle("confidence_database", root=Path(tmp) / "registry")
            self.assertTrue(validation.ok)
            status = benchmark_status("confidence_database", root=Path(tmp) / "registry")
            self.assertEqual(status.benchmark_state, "acceptance_ready")
            self.assertFalse(status.blockers)

    def test_import_igt_external_bundle_succeeds_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "incoming_igt"
            source_dir.mkdir(parents=True)
            manifest = {
                "benchmark_id": "iowa_gambling_task",
                "version": "1.0.0",
                "record_count": 6,
                "data_file": "igt_bundle.jsonl",
                "source_type": "external_bundle",
                "source_label": "igt_fixture",
                "grouping_fields": ["subject_id"],
                "default_split_unit": "subject_id",
                "fields": sorted(
                    [
                        "trial_id",
                        "subject_id",
                        "deck",
                        "reward",
                        "penalty",
                        "net_outcome",
                        "advantageous",
                        "trial_index",
                        "split",
                    ]
                ),
            }
            rows = [
                {"trial_id": "igt_001", "subject_id": "p01", "deck": "A", "reward": 100, "penalty": -150, "net_outcome": -50, "advantageous": False, "trial_index": 1, "split": "train"},
                {"trial_id": "igt_002", "subject_id": "p01", "deck": "C", "reward": 50, "penalty": -10, "net_outcome": 40, "advantageous": True, "trial_index": 2, "split": "train"},
                {"trial_id": "igt_003", "subject_id": "p02", "deck": "B", "reward": 100, "penalty": -125, "net_outcome": -25, "advantageous": False, "trial_index": 1, "split": "validation"},
                {"trial_id": "igt_004", "subject_id": "p02", "deck": "D", "reward": 50, "penalty": -5, "net_outcome": 45, "advantageous": True, "trial_index": 2, "split": "validation"},
                {"trial_id": "igt_005", "subject_id": "p03", "deck": "C", "reward": 50, "penalty": -15, "net_outcome": 35, "advantageous": True, "trial_index": 1, "split": "heldout"},
                {"trial_id": "igt_006", "subject_id": "p03", "deck": "D", "reward": 50, "penalty": -5, "net_outcome": 45, "advantageous": True, "trial_index": 2, "split": "heldout"},
            ]
            (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source_dir / "igt_bundle.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            registry_root = Path(tmp) / "registry"
            imported = import_benchmark_bundle(source_dir, destination_root=registry_root)
            self.assertEqual(imported.benchmark_id, "iowa_gambling_task")
            validation = validate_benchmark_bundle("iowa_gambling_task", root=registry_root)
            self.assertTrue(validation.ok)
            status = benchmark_status("iowa_gambling_task", root=registry_root)
            self.assertEqual(status.benchmark_state, "acceptance_ready")
            previous_root = os.environ.get("SEGMENTUM_BENCHMARK_ROOT")
            try:
                os.environ["SEGMENTUM_BENCHMARK_ROOT"] = str(registry_root)
                run = run_iowa_gambling_benchmark(CognitiveStyleParameters(), seed=44)
            finally:
                if previous_root is None:
                    os.environ.pop("SEGMENTUM_BENCHMARK_ROOT", None)
                else:
                    os.environ["SEGMENTUM_BENCHMARK_ROOT"] = previous_root
            self.assertEqual(run["benchmark_status"]["benchmark_state"], "acceptance_ready")
            self.assertEqual(run["claim_envelope"], "benchmark_eval")
            self.assertEqual(run["bundle_mode"], "external_bundle")
            self.assertEqual(run["trial_count"], 6)

    def test_repo_sample_is_not_default_benchmark_path(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_confidence_database()

    def test_igt_repo_sample_is_not_default_benchmark_path(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_iowa_gambling_task()

    def test_igt_smoke_fixture_can_run_with_explicit_opt_in(self) -> None:
        payload = preprocess_iowa_gambling_task(allow_smoke_test=True)
        self.assertEqual(payload["benchmark_status"]["benchmark_state"], "blocked_missing_external_bundle")
        self.assertEqual(payload["claim_envelope"], "smoke_only")
        self.assertTrue(payload["benchmark_status"]["blockers"][0].strip())

    def test_detect_subject_leakage_fails_for_bad_fixture(self) -> None:
        leakage = detect_subject_leakage(
            [
                {"subject_id": "s01", "session_id": "sess_1", "split": "train"},
                {"subject_id": "s01", "session_id": "sess_2", "split": "heldout"},
            ],
            key_field="subject_id",
        )
        self.assertFalse(leakage["ok"])

    def test_session_split_keeps_same_session_in_single_partition(self) -> None:
        payload = preprocess_confidence_database(allow_smoke_test=True)
        sessions = {}
        for row in payload["trials"]:
            sessions.setdefault(row["session_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in sessions.values()))


if __name__ == "__main__":
    unittest.main()
