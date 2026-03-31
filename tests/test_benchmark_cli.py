from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from segmentum.benchmark_cli import main


class TestBenchmarkCli(unittest.TestCase):
    def test_list_command_outputs_registered_bundles(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["list"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        bundle_ids = {bundle["benchmark_id"] for bundle in payload["bundles"]}
        self.assertIn("confidence_database", bundle_ids)

    def test_validate_command_reports_clean_repo_bundle(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["validate", "confidence_database"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["benchmark_status"]["benchmark_state"], "blocked_missing_external_bundle")
        self.assertTrue(payload["benchmark_status"]["blockers"])

    def test_import_command_loads_valid_external_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "incoming_confidence"
            source_dir.mkdir(parents=True)
            manifest = {
                "benchmark_id": "confidence_database",
                "version": "2.0.0",
                "record_count": 2,
                "data_file": "bundle.jsonl",
                "source_type": "external_bundle",
                "source_label": "cli_fixture",
                "fields": sorted(
                    [
                        "trial_id",
                        "subject_id",
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
                    "stimulus_strength": 0.5,
                    "correct_choice": "right",
                    "human_choice": "right",
                    "human_confidence": 0.7,
                    "rt_ms": 800,
                },
                {
                    "trial_id": "ext_002",
                    "subject_id": "s_ext",
                    "stimulus_strength": -0.4,
                    "correct_choice": "left",
                    "human_choice": "left",
                    "human_confidence": 0.68,
                    "rt_ms": 820,
                },
            ]
            (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source_dir / "bundle.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["import", str(source_dir), "--root", str(Path(tmp) / "registry")])
            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["imported"]["benchmark_id"], "confidence_database")
            self.assertEqual(payload["benchmark_status"]["benchmark_state"], "acceptance_ready")

    def test_import_command_reports_errors_on_invalid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "incoming_confidence"
            source_dir.mkdir(parents=True)
            manifest = {
                "benchmark_id": "confidence_database",
                "version": "2.0.0",
                "record_count": 3,
                "data_file": "bundle.jsonl",
                "source_type": "external_bundle",
                "source_label": "cli_fixture",
                "fields": sorted(
                    [
                        "trial_id",
                        "subject_id",
                        "stimulus_strength",
                        "correct_choice",
                        "human_choice",
                        "human_confidence",
                        "rt_ms",
                    ]
                ),
            }
            row = {
                "trial_id": "ext_001",
                "subject_id": "s_ext",
                "stimulus_strength": 0.5,
                "correct_choice": "right",
                "human_choice": "right",
                "human_confidence": 0.7,
                "rt_ms": 800,
            }
            (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source_dir / "bundle.jsonl").write_text(json.dumps(row), encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["import", str(source_dir), "--root", str(Path(tmp) / "registry")])
            self.assertEqual(exit_code, 1)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["error"], "ValueError")


if __name__ == "__main__":
    unittest.main()
