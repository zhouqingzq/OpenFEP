from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from segmentum.m44_audit import (
    M44_ARCHITECTURE_PATH,
    M44_DEGRADATION_PATH,
    M44_IGT_AGGREGATE_PATH,
    M44_JOINT_FIT_PATH,
    M44_PARAMETER_STABILITY_PATH,
    M44_REPORT_PATH,
    M44_SUMMARY_PATH,
    M44_WEIGHT_SENSITIVITY_PATH,
    _evaluate_acceptance,
    write_m44_acceptance_artifacts,
)
from segmentum.m44_cross_task import run_m44_cross_task_suite
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


class TestM44Acceptance(unittest.TestCase):
    def _official_output_snapshot(self) -> dict[Path, bytes | None]:
        tracked_paths = (
            Path(M44_REPORT_PATH),
            Path(M44_SUMMARY_PATH),
            Path(M44_JOINT_FIT_PATH),
            Path(M44_DEGRADATION_PATH),
            Path(M44_PARAMETER_STABILITY_PATH),
            Path(M44_WEIGHT_SENSITIVITY_PATH),
            Path(M44_IGT_AGGREGATE_PATH),
            Path(M44_ARCHITECTURE_PATH),
        )
        return {path: path.read_bytes() if path.exists() else None for path in tracked_paths}

    def _assert_official_outputs_unchanged(self, snapshot: dict[Path, bytes | None]) -> None:
        for path, original_bytes in snapshot.items():
            if original_bytes is None:
                self.assertFalse(path.exists(), str(path))
            else:
                self.assertEqual(path.read_bytes(), original_bytes, str(path))

    def _read_json(self, path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _assert_json_outputs_readable(self, outputs: dict[str, str]) -> None:
        for key in ("joint_fit", "degradation", "parameter_stability", "weight_sensitivity", "igt_aggregate", "architecture_assessment", "report"):
            payload = self._read_json(outputs[key])
            self.assertIsInstance(payload, dict, key)

    def _assert_gate_shapes(self, report: dict[str, object]) -> None:
        gates = dict(report["gates"])
        for gate_name, gate in gates.items():
            self.assertIn("passed", gate, gate_name)
            self.assertIn("blocking", gate, gate_name)
            self.assertIsInstance(gate.get("evidence"), dict, gate_name)
            self.assertTrue(gate["evidence"], gate_name)

    def test_missing_external_bundle_is_honestly_blocked(self) -> None:
        official_snapshot = self._official_output_snapshot()

        with TemporaryDirectory() as tmpdir:
            with patch("segmentum.m44_audit.default_acceptance_benchmark_root", return_value=None), patch(
                "segmentum.m44_cross_task._resolve_benchmark_root",
                return_value=None,
            ):
                outputs = write_m44_acceptance_artifacts(sample_limits=SMALL_EXTERNAL_LIMITS, output_root=tmpdir)
            report = self._read_json(outputs["report"])
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), str(path))
                self.assertTrue(Path(path).is_relative_to(Path(tmpdir).resolve()))

        self._assert_gate_shapes(report)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["acceptance_state"], "blocked_missing_external_bundle")
        self.assertEqual(report["recommendation"], "BLOCK")
        self.assertFalse(report["gates"]["joint_fit_exists"]["passed"])
        self.assertFalse(report["gates"]["degradation_bounded"]["passed"])
        self.assertTrue(report["gates"]["report_honesty"]["passed"])
        self._assert_official_outputs_unchanged(official_snapshot)

    @unittest.skipUnless(
        default_acceptance_benchmark_root() is not None and os.environ.get("SEGMENTUM_RUN_SLOW_EXTERNAL") == "1",
        "external acceptance bundle and SEGMENTUM_RUN_SLOW_EXTERNAL=1 required",
    )
    def test_external_bundle_report_matches_recomputed_suite_truth(self) -> None:
        official_snapshot = self._official_output_snapshot()

        with TemporaryDirectory() as tmpdir:
            outputs = write_m44_acceptance_artifacts(
                sample_limits=SMALL_EXTERNAL_LIMITS,
                output_root=tmpdir,
                benchmark_root=default_acceptance_benchmark_root(),
            )
            self._assert_json_outputs_readable(outputs)
            report = self._read_json(outputs["report"])
            joint_fit_artifact = self._read_json(outputs["joint_fit"])
            joint_fit_text = Path(outputs["joint_fit"]).read_text(encoding="utf-8")
            suite = run_m44_cross_task_suite(
                seed=44,
                benchmark_root=default_acceptance_benchmark_root(),
                sample_limits=SMALL_EXTERNAL_LIMITS,
            )
            evaluation = _evaluate_acceptance(suite)
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), str(path))
                self.assertTrue(Path(path).is_relative_to(Path(tmpdir).resolve()))

        self._assert_gate_shapes(report)
        self.assertEqual(report["status"], evaluation["status"])
        self.assertEqual(report["acceptance_state"], evaluation["acceptance_state"])
        self.assertEqual(report["gates"], evaluation["gates"])
        self.assertEqual(report["failed_gates"], evaluation["failed_gates"])
        self.assertEqual(report["findings"], evaluation["findings"])
        self.assertEqual(report["headline_metrics"], evaluation["headline_metrics"])
        self.assertEqual(report["recommendation"], evaluation["recommendation"])
        self.assertIn("joint_degradation", report["headline_metrics"])
        self.assertIn("stable_parameter_count", report["headline_metrics"])
        self.assertIn("task_sensitive_count", report["headline_metrics"])
        self.assertIsInstance(joint_fit_artifact.get("parameters"), dict)
        self.assertIsInstance(joint_fit_artifact.get("selected_parameters"), dict)
        self.assertNotIn("CognitiveStyleParameters", joint_fit_text)
        self.assertIn(f"Status: `{report['status']}`", summary)
        self.assertIn(f"Recommendation: `{report['recommendation']}`", summary)
        self._assert_official_outputs_unchanged(official_snapshot)


if __name__ == "__main__":
    unittest.main()
