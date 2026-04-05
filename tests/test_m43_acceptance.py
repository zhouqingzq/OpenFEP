from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from segmentum.m43_audit import (
    M43_CONFIDENCE_PATH,
    M43_FAILURE_PATH,
    M43_IGT_PATH,
    M43_PARAMETER_SENSITIVITY_PATH,
    M43_REPORT_PATH,
    M43_SUMMARY_PATH,
    _evaluate_acceptance,
    write_m43_acceptance_artifacts,
)
from segmentum.m4_benchmarks import default_acceptance_benchmark_root


SMALL_EXTERNAL_LIMITS = {
    "confidence_train_max_trials": 1500,
    "confidence_validation_max_trials": 1500,
    "confidence_heldout_max_trials": 2000,
    "sensitivity_confidence_max_trials": 800,
}


def _metric(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _competitive_parity(agent_metrics: dict[str, object], baseline_metrics: dict[str, object], *, primary_metric: str) -> bool:
    agent_primary = _metric(agent_metrics, primary_metric)
    baseline_primary = _metric(baseline_metrics, primary_metric)
    if agent_primary is None or baseline_primary is None:
        return False
    if agent_primary >= baseline_primary:
        return True
    baseline_brier = _metric(baseline_metrics, "brier_score") or 0.0
    agent_brier = _metric(agent_metrics, "brier_score")
    return bool(baseline_brier > 0.0 and agent_brier is not None and agent_brier <= baseline_brier * 1.05)


def _relative_worse_than(agent_value: float | None, baseline_value: float | None, *, tolerance: float = 0.15) -> bool:
    if agent_value is None or baseline_value is None or agent_value >= baseline_value:
        return False
    return abs(agent_value - baseline_value) / max(abs(baseline_value), 1e-6) > tolerance


class TestM43Acceptance(unittest.TestCase):
    def _official_output_snapshot(self) -> dict[Path, bytes | None]:
        tracked_paths = (
            Path(M43_REPORT_PATH),
            Path(M43_SUMMARY_PATH),
            Path(M43_CONFIDENCE_PATH),
            Path(M43_IGT_PATH),
            Path(M43_PARAMETER_SENSITIVITY_PATH),
            Path(M43_FAILURE_PATH),
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

    def _rebuild_payload(self, outputs: dict[str, str]) -> dict[str, object]:
        return {
            "blocked": False,
            "confidence": self._read_json(outputs["confidence_fit"]),
            "igt": self._read_json(outputs["igt_fit"]),
            "parameter_sensitivity": self._read_json(outputs["parameter_sensitivity"]),
            "failure_analysis": self._read_json(outputs["failure_modes"]),
        }

    def _assert_gate_shapes(self, report: dict[str, object]) -> None:
        gates = dict(report["gates"])
        for gate_name, gate in gates.items():
            self.assertIn("passed", gate, gate_name)
            self.assertIn("blocking", gate, gate_name)
            self.assertIsInstance(gate.get("evidence"), dict, gate_name)
            self.assertTrue(gate["evidence"], gate_name)

    def _assert_payload_derived_gate_truth(self, payload: dict[str, object], report: dict[str, object]) -> None:
        confidence = dict(payload["confidence"])
        igt = dict(payload["igt"])
        sensitivity = dict(payload["parameter_sensitivity"])
        confidence_metrics = dict(confidence["metrics"])
        igt_metrics = dict(igt["metrics"])
        confidence_subject_count = int(confidence["subject_summary"]["subject_count"])
        igt_subject_count = int(igt["subject_summary"]["subject_count"])

        confidence_lower_expected = all(
            _metric(confidence_metrics, "heldout_likelihood") > _metric(dict(confidence["baselines"][name]["metrics"]), "heldout_likelihood")
            for name in ("random", "stimulus_only")
        )
        igt_lower_expected = all(
            _metric(igt_metrics, "deck_match_rate") > _metric(dict(igt["baselines"][name]["metrics"]), "deck_match_rate")
            for name in ("random", "frequency_matching")
        )
        confidence_competitive_expected = any(
            _competitive_parity(confidence_metrics, dict(confidence["baselines"][name]["metrics"]), primary_metric="heldout_likelihood")
            for name in ("statistical_logistic", "human_match_ceiling")
        )
        igt_competitive_expected = _competitive_parity(
            igt_metrics,
            dict(igt["baselines"]["human_behavior"]["metrics"]),
            primary_metric="deck_match_rate",
        )
        confidence_review_block_expected = all(
            _relative_worse_than(
                _metric(confidence_metrics, "heldout_likelihood"),
                _metric(dict(confidence["baselines"][name]["metrics"]), "heldout_likelihood"),
            )
            for name in ("statistical_logistic", "human_match_ceiling")
        )
        igt_review_block_expected = _relative_worse_than(
            _metric(igt_metrics, "deck_match_rate"),
            _metric(dict(igt["baselines"]["human_behavior"]["metrics"]), "deck_match_rate"),
        )

        sample_size_expected = (
            int(confidence["trial_count"]) >= 1000
            and confidence_subject_count >= 10
            and igt_subject_count >= 3
        )
        fit_confidence_expected = (
            confidence["mode"] == "benchmark_eval"
            and confidence["source_type"] == "external_bundle"
            and confidence["claim_envelope"] == "benchmark_eval"
            and confidence.get("external_validation", False) is False
            and int(confidence["trial_count"]) >= 1000
            and confidence_subject_count >= 10
            and confidence_lower_expected
            and any(name in confidence["baselines"] for name in ("statistical_logistic", "human_match_ceiling"))
        )
        fit_igt_expected = (
            igt["mode"] == "benchmark_eval"
            and igt["protocol_mode"] == "standard_100"
            and igt["source_type"] == "external_bundle"
            and igt["claim_envelope"] == "benchmark_eval"
            and igt.get("external_validation", False) is False
            and igt_subject_count >= 3
            and igt_lower_expected
            and "human_behavior" in igt["baselines"]
        )
        parameter_sensitivity_expected = (
            sensitivity["source_type"] == "external_bundle"
            and sensitivity["claim_envelope"] == "benchmark_eval"
            and sensitivity.get("external_validation", False) is False
            and len(sensitivity["parameters"]) == 8
            and int(sensitivity["active_parameter_count"]) >= int(sensitivity["required_active_parameter_count"])
        )
        baseline_ladder_expected = (
            set(("random", "stimulus_only", "statistical_logistic", "human_match_ceiling")) <= set(confidence["baselines"])
            and set(("random", "frequency_matching", "human_behavior")) <= set(igt["baselines"])
            and confidence_lower_expected == bool(confidence["baseline_ladder"]["lower_baselines_beaten"])
            and igt_lower_expected == bool(igt["baseline_ladder"]["lower_baselines_beaten"])
            and confidence_competitive_expected == bool(confidence["baseline_ladder"]["competitive_baseline_matched"])
            and igt_competitive_expected == bool(igt["baseline_ladder"]["competitive_baseline_matched"])
            and confidence_review_block_expected == bool(confidence["baseline_ladder"]["competitive_review_block"])
            and igt_review_block_expected == bool(igt["baseline_ladder"]["competitive_review_block"])
        )

        self.assertEqual(report["gates"]["sample_size_sufficient"]["passed"], sample_size_expected)
        self.assertEqual(report["gates"]["fit_confidence_db"]["passed"], fit_confidence_expected)
        self.assertEqual(report["gates"]["fit_igt"]["passed"], fit_igt_expected)
        self.assertEqual(report["gates"]["parameter_sensitivity"]["passed"], parameter_sensitivity_expected)
        self.assertEqual(report["gates"]["baseline_ladder"]["passed"], baseline_ladder_expected)

    def test_missing_external_bundle_is_honestly_blocked(self) -> None:
        official_snapshot = self._official_output_snapshot()

        with TemporaryDirectory() as tmpdir:
            with patch("segmentum.m43_audit.default_acceptance_benchmark_root", return_value=None):
                outputs = write_m43_acceptance_artifacts(sample_limits=SMALL_EXTERNAL_LIMITS, output_root=tmpdir)
            report = self._read_json(outputs["report"])
            self.assertTrue(all(Path(path).is_relative_to(Path(tmpdir).resolve()) for path in outputs.values()))
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), str(path))

        self._assert_gate_shapes(report)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["acceptance_state"], "blocked_missing_external_bundle")
        self.assertEqual(report["recommendation"], "BLOCK")
        self.assertEqual(report["tracks"]["confidence_database"]["status"], "blocked")
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["status"], "blocked")
        self.assertEqual(report["tracks"]["parameter_sensitivity"]["status"], "blocked")
        self.assertFalse(report["gates"]["fit_confidence_db"]["passed"])
        self.assertFalse(report["gates"]["fit_igt"]["passed"])
        self.assertFalse(report["gates"]["sample_size_sufficient"]["passed"])
        self.assertTrue(report["gates"]["no_synthetic_claims"]["passed"])
        self.assertIn("fit_confidence_db", report["failed_gates"])
        self.assertNotIn("no_synthetic_claims", report["failed_gates"])
        self.assertEqual(report["readiness"]["deployment_readiness"], "NOT_READY")
        self.assertFalse(report["headline_metrics"]["confidence_database"]["external_bundle"])
        self.assertIsNone(report["headline_metrics"]["confidence_database"]["claim_envelope"])
        self.assertEqual(report["failure_modes"], [])
        self._assert_official_outputs_unchanged(official_snapshot)

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_external_bundle_report_matches_real_payload_and_artifacts(self) -> None:
        official_snapshot = self._official_output_snapshot()

        with TemporaryDirectory() as tmpdir:
            outputs = write_m43_acceptance_artifacts(sample_limits=SMALL_EXTERNAL_LIMITS, output_root=tmpdir)
            report = self._read_json(outputs["report"])
            rebuilt_payload = self._rebuild_payload(outputs)
            expected_evaluation = _evaluate_acceptance(rebuilt_payload)
            failure_artifact = self._read_json(outputs["failure_modes"])
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")
            self.assertTrue(all(Path(path).is_relative_to(Path(tmpdir).resolve()) for path in outputs.values()))
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), str(path))

        self._assert_gate_shapes(report)
        self._assert_payload_derived_gate_truth(rebuilt_payload, report)
        self.assertEqual(report["status"], expected_evaluation["status"])
        self.assertEqual(report["acceptance_state"], expected_evaluation["acceptance_state"])
        self.assertEqual(report["gates"], expected_evaluation["gates"])
        self.assertEqual(report["failed_gates"], expected_evaluation["failed_gates"])
        self.assertEqual(report["findings"], expected_evaluation["findings"])
        self.assertEqual(report["headline_metrics"], expected_evaluation["headline_metrics"])
        self.assertEqual(report["recommendation"], expected_evaluation["recommendation"])

        confidence = rebuilt_payload["confidence"]
        igt = rebuilt_payload["igt"]
        sensitivity = rebuilt_payload["parameter_sensitivity"]
        self.assertEqual(report["tracks"]["confidence_database"]["status"], confidence["mode"])
        self.assertEqual(report["tracks"]["confidence_database"]["trial_count"], confidence["trial_count"])
        self.assertEqual(report["tracks"]["confidence_database"]["subject_count"], confidence["subject_summary"]["subject_count"])
        self.assertEqual(report["tracks"]["confidence_database"]["claim_envelope"], confidence["claim_envelope"])
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["status"], igt["mode"])
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["subject_count"], igt["subject_summary"]["subject_count"])
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["protocol_mode"], igt["protocol_mode"])
        self.assertEqual(report["tracks"]["parameter_sensitivity"]["active_parameter_count"], sensitivity["active_parameter_count"])
        self.assertEqual(report["tracks"]["parameter_sensitivity"]["claim_envelope"], sensitivity["claim_envelope"])
        self.assertEqual(report["failure_modes"], failure_artifact["failure_modes"])
        self.assertIn("baseline_parameters", sensitivity)
        self.assertIn("noise_floor", sensitivity)
        self.assertIn("analysis_protocol", sensitivity)
        sensitivity_rows = {str(row["parameter"]): row for row in sensitivity["parameters"]}
        self.assertFalse(bool(sensitivity_rows["resource_pressure_sensitivity"]["active"]))
        for row in sensitivity_rows.values():
            self.assertIn("classification_reason", row)
            self.assertIn("signal_to_noise", row)
            self.assertIn("winning_metric", row)
            self.assertIn("relevant_metrics", row)

        self.assertIn(f"Status: `{report['status']}`", summary)
        self.assertIn(f"Recommendation: `{report['recommendation']}`", summary)
        if report["status"] == "PASS":
            self.assertIn("PASS:", summary)
        else:
            self.assertIn("FAIL:", summary)

        self._assert_official_outputs_unchanged(official_snapshot)


if __name__ == "__main__":
    unittest.main()
