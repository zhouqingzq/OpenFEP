from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m45_audit import (
    M45_ABLATION_PATH,
    M45_CANONICAL_TRACE_PATH,
    M45_FAILURE_INJECTION_PATH,
    M45_REPORT_PATH,
    M45_SUMMARY_PATH,
    _evaluate_acceptance,
    write_m45_acceptance_artifacts,
)
from segmentum.m45_acceptance_data import REGRESSION_TARGETS, build_m45_acceptance_payload


def _passing_regression_summary(*, files: list[str] | None = None) -> dict[str, object]:
    active_files = list(REGRESSION_TARGETS if files is None else files)
    return {
        "executed": True,
        "files": active_files,
        "returncode": 0,
        "passed": True,
        "passed_count": len(active_files),
        "duration_seconds": 0.1,
        "summary_line": f"{len(active_files)} passed in 0.10s",
        "stdout_tail": [f"{len(active_files)} passed in 0.10s"],
    }


class TestM45Acceptance(unittest.TestCase):
    def _read_json(self, path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _assert_report_matches_evaluation(
        self,
        report: dict[str, object],
        evaluation: dict[str, object],
    ) -> None:
        self.assertEqual(report["status"], evaluation["status"])
        self.assertEqual(report["acceptance_state"], evaluation["acceptance_state"])
        self.assertEqual(report["recommendation"], evaluation["recommendation"])
        self.assertEqual(report["gates"], evaluation["gates"])
        self.assertEqual(report["failed_gates"], evaluation["failed_gates"])
        self.assertEqual(report["headline_metrics"], evaluation["headline_metrics"])

    def _report_truth_mismatches(
        self,
        report: dict[str, object],
        evaluation: dict[str, object],
    ) -> list[str]:
        mismatches: list[str] = []
        for field in ("status", "acceptance_state", "recommendation", "gates", "failed_gates", "headline_metrics"):
            if report.get(field) != evaluation.get(field):
                mismatches.append(field)
        return mismatches

    def _official_payload(self) -> dict[str, object]:
        return {
            **self._read_json(str(M45_CANONICAL_TRACE_PATH)),
            "ablation": self._read_json(str(M45_ABLATION_PATH)),
            "failure_injection": self._read_json(str(M45_FAILURE_INJECTION_PATH)),
        }

    def test_acceptance_artifacts_match_gate_truth(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m45_acceptance_artifacts(output_root=tmpdir, round_started_at="2026-04-07T00:00:00+00:00")
            report = self._read_json(outputs["report"])
            rebuilt = {
                **self._read_json(outputs["canonical_trace"]),
                "ablation": self._read_json(outputs["ablation"]),
                "failure_injection": self._read_json(outputs["failure_injection"]),
            }
            evaluation = _evaluate_acceptance(rebuilt)
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self._assert_report_matches_evaluation(report, evaluation)
        self.assertIn("legacy_bridge", report["gates"])
        self.assertIn("report_honesty", report["gates"])
        self.assertIn("regression_passed", report["headline_metrics"])
        self.assertEqual(report["tests"]["regression"], REGRESSION_TARGETS)
        self.assertIn(f"Status: `{report['status']}`", summary)
        self.assertIn(f"Recommendation: `{report['recommendation']}`", summary)
        if report["status"] == "FAIL":
            self.assertIn("## 未完成项与风险说明", summary)
            self.assertTrue(report["residual_risks"])

    def test_regression_targets_cover_all_existing_m41_to_m44_tests(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected = [
            path.relative_to(repo_root).as_posix()
            for prefix in ("m41", "m42", "m43", "m44")
            for path in sorted((repo_root / "tests").glob(f"test_{prefix}*.py"))
        ]

        self.assertEqual(REGRESSION_TARGETS, expected)

    def test_official_artifacts_match_current_evaluator_truth(self) -> None:
        for path in (
            Path(M45_CANONICAL_TRACE_PATH),
            Path(M45_ABLATION_PATH),
            Path(M45_FAILURE_INJECTION_PATH),
            Path(M45_REPORT_PATH),
            Path(M45_SUMMARY_PATH),
        ):
            self.assertTrue(path.exists(), str(path))

        report = self._read_json(str(M45_REPORT_PATH))
        payload = self._official_payload()
        evaluation = _evaluate_acceptance(payload)
        summary = Path(M45_SUMMARY_PATH).read_text(encoding="utf-8")

        self._assert_report_matches_evaluation(report, evaluation)
        self.assertEqual(report["tests"]["regression"], REGRESSION_TARGETS)
        self.assertIn(f"Status: `{report['status']}`", summary)
        self.assertIn(f"Recommendation: `{report['recommendation']}`", summary)
        if report["status"] == "FAIL":
            self.assertIn("## 未完成项与风险说明", summary)
            self.assertTrue(report["residual_risks"])

    def test_integration_probe_can_flip_gate_even_with_passing_boundary_probe(self) -> None:
        payload = build_m45_acceptance_payload(regression_summary=_passing_regression_summary())
        payload["integration_probes"]["encoding_integration"]["observed"]["identity_store_level"] = "short"
        payload["integration_probes"]["encoding_integration"]["observed"]["noise_store_level"] = "short"

        evaluation = _evaluate_acceptance(payload)

        self.assertEqual(evaluation["status"], "FAIL")
        self.assertIn("encoding_pipeline", evaluation["failed_gates"])

    def test_narrow_regression_summary_fails_legacy_bridge_and_report_honesty(self) -> None:
        narrowed_files = list(REGRESSION_TARGETS[:4]) if len(REGRESSION_TARGETS) > 4 else list(REGRESSION_TARGETS[:-1])
        payload = build_m45_acceptance_payload(regression_summary=_passing_regression_summary(files=narrowed_files))

        evaluation = _evaluate_acceptance(payload)

        self.assertEqual(evaluation["status"], "FAIL")
        self.assertIn("legacy_bridge", evaluation["failed_gates"])
        self.assertIn("report_honesty", evaluation["failed_gates"])

    def test_fake_pass_report_is_detected_as_inconsistent_with_evaluator_truth(self) -> None:
        payload = build_m45_acceptance_payload(regression_summary=_passing_regression_summary())
        payload["boundary_probes"]["decay_boundary"]["observed"]["forgetting_paths"]["dormant_marked"] = []

        evaluation = _evaluate_acceptance(payload)
        fake_report = {
            "status": "PASS",
            "acceptance_state": "acceptance_pass",
            "recommendation": "ACCEPT",
            "gates": evaluation["gates"],
            "failed_gates": [],
            "headline_metrics": evaluation["headline_metrics"],
        }

        self.assertEqual(evaluation["status"], "FAIL")
        self.assertIn("dual_decay_correctness", evaluation["failed_gates"])
        self.assertTrue(self._report_truth_mismatches(fake_report, evaluation))

    def test_dual_decay_boundary_probe_marks_dormant_entry_on_real_decay_path(self) -> None:
        payload = build_m45_acceptance_payload(regression_summary=_passing_regression_summary())

        evaluation = _evaluate_acceptance(payload)
        forgetting_paths = evaluation["gates"]["dual_decay_correctness"]["evidence"]["boundary"]["forgetting_paths"]

        self.assertTrue(forgetting_paths["dormant_marked"])
        self.assertTrue(forgetting_paths["deleted_short_residue"])
        self.assertTrue(forgetting_paths["abstracted_entries"])
        self.assertTrue(forgetting_paths["source_confidence_drifted"])
        self.assertTrue(forgetting_paths["reality_confidence_drifted"])

    def test_store_transition_probe_requires_identity_delta_and_neutral_guardrail(self) -> None:
        payload = build_m45_acceptance_payload(regression_summary=_passing_regression_summary())
        probe = payload["integration_probes"]["store_transitions_integration"]["observed"]

        self.assertGreater(float(probe["identity_score_delta"]), 0.0)
        self.assertEqual(probe["identity_null_store_level"], "short")
        self.assertLess(float(probe["neutral_promotion_rate"]), 0.05)

    def test_probe_catalog_mismatch_fails_report_honesty(self) -> None:
        payload = build_m45_acceptance_payload(regression_summary=_passing_regression_summary())
        payload["probe_catalog"]["integration"] = payload["probe_catalog"]["integration"][:-1]

        evaluation = _evaluate_acceptance(payload)

        self.assertEqual(evaluation["status"], "FAIL")
        self.assertIn("report_honesty", evaluation["failed_gates"])


if __name__ == "__main__":
    unittest.main()
