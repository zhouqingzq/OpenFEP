from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from segmentum.m41_audit import M41_REPORT_PATH, write_m41_acceptance_artifacts
from segmentum.m4_cognitive_style import validate_acceptance_report


class TestM41Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.1")
        self.assertIn(report["status"], {"PASS", "FAIL"})
        self.assertEqual(report["analysis_scope"], "toy cognitive-style benchmark with falsifiable gates")
        self.assertIn("not a causal inference system", Path(report["artifacts"]["summary"]).read_text(encoding="utf-8"))
        blind_summary = Path(report["artifacts"]["blind_summary"]).read_text(encoding="utf-8")
        self.assertIn("Generator family", blind_summary)
        self.assertIn("External validation", blind_summary)
        for gate_name in (
            "schema_integrity",
            "trial_variation",
            "observability",
            "intervention_sensitivity",
            "blind_distinguishability",
            "log_completeness",
            "stress_behavior",
            "regression",
        ):
            self.assertIn(gate_name, report["gates"])
            self.assertIn("passed", report["gates"][gate_name])
            self.assertIn("evidence", report["gates"][gate_name])
        self.assertEqual(report["gates"]["intervention_sensitivity"]["evidence"]["analysis_type"], "intervention_sensitivity")
        self.assertEqual(report["gates"]["blind_distinguishability"]["evidence"]["analysis_type"], "toy_internal_distinguishability")
        self.assertEqual(report["gates"]["blind_distinguishability"]["evidence"]["generator_family"], "same_generator_family")
        self.assertFalse(report["gates"]["blind_distinguishability"]["evidence"]["external_validation"])
        self.assertIn("self_artifacts", report["gates"]["regression"]["evidence"])
        self.assertIn("dependencies", report["gates"]["regression"]["evidence"])
        self.assertTrue(report["report_validation"]["valid"], msg=report["report_validation"])

    def test_report_validation_rejects_gate_without_evidence(self) -> None:
        broken_report = {
            "status": "PASS",
            "findings": [],
            "gates": {
                "schema_integrity": {"passed": True},
                "trial_variation": {"passed": True, "evidence": {"ok": True}},
                "observability": {"passed": True, "evidence": {"ok": True}},
                "intervention_sensitivity": {"passed": True, "evidence": {"ok": True}},
                "blind_distinguishability": {"passed": True, "evidence": {"ok": True}},
                "log_completeness": {"passed": True, "evidence": {"ok": True}},
                "stress_behavior": {"passed": True, "evidence": {"ok": True}},
                "regression": {"passed": True, "evidence": {"ok": True}},
            },
        }
        validation = validate_acceptance_report(broken_report)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("missing_evidence" in item for item in validation["errors"]))

    def test_report_validation_rejects_pass_status_when_blocking_gate_failed(self) -> None:
        broken_report = {
            "status": "PASS",
            "findings": [],
            "gates": {
                "schema_integrity": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "trial_variation": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "observability": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "intervention_sensitivity": {
                    "passed": True,
                    "blocking": True,
                    "evidence": {"analysis_type": "intervention_sensitivity", "probes": {"x": {"analysis_type": "intervention_sensitivity"}}},
                },
                "blind_distinguishability": {
                    "passed": False,
                    "blocking": True,
                    "evidence": {
                        "analysis_type": "toy_internal_distinguishability",
                        "generator_family": "same_generator_family",
                        "external_validation": False,
                        "validation_limits": ["train/eval seed split only", "same generator family", "toy benchmark distinguishability"],
                        "train_eval_split": {"train_seeds": [1], "eval_seeds": [2]},
                    },
                },
                "log_completeness": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "stress_behavior": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "regression": {
                    "passed": True,
                    "blocking": True,
                    "evidence": {"self_artifacts": {"ok": True}, "dependencies": {"m35": {"passed": True}}},
                },
            },
        }
        validation = validate_acceptance_report(broken_report)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("status_mismatch:blocking_gate_failed" in item for item in validation["errors"]))

    def test_regression_gate_fails_when_dependency_evidence_is_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing_report = Path(tmpdir) / "missing_report.json"
            missing_summary = Path(tmpdir) / "missing_summary.md"
            missing_trace = Path(tmpdir) / "missing_trace.json"
            bad_dependency = {
                "report": str(missing_report),
                "summary": str(missing_summary),
                "trace": str(missing_trace),
            }
            with patch("segmentum.m41_audit.write_m35_acceptance_artifacts", return_value=bad_dependency):
                write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertFalse(report["gates"]["regression"]["passed"])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["recommendation"], "BLOCK")
        self.assertTrue(any(item["label"] == "regression_failed" for item in report["findings"]))

    def test_report_validation_rejects_misleading_causal_or_blind_claims(self) -> None:
        broken_report = {
            "status": "PASS",
            "findings": [],
            "gates": {
                "schema_integrity": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "trial_variation": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "observability": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "intervention_sensitivity": {
                    "passed": True,
                    "blocking": True,
                    "evidence": {"analysis_type": "causal_inference", "probes": {"x": {"analysis_type": "causal_inference"}}},
                },
                "blind_distinguishability": {
                    "passed": True,
                    "blocking": True,
                    "evidence": {
                        "analysis_type": "blind_validation",
                        "generator_family": "unknown",
                        "external_validation": True,
                        "validation_limits": [],
                        "train_eval_split": {},
                    },
                },
                "log_completeness": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "stress_behavior": {"passed": True, "blocking": True, "evidence": {"ok": True}},
                "regression": {
                    "passed": True,
                    "blocking": True,
                    "evidence": {"self_artifacts": {"ok": True}, "dependencies": {"m35": {"passed": True}}},
                },
            },
        }
        validation = validate_acceptance_report(broken_report)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("intervention_sensitivity:analysis_type_invalid" == item for item in validation["errors"]))
        self.assertTrue(any("blind_distinguishability:analysis_type_invalid" == item for item in validation["errors"]))
        self.assertTrue(any("blind_distinguishability:train_eval_split_incomplete" == item for item in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
