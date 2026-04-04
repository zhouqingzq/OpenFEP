from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from segmentum.m42_audit import (
    M42_CONFIDENCE_TRACE_PATH,
    M42_IGT_TRACE_PATH,
    M42_LEAKAGE_PATH,
    M42_REPRO_PATH,
    M42_REPORT_PATH,
    write_m42_acceptance_artifacts,
)
from segmentum.m4_benchmarks import default_acceptance_benchmark_root


class TestM42Acceptance(unittest.TestCase):
    def test_missing_external_bundle_is_honestly_blocked(self) -> None:
        with patch("segmentum.m42_audit.default_acceptance_benchmark_root", return_value=None):
            write_m42_acceptance_artifacts()
            report = json.loads(Path(M42_REPORT_PATH).read_text(encoding="utf-8"))
            leakage_report = json.loads(Path(M42_LEAKAGE_PATH).read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["acceptance_state"], "blocked_missing_external_bundle")
        self.assertEqual(report["tracks"]["confidence_database"]["status"], "blocked")
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["status"], "blocked")
        self.assertEqual(report["tracks"]["two_armed_bandit"]["status"], "smoke-only")
        self.assertEqual(report["tracks"]["reproducibility"]["status"], "blocked")
        self.assertFalse(report["gate_summary"]["leakage_passed"])
        self.assertFalse(report["gate_summary"]["reproducibility_passed"])
        self.assertEqual(leakage_report["status"], "blocked")

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_formal_acceptance_report_requires_external_bundle_and_standard_100(self) -> None:
        write_m42_acceptance_artifacts()
        report = json.loads(Path(M42_REPORT_PATH).read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["acceptance_state"], "acceptance_pass")
        self.assertEqual(report["tracks"]["confidence_database"]["status"], "pass")
        self.assertEqual(report["tracks"]["confidence_database"]["leakage_status"], "pass")
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["status"], "pass")
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["protocol_mode"], "standard_100")
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["trial_count"], 100)
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["standard_trial_count"], 100)
        self.assertEqual(report["tracks"]["iowa_gambling_task"]["leakage_status"], "pass")
        self.assertEqual(report["tracks"]["two_armed_bandit"]["status"], "smoke-only")
        self.assertEqual(report["tracks"]["two_armed_bandit"]["benchmark_state"], "smoke_only")
        self.assertEqual(report["tracks"]["reproducibility"]["status"], "pass")
        self.assertEqual(report["tracks"]["reproducibility"]["protocol_mode"], "standard_100")
        self.assertGreater(report["tracks"]["reproducibility"]["sequence_diff_summary"]["max_diff_count"], 0)
        self.assertGreaterEqual(report["tracks"]["reproducibility"]["varying_behavior_metric_count"], 2)
        self.assertTrue(report["tracks"]["reproducibility"]["reproducibility_gate"]["passed"])
        self.assertTrue(report["gate_summary"]["leakage_passed"])
        self.assertTrue(report["gate_summary"]["reproducibility_passed"])

    @unittest.skipUnless(default_acceptance_benchmark_root() is not None, "external acceptance bundle required")
    def test_formal_acceptance_artifacts_are_registered(self) -> None:
        write_m42_acceptance_artifacts()
        report = json.loads(Path(M42_REPORT_PATH).read_text(encoding="utf-8"))

        for path in (M42_REPORT_PATH, M42_CONFIDENCE_TRACE_PATH, M42_IGT_TRACE_PATH, M42_LEAKAGE_PATH, M42_REPRO_PATH):
            self.assertTrue(Path(path).exists(), str(path))
        self.assertEqual(report["artifacts"]["leakage"], str(M42_LEAKAGE_PATH))
        self.assertEqual(report["artifacts"]["reproducibility"], str(M42_REPRO_PATH))


if __name__ == "__main__":
    unittest.main()
