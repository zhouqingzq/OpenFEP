from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m42_audit import M42_REPORT_PATH, write_m42_acceptance_artifacts


class TestM42Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m42_acceptance_artifacts()
        report = json.loads(Path(M42_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.2")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["acceptance_state"], "blocked_missing_external_bundle")
        self.assertTrue(report["gates"]["schema"]["passed"])
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["causality"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertTrue(report["gates"]["benchmark_closed_loop"]["passed"])
        self.assertTrue(report["gates"]["subject_leakage_free"]["passed"])
        self.assertFalse(report["gates"]["external_bundle_used"]["passed"])
        self.assertFalse(report["gates"]["confidence_acceptance_ready"]["passed"])
        self.assertFalse(report["gates"]["igt_acceptance_ready"]["passed"])
        self.assertEqual(report["benchmarks"]["confidence_database"]["benchmark_state"], "blocked_missing_external_bundle")
        self.assertEqual(report["benchmarks"]["iowa_gambling_task"]["benchmark_state"], "blocked_missing_external_bundle")
        self.assertTrue(report["benchmarks"]["confidence_database"]["blockers"][0].strip())
        self.assertTrue(report["benchmarks"]["iowa_gambling_task"]["blockers"][0].strip())
        for finding in report["findings"]:
            self.assertTrue(str(finding["detail"]).strip())


if __name__ == "__main__":
    unittest.main()
