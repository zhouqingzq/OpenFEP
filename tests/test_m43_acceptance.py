from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m43_audit import M43_REPORT_PATH, write_m43_acceptance_artifacts


class TestM43Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m43_acceptance_artifacts()
        report = json.loads(Path(M43_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.3")
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["gates"]["schema"]["passed"])
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["causality"]["passed"])
        self.assertTrue(report["gates"]["upstream_parameter_causality"]["passed"])
        self.assertTrue(report["gates"]["upstream_log_completeness"]["passed"])
        self.assertTrue(report["gates"]["leakage_check_passed"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertFalse(report["gates"]["baseline_competitive"]["passed"])
        self.assertFalse(report["gates"]["sample_size_sufficient_for_claim"]["passed"])
        self.assertEqual(report["readiness"]["deployment_readiness"], "NOT_READY")


if __name__ == "__main__":
    unittest.main()
