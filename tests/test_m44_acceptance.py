from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m44_audit import M44_REPORT_PATH, write_m44_acceptance_artifacts


class TestM44Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m44_acceptance_artifacts()
        report = json.loads(Path(M44_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.4")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["schema"]["passed"])
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["causality"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertTrue(report["gates"]["shared_threshold"]["passed"])
        self.assertEqual(report["readiness"]["deployment_readiness"], "NOT_READY")


if __name__ == "__main__":
    unittest.main()
