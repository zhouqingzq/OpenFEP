from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m48_audit import M48_REPORT_PATH, write_m48_acceptance_artifacts


class TestM48Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m48_acceptance_artifacts()
        report = json.loads(Path(M48_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.8")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["schema"]["passed"])
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["causality"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertTrue(report["gates"]["parameter_projection"]["passed"])
        self.assertEqual(report["readiness"]["deployment_readiness"], "NOT_READY")
        self.assertTrue(report["headline_metrics"]["synthetic_probe"])
        self.assertTrue(report["headline_metrics"]["live_integration"])


if __name__ == "__main__":
    unittest.main()
