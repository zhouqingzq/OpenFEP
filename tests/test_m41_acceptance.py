from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m41_audit import M41_REPORT_PATH, write_m41_acceptance_artifacts


class TestM41Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.1")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["schema"]["passed"])
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["causality"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])


if __name__ == "__main__":
    unittest.main()
