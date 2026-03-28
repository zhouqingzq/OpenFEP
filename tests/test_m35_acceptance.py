from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m35_audit import M35_REPORT_PATH, write_m35_acceptance_artifacts


class TestM35Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m35_acceptance_artifacts()
        report = json.loads(Path(M35_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M3.5")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["schema"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])


if __name__ == "__main__":
    unittest.main()
