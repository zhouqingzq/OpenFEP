from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m33_audit import M33_REPORT_PATH, write_m33_acceptance_artifacts


class TestM33Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m33_acceptance_artifacts()
        report = json.loads(Path(M33_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M3.3")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["causality"]["passed"])


if __name__ == "__main__":
    unittest.main()
