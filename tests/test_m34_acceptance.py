from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m34_audit import M34_REPORT_PATH, write_m34_acceptance_artifacts


class TestM34Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m34_acceptance_artifacts()
        report = json.loads(Path(M34_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M3.4")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["causality"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])


if __name__ == "__main__":
    unittest.main()
