from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m32_audit import M32_REPORT_PATH, write_m32_acceptance_artifacts


class TestM32Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m32_acceptance_artifacts()
        report = json.loads(Path(M32_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M3.2")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["gates"]["schema"]["passed"])


if __name__ == "__main__":
    unittest.main()
