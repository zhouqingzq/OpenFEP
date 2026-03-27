from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m31_audit import M31_REPORT_PATH, write_m31_acceptance_artifacts


class TestM31Acceptance(unittest.TestCase):
    def test_acceptance_bundle_contains_required_fields(self) -> None:
        write_m31_acceptance_artifacts()
        report = json.loads(Path(M31_REPORT_PATH).read_text(encoding="utf-8"))
        for field in (
            "milestone_id",
            "status",
            "generated_at",
            "seed_set",
            "artifacts",
            "tests",
            "gates",
            "findings",
            "residual_risks",
            "freshness",
            "recommendation",
        ):
            self.assertIn(field, report)
        self.assertEqual(report["milestone_id"], "M3.1")
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
