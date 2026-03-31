from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m1_audit import M1_REPORT_PATH, write_m1_acceptance_artifacts


class TestM1Acceptance(unittest.TestCase):
    def test_acceptance_bundle_is_traceable_and_honest(self) -> None:
        write_m1_acceptance_artifacts()
        report = json.loads(Path(M1_REPORT_PATH).read_text(encoding="utf-8"))

        for field in (
            "milestone_id",
            "title",
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
            "canonical_files",
        ):
            self.assertIn(field, report)

        self.assertEqual(report["milestone_id"], "M1")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(any(finding["label"] == "reconstructed_spec" for finding in report["findings"]))
        self.assertIn("reconstructed minimal m1 scope", report["residual_risks"][0].lower())
        self.assertTrue(all(gate["passed"] for gate in report["gates"].values()))


if __name__ == "__main__":
    unittest.main()
