from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m227_audit import (
    M227_ABLATION_PATH,
    M227_REPORT_PATH,
    M227_STRESS_PATH,
    M227_SUMMARY_PATH,
    M227_TRACE_PATH,
    write_m227_acceptance_artifacts,
)


class TestM227AcceptanceArtifacts(unittest.TestCase):
    def test_report_contains_strict_audit_fields_and_current_round_artifacts(self) -> None:
        write_m227_acceptance_artifacts()
        report = json.loads(Path(M227_REPORT_PATH).read_text(encoding="utf-8"))

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

        self.assertEqual(report["milestone_id"], "M2.27")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["recommendation"], "ACCEPT")

        for path in (
            M227_TRACE_PATH,
            M227_ABLATION_PATH,
            M227_STRESS_PATH,
            M227_REPORT_PATH,
            M227_SUMMARY_PATH,
        ):
            self.assertTrue(Path(path).exists(), str(path))

        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])


if __name__ == "__main__":
    unittest.main()
