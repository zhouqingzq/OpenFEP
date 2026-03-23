from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m231_audit import (
    M231_ABLATION_PATH,
    M231_REPORT_PATH,
    M231_SPEC_PATH,
    M231_STRESS_PATH,
    M231_SUMMARY_PATH,
    M231_TRACE_PATH,
    write_m231_acceptance_artifacts,
)


class TestM231AcceptanceArtifacts(unittest.TestCase):
    def test_report_contains_strict_audit_fields_and_current_round_artifacts(self) -> None:
        write_m231_acceptance_artifacts()
        report = json.loads(Path(M231_REPORT_PATH).read_text(encoding="utf-8"))

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

        self.assertEqual(report["milestone_id"], "M2.31")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["recommendation"], "ACCEPT")
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertTrue(report["gates"]["artifact_freshness"]["passed"])

        for path in (
            M231_SPEC_PATH,
            M231_TRACE_PATH,
            M231_ABLATION_PATH,
            M231_STRESS_PATH,
            M231_REPORT_PATH,
            M231_SUMMARY_PATH,
        ):
            self.assertTrue(Path(path).exists(), str(path))

    def test_artifact_payloads_include_reconciliation_writeback_specific_evidence(self) -> None:
        written = write_m231_acceptance_artifacts()
        trace_lines = Path(written["trace"]).read_text(encoding="utf-8").strip().splitlines()
        ablation = json.loads(Path(written["ablation"]).read_text(encoding="utf-8"))
        stress = json.loads(Path(written["stress"]).read_text(encoding="utf-8"))

        self.assertTrue(trace_lines)
        self.assertIn('"current_chapter_reconciliation"', trace_lines[-1])
        self.assertIn('"core_summary"', trace_lines[-1])
        self.assertIn("degradation_checks", ablation)
        self.assertTrue(
            ablation["degradation_checks"]["core_summary_loses_reconciliation_clause_without_writeback"]
        )
        self.assertTrue(stress["stress_checks"]["unrelated_evidence_did_not_contaminate_other_thread"])
        self.assertTrue(stress["stress_checks"]["unmatched_repair_did_not_bind"])
        self.assertTrue(stress["stress_checks"]["narrative_writeback_survived_stress"])


if __name__ == "__main__":
    unittest.main()
