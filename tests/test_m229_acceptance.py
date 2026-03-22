from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m229_audit import (
    M229_ABLATION_PATH,
    M229_REPORT_PATH,
    M229_STRESS_PATH,
    M229_SUMMARY_PATH,
    M229_TRACE_PATH,
    write_m229_acceptance_artifacts,
)


class TestM229AcceptanceArtifacts(unittest.TestCase):
    def test_report_contains_strict_audit_fields_and_current_round_artifacts(self) -> None:
        write_m229_acceptance_artifacts()
        report = json.loads(Path(M229_REPORT_PATH).read_text(encoding="utf-8"))

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

        self.assertEqual(report["milestone_id"], "M2.29")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["recommendation"], "ACCEPT")
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertTrue(report["gates"]["artifact_freshness"]["passed"])

        for path in (
            M229_TRACE_PATH,
            M229_ABLATION_PATH,
            M229_STRESS_PATH,
            M229_REPORT_PATH,
            M229_SUMMARY_PATH,
        ):
            self.assertTrue(Path(path).exists(), str(path))

    def test_artifact_payloads_include_verification_specific_evidence(self) -> None:
        written = write_m229_acceptance_artifacts()
        trace_lines = Path(written["trace"]).read_text(encoding="utf-8").strip().splitlines()
        ablation = json.loads(Path(written["ablation"]).read_text(encoding="utf-8"))
        stress = json.loads(Path(written["stress"]).read_text(encoding="utf-8"))

        self.assertTrue(trace_lines)
        self.assertIn('"verification_loop"', trace_lines[-1])
        self.assertIn('"verification_payload"', trace_lines[-1])
        self.assertIn("degradation_checks", ablation)
        self.assertTrue(ablation["degradation_checks"]["evidence_seeking_removed_without_verification"])
        self.assertIn("timeout_update", stress)
        self.assertTrue(stress["timeout_update"]["expired_targets"])
        self.assertTrue(stress["trace_checks"]["last_record_has_verification_loop"])


if __name__ == "__main__":
    unittest.main()
