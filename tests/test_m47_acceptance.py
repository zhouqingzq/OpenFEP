from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m47_audit import (
    FORMAL_CONCLUSION_NOT_ISSUED,
    OFFICIAL_M47_ACCEPTANCE_NOTICE,
    build_m47_acceptance_report,
    write_m47_acceptance_artifacts,
)
from segmentum.m47_runtime import build_m47_runtime_snapshot


class TestM47Acceptance(unittest.TestCase):
    def _read_json(self, path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_official_acceptance_reuses_shared_runtime_snapshot_and_stays_not_issued(self) -> None:
        snapshot = build_m47_runtime_snapshot()
        report, evidence_report, failure_injection, ablation = build_m47_acceptance_report(runtime_snapshot=snapshot)

        self.assertEqual(report["mode"], "official_runtime_acceptance")
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertEqual(report["gate_summaries"]["regression"]["status"], "NOT_RUN")
        self.assertEqual(evidence_report["runtime_snapshot"]["generated_at"], snapshot["generated_at"])
        self.assertTrue(evidence_report["runtime_snapshot"]["diagnostic_only"])
        self.assertTrue(
            evidence_report["gate_summaries"]["behavioral_scenario_A_threat_learning"]["behavioral_claims_demoted"]
        )
        self.assertTrue(failure_injection["all_cases_failed_closed"])
        self.assertTrue(ablation["all_tampered_cases_failed_closed"])

    def test_acceptance_writer_emits_runtime_snapshot_and_primary_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m47_acceptance_artifacts(
                output_root=tmpdir,
                round_started_at="2026-04-09T00:00:00+00:00",
            )
            report = self._read_json(outputs["report"])
            runtime_snapshot = self._read_json(outputs["runtime_snapshot"])
            canonical_trace = self._read_json(outputs["canonical_trace"])
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self.assertEqual(report["artifact_lineage"], "official_runtime_evidence")
        self.assertEqual(report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertEqual(runtime_snapshot["mode"], "m47_shared_runtime_snapshot")
        self.assertEqual(canonical_trace["mode"], "independent_evidence_rebuild")
        self.assertIn("gate_summaries", canonical_trace)
        self.assertIn(OFFICIAL_M47_ACCEPTANCE_NOTICE, "\n".join(report["notes"]))
        self.assertIn("Formal Acceptance Conclusion: `NOT_ISSUED`", summary)

    def test_regression_request_does_not_convert_g9_into_pass_without_live_run(self) -> None:
        snapshot = build_m47_runtime_snapshot()
        report, evidence_report, _, _ = build_m47_acceptance_report(
            include_regressions=True,
            runtime_snapshot=snapshot,
        )

        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertEqual(report["gate_summaries"]["regression"]["status"], "NOT_RUN")
        self.assertTrue(evidence_report["regression_policy"]["live_only"])


if __name__ == "__main__":
    unittest.main()
