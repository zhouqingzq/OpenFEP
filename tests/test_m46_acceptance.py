from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m46_audit import (
    FORMAL_CONCLUSION_NOT_ISSUED,
    LEGACY_M46_ACCEPTANCE_NOTICE,
    OFFICIAL_M46_ACCEPTANCE_NOTICE,
    build_m46_acceptance_report,
    write_m46_acceptance_artifacts,
    write_m46_legacy_acceptance_artifacts,
)


class TestM46Acceptance(unittest.TestCase):
    def _read_json(self, path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_official_acceptance_defaults_to_not_issued_without_regressions(self) -> None:
        report, evidence_report, failure_injection, ablation = build_m46_acceptance_report()

        self.assertEqual(report["mode"], "official_runtime_acceptance")
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertEqual(report["recommendation"], "DEFER")
        self.assertIn("legacy_integration", report["not_run_gates"])
        self.assertTrue(report["primary_evidence_chain"])
        self.assertEqual(report["gate_summaries"]["legacy_integration"]["status"], "NOT_RUN")
        self.assertTrue(failure_injection["all_cases_failed_closed"])
        self.assertTrue(ablation["all_tampered_cases_failed_closed"])
        self.assertEqual(evidence_report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)

    def test_official_writer_emits_runtime_evidence_and_negative_controls(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m46_acceptance_artifacts(
                output_root=tmpdir,
                round_started_at="2026-04-08T00:00:00+00:00",
            )
            report = self._read_json(outputs["report"])
            canonical_trace = self._read_json(outputs["canonical_trace"])
            ablation = self._read_json(outputs["ablation"])
            failure_injection = self._read_json(outputs["failure_injection"])
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self.assertEqual(report["mode"], "official_runtime_acceptance")
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["formal_acceptance_conclusion"], FORMAL_CONCLUSION_NOT_ISSUED)
        self.assertEqual(report["artifact_lineage"], "official_runtime_evidence")
        self.assertTrue(report["primary_evidence_chain"])
        self.assertEqual(canonical_trace["mode"], "independent_evidence_rebuild")
        self.assertIn("gate_summaries", canonical_trace)
        self.assertIn("evidence_records", canonical_trace)
        self.assertTrue(failure_injection["all_cases_failed_closed"])
        self.assertTrue(ablation["all_tampered_cases_failed_closed"])
        self.assertIn("fake_regression_pass", {case["case"] for case in failure_injection["cases"]})
        self.assertIn("Official M4.6 runtime acceptance artifact", summary)
        self.assertIn("Formal Acceptance Conclusion: `NOT_ISSUED`", summary)
        self.assertIn("G7 `legacy_integration`: `NOT_RUN`", summary)

    def test_g4_g5_g6_runtime_evidence_contains_numeric_deltas_and_validation_linkage(self) -> None:
        report, evidence_report, _, _ = build_m46_acceptance_report()
        records = list(evidence_report["evidence_records"])

        reconsolidation_record = next(
            record for record in records if record["scenario_id"] == "reconsolidation_reinforcement_only"
        )
        numeric_delta = reconsolidation_record["observed"]["numeric_delta"]
        self.assertEqual(
            set(numeric_delta),
            {"accessibility", "trace_strength", "retrieval_count", "abstractness", "last_accessed"},
        )
        self.assertGreater(numeric_delta["last_accessed"], 0)

        consolidation_linkage = next(
            record for record in records if record["scenario_id"] == "consolidation_validation_linkage"
        )
        self.assertTrue(consolidation_linkage["criteria_checks"]["validated_inference_ids_present"])
        self.assertTrue(consolidation_linkage["criteria_checks"]["validated_entries_promoted_to_long"])

        inference_linkage = next(
            record for record in records if record["scenario_id"] == "inference_consolidation_validation_linkage"
        )
        self.assertTrue(inference_linkage["criteria_checks"]["validated_inference_ids_present"])
        self.assertTrue(inference_linkage["criteria_checks"]["metadata_scores_match_traceability"])
        self.assertEqual(report["gates"]["offline_consolidation_pipeline"]["status"], "PASS")
        self.assertEqual(report["gates"]["inference_validation_gate"]["status"], "PASS")

    def test_official_negative_controls_cover_required_honesty_fail_closed_cases(self) -> None:
        report, _, failure_injection, _ = build_m46_acceptance_report()
        failed_closed_cases = {case["case"] for case in failure_injection["cases"] if case["failed_closed"] is True}

        self.assertEqual(report["gates"]["report_honesty"]["status"], "PASS")
        self.assertTrue(report["gates"]["report_honesty"]["passed"])
        self.assertTrue(
            {
                "fake_regression_pass",
                "missing_integration_record",
                "tampered_candidate_ids",
                "tampered_validated_linkage",
                "empty_observed_payload",
                "duplicate_source_api_call_id",
            }.issubset(failed_closed_cases)
        )

    def test_official_acceptance_with_regressions_fail_closes_fake_regression_and_passes_g8(self) -> None:
        report, _, failure_injection, ablation = build_m46_acceptance_report(include_regressions=True)
        fake_regression_case = next(case for case in failure_injection["cases"] if case["case"] == "fake_regression_pass")

        self.assertTrue(fake_regression_case["failed_closed"])
        self.assertIn("legacy_regression_prereq", fake_regression_case["external_check_failures"])
        self.assertTrue(failure_injection["all_cases_failed_closed"])
        self.assertTrue(ablation["all_tampered_cases_failed_closed"])
        self.assertEqual(report["gates"]["report_honesty"]["status"], "PASS")
        self.assertTrue(report["gates"]["report_honesty"]["passed"])

    def test_legacy_writer_remains_historical_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            outputs = write_m46_legacy_acceptance_artifacts(
                output_root=tmpdir,
                round_started_at="2026-04-08T00:00:00+00:00",
                regression_summary={
                    "executed": True,
                    "files": [],
                    "returncode": 0,
                    "passed": True,
                    "duration_seconds": 0.1,
                    "summary_line": "synthetic pass",
                    "stdout_tail": ["synthetic pass"],
                },
            )
            report = self._read_json(outputs["report"])
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self.assertEqual(report["artifact_lineage"], "legacy_self_attested_acceptance")
        self.assertFalse(report["primary_evidence_chain"])
        self.assertEqual(report["legacy_notice"], LEGACY_M46_ACCEPTANCE_NOTICE)
        self.assertIn("historical self-attested acceptance view", summary)
        self.assertNotIn(OFFICIAL_M46_ACCEPTANCE_NOTICE, summary)


if __name__ == "__main__":
    unittest.main()
