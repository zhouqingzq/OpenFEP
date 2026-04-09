from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m47_evidence_chain_audit import (
    PRIMARY_M47_ARTIFACT_NAMES,
    build_m47_evidence_chain_audit,
    build_m47_evidence_requirement_map,
    write_m47_evidence_chain_audit,
)
from segmentum.m47_runtime import build_m47_runtime_snapshot


class TestM47EvidenceChainAudit(unittest.TestCase):
    def test_requirement_map_splits_mechanism_and_artifact_tests_for_each_gate(self) -> None:
        requirement_map = build_m47_evidence_requirement_map()
        snapshot = build_m47_runtime_snapshot()

        self.assertEqual(len(requirement_map), 12)
        self.assertEqual(snapshot["mode"], "m47_shared_runtime_snapshot")
        for item in requirement_map:
            self.assertTrue(item["required_scenario_ids"])
            self.assertTrue(item["required_observed_fields"])
            self.assertTrue(item["mechanism_tests"])
            self.assertTrue(item["artifact_tests"])
            self.assertEqual(
                set(item["supporting_tests"]),
                set(item["mechanism_tests"]) | set(item["artifact_tests"]),
            )

    def test_build_audit_rebuilds_artifacts_and_marks_g9_honestly_not_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            audit, output_paths = build_m47_evidence_chain_audit(
                round_started_at="2026-04-09T00:00:00+00:00",
                output_root=tmpdir,
                include_regressions=False,
            )

            self.assertTrue(Path(output_paths["runtime_snapshot"]).exists())

            self.assertFalse(audit["preexisting_artifacts"]["complete"])
            self.assertEqual(set(audit["preexisting_artifacts"]["missing"]), set(PRIMARY_M47_ARTIFACT_NAMES))
            self.assertTrue(audit["post_rebuild_artifacts"]["complete"])
            self.assertTrue(audit["executed_scope_acceptance_satisfied"])
            self.assertTrue(audit["anti_degeneration_review"]["mechanism_and_artifact_tests_split"])

            matrix = {row["dimension"]: row["status"] for row in audit["conclusion_matrix"]}
            self.assertEqual(matrix["preexisting_disk_artifact_completeness"], "INCOMPLETE")
            self.assertEqual(matrix["local_rebuild_evidence_chain_completeness"], "COMPLETE")
            self.assertEqual(matrix["executed_scope_acceptance_satisfaction"], "SATISFIED")

            regression_gate = next(result for result in audit["gate_results"] if result["gate"] == "regression")
            self.assertEqual(regression_gate["gate_summary_status"], "NOT_RUN")
            self.assertEqual(regression_gate["executed_scope_status"], "HONEST_NOT_RUN_BLOCKING_FORMAL_ISSUANCE")
            self.assertTrue(regression_gate["acceptance_requirement_met"])
            self.assertEqual(audit["runtime_snapshot"]["source"], "shared_workload_runtime_snapshot")

    def test_writer_emits_requirement_map_and_summary_with_split_coverage(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_paths = write_m47_evidence_chain_audit(
                round_started_at="2026-04-09T00:00:00+00:00",
                output_root=tmpdir,
                include_regressions=False,
            )
            requirement_map = json.loads(Path(output_paths["requirement_map"]).read_text(encoding="utf-8"))
            audit = json.loads(Path(output_paths["audit_json"]).read_text(encoding="utf-8"))
            summary = Path(output_paths["audit_summary"]).read_text(encoding="utf-8")

        self.assertEqual(requirement_map, audit["requirement_map"])
        self.assertIn("M4.7 Evidence Chain Audit", summary)
        self.assertIn("Mechanism / artifact test split recorded: `True`", summary)
        self.assertIn("HONEST_NOT_RUN_BLOCKING_FORMAL_ISSUANCE", summary)


if __name__ == "__main__":
    unittest.main()
