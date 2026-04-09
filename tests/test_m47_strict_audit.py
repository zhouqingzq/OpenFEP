from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m47_runtime import build_m47_runtime_snapshot
from segmentum.m47_strict_audit import build_m47_strict_audit, write_m47_strict_audit


LEGACY_FINDING_CATEGORIES = {
    "cached_regression",
    "self_generated_evidence",
    "synthetic_scenario_dependency",
    "rule_stack_promotion",
    "top_1_recall_backbone",
    "report_shape_tests",
}


class TestM47StrictAudit(unittest.TestCase):
    def test_build_strict_audit_only_blocks_g9_when_live_regression_is_skipped(self) -> None:
        snapshot = build_m47_runtime_snapshot()
        self.assertEqual(snapshot["mode"], "m47_shared_runtime_snapshot")

        with TemporaryDirectory() as tmpdir:
            audit, output_paths = build_m47_strict_audit(
                round_started_at="2026-04-09T00:00:00+00:00",
                output_root=tmpdir,
                run_m47_self_tests=False,
                run_live_regressions=False,
            )

            self.assertTrue(Path(audit["runtime_snapshot"]["output_paths"]["acceptance_report"]).exists())

            gate_results = {item["gate"]: item for item in audit["gate_results"]}
            self.assertEqual(audit["overall_conclusion"], "BLOCKED")
            for gate in (
                "behavioral_scenario_A_threat_learning",
                "behavioral_scenario_B_interference",
                "behavioral_scenario_C_consolidation",
                "long_term_subtypes",
                "identity_continuity_retention",
                "behavioral_scenario_E_natural_misattribution",
                "integration_interface",
            ):
                self.assertEqual(gate_results[gate]["strict_verdict"], "PASS")
            self.assertEqual(gate_results["regression"]["strict_verdict"], "BLOCKED")
            self.assertFalse(audit["evidence_appendix"]["live_regression"]["attempted"])
            self.assertFalse(
                LEGACY_FINDING_CATEGORIES & set(audit["summary_statistics"]["finding_category_counts"])
            )
            self.assertIn("m47_strict_runtime", output_paths["runtime_root"])

    def test_writer_emits_clean_summary_and_g9_follow_up(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_paths = write_m47_strict_audit(
                round_started_at="2026-04-09T00:00:00+00:00",
                output_root=tmpdir,
                run_m47_self_tests=False,
                run_live_regressions=False,
            )
            strict_json = json.loads(Path(output_paths["strict_json"]).read_text(encoding="utf-8"))
            strict_summary = Path(output_paths["strict_summary"]).read_text(encoding="utf-8")
            fix_priority = Path(output_paths["fix_priority"]).read_text(encoding="utf-8")

        self.assertEqual(strict_json["overall_conclusion"], "BLOCKED")
        self.assertIn("G9 `regression`: strict=`BLOCKED`, builder=`NOT_RUN`", strict_summary)
        self.assertIn("G9 remains BLOCKED", strict_summary)
        self.assertIn("Run a live M4.1-M4.6 regression suite", fix_priority)


if __name__ == "__main__":
    unittest.main()
