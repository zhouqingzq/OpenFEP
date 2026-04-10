from __future__ import annotations

import unittest

from segmentum.m47_reacceptance import (
    GATE_HONESTY,
    GATE_REGRESSION,
    STATUS_NOT_RUN,
    build_m47_evidence_records,
    build_m47_reacceptance_report,
)
from segmentum.m47_runtime import build_m47_runtime_snapshot


class TestM47Reacceptance(unittest.TestCase):
    def test_evidence_records_are_graded_from_shared_runtime_snapshot(self) -> None:
        snapshot = build_m47_runtime_snapshot()
        records = build_m47_evidence_records(runtime_snapshot=snapshot)

        self.assertEqual(len(records), 11)
        for record in records:
            self.assertIn("source_api_call_id", record)
            self.assertIn("source_input_set_id", record)
            self.assertTrue(record["observed"])

    def test_reacceptance_defaults_to_incomplete_with_honest_g9_not_run(self) -> None:
        report = build_m47_reacceptance_report()

        self.assertEqual(report["evidence_rebuild_status"], "INCOMPLETE")
        self.assertEqual(report["gate_summaries"][GATE_REGRESSION]["status"], STATUS_NOT_RUN)
        self.assertEqual(report["gate_summaries"][GATE_HONESTY]["status"], "PASS")
        self.assertIn("runtime_snapshot", report)
        self.assertTrue(report["runtime_snapshot"]["diagnostic_only"])

    def test_behavioral_gates_are_explicitly_marked_diagnostic_only(self) -> None:
        report = build_m47_reacceptance_report()
        summaries = report["gate_summaries"]

        self.assertTrue(summaries["behavioral_scenario_A_threat_learning"]["behavioral_claims_demoted"])
        self.assertTrue(summaries["behavioral_scenario_B_interference"]["behavioral_claims_demoted"])
        self.assertTrue(summaries["behavioral_scenario_C_consolidation"]["behavioral_claims_demoted"])
        self.assertEqual(summaries["behavioral_scenario_A_threat_learning"]["evidence_role"], "diagnostic_only")
        self.assertEqual(summaries["integration_interface"]["acceptance_layer"], "a_structural_self_consistency")


if __name__ == "__main__":
    unittest.main()
