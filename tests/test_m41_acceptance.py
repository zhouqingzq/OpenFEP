from __future__ import annotations

import json
import unittest
from pathlib import Path

from segmentum.m41_audit import M41_REPORT_PATH, write_m41_acceptance_artifacts
from segmentum.m4_cognitive_style import validate_acceptance_report


INTERFACE_GATES = [
    "g1_schema_completeness",
    "g2_trial_variability",
    "g3_observability",
    "g4_intervention_sensitivity",
    "g5_log_completeness",
    "g6_stress_behavior",
    "r1_report_structure",
]


class TestM41Acceptance(unittest.TestCase):
    def test_acceptance_gates_present_and_honest(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(report["milestone_id"], "M4.1")
        self.assertIn(report["status"], {"PASS", "FAIL"})

        # All interface gates must exist with passed flag and evidence
        for gate_name in INTERFACE_GATES:
            self.assertIn(gate_name, report["gates"], f"missing gate: {gate_name}")
            gate = report["gates"][gate_name]
            self.assertIn("passed", gate)
            self.assertIsInstance(gate["evidence"], dict)
            self.assertTrue(gate["evidence"], f"empty evidence for {gate_name}")

        # All interface gates are blocking
        for gate_name in INTERFACE_GATES:
            self.assertTrue(
                report["gates"][gate_name].get("blocking", False),
                f"{gate_name} must be blocking",
            )

        # failed_gates is consistent with gate results
        expected_failed = sorted(
            name for name, payload in report["gates"].items() if not payload["passed"]
        )
        self.assertEqual(sorted(report["failed_gates"]), expected_failed)

        # status is consistent with blocking gates
        has_blocking_failure = any(
            report["gates"][g]["blocking"] and not report["gates"][g]["passed"]
            for g in report["gates"]
        )
        if has_blocking_failure:
            self.assertEqual(report["status"], "FAIL")
        else:
            self.assertEqual(report["status"], "PASS")

        # report_validation passes
        self.assertTrue(report["report_validation"]["valid"], msg=report["report_validation"])

    def test_g1_evidence_has_numeric_values(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        g1 = report["gates"]["g1_schema_completeness"]["evidence"]
        self.assertIsInstance(g1["roundtrip_precision_loss"], float)
        self.assertIsInstance(g1["parameter_count"], int)
        self.assertEqual(g1["parameter_count"], 8)
        self.assertLess(g1["roundtrip_precision_loss"], 1e-6)
        self.assertEqual(g1["parameter_snapshot_complete_rate"], 1.0)

    def test_g4_evidence_has_per_parameter_results(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        g4 = report["gates"]["g4_intervention_sensitivity"]["evidence"]
        self.assertEqual(g4["total_parameters"], 8)
        self.assertIsInstance(g4["per_parameter"], dict)
        self.assertEqual(len(g4["per_parameter"]), 8)
        for name, entry in g4["per_parameter"].items():
            self.assertIn("delta", entry)
            self.assertIn("identifiable", entry)

    def test_g3_and_g5_evidence_include_honesty_checks(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        g3 = report["gates"]["g3_observability"]["evidence"]
        g5 = report["gates"]["g5_log_completeness"]["evidence"]
        self.assertIn("registry_executable", g3)
        self.assertIn("informative_observables_per_parameter", g3)
        self.assertIn("invalid_value_counts", g5)
        self.assertIn("semantic_invalid_counts", g5)

    def test_validate_acceptance_report_rejects_missing_gates(self) -> None:
        broken = {
            "status": "PASS",
            "gates": {"g1_schema_completeness": {"passed": True, "blocking": True, "evidence": {"ok": True}}},
            "findings": [],
            "failed_gates": [],
        }
        validation = validate_acceptance_report(broken)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("missing_gates" in e for e in validation["errors"]))

    def test_scope_describes_interface_layer(self) -> None:
        write_m41_acceptance_artifacts()
        report = json.loads(Path(M41_REPORT_PATH).read_text(encoding="utf-8"))
        scope = report["scope"]
        self.assertIn("parameter interface", scope["milestone_goal"])
        self.assertIn("observable interface", scope["milestone_goal"])
        self.assertIn("logging interface", scope["milestone_goal"])
        self.assertIsInstance(scope["deferred_to_later_milestones"], list)
        self.assertGreater(len(scope["deferred_to_later_milestones"]), 0)


if __name__ == "__main__":
    unittest.main()
