from __future__ import annotations

import unittest

from segmentum.m41_external_validation import acceptance_scope_note
from segmentum.m41_external_task_eval import (
    run_external_task_bundle_evaluation,
    run_minimal_external_task_validation,
    smoke_fixture_rejection_report,
)


class TestM41ExternalValidation(unittest.TestCase):
    def test_external_task_validation_uses_real_external_bundles(self) -> None:
        payload = run_external_task_bundle_evaluation()
        self.assertEqual(payload["analysis_type"], "m41_minimal_external_task_validation")
        self.assertEqual(payload["validation_type"], "external_task_bundle")
        self.assertEqual(payload["benchmark_scope"], "task-level evaluation on imported external human benchmark bundles")

        for benchmark_id in ("confidence_database", "iowa_gambling_task"):
            with self.subTest(benchmark=benchmark_id):
                evidence = payload["external_bundle_provenance"][benchmark_id]
                self.assertEqual(evidence["bundle"]["source_type"], "external_bundle")
                self.assertFalse(evidence["bundle"]["smoke_test_only"])
                self.assertFalse(evidence["bundle"]["is_synthetic"])
                self.assertTrue(evidence["validation"]["ok"])
                self.assertEqual(evidence["status"]["benchmark_state"], "acceptance_ready")
                self.assertTrue(evidence["subject_split_integrity"]["ok"])

        self.assertTrue(payload["confidence_benchmark"]["metric_gate"]["passed"])
        self.assertTrue(payload["igt_benchmark"]["metric_gate"]["passed"])
        self.assertTrue(payload["evaluation_chain_audit"]["all_clear"])

    def test_scope_boundary_explicitly_downgrades_synthetic_claims(self) -> None:
        payload = run_external_task_bundle_evaluation()
        replacements = {item["claim"]: item["replacement"] for item in payload["scope_boundary"]["downgraded_claims"]}
        self.assertEqual(replacements["external validation"], "same-framework synthetic holdout validation sidecar")
        self.assertEqual(replacements["identifiability"], "within synthetic family recoverability")
        self.assertEqual(replacements["blind classification"], "profile distinguishability on synthetic holdout")
        self.assertIn(
            "external validation of latent cognitive parameters",
            payload["scope_boundary"]["still_not_claimed"],
        )
        self.assertIn(
            "completion of M4.2 benchmark/task-layer recovery",
            payload["scope_boundary"]["still_not_claimed"],
        )

    def test_repo_smoke_fixtures_are_blocked_for_acceptance(self) -> None:
        payload = smoke_fixture_rejection_report()
        self.assertTrue(payload["all_blocked"])
        self.assertTrue(payload["checks"]["confidence_database"]["blocked"])
        self.assertTrue(payload["checks"]["iowa_gambling_task"]["blocked"])

    def test_acceptance_scope_note_preserves_m41_as_interface_layer(self) -> None:
        note = acceptance_scope_note()
        self.assertEqual(note["m41_acceptance_scope"], "interface_layer")
        self.assertEqual(note["m42_plus_scope"], "benchmark_environment_and_task_layer")
        self.assertEqual(note["legacy_synthetic_scope"], "same-framework synthetic holdout sidecars")
        self.assertIn("M4.1 acceptance is interface-layer only", note["note"])

    def test_legacy_external_task_entrypoint_remains_alias_only(self) -> None:
        self.assertEqual(run_minimal_external_task_validation(), run_external_task_bundle_evaluation())


if __name__ == "__main__":
    unittest.main()
