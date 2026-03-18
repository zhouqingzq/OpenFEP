from __future__ import annotations

import unittest

from segmentum.m223_benchmarks import SEED_SET, run_m223_self_consistency_benchmark


class TestM223AuditHardening(unittest.TestCase):
    def test_benchmark_refuses_pass_when_seed_set_incomplete(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=[223])
        self.assertEqual(payload["status"], "FAIL")
        self.assertFalse(payload["gates"]["protocol_integrity"])
        self.assertFalse(payload["protocol_integrity"]["seed_set_complete"])

    def test_false_positives_require_explicit_internal_conflict_output(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=[223, 242])
        aligned_records = [
            record
            for record in payload["artifacts"]["m223_self_consistency_trace"]["trace"]
            if record["condition"] == "aligned"
        ]
        self.assertTrue(aligned_records)
        for record in aligned_records:
            if not record["false_positive"]:
                self.assertFalse(record["false_positive_basis"]["counted"])
                self.assertFalse(record["false_positive_basis"]["explicit_conflict_output"])

    def test_repair_success_requires_explicit_chain_and_recovery_window(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=[223, 242])
        repair_records = [
            record
            for record in payload["artifacts"]["m223_self_consistency_trace"]["trace"]
            if record["repair_triggered"]
        ]
        self.assertTrue(repair_records)
        counted = [record for record in repair_records if record["repair_success_basis"]["counted"]]
        self.assertTrue(counted)
        for record in counted:
            self.assertTrue(record["repair_result"]["triggered"])
            self.assertTrue(record["repair_result"]["policy"])
            self.assertTrue(record["repair_result"]["outcome"])
            self.assertTrue(
                record["repair_result"]["recovery_window"]["improved"]
                or record["repair_result"]["recovery_window"]["returned_within_commitments"]
            )

    def test_long_horizon_samples_are_unique_per_seed(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=list(SEED_SET))
        checks = payload["sample_independence_checks"]
        self.assertTrue(checks["repeated_challenge"]["passes"])
        self.assertTrue(checks["chapter_transition"]["passes"])
        self.assertTrue(checks["bounded_identity_update"]["passes"])
        self.assertEqual(checks["bounded_identity_update"]["sample_count"], len(SEED_SET))

    def test_metric_counting_rules_are_present(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=list(SEED_SET))
        rules = payload["metric_counting_rules"]
        self.assertIn("self_inconsistency_detection_rate", rules)
        self.assertIn("false_positive_inconsistency_rate", rules)
        self.assertIn("repair_success_rate", rules)
        self.assertIn("bounded_identity_update", rules)

    def test_protocol_integrity_condition_set_is_derived_from_variants(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=list(SEED_SET))
        protocol_integrity = payload["protocol_integrity"]
        self.assertEqual(
            sorted(protocol_integrity["provided_condition_set"]),
            sorted(["with_commitments", "no_commitments", "with_repair", "no_repair"]),
        )
        self.assertTrue(protocol_integrity["condition_set_complete"])

    def test_required_seed_set_defaults_to_canonical_protocol(self) -> None:
        payload = run_m223_self_consistency_benchmark(seed_set=list(SEED_SET))
        self.assertEqual(payload["protocol_integrity"]["required_seed_set"], list(SEED_SET))
        self.assertTrue(payload["gates"]["protocol_integrity"])

    def test_required_seed_set_can_be_aligned_to_unified_audit_protocol(self) -> None:
        audit_seed_set = [226, 245, 323, 342, 420, 439]
        payload = run_m223_self_consistency_benchmark(
            seed_set=list(audit_seed_set),
            required_seed_set=list(audit_seed_set),
        )
        self.assertEqual(payload["protocol_integrity"]["required_seed_set"], audit_seed_set)
        self.assertTrue(payload["protocol_integrity"]["seed_set_complete"])
        self.assertTrue(payload["gates"]["protocol_integrity"])


if __name__ == "__main__":
    unittest.main()
