from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.m3_readiness_evaluation import (
    build_readiness_payload,
    derive_controlled_ready_status,
    derive_lifecycle_evidence,
    evaluate_runtime_lifecycle_evidence,
    evaluate_runtime_family_coverage,
    write_readiness_outputs,
)


def _followup_payload(
    *,
    eaa: float = 0.9725,
    caq: float = 1.0,
    vcus: float = 1.0,
    compression_removed: int = 0,
    archived: int = 0,
    pruned: int = 0,
) -> dict[str, object]:
    return {
        "new_metrics": {
            "ICI": 1.0,
            "EAA": eaa,
            "MUR": 1.0,
            "PSSR": 0.9997,
            "CAQ": caq,
            "VCUS": vcus,
        },
        "per_scenario_breakdown": {
            "sleep_reduction": {
                "sleep_summary": {
                    "compression_removed": compression_removed,
                    "episodes_archived": archived,
                    "episodes_deleted": pruned,
                }
            }
        },
    }


def _legacy_payload() -> dict[str, object]:
    return {
        "thresholds": {
            "ICI": 0.8,
            "EAA": 0.85,
            "MUR": 0.6,
            "PSSR": 0.3,
            "CAQ": 0.65,
            "VCUS": 0.85,
        },
        "metrics": {
            "ICI": 1.0,
            "EAA": 1.0,
            "MUR": 1.0,
            "PSSR": 0.9997,
            "CAQ": 1.0,
            "VCUS": 1.0,
        },
    }


def _previous_readiness_claim() -> dict[str, object]:
    return {
        "tests": {
            "total_passed": 163,
            "new_m3_readiness_tests": 24,
            "regression_tests_passing": True,
        }
    }


class TestReadinessAuditHardening(unittest.TestCase):
    def test_strict_baseline_is_not_overwritten_by_legacy_full_score(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(eaa=0.9725),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
        )
        self.assertEqual(payload["strict_metrics"]["EAA"]["value"], 0.9725)
        self.assertEqual(payload["legacy_metrics"]["EAA"]["value"], 1.0)

    def test_legacy_metric_is_non_gating(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
        )
        self.assertNotIn("legacy::EAA", payload["gating_metrics"])
        self.assertTrue(payload["legacy_metrics"]["EAA"]["non_gating"])
        self.assertFalse(payload["legacy_metrics"]["EAA"]["gating_metric"])

    def test_missing_current_round_evidence_downgrades_readiness(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
        )
        self.assertEqual(
            payload["controlled_ready_status"]["status"],
            "CONTROLLED_READY_CANDIDATE",
        )
        self.assertEqual(
            payload["final_recommendation"]["status"],
            "RECOMMEND_M3_WITH_CAUTION",
        )

    def test_claimed_not_revalidated_is_explicit(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
        )
        tests_claim = payload["claimed_not_revalidated"]["tests"]
        self.assertEqual(tests_claim["verification_status"], "CLAIMED_BUT_NOT_REVALIDATED")
        self.assertEqual(tests_claim["claimed_total_passed"], 163)

    def test_compression_is_distinguished_from_archive_and_prune(self) -> None:
        evidence = derive_lifecycle_evidence(
            _followup_payload(compression_removed=0, archived=2, pruned=3)
        )
        self.assertTrue(evidence["lifecycle_activity_observed"])
        self.assertFalse(evidence["compression_specifically_verified"])
        self.assertEqual(evidence["archived_count"], 2)
        self.assertEqual(evidence["pruned_count"], 3)
        self.assertIsNone(evidence["compressed_cluster_count"])

    def test_family_schema_count_and_runtime_validated_count_are_not_mixed(self) -> None:
        coverage = evaluate_runtime_family_coverage()
        self.assertEqual(coverage["family_schema_count"], 4)
        self.assertGreaterEqual(coverage["runtime_validated_family_count"], 2)
        self.assertEqual(coverage["family_coverage_status"], "RUNTIME_DIVERSITY_VALIDATED")

    def test_runtime_lifecycle_probe_distinguishes_compression_archive_and_prune(self) -> None:
        evidence = evaluate_runtime_lifecycle_evidence()
        self.assertEqual(evidence["verification_status"], "REVALIDATED_THIS_ROUND")
        self.assertTrue(evidence["lifecycle_activity_observed"])
        self.assertTrue(evidence["compression_specifically_verified"])
        self.assertGreater(evidence["compression_removed_count"], 0)
        self.assertGreater(evidence["compressed_cluster_count"], 0)
        self.assertGreater(evidence["archived_count"], 0)
        self.assertGreater(evidence["pruned_count"], 0)

    def test_controlled_ready_requires_gating_metrics_and_evidence_quality(self) -> None:
        strict_metrics = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness={},
        )["strict_metrics"]
        status = derive_controlled_ready_status(
            strict_metrics,
            {
                "family_coverage_status": "RUNTIME_DIVERSITY_VALIDATED",
                "runtime_validated_family_count": 2,
            },
            {
                "verification_status": "REVALIDATED_THIS_ROUND",
            },
        )
        self.assertEqual(status.status, "CONTROLLED_READY_VERIFIED")

    def test_build_payload_can_upgrade_to_controlled_ready_verified(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage={
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "family_schema_count": 4,
                "runtime_validated_family_count": 4,
                "family_coverage_status": "RUNTIME_DIVERSITY_VALIDATED",
            },
            lifecycle_evidence={
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "lifecycle_activity_observed": True,
                "compression_specifically_verified": True,
                "compression_removed_count": 2,
                "archived_count": 1,
                "pruned_count": 3,
                "compressed_cluster_count": 1,
                "compressed_cluster_count_status": "REVALIDATED_THIS_ROUND",
                "probe_results": [],
            },
        )
        self.assertEqual(payload["controlled_ready_status"]["status"], "CONTROLLED_READY_VERIFIED")
        self.assertEqual(payload["open_ready_status"]["status"], "OPEN_READY_VERIFIED")
        self.assertEqual(payload["final_recommendation"]["status"], "OPEN_READY_VERIFIED")

    def test_open_ready_is_not_verified_without_current_round_lifecycle_probe(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(compression_removed=4, archived=1, pruned=2),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage={
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "family_schema_count": 4,
                "runtime_validated_family_count": 4,
                "family_coverage_status": "RUNTIME_DIVERSITY_VALIDATED",
            },
        )
        self.assertEqual(payload["controlled_ready_status"]["status"], "CONTROLLED_READY_VERIFIED")
        self.assertEqual(payload["open_ready_status"]["status"], "NOT_VERIFIED")

    def test_generator_rebuilds_expected_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "m3_readiness_metrics.json"
            report_path = Path(tmp_dir) / "m3_readiness_repair_report.md"

            # Seed the comparison claim so the generator can carry it forward honestly.
            metrics_path.write_text(
                json.dumps(_previous_readiness_claim(), ensure_ascii=True),
                encoding="utf-8",
            )
            payload = write_readiness_outputs(
                metrics_path=metrics_path,
                report_path=report_path,
                test_evidence={
                    "verification_status": "REVALIDATED_THIS_ROUND",
                    "evidence_origin": "test",
                    "revalidated_this_round": True,
                    "passed_count": 31,
                    "failed_count": 0,
                    "output_summary": "31 passed in 0.35s",
                },
                family_coverage={
                    "verification_status": "REVALIDATED_THIS_ROUND",
                    "evidence_origin": "test",
                    "revalidated_this_round": True,
                    "family_schema_count": 4,
                    "runtime_validated_family_count": 4,
                    "family_coverage_status": "RUNTIME_DIVERSITY_VALIDATED",
                },
                lifecycle_evidence={
                    "verification_status": "REVALIDATED_THIS_ROUND",
                    "evidence_origin": "test",
                    "revalidated_this_round": True,
                    "lifecycle_activity_observed": True,
                    "compression_specifically_verified": True,
                    "compression_removed_count": 2,
                    "archived_count": 1,
                    "pruned_count": 3,
                    "compressed_cluster_count": 1,
                    "compressed_cluster_count_status": "REVALIDATED_THIS_ROUND",
                    "probe_results": [],
                },
            )
            self.assertTrue(metrics_path.exists())
            self.assertTrue(report_path.exists())
            restored = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                restored["final_recommendation"]["status"],
                payload["final_recommendation"]["status"],
            )
            self.assertEqual(restored["open_ready_status"]["status"], "OPEN_READY_VERIFIED")
            self.assertIn("evals/m3_readiness_evaluation.py", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
