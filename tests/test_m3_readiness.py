from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from evals.m3_readiness_evaluation import (
    build_readiness_payload,
    derive_controlled_ready_status,
    derive_historical_regression_evidence,
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


def _current_round_followup_payload() -> dict[str, object]:
    payload = _followup_payload()
    payload["_current_round_replay"] = True
    payload["_evidence_origin"] = "evals/m2_followup_repair.py:run_followup_evaluation"
    payload["seed_set"] = [11, 21, 31, 33, 42]
    return payload


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


def _historical_regressions(
    *,
    generated_at: str | None = None,
    runtime_family_coverage_passed: bool = False,
) -> dict[str, object]:
    return {
        "_artifact_path": "artifacts/pre_m3_regression_summary.json",
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "acceptance": {
            "passed": runtime_family_coverage_passed,
            "checks": {
                "soak_regression_passed": True,
                "snapshot_compatibility_passed": True,
                "runtime_lifecycle_passed": True,
                "runtime_family_coverage_passed": runtime_family_coverage_passed,
            },
        },
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
            "NOT_READY_FOR_M3",
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
        self.assertEqual(coverage["runtime_validated_family_count"], 4)
        self.assertEqual(coverage["family_coverage_status"], "RUNTIME_DIVERSITY_VALIDATED")
        self.assertEqual(coverage["evidence_kind"], "runtime_replay")
        self.assertTrue(coverage["fully_graduated"])
        self.assertEqual(coverage["missing_graduation_families"], [])

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
            historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
        )["strict_metrics"]
        status = derive_controlled_ready_status(
            strict_metrics,
            {
                "family_coverage_status": "RUNTIME_DIVERSITY_VALIDATED",
                "evidence_kind": "runtime_replay",
                "runtime_validated_family_count": 4,
                "fully_graduated": True,
                "missing_graduation_families": [],
            },
            {
                "verification_status": "REVALIDATED_THIS_ROUND",
            },
        )
        self.assertEqual(status.status, "CONTROLLED_READY_CANDIDATE")

    def test_build_payload_does_not_promote_pre_m3_gate_over_unverified_readiness(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "suite_type": "readiness_targets",
                "suite_scope": "targeted_readiness_tests_only",
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "targets": ["tests/test_m3_readiness.py"],
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage={
                "verification_status": "FRAMEWORK_SCHEMA_PROBE_ONLY",
                "evidence_origin": "test",
                "revalidated_this_round": False,
                "evidence_kind": "framework_schema_probe",
                "family_schema_count": 4,
                "runtime_validated_family_count": 0,
                "fully_graduated": False,
                "missing_graduation_families": list(["danger_avoidance", "resource_risk"]),
                "family_coverage_status": "FRAMEWORK_ONLY_NOT_RUNTIME_VALIDATED",
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
            pre_m3_readiness={
                "status": "READY_FOR_M3",
                "passed": True,
                "evidence_origin": "test",
                "revalidated_this_round": True,
            },
            historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
        )
        self.assertEqual(payload["controlled_ready_status"]["status"], "CONTROLLED_READY_CANDIDATE")
        self.assertEqual(payload["open_ready_status"]["status"], "NOT_VERIFIED")
        self.assertEqual(payload["final_recommendation"]["status"], "NOT_READY_FOR_M3")
        self.assertIn("controlled readiness is not verified", payload["final_recommendation"]["why_more_conservative"])

    def test_open_ready_is_not_verified_without_current_round_lifecycle_probe(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(compression_removed=4, archived=1, pruned=2),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "suite_type": "readiness_targets",
                "suite_scope": "targeted_readiness_tests_only",
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "targets": ["tests/test_m3_readiness.py"],
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage={
                "verification_status": "FRAMEWORK_SCHEMA_PROBE_ONLY",
                "evidence_origin": "test",
                "revalidated_this_round": False,
                "evidence_kind": "framework_schema_probe",
                "family_schema_count": 4,
                "runtime_validated_family_count": 0,
                "fully_graduated": False,
                "missing_graduation_families": list(["danger_avoidance", "resource_risk"]),
                "family_coverage_status": "FRAMEWORK_ONLY_NOT_RUNTIME_VALIDATED",
            },
            historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
        )
        self.assertEqual(payload["controlled_ready_status"]["status"], "CONTROLLED_READY_CANDIDATE")
        self.assertEqual(payload["open_ready_status"]["status"], "NOT_VERIFIED")

    def test_report_distinguishes_readiness_targets_from_historical_regressions(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "suite_type": "readiness_targets",
                "suite_scope": "targeted_readiness_tests_only",
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "targets": list(
                    [
                        "tests/test_m3_readiness.py",
                        "tests/test_memory.py",
                        "tests/test_counterfactual_artifact.py",
                        "tests/test_m23_ultimate_consolidation_loop.py",
                    ]
                ),
                "target_count": 4,
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage=evaluate_runtime_family_coverage(),
            lifecycle_evidence=evaluate_runtime_lifecycle_evidence(),
            historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
        )
        self.assertFalse(payload["tests"]["coverage_boundary"]["readiness_targets_are_full_regression"])
        self.assertEqual(payload["tests"]["readiness_targets"]["target_count"], 4)
        self.assertEqual(
            payload["tests"]["historical_regressions"]["suite_type"],
            "historical_regressions",
        )

    def test_report_contains_strict_audit_framework_fields(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
        )
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
            self.assertIn(field, payload)
        self.assertTrue(payload["seed_set"])

    def test_inherited_strict_metrics_keep_recommendation_consistent(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "suite_type": "readiness_targets",
                "suite_scope": "targeted_readiness_tests_only",
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "targets": ["tests/test_m3_readiness.py"],
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage=evaluate_runtime_family_coverage(),
            lifecycle_evidence=evaluate_runtime_lifecycle_evidence(),
            pre_m3_readiness={
                "status": "READY_FOR_M3",
                "passed": True,
                "evidence_origin": "test",
                "revalidated_this_round": True,
            },
            historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
        )
        self.assertFalse(payload["gates"]["strict_metrics_current_round_replayed"])
        self.assertEqual(payload["recommendation"]["status"], "NOT_READY_FOR_M3")

    def test_current_round_followup_payload_marks_strict_metrics_revalidated(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_current_round_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "suite_type": "readiness_targets",
                "suite_scope": "targeted_readiness_tests_only",
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "targets": ["tests/test_m3_readiness.py"],
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage=evaluate_runtime_family_coverage(),
            lifecycle_evidence=evaluate_runtime_lifecycle_evidence(),
            historical_regressions=derive_historical_regression_evidence(
                _historical_regressions(runtime_family_coverage_passed=True)
            ),
            pre_m3_readiness={
                "status": "READY_FOR_M3",
                "passed": True,
                "evidence_origin": "test",
                "revalidated_this_round": True,
            },
            strict_seed_set=[11, 21, 31, 33, 42],
        )
        self.assertTrue(payload["gates"]["strict_metrics_current_round_replayed"])
        self.assertEqual(payload["recommendation"]["status"], "READY_FOR_M3")
        self.assertEqual(payload["freshness"]["family_coverage"]["classification"], "current_round_replay")

    def test_partial_runtime_family_coverage_remains_blocking(self) -> None:
        payload = build_readiness_payload(
            followup_payload=_current_round_followup_payload(),
            legacy_payload=_legacy_payload(),
            previous_readiness=_previous_readiness_claim(),
            test_evidence={
                "suite_type": "readiness_targets",
                "suite_scope": "targeted_readiness_tests_only",
                "verification_status": "REVALIDATED_THIS_ROUND",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "targets": ["tests/test_m3_readiness.py"],
                "passed_count": 31,
                "failed_count": 0,
                "output_summary": "31 passed in 0.35s",
            },
            family_coverage={
                "verification_status": "LIMITED_RUNTIME_REPLAY",
                "evidence_origin": "test",
                "revalidated_this_round": True,
                "evidence_kind": "runtime_replay",
                "family_schema_count": 4,
                "runtime_validated_family_count": 3,
                "fully_graduated": False,
                "missing_graduation_families": ["resource_risk"],
                "family_coverage_status": "PARTIAL_RUNTIME_VALIDATED",
                "coverage_summary": {
                    "per_family_rates": {
                        "resource_risk": {"reviewed": 1, "graduated": 0},
                    }
                },
                "limitations": [
                    "Runtime replay is current-round evidence, but the following families still lack real graduation: resource_risk."
                ],
            },
            lifecycle_evidence=evaluate_runtime_lifecycle_evidence(),
            historical_regressions=derive_historical_regression_evidence(
                _historical_regressions(runtime_family_coverage_passed=False)
            ),
            pre_m3_readiness={
                "status": "NOT_READY_FOR_M3",
                "passed": False,
                "evidence_origin": "test",
                "revalidated_this_round": True,
            },
        )
        self.assertFalse(payload["gates"]["runtime_family_replay_verified"])
        self.assertEqual(payload["controlled_ready_status"]["status"], "CONTROLLED_READY_CANDIDATE")
        self.assertIn("resource_risk", " ".join(payload["residual_risks"]))
        self.assertTrue(
            any(
                finding["title"] == "Runtime family replay is still missing family graduation"
                for finding in payload["findings"]
            )
        )

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
                followup_payload=_followup_payload(),
                test_evidence={
                    "suite_type": "readiness_targets",
                    "suite_scope": "targeted_readiness_tests_only",
                    "verification_status": "REVALIDATED_THIS_ROUND",
                    "evidence_origin": "test",
                    "revalidated_this_round": True,
                    "targets": ["tests/test_m3_readiness.py"],
                    "passed_count": 31,
                    "failed_count": 0,
                    "output_summary": "31 passed in 0.35s",
                },
                family_coverage={
                    "verification_status": "FRAMEWORK_SCHEMA_PROBE_ONLY",
                    "evidence_origin": "test",
                    "revalidated_this_round": False,
                    "evidence_kind": "framework_schema_probe",
                    "family_schema_count": 4,
                    "runtime_validated_family_count": 0,
                    "family_coverage_status": "FRAMEWORK_ONLY_NOT_RUNTIME_VALIDATED",
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
                historical_regressions=derive_historical_regression_evidence(_historical_regressions()),
            )
            self.assertTrue(metrics_path.exists())
            self.assertTrue(report_path.exists())
            restored = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                restored["final_recommendation"]["status"],
                payload["final_recommendation"]["status"],
            )
            self.assertEqual(restored["open_ready_status"]["status"], "NOT_VERIFIED")
            self.assertIn("evals/m3_readiness_evaluation.py", report_path.read_text(encoding="utf-8"))
            self.assertEqual(restored["recommendation"]["status"], "NOT_READY_FOR_M3")


if __name__ == "__main__":
    unittest.main()
