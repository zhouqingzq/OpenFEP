from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import segmentum.m231_audit as m231_audit
from segmentum.m231_audit import (
    M231_ABLATION_PATH,
    M231_REPORT_PATH,
    M231_SPEC_PATH,
    M231_STRESS_PATH,
    M231_SUMMARY_PATH,
    M231_TRACE_PATH,
    write_m231_acceptance_artifacts,
)


def _suite_execution(paths: tuple[str, ...], *, passed: bool = True) -> dict[str, object]:
    return {
        "label": "test-suite",
        "paths": list(paths),
        "executed": True,
        "passed": passed,
        "returncode": 0 if passed else 1,
        "command": ["py", "-3.11", "-m", "pytest", *paths, "-q"],
        "stdout": "simulated pytest run",
        "stderr": "",
        "execution_source": "injected",
        "started_at": "2026-03-23T14:00:00+00:00",
        "completed_at": "2026-03-23T14:00:01+00:00",
    }


@contextmanager
def _isolated_m231_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        artifacts_dir = root / "artifacts"
        reports_dir = root / "reports"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        original_paths = {
            "M231_TRACE_PATH": m231_audit.M231_TRACE_PATH,
            "M231_ABLATION_PATH": m231_audit.M231_ABLATION_PATH,
            "M231_STRESS_PATH": m231_audit.M231_STRESS_PATH,
            "M231_REPORT_PATH": m231_audit.M231_REPORT_PATH,
            "M231_SUMMARY_PATH": m231_audit.M231_SUMMARY_PATH,
        }
        m231_audit.M231_TRACE_PATH = artifacts_dir / "m231_reconciliation_trace.jsonl"
        m231_audit.M231_ABLATION_PATH = artifacts_dir / "m231_reconciliation_ablation.json"
        m231_audit.M231_STRESS_PATH = artifacts_dir / "m231_reconciliation_stress.json"
        m231_audit.M231_REPORT_PATH = reports_dir / "m231_acceptance_report.json"
        m231_audit.M231_SUMMARY_PATH = reports_dir / "m231_acceptance_summary.md"
        try:
            yield {
                "trace": m231_audit.M231_TRACE_PATH,
                "ablation": m231_audit.M231_ABLATION_PATH,
                "stress": m231_audit.M231_STRESS_PATH,
                "report": m231_audit.M231_REPORT_PATH,
                "summary": m231_audit.M231_SUMMARY_PATH,
            }
        finally:
            for name, value in original_paths.items():
                setattr(m231_audit, name, value)


class TestM231AcceptanceArtifacts(unittest.TestCase):
    def test_report_contains_audit_fields_in_non_strict_mode(self) -> None:
        with _isolated_m231_outputs() as isolated:
            write_m231_acceptance_artifacts(
                strict=False,
                milestone_execution=_suite_execution(
                    (
                        "tests/test_m231_reconciliation_threads.py",
                        "tests/test_m231_narrative_writeback.py",
                        "tests/test_m231_acceptance.py",
                    )
                ),
                regression_execution=_suite_execution(
                    (
                        "tests/test_m229_acceptance.py",
                        "tests/test_m230_acceptance.py",
                        "tests/test_narrative_evolution.py",
                    )
                )
            )
            report = json.loads(Path(isolated["report"]).read_text(encoding="utf-8"))

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
                self.assertIn(field, report)

            self.assertEqual(report["milestone_id"], "M2.31")
            self.assertFalse(report["strict"])
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["recommendation"], "ACCEPT")
            self.assertTrue(report["gates"]["determinism"]["passed"])
            self.assertTrue(report["gates"]["long_horizon"]["passed"])
            self.assertTrue(report["gates"]["narrative_alignment"]["passed"])
            self.assertTrue(report["gates"]["claim_alignment"]["passed"])
            self.assertTrue(report["gates"]["ablation"]["passed"])
            self.assertTrue(report["gates"]["stress"]["passed"])
            self.assertTrue(report["gates"]["milestone_tests"]["passed"])
            self.assertTrue(report["gates"]["regression"]["passed"])
            self.assertTrue(report["gates"]["artifact_freshness"]["passed"])
            self.assertEqual(report["residual_risks"], [])

            self.assertTrue(Path(M231_SPEC_PATH).exists(), str(M231_SPEC_PATH))
            for path in isolated.values():
                self.assertTrue(Path(path).exists(), str(path))

    def test_artifact_payloads_include_reconciliation_writeback_specific_evidence(self) -> None:
        with _isolated_m231_outputs():
            written = write_m231_acceptance_artifacts(
                strict=False,
                milestone_execution=_suite_execution(
                    (
                        "tests/test_m231_reconciliation_threads.py",
                        "tests/test_m231_narrative_writeback.py",
                        "tests/test_m231_acceptance.py",
                    )
                ),
                regression_execution=_suite_execution(
                    (
                        "tests/test_m229_acceptance.py",
                        "tests/test_m230_acceptance.py",
                        "tests/test_narrative_evolution.py",
                    )
                ),
            )
            trace_lines = Path(written["trace"]).read_text(encoding="utf-8").strip().splitlines()
            ablation = json.loads(Path(written["ablation"]).read_text(encoding="utf-8"))
            stress = json.loads(Path(written["stress"]).read_text(encoding="utf-8"))
            report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))

            self.assertTrue(trace_lines)
            self.assertIn('"current_chapter_reconciliation"', trace_lines[-1])
            self.assertIn('"contradiction_reconciliation"', trace_lines[-1])
            self.assertIn('"target_thread_summary"', trace_lines[-1])
            self.assertIn('"narrative_alignment"', trace_lines[-1])
            self.assertIn('"claim_alignment"', trace_lines[-1])
            self.assertIn('"core_summary"', trace_lines[-1])
            self.assertIn("degradation_checks", ablation)
            self.assertGreaterEqual(
                len(ablation["full_mechanism"]["target_thread"]["linked_chapter_ids"]),
                2,
            )
            self.assertGreaterEqual(
                ablation["full_mechanism"]["target_thread"]["chapter_bridge_count"],
                1,
            )
            self.assertTrue(
                ablation["degradation_checks"]["core_summary_loses_reconciliation_clause_without_writeback"]
            )
            self.assertTrue(
                ablation["degradation_checks"]["cross_chapter_thread_survives_with_writeback"]
            )
            self.assertTrue(
                ablation["degradation_checks"]["writeback_targets_reconciled_thread_with_writeback"]
            )
            self.assertTrue(stress["stress_checks"]["unrelated_evidence_did_not_contaminate_other_thread"])
            self.assertTrue(stress["stress_checks"]["unmatched_repair_did_not_bind"])
            self.assertTrue(stress["stress_checks"]["cross_chapter_links_survived_stress"])
            self.assertTrue(stress["stress_checks"]["narrative_writeback_survived_stress"])
            self.assertTrue(stress["stress_checks"]["narrative_writeback_still_targets_intended_thread"])
            self.assertTrue(report["tests"]["milestone"]["executed"])
            self.assertTrue(report["tests"]["regressions"]["executed"])

    def test_strict_mode_rejects_injected_execution_records(self) -> None:
        with _isolated_m231_outputs():
            with self.assertRaisesRegex(ValueError, "refuses injected execution records"):
                write_m231_acceptance_artifacts(
                    milestone_execution=_suite_execution(
                        (
                            "tests/test_m231_reconciliation_threads.py",
                            "tests/test_m231_narrative_writeback.py",
                            "tests/test_m231_acceptance.py",
                        )
                    ),
                    regression_execution=_suite_execution(
                        (
                            "tests/test_m229_acceptance.py",
                            "tests/test_m230_acceptance.py",
                            "tests/test_narrative_evolution.py",
                        )
                    ),
                )

    def test_strict_mode_without_real_execution_blocks_report(self) -> None:
        with _isolated_m231_outputs() as isolated:
            write_m231_acceptance_artifacts(strict=True, execute_test_suites=False)
            report = json.loads(Path(isolated["report"]).read_text(encoding="utf-8"))

            self.assertTrue(report["strict"])
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["recommendation"], "BLOCK")
            self.assertFalse(report["gates"]["milestone_tests"]["passed"])
            self.assertFalse(report["gates"]["regression"]["passed"])
            self.assertFalse(report["gates"]["artifact_freshness"]["passed"])
            self.assertFalse(report["freshness"]["current_round"])


if __name__ == "__main__":
    unittest.main()
