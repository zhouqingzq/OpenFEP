from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import segmentum.m235_audit as m235_audit
from segmentum.m235_audit import (
    M235_REPORT_PATH,
    M235_SPEC_PATH,
    SCHEMA_VERSION,
    build_m235_runtime_evidence,
    write_m235_acceptance_artifacts,
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
        "started_at": "2026-03-25T13:40:00+00:00",
        "completed_at": "2026-03-25T13:40:01+00:00",
    }


@contextmanager
def _isolated_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        artifacts_dir = root / "artifacts"
        reports_dir = root / "reports"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        original_paths = {
            "M235_TRACE_PATH": m235_audit.M235_TRACE_PATH,
            "M235_ABLATION_PATH": m235_audit.M235_ABLATION_PATH,
            "M235_STRESS_PATH": m235_audit.M235_STRESS_PATH,
            "M235_REPORT_PATH": m235_audit.M235_REPORT_PATH,
            "M235_SUMMARY_PATH": m235_audit.M235_SUMMARY_PATH,
        }
        m235_audit.M235_TRACE_PATH = artifacts_dir / "m235_inquiry_scheduler_trace.jsonl"
        m235_audit.M235_ABLATION_PATH = artifacts_dir / "m235_inquiry_scheduler_ablation.json"
        m235_audit.M235_STRESS_PATH = artifacts_dir / "m235_inquiry_scheduler_stress.json"
        m235_audit.M235_REPORT_PATH = reports_dir / "m235_acceptance_report.json"
        m235_audit.M235_SUMMARY_PATH = reports_dir / "m235_acceptance_summary.md"
        try:
            yield {
                "trace": m235_audit.M235_TRACE_PATH,
                "ablation": m235_audit.M235_ABLATION_PATH,
                "stress": m235_audit.M235_STRESS_PATH,
                "report": m235_audit.M235_REPORT_PATH,
                "summary": m235_audit.M235_SUMMARY_PATH,
            }
        finally:
            for name, value in original_paths.items():
                setattr(m235_audit, name, value)


class TestM235AcceptanceArtifacts(unittest.TestCase):
    def test_runtime_evidence_captures_scheduler_ranking_and_downstream_consumption(self) -> None:
        evidence = build_m235_runtime_evidence()

        self.assertEqual(evidence["trace_records"][0]["event"], "scheduler_candidate_ranking")
        self.assertEqual(evidence["trace_records"][0]["schema_version"], SCHEMA_VERSION)
        self.assertTrue(evidence["trace_records"][0]["active_candidate_ids"])
        self.assertEqual(evidence["trace_records"][2]["event"], "runtime_consumption")
        self.assertTrue(evidence["trace_records"][2]["workspace_focus"])
        self.assertTrue(evidence["gates"]["cross_surface_ranking"]["passed"])
        self.assertTrue(evidence["gates"]["verification_budgeting"]["passed"])
        self.assertTrue(evidence["gates"]["workspace_allocation"]["passed"])
        self.assertTrue(evidence["gates"]["action_biasing"]["passed"])
        self.assertTrue(evidence["gates"]["downstream_causality"]["passed"])
        self.assertTrue(evidence["stress"]["stress_checks"]["replay_same_seed_equivalent"])
        self.assertTrue(evidence["stress"]["stress_checks"]["replay_multi_seed_equivalent"])
        self.assertEqual(
            len(evidence["stress"]["details"]["determinism_replay_signatures"]),
            len(m235_audit.SEED_SET),
        )

    def test_report_contains_expected_fields_in_non_strict_mode(self) -> None:
        with _isolated_outputs() as isolated:
            write_m235_acceptance_artifacts(
                strict=False,
                milestone_execution=_suite_execution(m235_audit.M235_TESTS),
                regression_execution=_suite_execution(m235_audit.M235_REGRESSIONS),
            )
            report = json.loads(Path(isolated["report"]).read_text(encoding="utf-8"))

            self.assertEqual(report["milestone_id"], "M2.35")
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertFalse(report["strict"])
            self.assertIn(report["status"], {"PASS", "PASS_WITH_RESIDUAL_RISK"})
            self.assertIn(report["recommendation"], {"ACCEPT", "ACCEPT_WITH_RESIDUAL_RISK"})
            self.assertTrue(report["gates"]["cross_surface_ranking"]["passed"])
            self.assertTrue(report["gates"]["verification_budgeting"]["passed"])
            self.assertTrue(report["gates"]["workspace_allocation"]["passed"])
            self.assertTrue(report["gates"]["action_biasing"]["passed"])
            self.assertTrue(report["gates"]["downstream_causality"]["passed"])
            self.assertTrue(report["gates"]["snapshot_roundtrip"]["passed"])
            self.assertTrue(report["gates"]["regression"]["passed"])
            self.assertTrue(report["gates"]["artifact_freshness"]["passed"])

            self.assertTrue(Path(M235_SPEC_PATH).exists(), str(M235_SPEC_PATH))
            for path in isolated.values():
                self.assertTrue(Path(path).exists(), str(path))

    def test_freshness_gate_rejects_stale_report_and_summary(self) -> None:
        with _isolated_outputs() as isolated:
            Path(isolated["report"]).write_text("{}", encoding="utf-8")
            Path(isolated["summary"]).write_text("# stale\n", encoding="utf-8")
            stale_mtime = time.time() - 3600
            os.utime(isolated["report"], (stale_mtime, stale_mtime))
            os.utime(isolated["summary"], (stale_mtime, stale_mtime))

            audit_started_at = m235_audit._now_iso()
            evidence = build_m235_runtime_evidence()
            Path(isolated["trace"]).write_text(
                "\n".join(json.dumps(record, ensure_ascii=True, sort_keys=True) for record in evidence["trace_records"])
                + "\n",
                encoding="utf-8",
            )
            Path(isolated["ablation"]).write_text(
                json.dumps(evidence["ablation"], indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            Path(isolated["stress"]).write_text(
                json.dumps(evidence["stress"], indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            generated_at = m235_audit._now_iso()

            freshness_ok, freshness = m235_audit._freshness_gate(
                artifacts={
                    "canonical_trace": str(isolated["trace"]),
                    "ablation": str(isolated["ablation"]),
                    "stress": str(isolated["stress"]),
                    "report": str(isolated["report"]),
                    "summary": str(isolated["summary"]),
                },
                audit_started_at=audit_started_at,
                generated_at=generated_at,
                milestone_execution=_suite_execution(m235_audit.M235_TESTS),
                regression_execution=_suite_execution(m235_audit.M235_REGRESSIONS),
                strict=False,
            )

            self.assertFalse(freshness_ok)
            self.assertTrue(freshness["evidence_times_within_round"])
            self.assertFalse(freshness["report_times_within_round"])
            self.assertFalse(freshness["current_round"])

    def test_strict_mode_rejects_injected_execution_records(self) -> None:
        with _isolated_outputs():
            with self.assertRaisesRegex(ValueError, "refuses injected execution records"):
                write_m235_acceptance_artifacts(
                    milestone_execution=_suite_execution(m235_audit.M235_TESTS),
                    regression_execution=_suite_execution(m235_audit.M235_REGRESSIONS),
                )

    def test_strict_dirty_findings_ignore_current_round_generated_artifacts(self) -> None:
        findings = m235_audit._strict_dirty_findings(
            [
                "?? artifacts/m235_inquiry_scheduler_trace.jsonl",
                "?? reports/m235_acceptance_report.json",
                "?? .pytest_m235_acceptance.log",
            ]
        )
        self.assertEqual(findings, [])

    def test_strict_dirty_findings_flag_code_and_spec_changes(self) -> None:
        findings = m235_audit._strict_dirty_findings(
            [
                "M segmentum/agent.py",
                "?? segmentum/inquiry_scheduler.py",
                "?? reports/m235_milestone_spec.md",
            ]
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["severity"], "S1")
        self.assertIn("segmentum/agent.py", findings[0]["paths"])


if __name__ == "__main__":
    unittest.main()
