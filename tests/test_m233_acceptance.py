from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import segmentum.m233_audit as m233_audit
from segmentum.m233_audit import (
    M233_REPORT_PATH,
    M233_SPEC_PATH,
    build_m233_runtime_evidence,
    write_m233_acceptance_artifacts,
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
        "started_at": "2026-03-25T10:00:00+00:00",
        "completed_at": "2026-03-25T10:00:01+00:00",
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
            "M233_TRACE_PATH": m233_audit.M233_TRACE_PATH,
            "M233_ABLATION_PATH": m233_audit.M233_ABLATION_PATH,
            "M233_STRESS_PATH": m233_audit.M233_STRESS_PATH,
            "M233_REPORT_PATH": m233_audit.M233_REPORT_PATH,
            "M233_SUMMARY_PATH": m233_audit.M233_SUMMARY_PATH,
        }
        m233_audit.M233_TRACE_PATH = artifacts_dir / "m233_uncertainty_trace.jsonl"
        m233_audit.M233_ABLATION_PATH = artifacts_dir / "m233_uncertainty_ablation.json"
        m233_audit.M233_STRESS_PATH = artifacts_dir / "m233_uncertainty_stress.json"
        m233_audit.M233_REPORT_PATH = reports_dir / "m233_acceptance_report.json"
        m233_audit.M233_SUMMARY_PATH = reports_dir / "m233_acceptance_summary.md"
        try:
            yield {
                "trace": m233_audit.M233_TRACE_PATH,
                "ablation": m233_audit.M233_ABLATION_PATH,
                "stress": m233_audit.M233_STRESS_PATH,
                "report": m233_audit.M233_REPORT_PATH,
                "summary": m233_audit.M233_SUMMARY_PATH,
            }
        finally:
            for name, value in original_paths.items():
                setattr(m233_audit, name, value)


class TestM233AcceptanceArtifacts(unittest.TestCase):
    def test_runtime_evidence_captures_unknowns_and_downstream_effects(self) -> None:
        evidence = build_m233_runtime_evidence()

        self.assertEqual(evidence["trace_records"][0]["event"], "unknowns_extracted")
        self.assertTrue(evidence["trace_records"][0]["unknowns"])
        self.assertEqual(evidence["trace_records"][1]["event"], "downstream_promotion")
        self.assertTrue(evidence["trace_records"][1]["ledger_created_predictions"])
        self.assertTrue(evidence["gates"]["competition"]["passed"])
        self.assertTrue(evidence["gates"]["downstream_causality"]["passed"])

    def test_report_contains_expected_fields_in_non_strict_mode(self) -> None:
        with _isolated_outputs() as isolated:
            write_m233_acceptance_artifacts(
                strict=False,
                milestone_execution=_suite_execution(m233_audit.M233_TESTS),
                regression_execution=_suite_execution(m233_audit.M233_REGRESSIONS),
            )
            report = json.loads(Path(isolated["report"]).read_text(encoding="utf-8"))

            self.assertEqual(report["milestone_id"], "M2.33")
            self.assertFalse(report["strict"])
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["recommendation"], "ACCEPT")
            self.assertTrue(report["gates"]["competition"]["passed"])
            self.assertTrue(report["gates"]["downstream_causality"]["passed"])
            self.assertTrue(report["gates"]["surface_latent_separation"]["passed"])
            self.assertTrue(report["gates"]["snapshot_roundtrip"]["passed"])
            self.assertTrue(report["gates"]["regression"]["passed"])
            self.assertTrue(report["gates"]["artifact_freshness"]["passed"])

            self.assertTrue(Path(M233_SPEC_PATH).exists(), str(M233_SPEC_PATH))
            for path in isolated.values():
                self.assertTrue(Path(path).exists(), str(path))

    def test_freshness_gate_rejects_stale_report_and_summary(self) -> None:
        with _isolated_outputs() as isolated:
            Path(isolated["report"]).write_text("{}", encoding="utf-8")
            Path(isolated["summary"]).write_text("# stale\n", encoding="utf-8")
            stale_mtime = time.time() - 3600
            os.utime(isolated["report"], (stale_mtime, stale_mtime))
            os.utime(isolated["summary"], (stale_mtime, stale_mtime))

            audit_started_at = m233_audit._now_iso()
            evidence = build_m233_runtime_evidence()
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
            generated_at = m233_audit._now_iso()

            freshness_ok, freshness = m233_audit._freshness_gate(
                artifacts={
                    "canonical_trace": str(isolated["trace"]),
                    "ablation": str(isolated["ablation"]),
                    "stress": str(isolated["stress"]),
                    "report": str(isolated["report"]),
                    "summary": str(isolated["summary"]),
                },
                audit_started_at=audit_started_at,
                generated_at=generated_at,
                milestone_execution=_suite_execution(m233_audit.M233_TESTS),
                regression_execution=_suite_execution(m233_audit.M233_REGRESSIONS),
                strict=False,
            )

            self.assertFalse(freshness_ok)
            self.assertTrue(freshness["evidence_times_within_round"])
            self.assertFalse(freshness["report_times_within_round"])
            self.assertFalse(freshness["current_round"])

    def test_strict_mode_rejects_injected_execution_records(self) -> None:
        with _isolated_outputs():
            with self.assertRaisesRegex(ValueError, "refuses injected execution records"):
                write_m233_acceptance_artifacts(
                    milestone_execution=_suite_execution(m233_audit.M233_TESTS),
                    regression_execution=_suite_execution(m233_audit.M233_REGRESSIONS),
                )


if __name__ == "__main__":
    unittest.main()
