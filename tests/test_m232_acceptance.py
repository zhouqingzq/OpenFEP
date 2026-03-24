from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import segmentum.m232_audit as m232_audit
from segmentum.m232_audit import (
    M232_REPORT_PATH,
    M232_SPEC_PATH,
    build_m232_runtime_evidence,
    write_m232_acceptance_artifacts,
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
        "started_at": "2026-03-24T10:00:00+00:00",
        "completed_at": "2026-03-24T10:00:01+00:00",
    }


@contextmanager
def _isolated_m232_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        artifacts_dir = root / "artifacts"
        reports_dir = root / "reports"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        original_paths = {
            "M232_TRACE_PATH": m232_audit.M232_TRACE_PATH,
            "M232_ABLATION_PATH": m232_audit.M232_ABLATION_PATH,
            "M232_STRESS_PATH": m232_audit.M232_STRESS_PATH,
            "M232_REPORT_PATH": m232_audit.M232_REPORT_PATH,
            "M232_SUMMARY_PATH": m232_audit.M232_SUMMARY_PATH,
        }
        m232_audit.M232_TRACE_PATH = artifacts_dir / "m232_threat_memory_trace.jsonl"
        m232_audit.M232_ABLATION_PATH = artifacts_dir / "m232_threat_memory_ablation.json"
        m232_audit.M232_STRESS_PATH = artifacts_dir / "m232_threat_memory_stress.json"
        m232_audit.M232_REPORT_PATH = reports_dir / "m232_acceptance_report.json"
        m232_audit.M232_SUMMARY_PATH = reports_dir / "m232_acceptance_summary.md"
        try:
            yield {
                "trace": m232_audit.M232_TRACE_PATH,
                "ablation": m232_audit.M232_ABLATION_PATH,
                "stress": m232_audit.M232_STRESS_PATH,
                "report": m232_audit.M232_REPORT_PATH,
                "summary": m232_audit.M232_SUMMARY_PATH,
            }
        finally:
            for name, value in original_paths.items():
                setattr(m232_audit, name, value)


class TestM232AcceptanceArtifacts(unittest.TestCase):
    def test_runtime_evidence_is_derived_from_real_mechanism_outputs(self) -> None:
        evidence = build_m232_runtime_evidence()

        trace_records = evidence["trace_records"]
        ablation = evidence["ablation"]
        stress = evidence["stress"]

        self.assertEqual(len(trace_records), 2)
        self.assertEqual(trace_records[0]["event"], "protected_anchor_created")
        self.assertTrue(trace_records[0]["restart_protected"])
        self.assertIn("structural_trace", trace_records[0]["continuity_tags"])
        self.assertIn("chronic_threat=", trace_records[0]["memory_context_summary"])
        self.assertEqual(trace_records[1]["event"], "attention_and_prediction_shift")
        self.assertIn("danger", trace_records[1]["attention_selected_channels"])
        self.assertGreater(trace_records[1]["prediction_delta"]["danger"], 0.0)
        self.assertTrue(trace_records[1]["protected_anchor_retrieved"])
        self.assertTrue(ablation["degradation_checks"]["prediction_shift_degrades_without_threat_trace"])
        self.assertTrue(ablation["degradation_checks"]["attention_shift_degrades_without_sensitive_pattern_bias"])
        self.assertTrue(stress["stress_checks"]["roundtrip_retrieval_preserves_anchor"])

    def test_report_contains_expected_fields_in_non_strict_mode(self) -> None:
        with _isolated_m232_outputs() as isolated:
            write_m232_acceptance_artifacts(
                strict=False,
                milestone_execution=_suite_execution(m232_audit.M232_TESTS),
                regression_execution=_suite_execution(m232_audit.M232_REGRESSIONS),
            )
            report = json.loads(Path(isolated["report"]).read_text(encoding="utf-8"))

            self.assertEqual(report["milestone_id"], "M2.32")
            self.assertFalse(report["strict"])
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["recommendation"], "ACCEPT")
            self.assertTrue(report["gates"]["schema"]["passed"])
            self.assertTrue(report["gates"]["protection"]["passed"])
            self.assertTrue(report["gates"]["causality"]["passed"])
            self.assertTrue(report["gates"]["attention_prediction_influence"]["passed"])
            self.assertTrue(report["gates"]["regression"]["passed"])
            self.assertTrue(report["gates"]["artifact_freshness"]["passed"])

            self.assertTrue(Path(M232_SPEC_PATH).exists(), str(M232_SPEC_PATH))
            for path in isolated.values():
                self.assertTrue(Path(path).exists(), str(path))

    def test_artifacts_capture_threat_trace_specific_evidence(self) -> None:
        with _isolated_m232_outputs() as isolated:
            write_m232_acceptance_artifacts(
                strict=False,
                milestone_execution=_suite_execution(m232_audit.M232_TESTS),
                regression_execution=_suite_execution(m232_audit.M232_REGRESSIONS),
            )
            trace_lines = Path(isolated["trace"]).read_text(encoding="utf-8").strip().splitlines()
            ablation = json.loads(Path(isolated["ablation"]).read_text(encoding="utf-8"))
            stress = json.loads(Path(isolated["stress"]).read_text(encoding="utf-8"))

            self.assertTrue(trace_lines)
            self.assertIn('"structural_trace"', trace_lines[0])
            self.assertIn('"prediction_delta"', trace_lines[-1])
            self.assertIn('"protected_anchor_retrieved": true', trace_lines[-1].lower())
            self.assertTrue(ablation["degradation_checks"]["protection_degrades_without_anchor_mechanism"])
            self.assertTrue(ablation["degradation_checks"]["prediction_shift_degrades_without_threat_trace"])
            self.assertTrue(ablation["degradation_checks"]["attention_shift_degrades_without_sensitive_pattern_bias"])
            self.assertTrue(stress["stress_checks"]["structural_trace_anchor_remains_restart_protected"])
            self.assertTrue(stress["stress_checks"]["memory_sensitive_pattern_promotes_threat_channel"])
            self.assertTrue(stress["stress_checks"]["roundtrip_retrieval_preserves_anchor"])

    def test_freshness_gate_rejects_stale_report_and_summary(self) -> None:
        with _isolated_m232_outputs() as isolated:
            Path(isolated["report"]).write_text("{}", encoding="utf-8")
            Path(isolated["summary"]).write_text("# stale\n", encoding="utf-8")
            stale_mtime = time.time() - 3600
            os.utime(isolated["report"], (stale_mtime, stale_mtime))
            os.utime(isolated["summary"], (stale_mtime, stale_mtime))

            audit_started_at = m232_audit._now_iso()
            evidence = build_m232_runtime_evidence()
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
            generated_at = m232_audit._now_iso()

            freshness_ok, freshness = m232_audit._freshness_gate(
                artifacts={
                    "canonical_trace": str(isolated["trace"]),
                    "ablation": str(isolated["ablation"]),
                    "stress": str(isolated["stress"]),
                    "report": str(isolated["report"]),
                    "summary": str(isolated["summary"]),
                },
                audit_started_at=audit_started_at,
                generated_at=generated_at,
                milestone_execution=_suite_execution(m232_audit.M232_TESTS),
                regression_execution=_suite_execution(m232_audit.M232_REGRESSIONS),
                strict=False,
            )

            self.assertFalse(freshness_ok)
            self.assertTrue(freshness["evidence_times_within_round"])
            self.assertFalse(freshness["report_times_within_round"])
            self.assertFalse(freshness["current_round"])

    def test_strict_mode_rejects_injected_execution_records(self) -> None:
        with _isolated_m232_outputs():
            with self.assertRaisesRegex(ValueError, "refuses injected execution records"):
                write_m232_acceptance_artifacts(
                    milestone_execution=_suite_execution(m232_audit.M232_TESTS),
                    regression_execution=_suite_execution(m232_audit.M232_REGRESSIONS),
                )

    def test_strict_mode_without_real_execution_blocks_report(self) -> None:
        with _isolated_m232_outputs() as isolated:
            write_m232_acceptance_artifacts(strict=True, execute_test_suites=False)
            report = json.loads(Path(isolated["report"]).read_text(encoding="utf-8"))

            self.assertTrue(report["strict"])
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["recommendation"], "BLOCK")
            self.assertFalse(report["gates"]["regression"]["passed"])
            self.assertFalse(report["gates"]["artifact_freshness"]["passed"])
            self.assertFalse(report["freshness"]["current_round"])


if __name__ == "__main__":
    unittest.main()
