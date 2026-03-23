from __future__ import annotations

import json
from datetime import datetime, timezone
import unittest
from pathlib import Path
from unittest.mock import patch

import segmentum.m230_audit as m230_audit
from segmentum.m230_audit import (
    M230_ABLATION_PATH,
    M230_REGRESSIONS,
    M230_REPORT_PATH,
    M230_SPEC_PATH,
    M230_STRESS_PATH,
    M230_SUMMARY_PATH,
    M230_TESTS,
    M230_TRACE_PATH,
    write_m230_acceptance_artifacts,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record(suite: str, *, category: str) -> dict[str, object]:
    timestamp = _now_iso()
    return {
        "suite": suite,
        "category": category,
        "command": f"py -m pytest -q {suite}",
        "returncode": 0,
        "passed": True,
        "started_at": timestamp,
        "completed_at": timestamp,
    }


def _passing_evidence() -> tuple[str, list[dict[str, object]]]:
    round_started_at = _now_iso()
    run_id = "test-run-id"
    milestone = m230_audit._stamp_provenance_records(
        suites=M230_TESTS,
        category="milestone",
        started_at=round_started_at,
        completed_at=round_started_at,
        returncode=0,
        run_id=run_id,
    )
    regressions = m230_audit._stamp_provenance_records(
        suites=M230_REGRESSIONS,
        category="regression",
        started_at=round_started_at,
        completed_at=round_started_at,
        returncode=0,
        run_id=run_id,
    )
    return round_started_at, [*milestone, *regressions]


class TestM230AcceptanceArtifacts(unittest.TestCase):
    def test_report_fails_closed_when_external_pytest_evidence_is_forged(self) -> None:
        now = _now_iso()
        forged_milestone = [_record(path, category="milestone") for path in M230_TESTS]
        forged_regressions = [_record(path, category="regression") for path in M230_REGRESSIONS]
        for record in forged_milestone + forged_regressions:
            record["started_at"] = now
            record["completed_at"] = now
        write_m230_acceptance_artifacts(
            executed_tests=forged_milestone,
            executed_regressions=forged_regressions,
            round_started_at=now,
        )
        report = json.loads(Path(M230_REPORT_PATH).read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["recommendation"], "BLOCK")
        self.assertFalse(report["gates"]["artifact_freshness"]["passed"])
        self.assertFalse(report["gates"]["artifact_freshness"]["passed"])
        titles = {finding["title"] for finding in report["findings"]}
        self.assertIn("Untrusted external pytest evidence rejected", titles)
        self.assertIn("Current-round artifact freshness not proven", titles)

    def test_report_self_executes_required_pytest_and_writes_current_round_artifacts(self) -> None:
        round_started_at, signed_records = _passing_evidence()
        Path(M230_SUMMARY_PATH).unlink(missing_ok=True)
        with patch("segmentum.m230_audit.run_required_m230_pytest_suites", return_value=signed_records):
            write_m230_acceptance_artifacts(round_started_at=round_started_at)
        report = json.loads(Path(M230_REPORT_PATH).read_text(encoding="utf-8"))

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

        self.assertEqual(report["milestone_id"], "M2.30")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["recommendation"], "ACCEPT")
        self.assertTrue(report["gates"]["determinism"]["passed"])
        self.assertTrue(report["gates"]["ablation"]["passed"])
        self.assertTrue(report["gates"]["stress"]["passed"])
        self.assertTrue(report["gates"]["artifact_freshness"]["passed"])
        self.assertEqual(
            report["tests"]["milestone"][0]["provenance_runner"],
            "segmentum.m230_audit.run_required_m230_pytest_suites.v1",
        )

        for path in (
            M230_SPEC_PATH,
            M230_TRACE_PATH,
            M230_ABLATION_PATH,
            M230_STRESS_PATH,
            M230_REPORT_PATH,
            M230_SUMMARY_PATH,
        ):
            self.assertTrue(Path(path).exists(), str(path))

    def test_artifact_payloads_include_slow_learning_specific_evidence(self) -> None:
        round_started_at, signed_records = _passing_evidence()
        with patch("segmentum.m230_audit.run_required_m230_pytest_suites", return_value=signed_records):
            written = write_m230_acceptance_artifacts(round_started_at=round_started_at)
        trace_lines = Path(written["trace"]).read_text(encoding="utf-8").strip().splitlines()
        ablation = json.loads(Path(written["ablation"]).read_text(encoding="utf-8"))
        stress = json.loads(Path(written["stress"]).read_text(encoding="utf-8"))

        self.assertTrue(trace_lines)
        self.assertIn('"slow_summary"', trace_lines[-1])
        self.assertIn('"continuity_consistent": true', trace_lines[-1].lower())
        self.assertIn("degradation_checks", ablation)
        self.assertTrue(ablation["degradation_checks"]["continuity_consistency_preserved"])
        self.assertIn("anti_collapse", stress)
        self.assertTrue(stress["anti_collapse"]["triggered"])
        self.assertTrue(stress["trace_checks"]["last_record_has_slow_learning"])
        self.assertTrue(stress["trace_checks"]["last_record_has_continuity"])


if __name__ == "__main__":
    unittest.main()
