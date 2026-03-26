from __future__ import annotations

import json
import tempfile
from pathlib import Path

import segmentum.m237_audit as m237_audit
from segmentum.m237_audit import write_m237_total_acceptance_artifacts


def _report(*, status: str = "PASS", recommendation: str = "ACCEPT", fresh: bool = True) -> dict[str, object]:
    freshness = {"current_round": fresh}
    return {
        "milestone_id": "stub",
        "status": status,
        "recommendation": recommendation,
        "freshness": freshness,
        "generated_at": "2026-03-26T07:30:00+00:00",
    }


def test_total_acceptance_report_writes_required_outputs() -> None:
    overrides = {
        milestone_id: _report()
        for milestone_id in m237_audit.DEPENDENCY_REPORTS
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        scorecard = root / "scorecard.json"
        report = root / "report.json"
        summary = root / "summary.md"
        original = (
            m237_audit.M237_SCORECARD_PATH,
            m237_audit.M237_REPORT_PATH,
            m237_audit.M237_SUMMARY_PATH,
        )
        m237_audit.M237_SCORECARD_PATH = scorecard
        m237_audit.M237_REPORT_PATH = report
        m237_audit.M237_SUMMARY_PATH = summary
        try:
            written = write_m237_total_acceptance_artifacts(report_overrides=overrides)
            payload = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
            assert payload["status"] == "PASS"
            assert payload["recommendation"] == "ACCEPT"
            assert payload["summary"]["passed_pillars"] == payload["summary"]["total_pillars"]
            assert all(Path(path).exists() for path in written.values())
        finally:
            (
                m237_audit.M237_SCORECARD_PATH,
                m237_audit.M237_REPORT_PATH,
                m237_audit.M237_SUMMARY_PATH,
            ) = original
