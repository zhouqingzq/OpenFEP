from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_m62_acceptance_artifacts import generate_acceptance_report


def test_m62_acceptance_report_satisfies_required_gates(tmp_path: Path) -> None:
    report = generate_acceptance_report(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
    )

    assert report["milestone_id"] == "M6.2"
    assert report["status"] == "PASS"
    assert report["decision"] == "PASS"
    assert report["recommendation"] == "ACCEPT"
    assert report["findings"] == []
    assert {gate["status"] for gate in report["gates"]} == {"PASS"}

    metrics = report["summary_metrics"]
    assert metrics["sample_turns"] == 2
    assert metrics["sample_trace_rows"] == 2
    assert metrics["sample_conscious_trace_rows"] == 2
    assert metrics["required_trace_keys_present"] is True
    assert metrics["debug_off_by_default"] is True
    assert metrics["raw_sensitive_text_absent"] is True
    assert metrics["conscious_chinese_readable"] is True
    assert metrics["persona_session_scoped"] is True


def test_m62_acceptance_artifacts_are_complete_and_readable(tmp_path: Path) -> None:
    report = generate_acceptance_report(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
    )

    for path_text in report["artifacts"].values():
        path = Path(path_text)
        assert path.exists(), path
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            assert path.read_text(encoding="utf-8").strip()

    trace_path = Path(report["artifacts"]["turn_trace_sample"])
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(row["persona_id"] == "sample_persona" for row in rows)
    assert all(row["session_id"] == "sample_session" for row in rows)

    markdown = Path(report["artifacts"]["conscious_markdown"]).read_text(
        encoding="utf-8"
    )
    assert "当前观察" in markdown
    assert "候选路径" in markdown
    assert "结果反馈" in markdown
