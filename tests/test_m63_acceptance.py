from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_m63_acceptance_artifacts import generate_acceptance_report


def test_m63_acceptance_report_satisfies_required_gates(tmp_path: Path) -> None:
    report = generate_acceptance_report(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
    )

    assert report["milestone_id"] == "M6.3"
    assert report["status"] == "PASS"
    assert report["decision"] == "PASS"
    assert report["recommendation"] == "ACCEPT"
    assert report["findings"] == []
    assert {gate["status"] for gate in report["gates"]} == {"PASS"}

    metrics = report["summary_metrics"]
    assert metrics["latest_state_present"] is True
    assert metrics["state_sections_present"] is True
    assert metrics["trace_has_cognitive_state"] is True
    assert metrics["compressed_self_prior_consumed"] is True
    assert metrics["full_self_prior_not_consumed"] is True
    assert metrics["self_prior_unchanged"] is True
    assert metrics["action_selection_unchanged"] is True
    assert metrics["bounded_affect"] is True
    assert metrics["bounded_meta_control"] is True


def test_m63_acceptance_artifacts_are_complete_and_readable(tmp_path: Path) -> None:
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

    state = json.loads(Path(report["artifacts"]["state_sample"]).read_text(encoding="utf-8"))
    assert set(state) == {"task", "memory", "gaps", "affect", "meta_control"}

    rows = [
        json.loads(line)
        for line in Path(report["artifacts"]["turn_trace_sample"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == report["summary_metrics"]["sample_trace_rows"]
    assert all("cognitive_state" in row for row in rows)
    assert "FULL SELF PRIOR SHOULD REMAIN OUTSIDE PER-TURN STATE" not in json.dumps(
        rows,
        ensure_ascii=False,
    )
