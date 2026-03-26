from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import segmentum.m236_open_continuity_trial as m236_trial
from segmentum.m236_open_continuity_trial import write_m236_acceptance_artifacts


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
        "started_at": "2026-03-26T10:00:00+00:00",
        "completed_at": "2026-03-26T10:00:01+00:00",
    }


@contextmanager
def _isolated_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        artifacts_dir = root / "artifacts"
        reports_dir = root / "reports"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        original = {
            "M236_TRACE_PATH": m236_trial.M236_TRACE_PATH,
            "M236_METRICS_PATH": m236_trial.M236_METRICS_PATH,
            "M236_ABLATION_PATH": m236_trial.M236_ABLATION_PATH,
            "M236_REPORT_PATH": m236_trial.M236_REPORT_PATH,
            "M236_SUMMARY_PATH": m236_trial.M236_SUMMARY_PATH,
        }
        m236_trial.M236_TRACE_PATH = artifacts_dir / "m236_trace.jsonl"
        m236_trial.M236_METRICS_PATH = artifacts_dir / "m236_metrics.json"
        m236_trial.M236_ABLATION_PATH = artifacts_dir / "m236_ablation.json"
        m236_trial.M236_REPORT_PATH = reports_dir / "m236_report.json"
        m236_trial.M236_SUMMARY_PATH = reports_dir / "m236_summary.md"
        try:
            yield
        finally:
            for key, value in original.items():
                setattr(m236_trial, key, value)


def test_acceptance_artifacts_and_report_reflect_trial_outcome() -> None:
    with _isolated_outputs():
        written = write_m236_acceptance_artifacts(
            strict=False,
            milestone_execution=_suite_execution(m236_trial.M236_TESTS),
            regression_execution=_suite_execution(m236_trial.M236_REGRESSIONS),
        )
        report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
        ablation = json.loads(Path(written["ablation"]).read_text(encoding="utf-8"))
        trace_lines = Path(written["trace"]).read_text(encoding="utf-8").strip().splitlines()

        assert report["milestone_id"] == "M2.36"
        assert report["schema_version"] == m236_trial.SCHEMA_VERSION
        assert report["gates"]["artifact_freshness"]["passed"] is True
        assert "trial" in report
        assert "ablation" in report
        assert trace_lines
        assert ablation["degradation_checks"]["survival_only_is_rejected"] is True
        assert ablation["degradation_checks"]["fractured_identity_is_rejected"] is True
        assert report["trial"]["aggregate_acceptance"]["gates"]["bounded_adaptation"]["passed"] is True
        assert report["trial"]["aggregate_acceptance"]["gates"]["bounded_continuity"]["passed"] is True


def test_strict_mode_rejects_injected_execution_records() -> None:
    with _isolated_outputs():
        try:
            write_m236_acceptance_artifacts(
                strict=True,
                milestone_execution=_suite_execution(m236_trial.M236_TESTS),
                regression_execution=_suite_execution(m236_trial.M236_REGRESSIONS),
            )
        except ValueError as exc:
            assert "refuses injected execution records" in str(exc)
        else:
            raise AssertionError("expected strict mode to reject injected execution records")
