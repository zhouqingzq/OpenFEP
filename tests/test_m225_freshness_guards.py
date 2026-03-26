from __future__ import annotations

import copy
import json
from pathlib import Path

from segmentum.m225_benchmarks import (
    SEED_SET,
    _finalize_report_decision,
    run_m225_open_world_transfer,
    write_m225_acceptance_artifacts,
)


def test_report_schema_contains_tests_and_findings_with_semantics() -> None:
    payload = run_m225_open_world_transfer(seed_set=list(SEED_SET))
    report = payload["acceptance_report"]
    assert isinstance(report["tests"], list)
    assert isinstance(report["pytest_tests"], list)
    assert isinstance(report["historical_regressions"], list) and report["historical_regressions"]
    assert isinstance(report["internal_checks"], list) and report["internal_checks"]
    assert isinstance(report["findings"], list) and report["findings"]
    assert any(item["name"] == "artifact_freshness" for item in report["internal_checks"])
    assert any(finding["title"] == "Freshness not yet verified" for finding in report["findings"])
    assert any(finding["title"] == "Current-round pytest evidence missing" for finding in report["findings"])
    assert any(finding["title"] == "Historical regression evidence missing" for finding in report["findings"])


def test_missing_schema_or_freshness_cannot_finalize_to_pass() -> None:
    written = write_m225_acceptance_artifacts(seed_set=list(SEED_SET))
    report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))

    freshness_blocked = copy.deepcopy(report)
    freshness_blocked["gates"]["freshness_generated_this_round"] = False
    freshness_blocked["freshness"]["generated_this_round"] = False
    assert _finalize_report_decision(freshness_blocked)["status"] != "PASS"

    schema_blocked = copy.deepcopy(report)
    schema_blocked["gates"]["artifact_schema_complete"] = False
    schema_blocked["artifact_schema_complete"] = {"passed": False, "missing": {"report": ["tests"]}}
    assert _finalize_report_decision(schema_blocked)["status"] != "PASS"


def test_irrelevant_failed_pytest_records_do_not_block_acceptance_write() -> None:
    written = write_m225_acceptance_artifacts(
        seed_set=list(SEED_SET),
        pytest_evidence=[
            {
                "name": "tests/test_unrelated_suite.py::test_old_failure",
                "nodeid": "tests/test_unrelated_suite.py::test_old_failure",
                "status": "failed",
                "category": "pytest",
                "details": "stale unrelated failure",
            }
        ],
    )
    report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["recommendation"] == "ACCEPT"
    assert all(
        not str(item.get("nodeid", "")).startswith("tests/test_unrelated_suite.py")
        for item in report["pytest_tests"]
    )
    assert not any(finding["title"] == "Report includes non-passing pytest evidence" for finding in report["findings"])
