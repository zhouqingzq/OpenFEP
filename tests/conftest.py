from __future__ import annotations

import os
from pathlib import Path

from segmentum.m225_benchmarks import (
    clear_m225_persisted_test_execution_log,
    clear_m225_test_execution_log,
    persist_m225_test_execution_log,
    record_m225_test_execution,
)


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    _ = session
    clear_m225_test_execution_log()
    target = os.environ.get("SEGMENTUM_M225_TEST_LOG")
    if target and os.environ.get("SEGMENTUM_M225_CLEAR_LOG") == "1":
        clear_m225_persisted_test_execution_log(Path(target))


def pytest_runtest_logreport(report) -> None:  # noqa: ANN001
    if report.when != "call":
        return
    record_m225_test_execution(
        name=report.nodeid,
        nodeid=report.nodeid,
        category="pytest",
        status=str(report.outcome),
        details=f"pytest {report.when} phase outcome for {report.nodeid}",
    )
    target = os.environ.get("SEGMENTUM_M225_TEST_LOG")
    if target:
        persist_m225_test_execution_log(Path(target))
