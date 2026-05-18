from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from segmentum.m225_benchmarks import (
    clear_m225_persisted_test_execution_log,
    clear_m225_test_execution_log,
    persist_m225_test_execution_log,
    record_m225_test_execution,
)

_M_MILESTONE_RE = re.compile(r"test_m(\d+)")


def _is_pre_m11_test(item: pytest.Item) -> bool:
    """Return True if the test belongs to a milestone before M11.0."""
    filename = item.path.name if hasattr(item, "path") else ""
    if not filename:
        return True

    # test_m1[1-9]* → M11–M19: run
    if re.match(r"test_m1[1-9]", filename):
        return False

    # test_m2* → M2.x (all pre-M11)
    if filename.startswith("test_m2"):
        return True

    # test_m3, test_m10 etc. → pre-M11
    match = _M_MILESTONE_RE.search(filename)
    if not match:
        return True  # non-M-prefixed tests (e.g. test_metacognitive)

    return int(match.group(1)) < 11


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if _is_pre_m11_test(item):
            item.add_marker(pytest.mark.pre_m11)


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
