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

_M_DOTTED_RE = re.compile(r"test_m(\d+)_(\d+)")
_M_COMPACT_RE = re.compile(r"test_m(\d+)")


def _milestone_major_from_filename(filename: str) -> int | None:
    dotted = _M_DOTTED_RE.search(filename)
    if dotted:
        return int(dotted.group(1))

    compact = _M_COMPACT_RE.search(filename)
    if compact is None:
        return None
    digits = compact.group(1)
    if len(digits) >= 2 and digits[0] in {"2", "3", "4", "5", "6", "7", "8", "9"}:
        return int(digits[0])
    return int(digits)


def _is_pre_m11_test(item: pytest.Item) -> bool:
    """Return True if the test belongs to a milestone before M11.0."""
    filename = item.path.name if hasattr(item, "path") else ""
    if not filename:
        return True
    if filename == "test_lint_no_subjective_language.py":
        return False

    if re.match(r"test_m1[1-9]", filename):
        return False

    major = _milestone_major_from_filename(filename)
    if major is None:
        return True
    return major < 11


def _is_inactive_path_a_test(item: pytest.Item) -> bool:
    """Return True for frozen Path A / M10-and-earlier tests."""
    filename = item.path.name if hasattr(item, "path") else ""
    if not filename or filename == "test_lint_no_subjective_language.py":
        return False
    major = _milestone_major_from_filename(filename)
    if major is not None and major <= 10:
        return True
    try:
        source = Path(str(item.path)).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(
        keyword in source
        for keyword in ("conversation_loop", "CognitiveLoop", "SelfThoughtProducer")
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if _is_pre_m11_test(item):
            item.add_marker(pytest.mark.pre_m11)
        if _is_inactive_path_a_test(item):
            item.add_marker(pytest.mark.inactive_path_a)


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
