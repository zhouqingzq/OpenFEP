"""Lint: engineering-layer surfaces must not claim subjective system experience."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_ENGINEERING_PHRASES = (
    "she is awake",
    "consciousness module",
    "self-awareness module",
    "true self",
    "free agent",
    "i needed to reach out",
    "she_misses_user",
)

SCAN_PATHS = (
    "segmentum/dialogue/runtime/m14_1_self_runner.py",
    "segmentum/dialogue/runtime/m14_1_background_continuity.py",
    "segmentum/dialogue/runtime/m14_self_continuity.py",
    "segmentum/dialogue/runtime/m14_idle_owners.py",
)


def test_engineering_modules_avoid_subjective_system_claims() -> None:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for rel in SCAN_PATHS:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").casefold()
        for phrase in FORBIDDEN_ENGINEERING_PHRASES:
            if phrase in text:
                violations.append(f"{rel}: {phrase}")
    assert not violations, violations


def test_persona_reply_wording_not_in_lint_denylist() -> None:
    """Roleplay phrases like 想你 are allowed in chat; denylist is engineering-only."""
    assert "想你" not in FORBIDDEN_ENGINEERING_PHRASES
