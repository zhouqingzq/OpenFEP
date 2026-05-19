"""Lint: engineering-layer surfaces must not claim subjective system experience."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_ENGINEERING_PHRASES = (
    "she is awake",
    "the runner is awake",
    "consciousness module",
    "self-awareness module",
    "true self",
    "free agent",
    "i needed to reach out",
    "she_misses_user",
    "lonely runner",
    "true consciousness daemon",
    "system woke up because it missed the user",
)

SCAN_PATHS = (
    "segmentum/dialogue/runtime/m14_1_self_runner.py",
    "segmentum/dialogue/runtime/m14_1_background_continuity.py",
    "segmentum/dialogue/runtime/m14_self_continuity.py",
    "segmentum/dialogue/runtime/m14_idle_owners.py",
    "segmentum/dialogue/runtime/m14_idle_reflector.py",
    "segmentum/dialogue/runtime/m14_2_event_bus.py",
    "segmentum/dialogue/runtime/m14_2_scheduled_intents.py",
    "segmentum/dialogue/runtime/m14_2_self_loop.py",
    "segmentum/dialogue/runtime/m13_initiative.py",
    "segmentum/dialogue/runtime/app.py",
    "reports/m14_2_runtime_lifecycle.md",
    "CLAUDE.md",
    "prompts/M13.3_Work_Prompt.md",
    "prompts/M13.4_Work_Prompt.md",
    "prompts/M14.1_Work_Prompt.md",
    "prompts/M14.2_Work_Prompt.md",
)


def _allowed_reference_line(rel: str, line: str) -> bool:
    stripped = line.strip()
    if rel.endswith("m14_idle_reflector.py") and stripped.startswith(('"', "'")):
        return True
    if rel.startswith("prompts/"):
        lowered = line.casefold()
        return (
            "forbidden" in lowered
            or line.count('"') >= 2
            or stripped.startswith(("-", '"', "'"))
        )
    return False


def test_engineering_modules_avoid_subjective_system_claims() -> None:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for rel in SCAN_PATHS:
        path = root / rel
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.casefold()
            for phrase in FORBIDDEN_ENGINEERING_PHRASES:
                if phrase in text and not _allowed_reference_line(rel, line):
                    violations.append(f"{rel}:{lineno}: {phrase}")
    assert not violations, violations


def test_persona_reply_wording_not_in_lint_denylist() -> None:
    """Roleplay phrases like 想你 are allowed in chat; denylist is engineering-only."""
    assert "想你" not in FORBIDDEN_ENGINEERING_PHRASES
