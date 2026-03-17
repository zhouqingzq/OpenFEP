from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_m212_program_plan.py"
REPORT_PATH = ROOT / "reports" / "m212_program_plan.json"


def test_m212_program_plan_generation_and_schema() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert REPORT_PATH.exists()

    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["program_id"] == "M2.12-M3.0"

    milestones = payload["milestones"]
    assert [item["milestone_id"] for item in milestones] == [
        "M2.12",
        "M2.13",
        "M2.14",
        "M2.15",
        "M2.16",
        "M2.17",
        "M2.18",
        "M3.0",
    ]

    required = {
        "schema",
        "determinism",
        "causality",
        "ablation",
        "stress",
        "regression",
        "artifact_freshness",
    }
    assert set(payload["required_evidence_categories"]) == required
    for milestone in milestones:
        assert milestone["title"]
        assert milestone["objective"]
        assert set(milestone["required_evidence"]) == required
        assert len(milestone["acceptance_focus"]) >= 3
