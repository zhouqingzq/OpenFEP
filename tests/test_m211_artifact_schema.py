from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_pre_m3_acceptance_artifacts import generate_artifacts


REQUIRED_KEYS = {
    "schema_version",
    "benchmark_id",
    "seed",
    "world_id",
    "world_pair",
    "cycles",
    "attention_enabled",
    "profile",
    "metrics",
    "summary",
    "acceptance",
    "generated_at",
    "codebase_version",
}


def test_all_pre_m3_artifacts_share_common_schema() -> None:
    written = generate_artifacts()
    for path in written.values():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert REQUIRED_KEYS.issubset(payload)
        assert payload["schema_version"] == "pre_m3_v1"
        assert isinstance(payload["metrics"], dict)
        assert isinstance(payload["summary"], dict)
        assert isinstance(payload["acceptance"], dict)


def test_pre_m3_readiness_report_references_current_round_artifacts() -> None:
    written = generate_artifacts()
    readiness = json.loads(Path(written["readiness"]).read_text(encoding="utf-8"))
    summary = readiness["summary"]
    assert summary["attention_summary_path"] == "artifacts/pre_m3_attention_summary.json"
    assert summary["transfer_summary_path"] == "artifacts/pre_m3_transfer_summary.json"
    assert summary["personality_summary_path"] == "artifacts/pre_m3_personality_summary.json"
