from __future__ import annotations

import json
from functools import lru_cache
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

STRICT_AUDIT_KEYS = {
    "milestone_id",
    "status",
    "seed_set",
    "artifacts",
    "tests",
    "gates",
    "findings",
    "residual_risks",
    "freshness",
    "recommendation",
}


@lru_cache(maxsize=1)
def _written_artifacts() -> dict[str, str]:
    return {name: str(path) for name, path in generate_artifacts().items()}


def test_all_pre_m3_artifacts_share_common_schema() -> None:
    written = _written_artifacts()
    for path in written.values():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert REQUIRED_KEYS.issubset(payload)
        assert payload["schema_version"] == "pre_m3_v1"
        assert isinstance(payload["metrics"], dict)
        assert isinstance(payload["summary"], dict)
        assert isinstance(payload["acceptance"], dict)
        assert "codebase_provenance" in payload
        assert payload["codebase_provenance"]["git_commit"] == payload["codebase_version"]
        assert payload["codebase_provenance"]["workspace_fingerprint"]


def test_pre_m3_readiness_report_references_current_round_artifacts() -> None:
    written = _written_artifacts()
    readiness = json.loads(Path(written["readiness"]).read_text(encoding="utf-8"))
    transfer = json.loads(Path(written["transfer"]).read_text(encoding="utf-8"))
    summary = readiness["summary"]
    assert STRICT_AUDIT_KEYS.issubset(readiness)
    assert summary["attention_summary_path"] == "artifacts/pre_m3_attention_summary.json"
    assert summary["transfer_summary_path"] == "artifacts/pre_m3_transfer_summary.json"
    assert summary["personality_summary_path"] == "artifacts/pre_m3_personality_summary.json"
    assert readiness["seed_set"]
    assert transfer["seed_set"]
    assert transfer["summary"]["protocol"]["seed_set"] == transfer["seed_set"]
    assert readiness["summary"]["transfer_seed_set"] == transfer["seed_set"]
    assert readiness["freshness"]["transfer_summary"]["seed_set"] == transfer["seed_set"]
    assert readiness["recommendation"]["status"] == "READY_FOR_M3"
    assert readiness["codebase_provenance"]["git_commit"] == readiness["codebase_version"]
    assert readiness["codebase_provenance"]["workspace_fingerprint"]
