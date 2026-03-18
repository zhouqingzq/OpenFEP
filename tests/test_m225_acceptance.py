from __future__ import annotations

import json
from pathlib import Path

from segmentum.m225_benchmarks import SEED_SET, write_m225_acceptance_artifacts


def test_m225_acceptance_report_is_complete_and_fresh_after_write() -> None:
    written = write_m225_acceptance_artifacts(seed_set=list(SEED_SET))
    report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
    identity = json.loads(Path(written["identity_preservation"]).read_text(encoding="utf-8"))

    for field in (
        "milestone_id",
        "status",
        "recommendation",
        "generated_at",
        "codebase_version",
        "seed_set",
        "protocols",
        "variants",
        "world_definitions",
        "holdout_worlds",
        "control_variants",
        "variant_metrics",
        "paired_comparisons",
        "significant_metric_count",
        "effect_metric_count",
        "protocol_breakdown",
        "holdout_breakdown",
        "adapter_breakdown",
        "deception_breakdown",
        "shortcut_control_breakdown",
        "gates",
        "goal_details",
        "artifacts",
        "tests",
        "pytest_tests",
        "historical_regressions",
        "internal_checks",
        "findings",
        "recommendation",
        "freshness",
        "residual_risks",
    ):
        assert field in report

    assert report["milestone_id"] == "M2.25"
    assert report["status"] == "BLOCKED"
    assert report["recommendation"] == "BLOCK"
    assert report["freshness"]["generated_this_round"] is True
    assert report["pytest_tests"] == report["tests"]
    assert report["tests"] == []
    assert any(finding["title"] == "Current-round pytest evidence missing" for finding in report["findings"])
    assert any(finding["title"] == "Historical regression evidence missing" for finding in report["findings"])
    assert {item["category"] for item in report["internal_checks"]} >= {
        "causality",
        "ablation",
        "determinism",
        "schema",
        "artifact_freshness",
    }
    assert all(item["required"] for item in report["historical_regressions"])
    assert all(Path(manifest["path"]).exists() for manifest in report["artifacts"].values())
    assert all(
        float(row["cross_world_commitment_alignment"]) >= 0.70
        for row in identity["identity_rows"]
    )
