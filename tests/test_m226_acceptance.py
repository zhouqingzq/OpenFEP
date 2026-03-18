from __future__ import annotations

import json
from pathlib import Path

from segmentum.m226_maturity_audit import SEED_SET, write_m226_maturity_audit_artifacts

from tests._m226_test_utils import base_standardized_replays


def test_final_report_emits_required_decision_fields_and_artifacts() -> None:
    written = write_m226_maturity_audit_artifacts(seed_set=list(SEED_SET), standardized_replay_overrides=base_standardized_replays())
    report = json.loads(Path(written["report"]).read_text(encoding="utf-8"))
    scorecard = json.loads(Path(written["maturity_scorecard"]).read_text(encoding="utf-8"))
    freshness = json.loads(Path(written["replay_freshness"]).read_text(encoding="utf-8"))

    for field in (
        "status",
        "final_status",
        "default_mature",
        "blocking_reasons",
        "why_not_default_mature",
        "residual_risks",
        "recommended_next_action",
        "dimension_scores",
        "critical_dimension_pass_count",
        "current_round_replay_coverage",
    ):
        assert field in report

    assert scorecard["weighted_total_score"] >= 0.85
    assert freshness["generated_this_round"] is True
    assert all(Path(path).exists() for path in written.values())
