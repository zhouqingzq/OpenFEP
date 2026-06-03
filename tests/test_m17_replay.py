from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from segmentum.dialogue.runtime.m17_replay import run_m17_replay


ROOT = Path(__file__).resolve().parents[1]


def _fixture_session(root: Path) -> Path:
    session = root / "session"
    session.mkdir(parents=True, exist_ok=True)
    (session / "conversation_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "MemoryEfeEvaluationEvent",
                        "turn_index": 2,
                        "traceable_expectation_id": "exp_1",
                        "bundle_linkage_diagnostics": {
                            "retrieval_eligible_count": 2,
                            "bundle_linkable_count": 2,
                            "unlinked_count": 0,
                        },
                        "bundle_candidate_rows": [
                            {
                                "id": "m1",
                                "memory_id": "m1",
                                "item_support": 0.53,
                                "prediction_ids": ["pred:p1"],
                                "expectation_ids": ["exp_1"],
                                "episode_ids": ["ep:m1"],
                                "evidence_refs": ["e1"],
                                "contradiction_risk": 0.0,
                            },
                            {
                                "id": "m2",
                                "memory_id": "m2",
                                "item_support": 0.49,
                                "prediction_ids": ["pred:p1"],
                                "expectation_ids": ["exp_1"],
                                "episode_ids": ["ep:m2"],
                                "evidence_refs": ["e2"],
                                "contradiction_risk": 0.0,
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        encoding="utf-8",
    )
    (session / "m11_user_models.json").write_text(
        json.dumps(
            {
                "default_user": {
                    "prediction_ledger": {
                        "entries": [
                            {
                                "prediction_id": "pred:p1",
                                "turn_id": 1,
                                "prediction_type": "intent_prediction",
                                "predicted_value_summary": "user will clarify benchmark scope",
                                "confidence_band": "high",
                                "raw_confidence": 0.78,
                                "committed_confidence": 0.78,
                                "confidence_cap_reason": "direct_user_statement",
                                "evidence_basis": ["direct_user_statement"],
                                "evidence_refs": ["e1", "e2"],
                                "expires_after_turns": 2,
                                "created_at_turn": 1,
                                "created_before_response": True,
                                "response_turn_id": 1,
                                "validation_status": "pending",
                                "settlement_outcome": "",
                                "settlement_confidence": 0.0,
                                "settlement_id": "",
                                "calibration_need_band": "med",
                                "source_proposal_id": "p1",
                                "event_kind": "prediction",
                            },
                            {
                                "prediction_id": "pred:p1",
                                "turn_id": 2,
                                "prediction_type": "intent_prediction",
                                "predicted_value_summary": "user will clarify benchmark scope",
                                "confidence_band": "high",
                                "raw_confidence": 0.78,
                                "committed_confidence": 0.78,
                                "confidence_cap_reason": "direct_user_statement",
                                "evidence_basis": ["direct_user_statement"],
                                "evidence_refs": ["e1", "e2"],
                                "expires_after_turns": 2,
                                "created_at_turn": 1,
                                "created_before_response": True,
                                "response_turn_id": 1,
                                "validation_status": "violated",
                                "settlement_outcome": "violated",
                                "settlement_confidence": 0.82,
                                "settlement_id": "settle:p1",
                                "m17_prediction_error": 1.514128,
                                "m17_brier_score": 0.6084,
                                "calibration_need_band": "med",
                                "source_proposal_id": "p1",
                                "event_kind": "judgment",
                                "source_episode_id": "ep:m1",
                            },
                        ]
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return session


def test_run_m17_replay_writes_metrics_and_honest_low_sample_warning(tmp_path: Path) -> None:
    session = _fixture_session(tmp_path)
    out_dir = tmp_path / "out"

    result = run_m17_replay(session_paths=[session], out_dir=out_dir, seed=17)

    assert result["coverage"]["structured_linkage_coverage"] == 1.0
    assert result["coverage"]["prediction_state_loaded_session_count"] == 1
    assert result["coverage"]["prediction_sample_empty_session_count"] == 0
    assert result["calibration"]["overall_brier_score"] == 0.6084
    assert result["calibration"]["low_sample_warning"] is True
    assert result["ablation"]["bundle_required_decision_rate"] == 1.0
    assert (out_dir / "calibration_metrics.json").exists()
    assert (out_dir / "ablation_metrics.json").exists()
    assert (out_dir / "policy_comparison.csv").exists()


def test_run_m17_replay_cli_supports_fixture_mode(tmp_path: Path) -> None:
    session = _fixture_session(tmp_path)
    out_dir = tmp_path / "cli_out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_m17_replay.py",
            "--session",
            str(session),
            "--out",
            str(out_dir),
            "--fixture-mode",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    payload = json.loads(completed.stdout)
    assert payload["coverage"]["bundle_linkable_prediction_rate"] == 1.0
    assert payload["calibration"]["low_sample_warning"] is True


def test_run_m17_replay_warns_when_prediction_state_exists_but_has_no_samples(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir(parents=True, exist_ok=True)
    (session / "conversation_log.jsonl").write_text("", encoding="utf-8")
    (session / "m11_user_models.json").write_text(
        json.dumps({"default_user": {"aliases": ["tester"]}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = run_m17_replay(session_paths=[session], out_dir=tmp_path / "out", seed=17)

    assert f"no_prediction_entries:{session}" in result["warnings"]
    assert result["coverage"]["prediction_state_loaded_session_count"] == 1
    assert result["coverage"]["prediction_sample_empty_session_count"] == 1
