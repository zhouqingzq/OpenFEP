import json
from pathlib import Path

from segmentum.reciprocal_role.acceptance import build_m12_2_acceptance_artifact


def test_m12_2_acceptance_artifact_exists_matches_builder_and_covers_phases():
    artifact_path = Path("artifacts/m12_2_acceptance_report.json")
    assert artifact_path.exists()
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    built = build_m12_2_acceptance_artifact()
    assert saved == built
    assert saved["phase_a"]["non_interference_diff"]["disabled_state_unchanged"] is True
    assert saved["phase_b"]["replay_determinism"]["state_byte_identical"] is True
    scenarios = saved["phase_c"]["calibration_audit_report"]["scenarios"]
    required = {
        "scenario_user_asks_how_persona_models_them",
        "scenario_user_tests_persona_memory_and_consistency",
        "scenario_high_gain_question_blocked_by_privacy_boundary",
        "scenario_persona_clarifies_self_without_overclaiming",
        "scenario_sparse_transcript_no_second_order_overfit",
        "scenario_contradicted_second_order_claim_is_downgraded",
        "scenario_user_requests_bidirectional_free_energy_analysis",
        "claim_group_persona_about_user_open_converging_resolved",
        "claim_group_user_about_persona_open_contradicted_reexpanded",
    }
    assert required <= set(scenarios)
    assert all(row["passed"] for row in scenarios.values())
