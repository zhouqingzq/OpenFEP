import importlib
import json
from pathlib import Path

import pytest

from segmentum.user_continuity.identity_profile import AliasObservation, IdentityProfile
from segmentum.user_model.user_model import EvidenceRef, UserHypothesis, UserModel
from segmentum.user_personality import (
    DEFAULT_HYPERPARAMS,
    InsufficientEvidence,
    M121RuntimeConfig,
    M121RuntimeState,
    PersonalityProfile,
    StepExtractorValidationError,
    TriggerPolicyInput,
    assemble_personality_report,
    bounded_confidence_band,
    build_step_extractor_prompt,
    decide_trigger,
    evidence_cards_from_personality_profile,
    lint_user_facing_text,
    ready_report_or_none,
    run_m12_1_tick,
    run_personality_orchestrator,
    validate_step_output,
)
from segmentum.user_personality.personality_orchestrator import build_base_snapshot


def _extractors(outputs):
    return {idx: (lambda _snapshot, payload=payload: payload) for idx, payload in outputs.items()}


def _base_snapshot():
    refs = [
        {"turn_id": "t1", "quote_id": "q1"},
        {"turn_id": "t2", "quote_id": "q2"},
        {"turn_id": "t3", "quote_id": "q3"},
    ]
    return build_base_snapshot(
        user_id="u1",
        display_name="User",
        turn_id="t3",
        current_turn_quotes={"q3": "I keep asking for careful evidence before deciding."},
        transcript_quote_refs=refs,
        m11_readonly_summary={
            "active_hypotheses": [
                {"hypothesis_id": "h1", "content_summary": "User values careful evidence.", "confidence_band": "med"}
            ]
        },
        m12_readonly_summary={
            "identity_state": "corroborated",
            "binding_confidence_band": "high",
            "recent_continuity_cues": [{"cue_id": "c1", "cue_kind": "style", "confidence_band": "med", "supports": "binds"}],
        },
    )


def _rich_step_outputs():
    refs = ["t1:q1", "t2:q2", "t3:q3"]
    return {
        1: {"summary": "Usually looks for careful evidence before trusting a direction.", "evidence_quote_refs": refs, "confidence_band": "high"},
        2: {
            "evidence_items": [
                {"kind": "conflict", "content_summary": "Asks for concrete checks before moving on.", "evidence_quote_refs": refs, "confidence_band": "high"},
                {"kind": "style", "content_summary": "Prefers direct review of actual code.", "evidence_quote_refs": ["t2:q2", "t3:q3"], "confidence_band": "med"},
            ]
        },
        3: {
            "wants": "wants work to be grounded in checked evidence",
            "fears": "fears polished claims that hide weak work",
            "hypersensitive_to": ["loose claims", "thin tests"],
            "ignores": ["decorative explanations"],
            "default_interpretation": "first checks whether the work is real",
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        4: {
            "about_self": {"content_summary": "I need evidence before trust.", "evidence_quote_refs": refs, "confidence_band": "high"},
            "about_others": {"content_summary": "Others may overstate finished work.", "evidence_quote_refs": refs, "confidence_band": "high"},
            "about_world": {"content_summary": "Good work survives inspection.", "evidence_quote_refs": refs, "confidence_band": "high"},
        },
        5: {
            "dominant_emotional_baseline": "alert but constructive",
            "threat_response": "asks for inspection before acceptance",
            "defenses": [
                {
                    "defense_kind": "control",
                    "protects_what": "trust in the result",
                    "short_term_benefit": "finds weak spots early",
                    "long_term_cost": "can slow handoff",
                    "evidence_quote_refs": refs,
                    "confidence_band": "high",
                }
            ],
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        6: {
            "close_relationship_role": "direct reviewer",
            "recurring_loop_summary": "asks for evidence, then accepts when checks pass",
            "conflict_style": "confront",
            "drawn_to": {"kind": "precise builders", "why": "they make work inspectable", "evidence_quote_refs": refs, "confidence_band": "high"},
            "clashes_with": {"kind": "vague finishers", "why": "they make trust harder", "evidence_quote_refs": refs, "confidence_band": "high"},
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        7: {
            "trigger_event": "sees a milestone claim",
            "interpretation": "checks whether the claim has proof",
            "emotion": "becomes alert",
            "action": "asks for targeted inspection",
            "outcome": "weak parts are repaired",
            "belief_reinforcement": "inspection makes progress believable",
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
        8: {
            "stable_parts": ["needs real checks"],
            "fragile_spots": ["thin acceptance claims"],
            "soft_spots": ["clear evidence changes stance"],
            "communication_styles_likely_accepted": ["show concrete files and tests"],
            "communication_styles_that_trigger_defenses": ["claiming completion without proof"],
            "evidence_quote_refs": refs,
            "confidence_band": "high",
        },
    }


def test_profile_distinguishes_never_computed_from_insufficient_evidence_and_roundtrips_json():
    profile = PersonalityProfile(user_id="u1")
    assert profile.step_1_summary is None
    marker = InsufficientEvidence(reason="too_sparse", last_updated_turn_id="t1")
    profile = profile.with_insufficient("step_1", marker)
    assert isinstance(profile.step_1_summary, InsufficientEvidence)
    loaded = PersonalityProfile.from_json(profile.to_json())
    assert loaded.to_dict() == profile.to_dict()


def test_single_snippet_does_not_promote_confidence_to_high():
    band = bounded_confidence_band(
        "high",
        evidence_refs=[],
        existing_band="low",
        hyperparams=DEFAULT_HYPERPARAMS,
    )
    assert band != "high"


def test_report_assembled_deterministically_and_ready_channel_blocks_failed_report():
    profile = PersonalityProfile(user_id="u1")
    result = run_personality_orchestrator(
        profile,
        turn_id="t3",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(_rich_step_outputs()),
    )
    report_a = assemble_personality_report(result.profile, turn_id="t3", trigger_kind="explicit_request")
    report_b = assemble_personality_report(result.profile, turn_id="t3", trigger_kind="explicit_request")
    assert report_a.to_json() == report_b.to_json()
    assert report_a.report_status == "ready"
    assert ready_report_or_none(report_a) is not None

    bad_outputs = _rich_step_outputs()
    bad_outputs[1] = {"summary": "This predictive system is careful.", "evidence_quote_refs": ["t1:q1"], "confidence_band": "low"}
    bad = run_personality_orchestrator(
        PersonalityProfile(user_id="u1"),
        turn_id="t3",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(bad_outputs),
    )
    assert bad.report.report_status == "linter_failed"
    assert ready_report_or_none(bad.report) is None


def test_step_4_each_belief_has_evidence_refs_and_step_7_has_exactly_six_stages():
    result = run_personality_orchestrator(
        PersonalityProfile(user_id="u1"),
        turn_id="t3",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(_rich_step_outputs()),
    )
    beliefs = result.profile.step_4_core_beliefs
    loop = result.profile.step_7_core_loop
    assert beliefs is not None and not isinstance(beliefs, InsufficientEvidence)
    assert beliefs.about_self and beliefs.about_self.evidence_refs
    assert beliefs.about_others and beliefs.about_others.evidence_refs
    assert beliefs.about_world and beliefs.about_world.evidence_refs
    assert loop is not None and not isinstance(loop, InsufficientEvidence)
    assert len(loop.stages) == 6


def test_step_extractor_rejects_float_unknown_field_jargon_and_unknown_quote():
    with pytest.raises(StepExtractorValidationError):
        validate_step_output(1, {"summary": "Careful", "evidence_quote_refs": ["t1:q1"], "confidence_band": 0.7}, snapshot=_base_snapshot())
    with pytest.raises(StepExtractorValidationError):
        validate_step_output(1, {"summary": "Careful", "evidence_quote_refs": ["t1:q1"], "confidence_band": "low", "extra": "x"}, snapshot=_base_snapshot())
    with pytest.raises(StepExtractorValidationError):
        validate_step_output(1, {"summary": "Uses a model of people.", "evidence_quote_refs": ["t1:q1"], "confidence_band": "low"}, snapshot=_base_snapshot())
    with pytest.raises(StepExtractorValidationError):
        validate_step_output(1, {"summary": "Careful.", "evidence_quote_refs": ["missing:q9"], "confidence_band": "low"}, snapshot=_base_snapshot())


def test_step_extractor_prompt_is_bounded_to_snapshot_and_allowed_refs():
    system_prompt, user_prompt = build_step_extractor_prompt(7, _base_snapshot())
    assert "Return only one JSON object" in system_prompt
    assert "allowed_evidence_quote_refs" in user_prompt
    assert "t1:q1" in user_prompt
    assert "memory_dynamics" not in user_prompt


def test_linter_catches_every_configured_token_and_ignores_schema_names():
    for token in DEFAULT_HYPERPARAMS.forbidden_user_facing_tokens_extra:
        assert lint_user_facing_text(f"visible {token}", section="step_1"), token
    for token in DEFAULT_HYPERPARAMS.forbidden_clinical_label_tokens:
        assert lint_user_facing_text(f"visible {token}", section="step_1"), token
    for token in DEFAULT_HYPERPARAMS.forbidden_moral_or_chicken_soup_tokens:
        assert lint_user_facing_text(f"visible {token}", section="step_1"), token
    assert not lint_user_facing_text("defense_kind confidence_band loop_stage", section="internal.schema")


def test_trigger_policy_rules_are_pure_and_respect_caps_and_sparse_suspension():
    base = TriggerPolicyInput(user_id="u1", current_turn_index=12, current_hour_bucket=4, last_successful_report_turn_index=1)
    assert decide_trigger(base).kind == "turn_count_cadence"
    capped = TriggerPolicyInput(user_id="u1", current_turn_index=12, current_hour_bucket=4, run_hour_buckets=(4, 4), explicit_request=True)
    assert decide_trigger(capped).reason == "per_hour_cap"
    sparse = TriggerPolicyInput(
        user_id="u1",
        current_turn_index=30,
        current_hour_bucket=6,
        last_successful_report_turn_index=1,
        consecutive_step1_insufficient=2,
    )
    assert decide_trigger(sparse).reason == "cadence_suspended_after_sparse_step1"
    explicit = TriggerPolicyInput(user_id="u1", current_turn_index=2, current_hour_bucket=1, explicit_request=True)
    assert decide_trigger(explicit).kind == "explicit_request"


def test_evidence_cards_never_emit_for_linter_failed_and_step4_or_7_insufficient_is_strategy_only():
    result = run_personality_orchestrator(
        PersonalityProfile(user_id="u1"),
        turn_id="t3",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(_rich_step_outputs()),
    )
    cards = evidence_cards_from_personality_profile(result.profile, report_status="ready")
    assert cards
    assert all("model" not in card.content_summary.casefold() for card in cards)
    assert evidence_cards_from_personality_profile(result.profile, report_status="linter_failed") == ()

    sparse_outputs = _rich_step_outputs()
    sparse_outputs[4] = {"status": "insufficient_evidence", "reason": "not enough stable belief evidence"}
    sparse = run_personality_orchestrator(
        PersonalityProfile(user_id="u1"),
        turn_id="t3",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(sparse_outputs),
    )
    sparse_cards = evidence_cards_from_personality_profile(sparse.profile, report_status=sparse.report.report_status)
    assert sparse_cards
    assert all(card.permitted_use == "strategy_only" for card in sparse_cards)


def test_runtime_disabled_is_noop_and_enabled_initializes_without_first_turn_report():
    state = M121RuntimeState.clean()
    disabled_next, disabled = run_m12_1_tick(
        state,
        user_id="u1",
        display_name="User",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        config=M121RuntimeConfig(m12_1_personality_enabled=False),
    )
    assert disabled_next.to_dict() == state.to_dict()
    assert disabled.enabled is False

    enabled_next, enabled = run_m12_1_tick(
        state,
        user_id="u1",
        display_name="User",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        config=M121RuntimeConfig(m12_1_personality_enabled=True),
    )
    assert "u1" in enabled_next.profiles_by_user
    assert enabled.orchestrator_result is None
    assert enabled.trigger_decision.should_run is False


def test_runtime_explicit_run_does_not_mutate_m11_or_m12_state_rows():
    m11 = UserModel(
        user_id="u1",
        cognitive_style_hypotheses=(
            UserHypothesis("h1", "User values checks.", "task_requirements", evidence_refs=(EvidenceRef("t1", "q1"),)),
        ),
    )
    m12 = IdentityProfile(
        user_id="u1",
        display_name="User",
        aliases_observed=(AliasObservation("a1", "User", "t1", "t1", 1, "factual", "accept"),),
    )
    m11_before = m11.to_dict()
    m12_before = m12.to_dict()
    state, result = run_m12_1_tick(
        M121RuntimeState.clean(),
        user_id="u1",
        display_name="User",
        turn_id="t3",
        turn_index=3,
        hour_bucket=1,
        current_turn_quotes={"q3": "check it"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}, {"turn_id": "t2", "quote_id": "q2"}, {"turn_id": "t3", "quote_id": "q3"}],
        m11_readonly_summary={"active_hypotheses": [m11.all_hypotheses()[0].to_dict()]},
        m12_readonly_summary=m12.to_dict(),
        trigger_input=TriggerPolicyInput(user_id="u1", current_turn_index=3, current_hour_bucket=1, explicit_request=True),
        extractors=_extractors(_rich_step_outputs()),
        config=M121RuntimeConfig(m12_1_personality_enabled=True),
    )
    assert result.trigger_decision.kind == "explicit_request"
    assert state.profiles_by_user["u1"].last_full_report_status == "ready"
    assert m11.to_dict() == m11_before
    assert m12.to_dict() == m12_before


def test_roleplay_heavy_window_does_not_destabilise_ready_profile():
    ready = run_personality_orchestrator(
        PersonalityProfile(user_id="u1"),
        turn_id="t3",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(_rich_step_outputs()),
    ).profile
    roleplay_outputs = {idx: {"status": "insufficient_evidence", "reason": "roleplay window not stable evidence"} for idx in range(1, 9)}
    replay = run_personality_orchestrator(
        ready,
        turn_id="t4",
        trigger_kind="explicit_request",
        base_snapshot=_base_snapshot(),
        extractors=_extractors(roleplay_outputs),
    )
    assert replay.profile.step_1_summary == ready.step_1_summary
    assert replay.profile.section_last_insufficient["step_1"]["reason"] == "roleplay window not stable evidence"


def test_m12_1_does_not_import_forbidden_memory_modules_or_call_mutating_neighbors():
    root = Path("segmentum/user_personality")
    forbidden = ("memory_dynamics", "value_memory", "memory_retrieval")
    mutation_names = ("apply_claims_to_user_model", "apply_contradictions", "run_m11_turn", "run_m12_turn", "detect_identity_conflicts")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(f"segmentum.{name}" in text or f"..{name}" in text or f"import {name}" in text for name in forbidden), path
        assert not any(name in text for name in mutation_names), path
    importlib.import_module("segmentum.user_personality")


def test_acceptance_artifact_exists_and_covers_required_scenarios():
    artifact = Path("artifacts/m12_1_acceptance_report.json")
    if not artifact.exists():
        pytest.skip("acceptance artifact has not been generated yet")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    required = {
        "rich_evidence_full_report",
        "sparse_evidence_insufficient_at_step_8",
        "engineering_jargon_caught_by_linter",
        "dsm_label_caught_by_linter",
        "moral_verdict_caught_by_linter",
        "roleplay_does_not_destabilise_profile",
    }
    assert required <= set(payload["calibration_audit_report"]["scenarios"])
    assert all(row["passed"] for row in payload["calibration_audit_report"]["scenarios"].values())
