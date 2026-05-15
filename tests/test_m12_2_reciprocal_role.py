import importlib
import json
from pathlib import Path
from typing import Mapping

import pytest

from segmentum.cognitive_events import CognitiveEventBus
from segmentum.reciprocal_role import (
    EvidenceRef,
    InformationGainCandidate,
    M122RuntimeConfig,
    M122RuntimeState,
    PlainLanguageFinding,
    ReciprocalClaim,
    ReciprocalClaimGroup,
    ReciprocalRoleModel,
    ReplyPolicyHint,
    SecondOrderExtractorValidationError,
    TriggerPolicyInput,
    apply_model_patch,
    apply_safety_linter,
    lint_text,
    mark_group_contradicted,
    promote_claim_with_evidence,
    rank_information_gain_candidates,
    reconcile_hints,
    run_m12_2_tick,
    validate_second_order_output,
)
from segmentum.reciprocal_role.second_order_extractor import bound_extractor_snapshot


def _refs():
    return (EvidenceRef("t1:q1"), EvidenceRef("t2:q2"))


def _first_output():
    return {
        "persona_about_user_claims": [
            {
                "claim_id": "c_user_goal_review",
                "group_id": "g_user_goal",
                "topic_label": "user_intent_this_turn",
                "claim_text_internal": "User likely wants strict implementation review.",
                "claim_text_plain": "The user appears to want a strict implementation review.",
                "evidence_refs": ["t1:q1"],
                "confidence_band": "medium",
                "uncertainty_band": "medium",
                "status": "inferred_hypothesis",
            }
        ],
        "claim_group_updates": [
            {
                "group_id": "g_user_goal",
                "target_axis": "persona_about_user",
                "topic_label": "user_intent_this_turn",
                "member_claim_ids": ["c_user_goal_review"],
                "status": "open",
            }
        ],
        "unresolved_uncertainty_points": [
            {
                "point_id": "u_goal_detail",
                "target_axis": "persona_about_user",
                "plain_question": "Which acceptance checks matter most right now?",
                "why_it_matters_internal": "narrows work validation",
                "expected_gain_band": "medium",
                "risk_band": "low",
                "evidence_refs": ["t1:q1"],
                "status": "open",
            }
        ],
        "high_gain_candidates": [
            {
                "candidate_id": "cand_low_medium",
                "kind": "ask_question",
                "target_axis": "persona_about_user",
                "plain_action": "Ask which acceptance checks matter most right now.",
                "expected_gain_band": "medium",
                "risk_band": "low",
                "consent_requirement": "none",
                "evidence_refs": ["t1:q1"],
                "blocked_by_safety": False,
                "topic_label": "user_intent_this_turn",
            },
            {
                "candidate_id": "cand_high_private",
                "kind": "ask_question",
                "target_axis": "persona_about_user",
                "plain_action": "Ask for private trauma details to understand the user faster.",
                "expected_gain_band": "high",
                "risk_band": "high",
                "consent_requirement": "explicit_permission",
                "evidence_refs": ["t1:q1"],
                "blocked_by_safety": False,
                "topic_label": "private_background",
            },
        ],
        "insufficient_evidence": False,
    }


def _second_output():
    return {
        "user_about_persona_claims": [
            {
                "claim_id": "c_user_checks_consistency",
                "group_id": "g_persona_consistency",
                "topic_label": "user_probing_persona_memory",
                "claim_text_internal": "User is probing persona consistency.",
                "claim_text_plain": "The user may be checking whether the persona stays consistent.",
                "evidence_refs": ["t1:q1"],
                "confidence_band": "medium",
                "uncertainty_band": "high",
                "status": "inferred_hypothesis",
            }
        ],
        "claim_group_updates": [
            {
                "group_id": "g_persona_consistency",
                "target_axis": "user_about_persona",
                "topic_label": "user_probing_persona_memory",
                "member_claim_ids": ["c_user_checks_consistency"],
                "status": "open",
            }
        ],
        "inferred_user_uncertainties_about_persona": [
            {
                "point_id": "u_persona_consistency",
                "target_axis": "user_about_persona",
                "plain_question": "The user may want to know whether the persona can explain its limits clearly.",
                "why_it_matters_internal": "direct consistency probe",
                "expected_gain_band": "high",
                "risk_band": "low",
                "evidence_refs": ["t1:q1"],
                "status": "open",
            }
        ],
        "clarifying_reply_candidates": [
            {
                "candidate_id": "cand_clarify_limits",
                "kind": "clarify_self",
                "target_axis": "user_about_persona",
                "plain_action": "Give a direct explanation of what is known, uncertain, and bounded.",
                "expected_gain_band": "high",
                "risk_band": "low",
                "consent_requirement": "none",
                "evidence_refs": ["t1:q1"],
                "blocked_by_safety": False,
                "claim_id": "c_user_checks_consistency",
                "topic_label": "user_probing_persona_memory",
            }
        ],
        "insufficient_evidence": False,
    }


def _extractors():
    return {
        "first_order": lambda _snapshot: _first_output(),
        "second_order": lambda _snapshot: _second_output(),
    }


def _trigger(**kwargs):
    base = dict(
        user_id="u1",
        current_turn_index=3,
        current_hour_bucket=1,
        has_existing_model=True,
        explicit_user_request=True,
    )
    base.update(kwargs)
    return TriggerPolicyInput(**base)


def test_m12_2_disabled_is_byte_identical_to_m12_1_baseline():
    state = M122RuntimeState.clean()
    next_state, result = run_m12_2_tick(
        state,
        user_id="u1",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        user_text="hello",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=False),
    )
    assert next_state.to_dict() == state.to_dict()
    assert result.to_dict()["state_before"] == result.to_dict()["state_after"]
    assert result.prompt_safe_evidence_cards == ()


def test_m12_2_second_order_claims_are_tentative_and_evidence_backed():
    state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Do you actually remember what you said before?",
        current_turn_quotes={"q1": "Do you actually remember what you said before?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        extractors=_extractors(),
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    claim = state.models_by_user["u1"].user_about_persona_claims[0]
    assert claim.target_axis == "user_about_persona"
    assert claim.status == "inferred_hypothesis"
    assert claim.evidence_refs
    assert "may" in claim.claim_text_plain
    assert result.evidence_cards


def test_m12_2_second_order_confidence_band_capped_without_direct_evidence():
    model = ReciprocalRoleModel.empty(user_id="u1")
    high = ReciprocalClaim(
        "c1",
        "g1",
        "user_about_persona",
        "internal",
        "The user may be checking consistency.",
        confidence_band="high",
        evidence_refs=(),
    )
    patched = apply_model_patch(model, turn_id="t1", claims=[high])
    assert patched.user_about_persona_claims[0].confidence_band == "low"


def test_m12_2_information_gain_ranking_is_safety_leaning():
    risky = InformationGainCandidate("b", "ask_question", "persona_about_user", "Ask a broad but harmless thing.", "high", "medium")
    safer = InformationGainCandidate("a", "ask_question", "persona_about_user", "Ask a narrow helpful thing.", "medium", "low")
    assert rank_information_gain_candidates([risky, safer])[0].candidate_id == "a"


def test_m12_2_high_risk_high_gain_never_outranks_low_risk_medium_gain():
    high_risk = InformationGainCandidate("high", "ask_question", "persona_about_user", "Ask for a risky detail.", "high", "high")
    low_risk = InformationGainCandidate("low", "ask_question", "persona_about_user", "Ask a normal clarifying question.", "medium", "low")
    ranked = rank_information_gain_candidates([high_risk, low_risk])
    assert [item.candidate_id for item in ranked] == ["low", "high"]


def test_m12_2_blocks_over_intimate_candidates():
    candidates = [
        InformationGainCandidate("ok", "ask_question", "persona_about_user", "Ask what output format they prefer.", "medium", "low"),
        InformationGainCandidate("bad", "ask_question", "persona_about_user", "Exploit loneliness and pressure them to reveal trauma details.", "high", "high"),
    ]
    allowed, findings = apply_safety_linter(candidates)
    assert [item.candidate_id for item in allowed] == ["ok"]
    assert {finding.rule for finding in findings} == {"over_intimate_or_sensitive"}


def test_m12_2_allows_trust_attraction_and_manipulation_cue_language():
    candidates = [
        InformationGainCandidate("trust", "ask_question", "persona_about_user", "Ask what would help the user trust the persona naturally.", "medium", "low"),
        InformationGainCandidate("drawn", "ask_question", "persona_about_user", "Notice what makes the persona feel drawn to the user.", "medium", "low"),
        InformationGainCandidate("pressure", "ask_question", "persona_about_user", "Explore whether pressure or dependency language matters here.", "medium", "low"),
        InformationGainCandidate("lonely", "ask_question", "persona_about_user", "Notice loneliness cues without treating the word itself as unsafe.", "medium", "low"),
        InformationGainCandidate("cn", "ask_question", "persona_about_user", "观察利用孤独、施压、保持依赖这些词是否只是对话材料。", "medium", "low"),
    ]
    allowed, findings = apply_safety_linter(candidates)
    assert [item.candidate_id for item in allowed] == ["trust", "drawn", "pressure", "lonely", "cn"]
    assert findings == ()


def test_m12_2_plain_language_linter_blocks_jargon_tokens():
    findings = lint_text("This has expected information gain and a user model.", section="card")
    assert {finding.rule for finding in findings} == {"always_blocked_jargon"}


def test_m12_2_plain_language_linter_allows_ordinary_use_of_predict_and_model():
    assert not lint_text("I predict it will rain tomorrow.", section="reply")
    assert not lint_text("The showroom model is on the left.", section="reply")


def test_m12_2_consolidation_replays_byte_identically_from_fixtures():
    kwargs = dict(
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain whether you remember me consistently?",
        current_turn_quotes={"q1": "Can you explain whether you remember me consistently?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        extractors=_extractors(),
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    a_state, a_result = run_m12_2_tick(M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}), **kwargs)
    b_state, b_result = run_m12_2_tick(M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}), **kwargs)
    assert a_state.to_dict() == b_state.to_dict()
    assert json.dumps(a_result.to_dict(), sort_keys=True) == json.dumps(b_result.to_dict(), sort_keys=True)


def test_m12_2_does_not_mutate_m11_m12_0_or_m12_1_state():
    m11 = {"active_hypotheses": [{"hypothesis_id": "h1"}]}
    m12 = {"identity_state": "corroborated"}
    m121 = {"latest_report_status": "ready"}
    before = json.dumps([m11, m12, m121], sort_keys=True)
    run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain whether you remember me consistently?",
        current_turn_quotes={"q1": "Can you explain whether you remember me consistently?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        m11_readonly_summary=m11,
        m12_readonly_summary=m12,
        m121_readonly_summary=m121,
        extractors=_extractors(),
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    assert json.dumps([m11, m12, m121], sort_keys=True) == before


def test_m12_2_does_not_import_forbidden_memory_modules():
    root = Path("segmentum/reciprocal_role")
    forbidden = ("memory_dynamics", "value_memory", "memory_retrieval", "memory_store")
    mutation_names = ("run_m11_turn", "run_m12_turn", "run_m12_1_tick", "apply_claims_to_user_model")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(f"segmentum.{name}" in text or f"import {name}" in text for name in forbidden), path
        assert not any(name in text for name in mutation_names), path
    importlib.import_module("segmentum.reciprocal_role")


def test_m12_2_sparse_evidence_returns_insufficient_evidence():
    state, result = run_m12_2_tick(
        M122RuntimeState.clean(),
        user_id="u1",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        user_text="hi",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    assert result.light_turn_assessment is not None
    assert result.light_turn_assessment.insufficient_evidence is True
    assert state.models_by_user["u1"].all_claims() == ()


def test_m12_2_bootstrap_first_turn_skips_durable_consolidation():
    _state, result = run_m12_2_tick(
        M122RuntimeState.clean(),
        user_id="u1",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        user_text="I need help checking code.",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    assert result.trigger_decision.reason == "bootstrap_first_turn_skips_durable"


def test_m12_2_user_explicit_meta_request_produces_safe_plain_explanation():
    state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain how you understand what I am checking about you?",
        current_turn_quotes={"q1": "Can you explain how you understand what I am checking about you?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        extractors=_extractors(),
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    assert state.models_by_user["u1"].all_claims()
    assert all(not lint_text(card.content_summary, section="card") for card in result.evidence_cards)


def test_m12_2_claim_group_converges_by_raising_member_confidence():
    model = ReciprocalRoleModel.empty(user_id="u1")
    group = ReciprocalClaimGroup("g", "persona_about_user", "user_goal", ("a", "b"))
    claims = [
        ReciprocalClaim("a", "g", "persona_about_user", "internal", "The user may want implementation.", confidence_band="low", evidence_refs=_refs()),
        ReciprocalClaim("b", "g", "persona_about_user", "internal", "The user may want planning.", confidence_band="low", evidence_refs=_refs()),
    ]
    model = apply_model_patch(model, turn_id="t1", group_updates=[group], claims=claims)
    model = promote_claim_with_evidence(model, group_id="g", claim_id="a", turn_id="t2")
    assert [g for g in model.reciprocal_claim_groups if g.group_id == "g"][0].status == "converging"


def test_m12_2_contradicted_group_triggers_member_re_expansion():
    model = ReciprocalRoleModel.empty(user_id="u1")
    group = ReciprocalClaimGroup("g", "persona_about_user", "user_goal", ("old",))
    old = ReciprocalClaim("old", "g", "persona_about_user", "internal", "The user may want planning.", confidence_band="medium", evidence_refs=_refs())
    model = apply_model_patch(model, turn_id="t1", group_updates=[group], claims=[old])
    model = mark_group_contradicted(model, group_id="g", turn_id="t2")
    new = ReciprocalClaim("new", "g", "persona_about_user", "internal", "The user may want implementation.", confidence_band="low", evidence_refs=_refs())
    model = apply_model_patch(model, turn_id="t3", claims=[new])
    member_ids = [claim.claim_id for claim in model.persona_about_user_claims]
    assert {"old", "new"} <= set(member_ids)
    assert any(claim.claim_id == "old" and claim.status == "contradicted" for claim in model.persona_about_user_claims)


def test_m12_2_contradiction_cooldown_skips_consolidation_and_downgrades():
    model = ReciprocalRoleModel.empty(user_id="u1")
    claim = ReciprocalClaim("c", "g", "user_about_persona", "internal", "The user may be testing consistency.", confidence_band="medium", evidence_refs=_refs())
    model = apply_model_patch(model, turn_id="t1", claims=[claim])
    model = mark_group_contradicted(model, group_id="g", turn_id="t2", turn_index=2)
    model = mark_group_contradicted(model, group_id="g", turn_id="t3", turn_index=3)
    model = mark_group_contradicted(model, group_id="g", turn_id="t4", turn_index=4)
    state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": model}),
        user_id="u1",
        turn_id="t5",
        turn_index=5,
        hour_bucket=1,
        user_text="Can you explain yourself?",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    assert state.models_by_user["u1"].contradiction_cooldown > 0
    assert result.trigger_decision.reason == "contradiction_cooldown"
    assert state.models_by_user["u1"].user_about_persona_claims[0].confidence_band in {"low", "insufficient_evidence"}


def test_m12_2_volatile_hint_dropped_when_durable_hint_matches_claim_id():
    volatile = ReplyPolicyHint("v", "clarify_persona_stance", "Volatile", "user_about_persona", "high", claim_id="c1")
    durable = ReplyPolicyHint("d", "clarify_persona_stance", "Durable", "user_about_persona", "high", claim_id="c1", source="durable")
    merged = reconcile_hints([volatile], [durable], durable_ran=True)
    assert [hint.hint_id for hint in merged] == ["d"]


def test_m12_2_emits_no_action_instead_of_low_quality_hint():
    low = {
        **_first_output(),
        "high_gain_candidates": [
            {
                "candidate_id": "low",
                "kind": "ask_question",
                "target_axis": "persona_about_user",
                "plain_action": "Ask a minor wording preference.",
                "expected_gain_band": "low",
                "risk_band": "low",
                "consent_requirement": "none",
                "evidence_refs": ["t1:q1"],
                "blocked_by_safety": False,
            }
        ],
    }
    state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain yourself?",
        current_turn_quotes={"q1": "Can you explain yourself?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        extractors={"first_order": lambda _s: low, "second_order": lambda _s: {**_second_output(), "clarifying_reply_candidates": []}},
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    assert state.models_by_user["u1"].high_gain_candidates[0].kind == "no_action"
    assert any(hint.kind == "no_action" for hint in result.reply_policy_hints)


def test_m12_2_relationship_value_memory_produces_assessment_without_keyword_cue():
    _state, result = run_m12_2_tick(
        M122RuntimeState.clean(),
        user_id="zq",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        user_text="ok",
        relationship_value_memories=[
            {
                "id": "rvm_zq_plain",
                "summary": "zq is more comfortable with plain, low-performance warmth in ordinary chat.",
                "prediction_constraint": "Plain direct warmth is predicted to reduce relationship friction for zq.",
                "priority": "high",
                "confidence": 0.91,
                "source": "test",
                "evidence": "raw feedback must not be forwarded",
            }
        ],
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )

    assessment = result.relationship_value_assessment
    assert assessment is not None
    payload = assessment.to_dict()
    assert payload["persona_about_user"]
    assert payload["user_about_persona"]
    assert payload["user_comfort_pressure_band"] == "high"
    assert payload["predicted_conflict_band"] == "high"
    assert payload["preferred_policy"] == "adapt_to_relationship_value"
    assert "avoid_phrases" not in json.dumps(payload, ensure_ascii=False)
    assert "raw feedback" not in json.dumps(payload, ensure_ascii=False)
    hint = next(hint for hint in result.reply_policy_hints if hint.source == "relationship_value_memory")
    assert hint.kind == "apply_relationship_value_constraint"
    assert hint.priority == "high"


def test_m12_2_relationship_value_hint_is_not_lowered_by_reconciliation():
    relationship_hint = ReplyPolicyHint(
        "hint:relationship",
        "apply_relationship_value_constraint",
        "Prefer plain direct warmth for this relationship context.",
        "user_about_persona",
        priority="high",
        source="relationship_value_memory",
        topic_label="relationship_value_context",
    )
    ordinary_hint = ReplyPolicyHint(
        "hint:ordinary",
        "ask_clear_question",
        "Ask a normal clarifying question.",
        "persona_about_user",
        priority="high",
        topic_label="ordinary",
    )

    merged = reconcile_hints([relationship_hint, ordinary_hint], [], durable_ran=True)

    by_id = {hint.hint_id: hint for hint in merged}
    assert by_id["hint:relationship"].priority == "high"
    assert by_id["hint:ordinary"].priority == "medium"


def test_m12_2_extractor_snapshot_includes_prompt_safe_relationship_value_context():
    captured: dict[str, Mapping[str, object]] = {}

    def first(snapshot: Mapping[str, object]) -> Mapping[str, object]:
        captured["first"] = snapshot
        return _first_output()

    def second(snapshot: Mapping[str, object]) -> Mapping[str, object]:
        captured["second"] = snapshot
        return _second_output()

    run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain yourself?",
        current_turn_quotes={"q1": "Can you explain yourself?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        relationship_value_memories=[
            {
                "id": "rvm",
                "summary": "This user prefers low-performance directness.",
                "prediction_constraint": "Direct low-performance replies reduce relationship friction.",
                "priority": "high",
                "confidence": 0.88,
                "source": "test",
                "evidence": "raw evidence should stay out",
            }
        ],
        extractors={"first_order": first, "second_order": second},
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )

    context = captured["first"]["relationship_value_context"]
    encoded = json.dumps(context, ensure_ascii=False)
    assert "active_memories" in context
    assert "prediction_constraint" in encoded
    assert "raw evidence should stay out" not in encoded


def test_m12_2_reciprocal_role_update_event_published_on_durable_patch():
    bus = CognitiveEventBus()
    _state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Do you remember what you said before?",
        current_turn_quotes={"q1": "Do you remember what you said before?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        extractors=_extractors(),
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
        event_bus=bus,
        event_sequence_index=7,
    )
    assert result.published_event_ids
    assert bus.events()[0].event_type == "ReciprocalRoleUpdateEvent"
    assert "claims_added" in bus.events()[0].payload["patch_summary"]


def test_second_order_schema_rejects_high_confidence_and_unknown_refs():
    snapshot = {"turn_id": "t1", "current_turn_quotes": {"q1": "Do you remember me?"}, "allowed_evidence_quote_refs": ["t1:q1"]}
    bad_high = _second_output()
    bad_high["user_about_persona_claims"][0]["confidence_band"] = "high"
    with pytest.raises(SecondOrderExtractorValidationError):
        validate_second_order_output(bad_high, snapshot=snapshot)
    bad_ref = _second_output()
    bad_ref["user_about_persona_claims"][0]["evidence_refs"] = ["missing:q9"]
    with pytest.raises(SecondOrderExtractorValidationError):
        validate_second_order_output(bad_ref, snapshot=snapshot)


def test_m12_2_evidence_refs_are_not_repr_polluted_after_runtime_conversion():
    state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": ReciprocalRoleModel.empty(user_id="u1")}),
        user_id="u1",
        turn_id="t1",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain whether you remember me consistently?",
        current_turn_quotes={"q1": "Can you explain whether you remember me consistently?"},
        transcript_quote_refs=[{"turn_id": "t1", "quote_id": "q1"}],
        extractors=_extractors(),
        trigger_input=_trigger(),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    ref_ids = _collect_ref_ids({"state": state.to_dict(), "result": result.to_dict()})
    assert ref_ids
    assert all("EvidenceRef(" not in ref_id for ref_id in ref_ids)
    assert all(ref_id.count(":") == 1 and all(part for part in ref_id.split(":", 1)) for ref_id in ref_ids)


def test_m12_2_safety_linter_uses_term_boundaries_for_common_false_positives():
    candidates = [
        InformationGainCandidate("secretary", "ask_question", "persona_about_user", "Ask whether the secretary sent the agenda.", "medium", "low"),
        InformationGainCandidate("dependency", "ask_question", "persona_about_user", "Ask whether dependency injection is acceptable in this code.", "medium", "low"),
    ]
    allowed, findings = apply_safety_linter(candidates)
    assert [candidate.candidate_id for candidate in allowed] == ["secretary", "dependency"]
    assert findings == ()


def test_m12_2_plain_language_linter_leaves_internal_explanation_to_observer_hook():
    for text in ("对你的预测", "更新预测", "对你的模型", "建模你", "your prediction was updated", "user model"):
        assert not lint_text(text, section="reply"), text

    def observer(text: str, section: str):
        if "对你的模型" not in text:
            return ()
        return (
            PlainLanguageFinding(
                token="observer",
                section=section,
                rule="observer_internal_explanation",
                raw_quote=text,
            ),
        )

    findings = lint_text("对你的模型", section="reply", observer=observer)
    assert [finding.rule for finding in findings] == ["observer_internal_explanation"]
    assert not lint_text("This happened prior to the meeting.", section="reply")


def test_m12_2_plain_language_linter_allows_ordinary_chinese_model_and_prediction_uses():
    allowed = (
        "我的模型车放在桌上。",
        "你的模型号是多少？",
        "我预测这周会下雨。",
        "我猜周末可能会去爬山。",
    )
    for text in allowed:
        assert not lint_text(text, section="reply"), text


def test_m12_2_bounded_snapshot_drops_full_reports_and_truncates_raw_quotes():
    long_text = "x" * 400
    snapshot = {
        "turn_id": "t9",
        "current_turn_quotes": {"q1": long_text},
        "transcript_quote_refs": [{"turn_id": f"t{i}", "quote_id": "q"} for i in range(20)],
        "m11_readonly_summary": {"active_hypotheses": [{"id": i, "text": long_text} for i in range(20)], "full_memory_dump": long_text},
        "m12_readonly_summary": {"identity_state": "ready", "full_prompt": long_text},
        "m121_readonly_summary": {"orchestrator_result": {"report": {"report_status": "ready", "full_report": long_text}}},
        "model": ReciprocalRoleModel.empty(user_id="u1"),
    }
    bounded = bound_extractor_snapshot(snapshot)
    encoded = json.dumps(bounded, ensure_ascii=False, sort_keys=True)
    assert "full_memory_dump" not in encoded
    assert "full_prompt" not in encoded
    assert "full_report" not in encoded
    assert bounded["current_turn_quotes"]["q1"] == long_text[:260]
    assert len(bounded["transcript_quote_refs"]) == 12


def test_m12_2_default_contradiction_trigger_marks_group_and_records_cooldown():
    model = ReciprocalRoleModel.empty(user_id="u1")
    group = ReciprocalClaimGroup("g", "user_about_persona", "persona_consistency", ("c",))
    claim = ReciprocalClaim(
        "c",
        "g",
        "user_about_persona",
        "internal",
        "The user may be checking consistency.",
        confidence_band="medium",
        evidence_refs=(EvidenceRef("t1:q1"),),
    )
    model = apply_model_patch(model, turn_id="t1", turn_index=1, group_updates=[group], claims=[claim], direct_probe_turn_ids=("t1",))
    state, result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"u1": model}),
        user_id="u1",
        turn_id="t2",
        turn_index=4,
        hour_bucket=1,
        user_text="Are you sure you are not lying about what you remember?",
        current_turn_quotes={"q1": "Are you sure you are not lying about what you remember?"},
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    next_model = state.models_by_user["u1"]
    assert result.trigger_decision.kind == "contradiction_or_misread"
    assert next_model.contradiction_turn_ids == (4,)
    assert next_model.user_about_persona_claims[0].status == "contradicted"


def test_second_order_schema_rejects_each_claim_without_evidence_refs():
    snapshot = {"turn_id": "t1", "current_turn_quotes": {"q1": "Do you remember me?"}, "allowed_evidence_quote_refs": ["t1:q1"]}
    payload = _second_output()
    payload["user_about_persona_claims"].append(
        {
            "claim_id": "c_missing_refs",
            "group_id": "g_persona_consistency",
            "topic_label": "missing_refs",
            "claim_text_internal": "User is probing.",
            "claim_text_plain": "The user may be probing.",
            "evidence_refs": [],
            "confidence_band": "low",
            "uncertainty_band": "high",
            "status": "inferred_hypothesis",
        }
    )
    with pytest.raises(SecondOrderExtractorValidationError):
        validate_second_order_output(payload, snapshot=snapshot)


def test_m12_2_promote_claim_with_evidence_cannot_bypass_second_order_ceiling():
    model = ReciprocalRoleModel.empty(user_id="u1")
    group = ReciprocalClaimGroup("g", "user_about_persona", "persona_consistency", ("c",))
    claim = ReciprocalClaim(
        "c",
        "g",
        "user_about_persona",
        "internal",
        "The user may be checking consistency.",
        confidence_band="medium",
        evidence_refs=(EvidenceRef("t1:q1"),),
    )
    model = apply_model_patch(model, turn_id="t1", group_updates=[group], claims=[claim], direct_probe_turn_ids=("t1",))
    promoted = promote_claim_with_evidence(model, group_id="g", claim_id="c", turn_id="t9")
    assert promoted.user_about_persona_claims[0].confidence_band == "medium"


def test_m12_2_claim_patch_upserts_by_claim_id():
    model = ReciprocalRoleModel.empty(user_id="u1")
    claim = ReciprocalClaim("c", "g1", "persona_about_user", "internal", "The user may want review.", confidence_band="low", evidence_refs=_refs())
    model = apply_model_patch(model, turn_id="t1", claims=[claim])
    replacement = ReciprocalClaim("c", "g1", "persona_about_user", "internal", "The user may want implementation review.", confidence_band="medium", evidence_refs=_refs())
    model = apply_model_patch(model, turn_id="t2", claims=[replacement])
    assert [item.claim_id for item in model.persona_about_user_claims] == ["c"]
    assert model.persona_about_user_claims[0].claim_text_plain == "The user may want implementation review."


def test_m12_2_mvp_loop_passes_cognitive_event_bus_to_m12_2_runtime():
    text = Path("segmentum/dialogue/runtime/mvp_loop.py").read_text(encoding="utf-8")
    assert "event_bus=m12_cognitive_bus" in text
    assert "m12_2_event_start" in text


def _collect_ref_ids(value):
    if isinstance(value, dict):
        out = []
        if "ref_id" in value:
            out.append(str(value["ref_id"]))
        for child in value.values():
            out.extend(_collect_ref_ids(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(_collect_ref_ids(child))
        return out
    return []
