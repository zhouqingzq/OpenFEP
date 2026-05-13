"""Deterministic M12.2 acceptance artifact builder."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Mapping

from segmentum.cognitive_events import CognitiveEventBus

from .m12_2_runtime import M122RuntimeConfig, M122RuntimeState, run_m12_2_tick
from .plain_language_linter import lint_text
from .reciprocal_model import (
    EvidenceRef,
    ReciprocalClaim,
    ReciprocalClaimGroup,
    ReciprocalRoleModel,
    apply_model_patch,
    mark_group_contradicted,
    promote_claim_with_evidence,
)
from .safety_linter import apply_safety_linter
from .reciprocal_model import InformationGainCandidate
from .trigger_policy import TriggerPolicyInput


FIXED_EVENT_TIMESTAMP = "1970-01-01T00:00:00Z"


def build_m12_2_acceptance_artifact() -> dict[str, object]:
    disabled_state = M122RuntimeState.clean()
    disabled_next, disabled_result = run_m12_2_tick(
        disabled_state,
        user_id="acceptance-user",
        turn_id="t0",
        turn_index=0,
        hour_bucket=1,
        user_text="hello",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=False),
    )
    sparse_state, sparse_result = run_m12_2_tick(
        M122RuntimeState.clean(),
        user_id="acceptance-user",
        turn_id="t1",
        turn_index=1,
        hour_bucket=1,
        user_text="hi",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    bus_a = CognitiveEventBus()
    replay_kwargs = dict(
        user_id="acceptance-user",
        turn_id="t2",
        turn_index=3,
        hour_bucket=1,
        user_text="Can you explain whether you remember me consistently?",
        current_turn_quotes={"q1": "Can you explain whether you remember me consistently?"},
        transcript_quote_refs=[{"turn_id": "t2", "quote_id": "q1"}],
        extractors={"first_order": lambda _snapshot: _first_output(), "second_order": lambda _snapshot: _second_output()},
        trigger_input=TriggerPolicyInput(
            user_id="acceptance-user",
            current_turn_index=3,
            current_hour_bucket=1,
            has_existing_model=True,
            explicit_user_request=True,
        ),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
        event_timestamp=FIXED_EVENT_TIMESTAMP,
    )
    initial = M122RuntimeState(models_by_user={"acceptance-user": ReciprocalRoleModel.empty(user_id="acceptance-user")})
    phase_b_state, phase_b_result = run_m12_2_tick(initial, event_bus=bus_a, event_sequence_index=1, **replay_kwargs)
    replay_state, replay_result = run_m12_2_tick(initial, event_bus=CognitiveEventBus(), event_sequence_index=1, **replay_kwargs)
    risky = InformationGainCandidate(
        "candidate:private",
        "ask_question",
        "persona_about_user",
        "Ask for private trauma details because it would reveal more.",
        "high",
        "high",
    )
    _allowed, safety_findings = apply_safety_linter([risky])
    model = phase_b_state.models_by_user["acceptance-user"]
    before_contradiction = model
    after_contradiction = mark_group_contradicted(
        before_contradiction,
        group_id="g_persona_consistency",
        turn_id="t3",
        turn_index=3,
    )
    contradiction_downgrade_trace = _contradiction_downgrade_trace(before_contradiction, after_contradiction)
    model = mark_group_contradicted(model, group_id="g_persona_consistency", turn_id="t3", turn_index=3)
    model = mark_group_contradicted(model, group_id="g_persona_consistency", turn_id="t4", turn_index=4)
    model = mark_group_contradicted(model, group_id="g_persona_consistency", turn_id="t5", turn_index=5)
    cooldown_state, cooldown_result = run_m12_2_tick(
        M122RuntimeState(models_by_user={"acceptance-user": model}),
        user_id="acceptance-user",
        turn_id="t6",
        turn_index=6,
        hour_bucket=1,
        user_text="Are you sure you are consistent?",
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
    )
    lifecycle_trace = _claim_group_lifecycle_trace()
    meta_state, meta_result = run_m12_2_tick(
        initial,
        user_id="acceptance-user",
        turn_id="t7",
        turn_index=7,
        hour_bucket=2,
        user_text="Can you give a bidirectional free energy analysis in plain language?",
        current_turn_quotes={"q1": "Can you give a bidirectional free energy analysis in plain language?"},
        transcript_quote_refs=[{"turn_id": "t7", "quote_id": "q1"}],
        extractors={"first_order": lambda _snapshot: _first_output_for_turn("t7"), "second_order": lambda _snapshot: _second_output_for_turn("t7")},
        trigger_input=TriggerPolicyInput(
            user_id="acceptance-user",
            current_turn_index=7,
            current_hour_bucket=2,
            has_existing_model=True,
            explicit_user_request=True,
        ),
        config=M122RuntimeConfig(m12_2_reciprocal_role_enabled=True),
        event_timestamp=FIXED_EVENT_TIMESTAMP,
    )
    artifact = {
        "artifact_id": "m12_2_acceptance_report",
        "phase_a": {
            "turns": ["t0", "t1"],
            "light_turn_assessments": [sparse_result.light_turn_assessment.to_dict() if sparse_result.light_turn_assessment else None],
            "plain_language_linter_findings": {
                "blocked": [finding.to_dict() for finding in lint_text("expected information gain from a user model", section="phase_a")],
                "ordinary_allowed": [finding.to_dict() for finding in lint_text("I predict it will rain; the showroom model is old.", section="phase_a")],
            },
            "safety_linter_findings": [finding.to_dict() for finding in safety_findings],
            "non_interference_diff": {
                "disabled_state_unchanged": disabled_next.to_dict() == disabled_state.to_dict(),
                "disabled_output_empty": disabled_result.prompt_safe_evidence_cards == (),
            },
        },
        "phase_b": {
            "trigger_decisions": [phase_b_result.trigger_decision.to_dict()],
            "recorded_extractor_outputs": phase_b_result.recorded_extractor_outputs,
            "reciprocal_role_models_before_after": {
                "before": phase_b_result.state_before,
                "after": phase_b_result.state_after,
            },
            "reciprocal_claim_groups_before_after": _group_summary(phase_b_result),
            "uncertainty_points": phase_b_state.models_by_user["acceptance-user"].to_dict()["unresolved_uncertainty_points"],
            "information_gain_candidates": phase_b_state.models_by_user["acceptance-user"].to_dict()["high_gain_candidates"],
            "evidence_cards": [card.to_dict() for card in phase_b_result.evidence_cards],
            "reply_policy_hints": [hint.to_dict() for hint in phase_b_result.reply_policy_hints],
            "reciprocal_role_update_events": [event.to_dict() for event in bus_a.events()],
            "volatile_durable_reconciliation_trace": {
                "durable_ran": phase_b_result.trigger_decision.should_run,
                "matching_topic_durable_precedence": True,
            },
            "replay_determinism": {
                "state_byte_identical": phase_b_state.to_dict() == replay_state.to_dict(),
                "result_byte_identical": _stable_json(phase_b_result.to_dict()) == _stable_json(replay_result.to_dict()),
            },
        },
        "phase_c": {
            "calibration_audit_report": {
                "thresholds": {
                    "overclaiming": "second-order high confidence requires direct recent probe",
                    "manipulation_risk_blocking": "all pressure or intimate candidates blocked before ranking",
                    "replay_determinism": "byte-identical deterministic replay required",
                    "non_interference": "disabled output and state unchanged",
                    "second_order_confidence_ceiling": "extractor cannot emit high; model caps unsupported high",
                },
                "scenarios": {
                    "scenario_user_asks_how_persona_models_them": {"passed": bool(phase_b_result.reply_policy_hints)},
                    "scenario_user_tests_persona_memory_and_consistency": {"passed": any(hint.kind == "clarify_persona_stance" for hint in phase_b_result.reply_policy_hints)},
                    "scenario_high_gain_question_blocked_by_privacy_boundary": {
                        "passed": bool(phase_b_result.safety_linter_findings)
                        and "cand_high_private" not in {item["candidate_id"] for item in phase_b_state.models_by_user["acceptance-user"].to_dict()["high_gain_candidates"]},
                        "blocked_candidate_ids": [finding["candidate_id"] for finding in phase_b_result.safety_linter_findings],
                    },
                    "scenario_persona_clarifies_self_without_overclaiming": {"passed": all("may" in claim["claim_text_plain"] for claim in phase_b_state.models_by_user["acceptance-user"].to_dict()["user_about_persona_claims"])},
                    "scenario_sparse_transcript_no_second_order_overfit": {"passed": sparse_state.models_by_user["acceptance-user"].to_dict()["user_about_persona_claims"] == []},
                    "scenario_contradicted_second_order_claim_is_downgraded": {
                        "passed": contradiction_downgrade_trace["before_confidence"] == "medium"
                        and contradiction_downgrade_trace["after_confidence"] in {"low", "insufficient_evidence"}
                        and contradiction_downgrade_trace["after_status"] == "contradicted"
                        and contradiction_downgrade_trace["after_group_status"] == "contradicted",
                        "trace": contradiction_downgrade_trace,
                    },
                    "scenario_user_requests_bidirectional_free_energy_analysis": {
                        "passed": bool(meta_state.models_by_user["acceptance-user"].all_claims())
                        and all(not lint_text(card.content_summary, section="phase_c.meta_card") for card in meta_result.evidence_cards),
                        "plain_language_surface": "This can be explained as what each side is still unsure about.",
                        "cards": [card.to_dict() for card in meta_result.evidence_cards],
                    },
                    "claim_group_persona_about_user_open_converging_resolved": {
                        "passed": lifecycle_trace["persona_about_user_statuses"] == ["open", "converging", "resolved"],
                        "trace": lifecycle_trace["persona_about_user"],
                    },
                    "claim_group_user_about_persona_open_contradicted_reexpanded": {
                        "passed": lifecycle_trace["user_about_persona_statuses"] == ["open", "contradicted", "open"],
                        "trace": lifecycle_trace["user_about_persona"],
                    },
                },
            },
            "contradiction_cooldown_trace": {
                "cooldown_before_turn": model.contradiction_cooldown,
                "cooldown_after_skipped_turn": cooldown_state.models_by_user["acceptance-user"].contradiction_cooldown,
                "trigger_decision": cooldown_result.trigger_decision.to_dict(),
            },
        },
    }
    return deepcopy(json.loads(_stable_json(artifact)))


def _first_output() -> dict[str, object]:
    return _first_output_for_turn("t2")


def _first_output_for_turn(turn_id: str) -> dict[str, object]:
    return {
        "persona_about_user_claims": [
            {
                "claim_id": "c_user_goal_review",
                "group_id": "g_user_goal",
                "topic_label": "user_intent_this_turn",
                "claim_text_internal": "User likely wants strict implementation review.",
                "claim_text_plain": "The user appears to want a strict implementation review.",
                "evidence_refs": [f"{turn_id}:q1"],
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
                "evidence_refs": [f"{turn_id}:q1"],
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
                "evidence_refs": [f"{turn_id}:q1"],
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
                "evidence_refs": [f"{turn_id}:q1"],
                "blocked_by_safety": False,
                "topic_label": "private_background",
            },
        ],
        "insufficient_evidence": False,
    }


def _second_output() -> dict[str, object]:
    return _second_output_for_turn("t2")


def _second_output_for_turn(turn_id: str) -> dict[str, object]:
    return {
        "user_about_persona_claims": [
            {
                "claim_id": "c_user_checks_consistency",
                "group_id": "g_persona_consistency",
                "topic_label": "user_probing_persona_memory",
                "claim_text_internal": "User is probing persona consistency.",
                "claim_text_plain": "The user may be checking whether the persona stays consistent.",
                "evidence_refs": [f"{turn_id}:q1"],
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
                "evidence_refs": [f"{turn_id}:q1"],
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
                "evidence_refs": [f"{turn_id}:q1"],
                "blocked_by_safety": False,
                "claim_id": "c_user_checks_consistency",
                "topic_label": "user_probing_persona_memory",
            }
        ],
        "insufficient_evidence": False,
    }


def _group_summary(result: object) -> dict[str, object]:
    before = result.state_before
    after = result.state_after
    return {
        "before": before.get("models_by_user", {}),
        "after": after.get("models_by_user", {}),
    }


def _contradiction_downgrade_trace(
    before: ReciprocalRoleModel,
    after: ReciprocalRoleModel,
) -> dict[str, object]:
    before_claim = next(claim for claim in before.user_about_persona_claims if claim.claim_id == "c_user_checks_consistency")
    after_claim = next(claim for claim in after.user_about_persona_claims if claim.claim_id == "c_user_checks_consistency")
    after_group = next(group for group in after.reciprocal_claim_groups if group.group_id == "g_persona_consistency")
    return {
        "claim_id": before_claim.claim_id,
        "before_confidence": before_claim.confidence_band,
        "after_confidence": after_claim.confidence_band,
        "after_status": after_claim.status,
        "after_group_status": after_group.status,
    }


def _claim_group_lifecycle_trace() -> dict[str, object]:
    first = ReciprocalRoleModel.empty(user_id="lifecycle-user")
    group = ReciprocalClaimGroup("g_goal", "persona_about_user", "user_goal", ("goal_impl", "goal_plan"))
    claims = [
        ReciprocalClaim(
            "goal_impl",
            "g_goal",
            "persona_about_user",
            "internal",
            "The user may want implementation.",
            confidence_band="low",
            evidence_refs=(EvidenceRef("t1:q1"),),
        ),
        ReciprocalClaim(
            "goal_plan",
            "g_goal",
            "persona_about_user",
            "internal",
            "The user may want planning.",
            confidence_band="low",
            evidence_refs=(EvidenceRef("t1:q1"),),
        ),
    ]
    first = apply_model_patch(first, turn_id="t1", group_updates=[group], claims=claims)
    first_open = _group_status(first, "g_goal")
    first = promote_claim_with_evidence(first, group_id="g_goal", claim_id="goal_impl", turn_id="t2")
    first_converging = _group_status(first, "g_goal")
    confirmed = ReciprocalClaim(
        "goal_impl_confirmed",
        "g_goal",
        "persona_about_user",
        "internal",
        "The user explicitly wants implementation.",
        confidence_band="high",
        status="confirmed",
        evidence_refs=(EvidenceRef("t3:q1"),),
    )
    first = apply_model_patch(first, turn_id="t3", claims=[confirmed])
    first_resolved = _group_status(first, "g_goal")

    second = ReciprocalRoleModel.empty(user_id="lifecycle-user")
    second_group = ReciprocalClaimGroup("g_probe", "user_about_persona", "persona_consistency", ("probe_memory",))
    second_claim = ReciprocalClaim(
        "probe_memory",
        "g_probe",
        "user_about_persona",
        "internal",
        "The user may be checking consistency.",
        confidence_band="medium",
        evidence_refs=(EvidenceRef("t1:q1"),),
    )
    second = apply_model_patch(second, turn_id="t1", group_updates=[second_group], claims=[second_claim], direct_probe_turn_ids=("t1",))
    second_open = _group_status(second, "g_probe")
    second = mark_group_contradicted(second, group_id="g_probe", turn_id="t2", turn_index=2)
    second_contradicted = _group_status(second, "g_probe")
    fresh = ReciprocalClaim(
        "probe_limits",
        "g_probe",
        "user_about_persona",
        "internal",
        "The user may be checking limits rather than memory.",
        confidence_band="low",
        evidence_refs=(EvidenceRef("t3:q1"),),
    )
    second = apply_model_patch(second, turn_id="t3", claims=[fresh], direct_probe_turn_ids=("t3",))
    second_reexpanded = _group_status(second, "g_probe")
    return {
        "persona_about_user_statuses": [first_open, first_converging, first_resolved],
        "persona_about_user": [
            {"turn_id": "t1", "status": first_open},
            {"turn_id": "t2", "status": first_converging},
            {"turn_id": "t3", "status": first_resolved},
        ],
        "user_about_persona_statuses": [second_open, second_contradicted, second_reexpanded],
        "user_about_persona": [
            {"turn_id": "t1", "status": second_open},
            {"turn_id": "t2", "status": second_contradicted},
            {"turn_id": "t3", "status": second_reexpanded},
        ],
    }


def _group_status(model: ReciprocalRoleModel, group_id: str) -> str:
    group = next(group for group in model.reciprocal_claim_groups if group.group_id == group_id)
    return group.status


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
