import importlib
from pathlib import Path

import pytest

from segmentum.cognitive_events import CognitiveEventBus
from segmentum.user_continuity import (
    M12RuntimeConfig,
    M12RuntimeState,
    ReplyPolicyDecision,
    run_m12_turn,
    select_reply_policy,
    validate_extractor_output,
)
from segmentum.user_continuity.evidence_cards import cards_to_prompt_safe_memory_evidence
from segmentum.user_continuity.identity_profile import AliasObservation, IdentityProfile
from segmentum.user_continuity.llm_identity_extractor import ExtractorValidationError


def _extractor(*, claims=None, cues=None, band="low", surprise="") -> object:
    payload = {
        "identity_claims": claims or [],
        "continuity_cues": cues or [],
        "strangeness_band": band,
        "surprise_explanation": surprise,
    }
    return lambda snapshot: payload


def test_identity_extractor_output_with_any_float_is_rejected():
    payload = _extractor(claims=[{"id": "c1", "claimant_user_id": "u", "asserted_alias": "A", "modality": "factual", "evidence_quote_ids": ["q1"], "confidence_band": 0.7}])({})
    with pytest.raises(ExtractorValidationError):
        validate_extractor_output(payload)


def test_identity_extractor_output_with_unknown_field_is_rejected():
    payload = _extractor()({})
    payload["unknown"] = "x"
    with pytest.raises(ExtractorValidationError):
        validate_extractor_output(payload)


def test_single_utterance_alias_claim_does_not_promote_to_corroborated():
    state = M12RuntimeState.clean()
    state, result = run_m12_turn(
        state,
        user_id="u1",
        display_name="U1",
        turn_id="turn_0001",
        extractor=_extractor(
            claims=[
                {
                    "id": "claim:1",
                    "claimant_user_id": "u1",
                    "asserted_alias": "Alice",
                    "modality": "factual",
                    "evidence_quote_ids": ["q1"],
                    "confidence_band": "high",
                }
            ]
        ),
        config=M12RuntimeConfig(m12_identity_continuity_enabled=True),
    )
    profile = state.profiles_by_user["u1"]
    assert profile.identity_state in {"asserted", "unverified"}
    assert profile.identity_state != "corroborated"
    assert result.reply_policy.permitted_response in {"hedge", "ask", "probe", "observe"}


def test_three_bind_cues_single_turn_do_not_promote_to_corroborated_via_cues():
    """Cue-only binding must not hit corroborated from one turn even with many bind cues."""
    state = M12RuntimeState.clean()
    cues = [
        {
            "id": f"cue:{i}",
            "cue_kind": "history",
            "supports": "binds",
            "content_summary": f"Same-turn bind {i}",
            "evidence_quote_ids": ["q1"],
            "confidence_band": "med",
        }
        for i in range(3)
    ]
    state, result = run_m12_turn(
        state,
        user_id="u1",
        display_name="U1",
        turn_id="turn_0001",
        extractor=_extractor(cues=cues),
        config=M12RuntimeConfig(m12_identity_continuity_enabled=True),
    )
    prof = state.profiles_by_user["u1"]
    assert prof.identity_state != "corroborated"


def test_continuity_cues_bind_alias_over_multiple_turns_under_threshold():
    state = M12RuntimeState.clean()
    config = M12RuntimeConfig(m12_identity_continuity_enabled=True)
    for idx in range(1, 4):
        state, _ = run_m12_turn(
            state,
            user_id="u1",
            display_name="U1",
            turn_id=f"turn_000{idx}",
            extractor=_extractor(
                claims=[
                    {
                        "id": f"claim:{idx}",
                        "claimant_user_id": "u1",
                        "asserted_alias": "Alice",
                        "modality": "factual",
                        "evidence_quote_ids": [f"q{idx}"],
                        "confidence_band": "med",
                    }
                ],
                cues=[
                    {
                        "id": f"cue:{idx}",
                        "cue_kind": "history",
                        "supports": "binds",
                        "content_summary": "Multi-turn continuity consistent",
                        "evidence_quote_ids": [f"q{idx}"],
                        "confidence_band": "med",
                    }
                ],
            ),
            config=config,
        )
    assert state.profiles_by_user["u1"].identity_state == "corroborated"


def test_new_user_id_asserting_corroborated_alias_emits_exactly_one_conflict_record():
    incumbent = IdentityProfile(
        user_id="u1",
        display_name="U1",
        aliases_observed=(
            AliasObservation("a1", "Alice", "turn_0001", "turn_0003", 3, "factual", "accept"),
        ),
        binding_confidence_band="high",
        identity_state="corroborated",
        last_updated_turn_id="turn_0003",
    )
    state = M12RuntimeState(profiles_by_user={"u1": incumbent})
    state, result = run_m12_turn(
        state,
        user_id="u2",
        display_name="U2",
        turn_id="turn_0004",
        extractor=_extractor(
            claims=[
                {
                    "id": "claim:impostor",
                    "claimant_user_id": "u2",
                    "asserted_alias": "Alice",
                    "modality": "factual",
                    "evidence_quote_ids": ["q4"],
                    "confidence_band": "high",
                }
            ]
        ),
        config=M12RuntimeConfig(m12_identity_continuity_enabled=True),
        identity_anchored_action=True,
    )
    assert len(result.conflict_records_created) == 1
    assert result.conflict_records_created[0].severity_band == "major"
    assert result.reply_policy.permitted_response in {"refuse", "ask"}


def test_roleplay_modality_never_produces_major_severity_conflict():
    incumbent = IdentityProfile(
        user_id="u1",
        display_name="U1",
        aliases_observed=(AliasObservation("a1", "Alice", "t1", "t3", 3, "factual", "accept"),),
        binding_confidence_band="high",
        identity_state="corroborated",
        last_updated_turn_id="t3",
    )
    state = M12RuntimeState(profiles_by_user={"u1": incumbent})
    _, result = run_m12_turn(
        state,
        user_id="u2",
        display_name="U2",
        turn_id="turn_0004",
        extractor=_extractor(
            claims=[
                {
                    "id": "claim:rp",
                    "claimant_user_id": "u2",
                    "asserted_alias": "Alice",
                    "modality": "roleplay",
                    "evidence_quote_ids": ["q4"],
                    "confidence_band": "high",
                }
            ]
        ),
        config=M12RuntimeConfig(m12_identity_continuity_enabled=True),
    )
    assert result.conflict_records_created
    assert all(item.severity_band != "major" for item in result.conflict_records_created)


def test_major_conflict_publishes_identity_strangeness_signal():
    incumbent = IdentityProfile(
        user_id="u1",
        display_name="U1",
        aliases_observed=(AliasObservation("a1", "Alice", "t1", "t3", 3, "factual", "accept"),),
        binding_confidence_band="high",
        identity_state="corroborated",
        last_updated_turn_id="t3",
    )
    bus = CognitiveEventBus()
    _, result = run_m12_turn(
        M12RuntimeState(profiles_by_user={"u1": incumbent}),
        user_id="u2",
        display_name="U2",
        turn_id="turn_0004",
        extractor=_extractor(
            claims=[
                {
                    "id": "claim:1",
                    "claimant_user_id": "u2",
                    "asserted_alias": "Alice",
                    "modality": "factual",
                    "evidence_quote_ids": ["q4"],
                    "confidence_band": "high",
                }
            ]
        ),
        config=M12RuntimeConfig(m12_identity_continuity_enabled=True),
        event_bus=bus,
    )
    assert result.strangeness_signal is not None
    assert any(event.event_type == "SelfThoughtEvent" for event in bus.events())


def test_identity_cards_with_user_facing_jargon_are_rejected():
    profile = IdentityProfile(
        user_id="u1",
        display_name="U1",
        aliases_observed=(AliasObservation("a1", "prediction error", "t1", "t1", 1, "factual", "accept"),),
        binding_confidence_band="low",
        identity_state="asserted",
    )
    with pytest.raises(ValueError):
        cards_to_prompt_safe_memory_evidence(profile=profile, open_conflicts=())


def test_deterministic_layer_runs_without_any_llm_call_using_stub_extractor():
    state = M12RuntimeState.clean()
    config = M12RuntimeConfig(m12_identity_continuity_enabled=True)
    for idx in range(1, 3):
        state, _ = run_m12_turn(
            state,
            user_id="u1",
            display_name="U1",
            turn_id=f"turn_{idx:04d}",
            extractor=_extractor(),
            config=config,
        )
    assert "u1" in state.profiles_by_user


def test_decisions_byte_identical_across_runs_with_same_extractor_fixture():
    config = M12RuntimeConfig(m12_identity_continuity_enabled=True)
    fixture = _extractor(
        claims=[{"id": "c1", "claimant_user_id": "u1", "asserted_alias": "Alice", "modality": "factual", "evidence_quote_ids": ["q1"], "confidence_band": "med"}],
        cues=[{"id": "k1", "cue_kind": "history", "supports": "binds", "content_summary": "same", "evidence_quote_ids": ["q1"], "confidence_band": "med"}],
    )
    state_a, result_a = run_m12_turn(
        M12RuntimeState.clean(),
        user_id="u1",
        display_name="U1",
        turn_id="turn_0001",
        extractor=fixture,
        config=config,
    )
    state_b, result_b = run_m12_turn(
        M12RuntimeState.clean(),
        user_id="u1",
        display_name="U1",
        turn_id="turn_0001",
        extractor=fixture,
        config=config,
    )
    assert state_a.to_dict() == state_b.to_dict()
    assert result_a.to_dict() == result_b.to_dict()


def test_m12_does_not_import_memory_dynamics_or_value_memory_or_memory_retrieval():
    root = Path("segmentum/user_continuity")
    forbidden = ("memory_dynamics", "value_memory", "memory_retrieval")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(
            f"segmentum.{name}" in text or f"..{name}" in text or f"import {name}" in text
            for name in forbidden
        ), path
    importlib.import_module("segmentum.user_continuity")


def test_m12_disabled_is_noop_state():
    state = M12RuntimeState.clean()
    next_state, result = run_m12_turn(
        state,
        user_id="u1",
        display_name="U1",
        turn_id="turn_0001",
        extractor=_extractor(),
        config=M12RuntimeConfig(m12_identity_continuity_enabled=False),
    )
    assert next_state.to_dict() == state.to_dict()
    assert result.enabled is False


def test_unverified_alias_never_selects_accept_for_identity_anchored_action():
    profile = IdentityProfile(user_id="u1", display_name="U1", identity_state="unverified")
    decision = select_reply_policy(
        profile=profile,
        active_conflicts=(),
        strangeness_signal=None,
        identity_anchored_action=True,
    )
    assert isinstance(decision, ReplyPolicyDecision)
    assert decision.permitted_response != "accept"
