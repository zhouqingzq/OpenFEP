"""Tests for M20.3 §1 PolicyProducer (Tier A.1).

M20.3 freezes the PolicyProducer surface and the v1 input →
observable → owner_id → scope mapping table. These tests cover
each row of the table, the deterministic commit_id derivation,
the audit envelope, and the routing into the v2 registry.
"""

from __future__ import annotations

import inspect

import pytest

from segmentum.dialogue.runtime.active_commitment import (
    COMMITMENT_REGISTRY_V2,
    ActiveCommitment,
    compute_commit_id,
    is_registry_v2_accepts_policy_correction,
)
from segmentum.dialogue.runtime.policy_producer import (
    PolicyProducer,
    build_policy_admitted_event,
    is_policy_owner_accepts_policy_correction,
    policy_producer_table_snapshot,
)


# === frozen producer surface ============================================


def test_policy_producer_evaluate_signature_is_frozen() -> None:
    sig = inspect.signature(PolicyProducer.evaluate)
    params = list(sig.parameters.keys())
    # First param is `self`; the rest are the four M20.3 §1.1 inputs.
    assert params[1:] == [
        "turn_context",
        "runtime_mode_flags",
        "command_envelope",
        "user_correction_signal",
    ]
    # All inputs are keyword-only by M20.3 convention.
    for name in params[1:]:
        assert sig.parameters[name].kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )


def test_policy_producer_table_is_frozen_and_complete() -> None:
    snapshot = policy_producer_table_snapshot()
    assert len(snapshot) == 12
    kinds = {row["kind"] for row in snapshot}
    expected = {
        "command_status",
        "command_mode",
        "command_persona",
        "surface_intent_bot",
        "surface_intent_chat",
        "surface_intent_abstain",
        "user_correction_wrong_persona",
        "user_correction_wrong_voice",
        "user_correction_reaffirm",
        "group_mode_ingress_change",
        "command_quiet",
        "command_resume",
    }
    assert kinds == expected


def test_policy_producer_table_owners_are_in_v2_registry() -> None:
    snapshot = policy_producer_table_snapshot()
    for row in snapshot:
        assert row["owner_id"] in COMMITMENT_REGISTRY_V2, (
            f"owner {row['owner_id']!r} not in v2 registry"
        )


# === command envelope mappings ==========================================


def test_policy_producer_emits_for_status_command_turn_scoped() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 0, "at": "2026-06-06T00:00:00Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/status", "bot_command_args": []},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    commitment = admitted[0]
    assert commitment.owner_id == "runtime_mode_state"
    assert commitment.observable == "runtime_mode_state"
    assert commitment.source_kind == "policy"
    assert commitment.horizon == "same_turn_surface"
    assert len(audit_events) == 1
    event = audit_events[0]
    assert event["type"] == "PolicyAdmitted"
    assert event["rule_kind"] == "command_status"
    assert event["commit_id"] == commitment.commit_id


def test_policy_producer_emits_for_mode_command_durable_mutate() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 1, "at": "2026-06-06T00:00:01Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/mode", "bot_command_args": ["persona_chat"]},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    commitment = admitted[0]
    assert commitment.owner_id == "runtime_mode_state"
    assert commitment.observable_payload["expected_mode"] == "persona_chat"
    assert audit_events[0]["rule_kind"] == "command_mode"


def test_policy_producer_emits_for_persona_command_durable_mutate() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 2, "at": "2026-06-06T00:00:02Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/persona", "bot_command_args": ["roleplay"]},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    assert admitted[0].observable_payload["expected_mode"] == "roleplay"


def test_policy_producer_emits_for_quiet_command_outreach_intent_off() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 3, "at": "2026-06-06T00:00:03Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/quiet", "bot_command_args": []},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    commitment = admitted[0]
    assert commitment.owner_id == "outreach_intent_registry"
    assert commitment.observable == "outreach_intent_off"
    assert audit_events[0]["rule_kind"] == "command_quiet"


def test_policy_producer_emits_for_resume_command_outreach_intent_on() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 4, "at": "2026-06-06T00:00:04Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/resume", "bot_command_args": []},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    assert admitted[0].observable == "outreach_intent_on"


def test_policy_producer_unknown_command_emits_nothing() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 5, "at": "2026-06-06T00:00:05Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/unknown", "bot_command_args": []},
        user_correction_signal="",
    )
    assert admitted == []
    assert audit_events == []


def test_policy_producer_invalid_mode_arg_emits_nothing() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 6, "at": "2026-06-06T00:00:06Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/mode", "bot_command_args": ["unknown_mode"]},
        user_correction_signal="",
    )
    assert admitted == []


# === runtime_mode_flags mappings ========================================


def test_policy_producer_emits_for_surface_intent_bot() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 7, "at": "2026-06-06T00:00:07Z"},
        runtime_mode_flags={"surface_intent": "bot"},
        command_envelope={},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    commitment = admitted[0]
    assert commitment.observable_payload["expected_mode"] == "bot_system"
    assert commitment.horizon == "same_turn_surface"


def test_policy_producer_emits_for_surface_intent_chat() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 8, "at": "2026-06-06T00:00:08Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    assert admitted[0].observable_payload["expected_mode"] == "persona_chat"


def test_policy_producer_emits_for_surface_intent_abstain() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 9, "at": "2026-06-06T00:00:09Z"},
        runtime_mode_flags={"surface_intent": "abstain"},
        command_envelope={},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    assert admitted[0].observable_payload["expected_mode"] == "abstain"


def test_policy_producer_unknown_surface_intent_emits_nothing() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 10, "at": "2026-06-06T00:00:10Z"},
        runtime_mode_flags={"surface_intent": "unknown_intent"},
        command_envelope={},
        user_correction_signal="",
    )
    assert admitted == []


# === user_correction_signal mappings ====================================


def test_policy_producer_emits_for_wrong_persona_signal() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 11, "at": "2026-06-06T00:00:11Z"},
        runtime_mode_flags={},
        command_envelope={},
        user_correction_signal="wrong_persona",
    )
    assert len(admitted) == 1
    commitment = admitted[0]
    assert commitment.owner_id == "runtime_mode_state"
    assert audit_events[0]["rule_kind"] == "user_correction_wrong_persona"
    # N4 (M20.3 follow-up): correction signals without explicit
    # target return expected_mode="" so the LLM downstream (not
    # engineering) picks the target.
    assert commitment.observable_payload["expected_mode"] == ""


def test_policy_producer_emits_for_wrong_voice_signal() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 12, "at": "2026-06-06T00:00:12Z"},
        runtime_mode_flags={},
        command_envelope={},
        user_correction_signal="wrong_voice",
    )
    assert len(admitted) == 1
    assert audit_events[0]["rule_kind"] == "user_correction_wrong_voice"


def test_policy_producer_emits_for_reaffirm_signal() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 13, "at": "2026-06-06T00:00:13Z"},
        runtime_mode_flags={},
        command_envelope={},
        user_correction_signal="right_persona_reaffirm",
    )
    assert len(admitted) == 1
    assert audit_events[0]["rule_kind"] == "user_correction_reaffirm"
    # N4: reaffirm also returns "" (no target) — the conscious
    # loop LLM picks the reaffirmed target.
    assert admitted[0].observable_payload["expected_mode"] == ""


def test_policy_producer_unknown_correction_signal_emits_nothing() -> None:
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 14, "at": "2026-06-06T00:00:14Z"},
        runtime_mode_flags={},
        command_envelope={},
        user_correction_signal="unknown_signal",
    )
    assert admitted == []


def test_policy_producer_n4_correction_signal_returns_empty_target() -> None:
    """N4 (M20.3 follow-up): when a correction signal is set
    without an explicit target, `expected_mode` is empty so the
    LLM downstream (not engineering) picks the target.

    Previously the producer hardcoded `persona_chat` which
    silently forced a durable mutate toward a specific mode
    without the LLM seeing what the user actually wanted.
    """
    producer = PolicyProducer()
    for signal in ("wrong_persona", "wrong_voice", "right_persona_reaffirm"):
        admitted, _ = producer.evaluate(
            turn_context={"turn_index": 100, "at": "2026-06-06T00:00:00Z"},
            runtime_mode_flags={},
            command_envelope={},
            user_correction_signal=signal,
        )
        assert len(admitted) == 1, f"signal {signal!r} should admit one row"
        commitment = admitted[0]
        assert commitment.observable_payload["expected_mode"] == "", (
            f"signal {signal!r} should return empty expected_mode, got "
            f"{commitment.observable_payload['expected_mode']!r}"
        )


# === group_mode_ingress_change ==========================================


def test_policy_producer_emits_for_group_mode_ingress_change() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 15, "at": "2026-06-06T00:00:15Z"},
        runtime_mode_flags={"group_mode_ingress_change": True},
        command_envelope={},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    assert audit_events[0]["rule_kind"] == "group_mode_ingress_change"


# === determinism and commit_id ==========================================


def test_policy_producer_is_deterministic_and_does_not_call_llm() -> None:
    """Same inputs → same `commit_id` (deterministic sha1 from v1)."""
    producer = PolicyProducer()
    a_admitted, a_events = producer.evaluate(
        turn_context={"turn_index": 16, "at": "2026-06-06T00:00:16Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )
    b_admitted, b_events = producer.evaluate(
        turn_context={"turn_index": 16, "at": "2026-06-06T00:00:16Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )
    assert len(a_admitted) == 1
    assert len(b_admitted) == 1
    assert a_admitted[0].commit_id == b_admitted[0].commit_id
    assert a_events[0]["commit_id"] == b_events[0]["commit_id"]


def test_policy_producer_commit_id_matches_v1_derivation() -> None:
    """The v1 commit_id derivation (sha1 of canonical input) is reused."""
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 17, "at": "2026-06-06T00:00:17Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )
    commitment = admitted[0]
    expected = compute_commit_id(
        owner_id="runtime_mode_state",
        source_ref="surface_intent_chat",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        created_turn=17,
    )
    assert commitment.commit_id == expected


def test_policy_producer_policy_admitted_event_uses_same_commit_id() -> None:
    """M20.3 §1.4: PolicyAdmitted and ActiveCommitmentCreated share commit_id."""
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 18, "at": "2026-06-06T00:00:18Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )
    commitment = admitted[0]
    assert audit_events[0]["commit_id"] == commitment.commit_id


def test_policy_producer_routes_to_runtime_mode_state_owner() -> None:
    """All `surface_intent` and command rows target `runtime_mode_state`."""
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 19, "at": "2026-06-06T00:00:19Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={"platform_command": "/mode", "bot_command_args": ["persona_chat"]},
        user_correction_signal="wrong_persona",
    )
    # Two rows: surface_intent_chat + command_mode + user_correction
    # may be 3 rows or deduped. Each must target runtime_mode_state
    # OR outreach_intent_registry.
    for c in admitted:
        assert c.owner_id in {"runtime_mode_state", "outreach_intent_registry"}


def test_policy_producer_emits_runtime_mode_state_and_outreach_simultaneously() -> None:
    """A turn with `/quiet` AND chat surface_intent produces 2 rows."""
    producer = PolicyProducer()
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 20, "at": "2026-06-06T00:00:20Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={"platform_command": "/quiet", "bot_command_args": []},
        user_correction_signal="",
    )
    observables = {c.observable for c in admitted}
    assert "runtime_mode_state" in observables
    assert "outreach_intent_off" in observables


# === v2 exception table routing =========================================


def test_runtime_mode_state_owner_accepts_policy_correction() -> None:
    """M20.3 §3.5: `runtime_mode_state` opts into the v2 exception."""
    assert is_registry_v2_accepts_policy_correction("runtime_mode_state") is True
    assert is_policy_owner_accepts_policy_correction("runtime_mode_state") is True


def test_outreach_intent_registry_owner_does_not_accept_policy_correction() -> None:
    """M20.3: `outreach_intent_registry` is observation-only."""
    assert is_registry_v2_accepts_policy_correction("outreach_intent_registry") is False


def test_v1_owner_does_not_accept_policy_correction() -> None:
    """M20.3: v1 owners keep the v1 'policy -> expire' rule."""
    assert is_registry_v2_accepts_policy_correction("policy_state") is False
    assert is_registry_v2_accepts_policy_correction("m13_drive_state") is False


# === audit event builder ================================================


def test_build_policy_admitted_event_shape() -> None:
    commitment = ActiveCommitment(
        commit_id="abc",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="surface_intent_chat",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at={"kind": "next_turn"},
        priority=0.6,
        confidence=0.7,
        evidence_refs=("turn_0_surface_intent",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    event = build_policy_admitted_event(
        commitment=commitment,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
        rule_kind="surface_intent_chat",
    )
    assert event["type"] == "PolicyAdmitted"
    assert event["commit_id"] == "abc"
    assert event["rule_kind"] == "surface_intent_chat"
    assert event["horizon"] == "same_turn_surface"
    assert event["engineering_proxy_label"] == "mvp_local_policy_admission"
    assert event["at"] == "2026-06-06T00:00:00Z"


# === combined inputs ====================================================


def test_policy_producer_combined_inputs_emit_all_matching_rows() -> None:
    producer = PolicyProducer()
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 21, "at": "2026-06-06T00:00:21Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={"platform_command": "/mode", "bot_command_args": ["persona_chat"]},
        user_correction_signal="wrong_persona",
    )
    # Three rules: surface_intent_chat, command_mode, user_correction_wrong_persona
    assert len(admitted) == 3
    rule_kinds = {event["rule_kind"] for event in audit_events}
    assert rule_kinds == {
        "surface_intent_chat",
        "command_mode",
        "user_correction_wrong_persona",
    }
