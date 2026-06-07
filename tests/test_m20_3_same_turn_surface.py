"""Tests for M20.3 §3 SameTurnSurfaceSettler (Tier A.2).

M20.3 freezes the pre-send gate (can block only for
`runtime_mode_state` with `accepts_same_turn_block = true`) and
the post-send advisory (never blocks; writes to next-turn
`control_guidance`). The horizon attribute defaults to
`"next_turn"` for v1 commitments; v2 producers set
`"same_turn_surface"`.
"""

from __future__ import annotations

from types import MappingProxyType

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    HORIZON_V1,
    is_registry_v2_accepts_same_turn_block,
)
from segmentum.dialogue.runtime.same_turn_surface import (
    SameTurnSurfaceSettler,
    SameTurnSurfaceVerdict,
    build_same_turn_surface_verdict_event,
)


# === horizon attribute ==================================================


def test_horizon_attribute_is_v2_only() -> None:
    assert HORIZON_V1 == frozenset({"same_turn_surface", "next_turn", "natural_context"})


def test_horizon_defaults_to_next_turn() -> None:
    commitment = ActiveCommitment(
        commit_id="cid1",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="ref1",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        observable_payload={},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref1",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("self_expectation_formation",),
        engineering_proxy_label="mvp_local_self_expectation",
    )
    assert commitment.horizon == "next_turn"


def test_horizon_violated_outcomes_can_be_same_turn_surface() -> None:
    """M20.3 §3.0: v2 producers can set horizon = same_turn_surface."""
    commitment = ActiveCommitment(
        commit_id="cid2",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="ref2",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref2",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    assert commitment.horizon == "same_turn_surface"


# === pre-send gate ======================================================


def test_pre_send_returns_none_when_no_horizon_commitments() -> None:
    settler = SameTurnSurfaceSettler()
    verdict = settler.run_pre_send(
        draft_reply="hello",
        horizon_commitments=[],
    )
    assert verdict is None


def test_pre_send_passes_when_no_violation() -> None:
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_pass",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="surface_intent_chat",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "consistent",
            "committed_surface_intent": "persona_chat",
        },
    }
    verdict = settler.run_pre_send(
        draft_reply="你好，我在。",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.decision == "pass"
    assert verdict.horizon == "pre_send"
    assert verdict.owner_id == "runtime_mode_state"


def test_pre_send_can_block_when_owner_is_runtime_mode_state() -> None:
    """M20.3 §3.2: the gate can `block` for `runtime_mode_state`
    with `accepts_same_turn_block = true`.
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_block",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="surface_intent_chat",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            # The committed surface is bot_system, but expected was
            # persona_chat → drift_voice → "violated".
            "surface_intent_outcome": "drifted_voice",
            "committed_surface_intent": "bot_system",
        },
    }
    verdict = settler.run_pre_send(
        draft_reply="在线，路由正常，待命中。",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.decision == "block"
    assert verdict.replacement  # the bounded persona fallback
    assert "pre_send_block_runtime_mode_state" in verdict.reason_codes


def test_pre_send_cannot_block_for_non_runtime_owners() -> None:
    """M20.3 §3.2: non-runtime owners can only return `pass` or
    `advisory_guidance`. Even on a violated outcome, the gate
    returns `advisory_guidance` (no block, no replacement).
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_other",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="identity_voice_match",
        observable_payload={"expected_voice": "persona_chat", "actual_voice": "bot_system"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("boundary_check",),
        engineering_proxy_label="mvp_local_identity_voice",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_voice",
            "committed_surface_intent": "bot_system",
        },
    }
    verdict = settler.run_pre_send(
        draft_reply="在线。",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.decision == "advisory_guidance"
    assert not verdict.replacement  # advisory has no replacement


def test_pre_send_advisory_when_audit_absent() -> None:
    """Without a surface audit, the gate falls back to advisory
    (never block) so a missing audit does not flip the reply.
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_no_audit",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="surface_intent_chat",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    verdict = settler.run_pre_send(
        draft_reply="你好。",
        horizon_commitments=[commitment],
        observation_context={},  # no audit
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.decision == "advisory_guidance"
    assert "runtime_mode_state_audit_absent" in verdict.reason_codes


def test_r4_pre_send_block_defeated_by_missing_audit() -> None:
    """R4 (M20.3 follow-up): when the surface_consistency_verification
    audit is absent (e.g. fast_chat didn't run the LLM self-audit),
    the pre-send gate CANNOT block, even for `runtime_mode_state`
    with a clear persona drift. This is a real, documented gap.

    Implication: the only `runtime_mode_state` block protection
    depends on the upstream M19.x LLM self-audit having run and
    produced a `drifted_*` outcome. The fast_chat path may skip
    the audit (latency optimization), in which case a persona
    drift in the draft reply would still be sent. This is
    intentional — blocking on a missing audit would be too eager.
    A future M20.x milestone can either (a) force the LLM audit
    to always run, or (b) add a fast_chat-aware fallback
    (e.g. compare draft persona markers against the expected_mode
    heuristically).
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_r4",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="surface_intent_chat",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    # The draft reply is in bot_system voice; the LLM surface
    # audit is absent. The gate should NOT block.
    verdict = settler.run_pre_send(
        draft_reply="在线，路由正常，待命中。",
        horizon_commitments=[commitment],
        observation_context={},  # no audit
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.decision != "block", (
        "R4 regression: pre-send gate blocked despite missing audit"
    )
    assert verdict.decision == "advisory_guidance"
    assert "runtime_mode_state_audit_absent" in verdict.reason_codes
    assert not verdict.replacement, (
        "R4: no replacement string when audit is absent"
    )


# === post-send advisory =================================================


def test_post_send_advisory_is_added_to_next_turn_guidance() -> None:
    """M20.3 §3.3: post-send returns `advisory_guidance` (no
    block) and the caller writes `verdict.guidance` to next-turn
    `control_guidance`.
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_post",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="identity_voice_match",
        observable_payload={"expected_voice": "persona_chat", "actual_voice": "bot_system"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("boundary_check",),
        engineering_proxy_label="mvp_local_identity_voice",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_voice",
            "committed_surface_intent": "bot_system",
        },
    }
    verdict = settler.run_post_send(
        committed_reply="在线。",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    assert verdict.decision == "advisory_guidance"
    assert verdict.horizon == "post_send"
    assert not verdict.replacement  # post-send never blocks


def test_post_send_cannot_block() -> None:
    """M20.3 §3.3: post-send path is advisory only; no decision is `block`."""
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_post_block",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_voice",
            "committed_surface_intent": "bot_system",
        },
    }
    verdict = settler.run_post_send(
        committed_reply="在线。",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    # Post-send handles non-runtime owners only; runtime owner
    # already had its chance pre-send.
    assert verdict is None


# === dedup invariant ====================================================


def test_same_turn_surface_does_not_double_emit_per_turn() -> None:
    """M20.3 §3.4: a commit_id may appear in at most one verdict
    per turn. Calling run_pre_send twice with the same commitment
    MUST NOT double-emit.
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_dedup",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "consistent",
            "committed_surface_intent": "persona_chat",
        },
    }
    first = settler.run_pre_send(
        draft_reply="hi",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    second = settler.run_pre_send(
        draft_reply="hi",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert first is not None
    assert second is None  # dedup


def test_reset_turn_dedup_clears_state() -> None:
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_reset",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "consistent",
            "committed_surface_intent": "persona_chat",
        },
    }
    settler.run_pre_send(
        draft_reply="hi",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    settler.reset_turn_dedup()
    # Now the same commitment can be admitted again.
    again = settler.run_pre_send(
        draft_reply="hi",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert again is not None


# === audit event builder ================================================


def test_same_turn_surface_audit_event_emitted() -> None:
    commitment = ActiveCommitment(
        commit_id="cid_event",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": "persona_chat"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_voice",
            "committed_surface_intent": "bot_system",
        },
    }
    settler = SameTurnSurfaceSettler()
    verdict = settler.run_pre_send(
        draft_reply="在线。",
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert verdict is not None
    event = build_same_turn_surface_verdict_event(verdict)
    assert event["type"] == "SameTurnSurfaceVerdict"
    assert event["turn_index"] == 0
    assert event["horizon"] == "pre_send"
    assert event["decision"] == "block"
    assert "cid_event" in event["commit_ids"]
    assert event["engineering_proxy_label"] == "mvp_local_same_turn_surface"


# === accept flag ========================================================


def test_runtime_mode_state_owner_accepts_same_turn_block() -> None:
    assert is_registry_v2_accepts_same_turn_block("runtime_mode_state") is True


def test_v1_owner_does_not_accept_same_turn_block() -> None:
    assert is_registry_v2_accepts_same_turn_block("mismatch_memory_fast") is False
    assert is_registry_v2_accepts_same_turn_block("policy_state") is False


# === post-send does not modify committed reply ==========================


def test_same_turn_surface_does_not_modify_committed_reply() -> None:
    """M20.3 §3.3: post-send only emits `guidance`; the committed
    reply string is untouched. The settler has no mutation method.
    """
    settler = SameTurnSurfaceSettler()
    commitment = ActiveCommitment(
        commit_id="cid_pure",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="ref",
        layer="B_per_turn_commitment",
        observable="identity_voice_match",
        observable_payload={"expected_voice": "persona_chat", "actual_voice": "bot_system"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("boundary_check",),
        engineering_proxy_label="mvp_local_identity_voice",
        horizon="same_turn_surface",
    )
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            "surface_intent_outcome": "drifted_voice",
            "committed_surface_intent": "bot_system",
        },
    }
    committed = "原始回复文本不应被修改。"
    verdict = settler.run_post_send(
        committed_reply=committed,
        horizon_commitments=[commitment],
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    # Settler has no mutation hook; the committed_reply is just a
    # parameter that the caller has already committed.
    assert verdict is not None
    # The verdict's `replacement` field is empty for post-send.
    assert not verdict.replacement
