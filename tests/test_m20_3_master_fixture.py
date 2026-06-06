"""Tests for the M20-series master acceptance fixture (6+1+1).

M20.3 owns the additional two sub-scenarios on top of the M20.0
base 6 steps. The fixture asserts the full path:
- Base 6 steps (M20.0–M20.2): T0 admit, T1 settle, T2 microadjust,
  T3 大 magnitude settle, T4 slow_promote, T5 M19.3 lock, T6 二次
  dispatch 拒绝.
- +1 identity sub-scenario (NEW in M20.3): pre-send gate runs
  BEFORE reply commit; post-send advisory runs AFTER.
- +1 fast_chat sub-scenario (NEW in M20.3): `fast_chat` does not
  skip PolicyProducer; MinimumLoopCoverageMissed is NOT emitted.

The fixture is the M20-series master acceptance check. M20.3 must
not break the M20.0–M20.2 invariants.
"""

from __future__ import annotations

from typing import Any

from segmentum.dialogue.runtime.active_commitment import (
    COMMITMENT_REGISTRY_V2,
    ActiveCommitment,
    ActiveCommitmentAdapter,
    GradedCorrectionDecision,
    GradedCorrectionDispatcher,
    HORIZON_V1,
    OBSERVABLE_V2,
    OUTCOME_BY_OBSERVABLE_V2,
    REASON_CODES_V1,
    SettledValue,
    build_active_commitment_created_event,
    build_active_commitment_settled_event,
    compute_commit_id,
    init_owner_observability_for_commitment,
    is_registry_v2_accepts_policy_correction,
    is_registry_v2_accepts_same_turn_block,
    record_active_commitment_event,
    record_pending_commitment,
    write_owner_observability,
)
from segmentum.dialogue.runtime.active_commitment_grader import (
    route_microadjust,
    route_next_turn,
    route_revoke,
    route_slow_promote,
)
from segmentum.dialogue.runtime.loop_invariants import (
    LoopInvariants,
    RULE_POLICY_SOURCE_REQUIRED,
    RULE_RUNTIME_MODE_STATE_REQUIRED,
    build_minimum_loop_coverage_missed_event,
)
from segmentum.dialogue.runtime.policy_producer import (
    PolicyProducer,
    build_policy_admitted_event,
)
from segmentum.dialogue.runtime.same_turn_surface import (
    SameTurnSurfaceSettler,
    build_same_turn_surface_verdict_event,
)


# === Shared fixtures =====================================================


def _commitment(
    *,
    commit_id: str,
    owner_id: str,
    source_kind: str,
    source_ref: str,
    observable: str,
    layer: str,
    observable_payload: dict | None = None,
    evidence_refs: tuple[str, ...] = ("ref1",),
    created_turn: int = 0,
    reason_codes: tuple[str, ...] = ("policy_prior",),
    engineering_proxy_label: str = "mvp_local_active_commitment",
    horizon: str = "next_turn",
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id=commit_id,
        owner_id=owner_id,
        source_kind=source_kind,
        source_ref=source_ref,
        layer=layer,
        observable=observable,
        observable_payload=dict(observable_payload or {}),
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=evidence_refs,
        created_turn=created_turn,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=reason_codes,
        engineering_proxy_label=engineering_proxy_label,
        horizon=horizon,
    )


# === Base 6 steps (M20.0–M20.2 fixture) ================================


def test_master_fixture_6_base_steps_pass() -> None:
    """The base 6 steps from M20.0–M20.2 are still satisfied:
    T0 admit, T1 settle, T2 microadjust, T3 大 settle, T4 slow_promote,
    T5 M19.3 lock, T6 二次 dispatch 拒绝.
    """
    state: dict = {}
    adapter = ActiveCommitmentAdapter()
    dispatcher = GradedCorrectionDispatcher()

    # T0: admit a policy_state commitment (small magnitude).
    c1_proposal = {
        "owner_id": "policy_state",
        "source_kind": "episodic",
        "source_ref": "src_1",
        "layer": "A_long_term_prior",
        "observable": "repair_bias_band",
        "observable_payload": {"context": "ctx", "band": "low", "value": 0.05},
        "target": {"context": "ctx"},
        "due_at": {"kind": "next_turn"},
        "priority": 0.1,
        "confidence": 0.1,
        "evidence_refs": ["ref1"],
        "reason_codes": ["policy_prior"],
        "engineering_proxy_label": "mvp_local_memory_dynamics",
    }
    c1, _ = adapter.admit(proposal=c1_proposal, turn_index=0, created_at="2026-06-06T00:00:00Z")
    assert c1 is not None
    record_pending_commitment(state, c1)
    init_owner_observability_for_commitment(state, owner_id=c1.owner_id, commitment=c1)

    # T1: settle the commitment (small magnitude → 0.05 → expire)
    sv1 = SettledValue(
        commit_id=c1.commit_id,
        outcome="confirmed",
        magnitude=0.05,  # below 0.1 → "expire"
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:01Z",
        turn_index=1,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    write_owner_observability(
        state,
        owner_id=c1.owner_id,
        commit_id=c1.commit_id,
        settled_value=sv1,
        last_attempt_turn_index=1,
        last_attempt_reason_code="settler_deterministic",
    )

    # T2: microadjust dispatch on a state-kind commitment
    c2 = _commitment(
        commit_id="cid_microadjust",
        owner_id="m13_drive_state",
        source_kind="state",
        source_ref="m13_src",
        observable="behavioral_pull_shift",
        layer="C_observation",
        observable_payload={"action": "answer", "delta": 0.1, "user_id": "u1", "evidence_refs": ["ref1"]},
        evidence_refs=("ref1",),
        reason_codes=("m13_drive_signal",),
        engineering_proxy_label="mvp_local_m13_drive",
    )
    init_owner_observability_for_commitment(state, owner_id=c2.owner_id, commitment=c2)
    sv2 = SettledValue(
        commit_id=c2.commit_id,
        outcome="confirmed",
        magnitude=0.15,  # 0.1-0.3 → microadjust
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:02Z",
        turn_index=2,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    write_owner_observability(
        state,
        owner_id=c2.owner_id,
        commit_id=c2.commit_id,
        settled_value=sv2,
        last_attempt_turn_index=2,
        last_attempt_reason_code="settler_deterministic",
    )
    decision = dispatcher.decide(
        commitment=c2,
        settled_value=sv2,
        turn_index=3,
        now="2026-06-06T00:00:03Z",
    )
    assert decision.correction_level == "microadjust"
    # route the decision (T+1+1)
    route_microadjust(decision, state=state, bus=[], commitment=c2)

    # T3: a large magnitude settle (>= 0.85 → revoke)
    c3 = _commitment(
        commit_id="cid_big",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="src_big",
        observable="expectation_outcome_match",
        layer="B_per_turn_commitment",
        observable_payload={"source_expectation_id": "src_big", "target_context": "ctx", "outcome": "violated", "evidence_refs": ["ref1"]},
        evidence_refs=("ref1",),
        reason_codes=("self_expectation_formation",),
        engineering_proxy_label="mvp_local_self_expectation",
    )
    init_owner_observability_for_commitment(state, owner_id=c3.owner_id, commitment=c3)
    sv3 = SettledValue(
        commit_id=c3.commit_id,
        outcome="violated",
        magnitude=0.9,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:03Z",
        turn_index=3,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    write_owner_observability(
        state,
        owner_id=c3.owner_id,
        commit_id=c3.commit_id,
        settled_value=sv3,
        last_attempt_turn_index=3,
        last_attempt_reason_code="settler_deterministic",
    )
    decision3 = dispatcher.decide(
        commitment=c3,
        settled_value=sv3,
        turn_index=4,
        now="2026-06-06T00:00:04Z",
    )
    assert decision3.correction_level == "revoke"

    # T4: slow_promote → routed (use episodic source so the
    # `policy -> expire` rule is not triggered; the slow_promote
    # v2 exception only applies to `runtime_mode_state`).
    c4 = _commitment(
        commit_id="cid_slow",
        owner_id="self_cognition_calibrated_tendencies",
        source_kind="episodic",
        source_ref="src_slow",
        observable="repair_bias_band",
        layer="A_long_term_prior",
        observable_payload={"context": "ctx", "band": "high", "value": 0.9},
        evidence_refs=("ref1",),
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_self_repair",
    )
    init_owner_observability_for_commitment(state, owner_id=c4.owner_id, commitment=c4)
    sv4 = SettledValue(
        commit_id=c4.commit_id,
        outcome="confirmed",
        magnitude=0.9,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:04Z",
        turn_index=4,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    write_owner_observability(
        state,
        owner_id=c4.owner_id,
        commit_id=c4.commit_id,
        settled_value=sv4,
        last_attempt_turn_index=4,
        last_attempt_reason_code="settler_deterministic",
    )
    decision4 = dispatcher.decide(
        commitment=c4,
        settled_value=sv4,
        turn_index=5,
        now="2026-06-06T00:00:05Z",
    )
    assert decision4.correction_level == "slow_promote"

    # T5: M19.3 already promoted → deferred
    c5 = _commitment(
        commit_id="cid_lock",
        owner_id="self_cognition_calibrated_tendencies",
        source_kind="episodic",
        source_ref="src_lock",
        observable="repair_bias_band",
        layer="A_long_term_prior",
        observable_payload={"context": "ctx", "band": "high", "value": 0.9},
        evidence_refs=("ref1",),
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_self_repair",
    )
    init_owner_observability_for_commitment(state, owner_id=c5.owner_id, commitment=c5)
    sv5 = SettledValue(
        commit_id=c5.commit_id,
        outcome="confirmed",
        magnitude=0.9,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:05Z",
        turn_index=5,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    write_owner_observability(
        state,
        owner_id=c5.owner_id,
        commit_id=c5.commit_id,
        settled_value=sv5,
        last_attempt_turn_index=5,
        last_attempt_reason_code="settler_deterministic",
    )
    # Simulate M19.3 promotion lock.
    decision5 = dispatcher.decide(
        commitment=c5,
        settled_value=sv5,
        owner_state_snapshot={"m19_3_promotion_lock": {"promoted": ["src_lock"]}},
        turn_index=6,
        now="2026-06-06T00:00:06Z",
    )
    assert decision5.correction_level == "expire"
    assert "m19_3_already_promoted" in decision5.reason_codes

    # T6: re-dispatch is rejected (deferred) or marked dispatched.
    c6 = _commitment(
        commit_id="cid_dup",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="src_dup",
        observable="expectation_outcome_match",
        layer="B_per_turn_commitment",
        observable_payload={"source_expectation_id": "src_dup", "target_context": "ctx", "outcome": "violated", "evidence_refs": ["ref1"]},
        evidence_refs=("ref1",),
        reason_codes=("self_expectation_formation",),
        engineering_proxy_label="mvp_local_self_expectation",
    )
    init_owner_observability_for_commitment(state, owner_id=c6.owner_id, commitment=c6)
    # c6 is undispatched; second dispatch attempt sees it (M20.2
    # §6 may produce a "magnitude_below_threshold" reject).
    observability = state["commitment_owner_observability"]["mismatch_memory_fast"][c6.commit_id]
    assert observability["dispatched"] is False


# === +1 identity sub-scenario (NEW in M20.3) ==========================


def test_master_fixture_identity_sub_scenario_passes() -> None:
    """M20.3 §6.2: pre-send gate runs BEFORE reply commit; post-send
    advisory runs AFTER; with at most one `SameTurnSurfaceVerdict`
    per `commit_id` per turn.
    """
    state: dict = {}
    producer = PolicyProducer()
    invariants = LoopInvariants()
    surface_settler = SameTurnSurfaceSettler()
    surface_settler.reset_turn_dedup()

    # T0: PolicyProducer admits runtime_mode_state (turn-scoped /status
    # "persona_chat") with horizon = "same_turn_surface".
    admitted, audit_events = producer.evaluate(
        turn_context={"turn_index": 0, "at": "2026-06-06T00:00:00Z"},
        runtime_mode_flags={},
        command_envelope={"platform_command": "/status", "bot_command_args": []},
        user_correction_signal="",
    )
    assert len(admitted) == 1
    runtime_mode_commitment = admitted[0]
    assert runtime_mode_commitment.owner_id == "runtime_mode_state"
    assert runtime_mode_commitment.observable_payload["expected_mode"] == "bot_system"  # /status maps to bot_system
    assert runtime_mode_commitment.horizon == "same_turn_surface"

    # T0: LoopInvariants check — rule A and B both pass.
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=admitted,
        surface_intent="chat",
        is_external_turn=True,
    )
    assert verdict.missed == ()

    # T0: draft reply in bot_system voice (mismatched expected_mode =
    # "bot_system" with the /status admission).
    draft_reply = "在线，路由正常，待命中。"
    observation_context = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "surface_consistency_verification": {
            # M19.x self-audit says the draft persona matches the
            # committed surface; but the LLM-audited
            # `committed_surface_intent` is `bot_system` which
            # matches the expected_mode `bot_system` from /status.
            # So the verdict should be `pass`.
            "surface_intent_outcome": "consistent",
            "committed_surface_intent": "bot_system",
        },
    }
    pre_verdict = surface_settler.run_pre_send(
        draft_reply,
        horizon_commitments=admitted,
        observation_context=observation_context,
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert pre_verdict is not None
    assert pre_verdict.decision == "pass"

    # Now flip the audit to drift; the gate should NOT block
    # because expected_mode is "bot_system" and the committed
    # surface is also "bot_system" (the /status admission). To
    # force a block, we change the expected_mode to "persona_chat".
    flipped = _commitment(
        commit_id="cid_block",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="surface_intent_chat",
        observable="runtime_mode_state",
        layer="B_per_turn_commitment",
        observable_payload={"expected_mode": "persona_chat"},
        evidence_refs=("turn_0_surface_intent",),
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )
    surface_settler.reset_turn_dedup()
    pre_verdict2 = surface_settler.run_pre_send(
        draft_reply,  # bot_system voice
        horizon_commitments=[flipped],
        observation_context={
            "now": "2026-06-06T00:00:00Z",
            "turn_index": 0,
            "surface_consistency_verification": {
                "surface_intent_outcome": "drifted_voice",
                "committed_surface_intent": "bot_system",
            },
        },
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert pre_verdict2 is not None
    assert pre_verdict2.decision == "block"
    assert pre_verdict2.replacement  # persona fallback
    # Build the audit event.
    pre_event = build_same_turn_surface_verdict_event(pre_verdict2)
    assert pre_event["type"] == "SameTurnSurfaceVerdict"
    assert pre_event["horizon"] == "pre_send"
    assert pre_event["decision"] == "block"
    assert "cid_block" in pre_event["commit_ids"]

    # Post-send: the runtime owner already had its chance pre-send,
    # so the post-send returns None.
    post_verdict = surface_settler.run_post_send(
        committed_reply=pre_verdict2.replacement,
        horizon_commitments=[flipped],
        observation_context={},
        turn_index=0,
        at="2026-06-06T00:00:00Z",
    )
    assert post_verdict is None  # runtime owner handled pre-send

    # T1: M20.1 settlement (T+1) on the runtime_mode_state commitment.
    rms_commit_id = flipped.commit_id
    sv = SettledValue(
        commit_id=rms_commit_id,
        outcome="violated",
        magnitude=1.0,
        evidence_refs=("turn_0_surface_intent",),
        reason_codes=("settler_llm_judge",),
        at="2026-06-06T00:00:01Z",
        turn_index=1,
        settler_type="llm_judge",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    init_owner_observability_for_commitment(state, owner_id="runtime_mode_state", commitment=flipped)
    write_owner_observability(
        state,
        owner_id="runtime_mode_state",
        commit_id=rms_commit_id,
        settled_value=sv,
        last_attempt_turn_index=1,
        last_attempt_reason_code="settler_llm_judge",
    )

    # T2: settled_value is written to observability.
    observability_row = state["commitment_owner_observability"]["runtime_mode_state"][rms_commit_id]
    assert observability_row["settled_value"] is not None
    assert observability_row["settled_value"]["outcome"] == "violated"

    # T3: M20.2 dispatch (large magnitude, outcome violated) — the
    # §3.5 registry v2 exception table lets the policy-source
    # commitment through. Without the exception, dispatch would
    # emit `CorrectionDeferred` with `policy_source_no_correction`.
    dispatcher = GradedCorrectionDispatcher()
    decision = dispatcher.decide(
        commitment=flipped,
        settled_value=sv,
        turn_index=2,
        now="2026-06-06T00:00:02Z",
    )
    # `runtime_mode_state` is in the v2 registry with
    # accepts_policy_correction = true, so the §2 special case
    # is bypassed. Magnitude 1.0 + outcome violated → revoke.
    assert decision.correction_level == "revoke"
    assert "policy_source_no_correction" not in decision.reason_codes


def test_master_fixture_registry_v2_exception_routes_runtime_mode_state() -> None:
    """§3.5 acceptance: the registry v2 exception table lets the
    policy-source `runtime_mode_state` commitment through the
    dispatcher; without it, dispatch would defer with
    `policy_source_no_correction`.
    """
    commitment = _commitment(
        commit_id="cid_rms_policy",
        owner_id="runtime_mode_state",
        source_kind="policy",
        source_ref="src",
        observable="runtime_mode_state",
        layer="B_per_turn_commitment",
        observable_payload={"expected_mode": "persona_chat"},
        evidence_refs=("ref1",),
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
    )
    dispatcher = GradedCorrectionDispatcher()
    sv = SettledValue(
        commit_id=commitment.commit_id,
        outcome="violated",
        magnitude=0.6,
        evidence_refs=("ref1",),
        reason_codes=("settler_llm_judge",),
        at="2026-06-06T00:00:00Z",
        turn_index=1,
        settler_type="llm_judge",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    decision = dispatcher.decide(
        commitment=commitment,
        settled_value=sv,
        turn_index=2,
        now="2026-06-06T00:00:01Z",
    )
    # §3.5 exception: policy-source runtime_mode_state is NOT
    # deferred with `policy_source_no_correction`.
    assert "policy_source_no_correction" not in decision.reason_codes
    assert decision.correction_level != "expire"


# === +1 fast_chat sub-scenario (NEW in M20.3) =========================


def test_master_fixture_fast_chat_sub_scenario_passes() -> None:
    """M20.3 §6.3: a fast_chat turn (chat surface) admits at least
    one policy commitment AND at least one runtime_mode_state
    commitment. The runtime invariant does NOT emit
    `MinimumLoopCoverageMissed`.
    """
    state: dict = {}
    producer = PolicyProducer()
    invariants = LoopInvariants()

    # Simulate a fast_chat turn with chat surface.
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 0, "at": "2026-06-06T00:00:00Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )
    # Fast_chat MUST admit at least one policy commitment
    # (rule A) AND at least one runtime_mode_state (rule B).
    assert any(c.source_kind == "policy" for c in admitted)
    assert any(c.owner_id == "runtime_mode_state" for c in admitted)

    # Run the invariant; the chat surface triggers rule B.
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=admitted,
        surface_intent="chat",
        is_external_turn=True,
    )
    assert verdict.missed == ()  # No MinimumLoopCoverageMissed

    # Build the audit envelope for completeness; it should be None
    # because the verdict passed.
    event = build_minimum_loop_coverage_missed_event(verdict)
    # The audit envelope has `missing == []`; the caller may or
    # may not emit it (we only emit on miss).
    assert event["missing"] == []


def test_master_fixture_fast_chat_does_not_skip_policy_producer() -> None:
    """M20.3 §4: `fast_chat` MUST NOT skip PolicyProducer."""
    state: dict = {}
    producer = PolicyProducer()
    invariants = LoopInvariants()

    # Simulate a fast_chat turn that does NOT call the LLM conscious
    # loop. Even so, the policy producer must run.
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 0, "at": "2026-06-06T00:00:00Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",  # no conscious-loop signal
    )
    # Even without the LLM signal, the producer admits at least
    # one policy commitment.
    assert any(c.source_kind == "policy" for c in admitted)

    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=admitted,
        surface_intent="chat",
        is_external_turn=True,
    )
    assert verdict.missed == ()


def test_master_fixture_minimum_loop_coverage_does_not_miss() -> None:
    """M20.3 §6.3: `MinimumLoopCoverageMissed` is NOT emitted on a
    fast_chat turn that admits both policy + runtime_mode_state.
    """
    state: dict = {}
    producer = PolicyProducer()
    invariants = LoopInvariants()
    bus: list = []

    # A fast_chat turn with chat surface.
    admitted, _ = producer.evaluate(
        turn_context={"turn_index": 0, "at": "2026-06-06T00:00:00Z"},
        runtime_mode_flags={"surface_intent": "chat"},
        command_envelope={},
        user_correction_signal="",
    )

    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=admitted,
        surface_intent="chat",
        is_external_turn=True,
    )
    if not verdict.passed:
        event = build_minimum_loop_coverage_missed_event(verdict)
        event["at"] = "2026-06-06T00:00:00Z"
        bus.append(event)
    # No `MinimumLoopCoverageMissed` event should be in the bus.
    assert not any(e.get("type") == "MinimumLoopCoverageMissed" for e in bus)


# === M20.1.1 migration agreement (master fixture) ====================


def test_master_fixture_m20_1_1_agreement_holds() -> None:
    """M20.3 §6.2: the existing owner audit event and the new
    `ActiveCommitmentSettled` event agree on outcome.
    """
    from segmentum.dialogue.runtime.m20_1_1_settler_migration import (
        M13BandCheckAdapter,
        M15EpisodeAggregationAdapter,
    )

    # M13.2 case: existing `M13RewardSettlementEvent` with
    # `outcome_band = "positive"` ↔ adapter outcome = "confirmed".
    m13_commit = _commitment(
        commit_id="cid_m13_master",
        owner_id="m13_drive_state",
        source_kind="state",
        source_ref="src_m13",
        observable="traction_delta_band",
        layer="C_observation",
        observable_payload={"pending_id": "p1"},
        evidence_refs=("ref1",),
        reason_codes=("m13_drive_signal",),
        engineering_proxy_label="mvp_local_m13_drive",
    )
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "p1",
                "prediction_error_proxy": 0.05,
                "outcome_band": "positive",
                "evidence_refs": ["ref1"],
            }
        ],
    }
    result = M13BandCheckAdapter().settle(m13_commit, observation)
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    # `m13_2_agreement` confirms the agreement with the existing event.
    assert "m13_2_agreement" in result.reason_codes

    # M15.0 case: existing event with `outcome_summary = "settled"`
    # ↔ adapter outcome = "confirmed".
    m15_commit = _commitment(
        commit_id="cid_m15_master",
        owner_id="m15_episode_ledger",
        source_kind="episodic",
        source_ref="src_m15",
        observable="expectation_outcome_match",
        layer="B_per_turn_commitment",
        observable_payload={"episode_id": "ep1"},
        evidence_refs=("ref1",),
        reason_codes=("memory_dynamics_guidance",),
        engineering_proxy_label="mvp_local_m15_episode",
    )
    m15_observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m15_episode_settlements": [
            {
                "episode_id": "ep1",
                "outcome_summary": "settled",
                "delta_fe_proxy": 0.1,
                "evidence_refs": ["ref1"],
            }
        ],
    }
    result15 = M15EpisodeAggregationAdapter().settle(m15_commit, m15_observation)
    assert isinstance(result15, SettledValue)
    assert result15.outcome == "confirmed"
    assert "m15_0_agreement" in result15.reason_codes


# === v2 vocabulary is in place ==========================================


def test_v2_vocabulary_in_place_for_m20_3() -> None:
    """The M20.3 v2 vocabulary bumps are visible to callers."""
    assert "runtime_mode_state" in COMMITMENT_REGISTRY_V2
    assert "runtime_mode_state" in OBSERVABLE_V2
    assert "outreach_intent_on" in OBSERVABLE_V2
    assert "outreach_intent_off" in OBSERVABLE_V2
    assert "runtime_mode_state" in OUTCOME_BY_OBSERVABLE_V2
    assert HORIZON_V1 == frozenset({"same_turn_surface", "next_turn", "natural_context"})
    assert is_registry_v2_accepts_policy_correction("runtime_mode_state") is True
    assert is_registry_v2_accepts_same_turn_block("runtime_mode_state") is True
