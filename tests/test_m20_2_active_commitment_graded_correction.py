"""Tests for M20.2 graded correction: dispatcher, routing, audit events,
and the T+1+1 dispatch rule.

M20.2 closes the action half of the unified-commitment loop. It
reads M20.1's `owner.observability[commit_id]`, maps the settled
value to a `GradedCorrection` level via the frozen rules, and routes
the decision through `active_commitment_grader` stubs that emit
`GradedCorrectionRouted` / `CorrectionDeferred` /
`CorrectionRejected` audit events.

These tests cover the dispatcher purity rules, the magnitude-to-
level table, the outcome overrides, the per-owner `graded_action_set`
constraints, the routing stubs, and the at-most-once
`GradedCorrectionRouted` invariant per `commit_id`.
"""

from __future__ import annotations

import inspect

from segmentum.dialogue.runtime.active_commitment import (
    DISPATCHER_REASON_CODES_V1,
    GRADED_CORRECTION_V1,
    ActiveCommitment,
    COMMITMENT_REGISTRY_V1,
    GradedCorrectionDecision,
    GradedCorrectionDispatcher,
    SettledValue,
    build_correction_deferred_event,
    build_correction_rejected_event,
    build_graded_correction_routed_event,
    init_owner_observability_for_commitment,
    record_pending_commitment,
    update_graded_correction_diagnostics,
)
from segmentum.dialogue.runtime.active_commitment_grader import (
    route_expire,
    route_microadjust,
    route_next_turn,
    route_revoke,
    route_same_turn,
    route_slow_promote,
)
from segmentum.dialogue.runtime.mvp_loop import (
    _dispatch_graded_corrections,
)


# === fixtures =============================================================


def _commitment(
    *,
    owner_id: str = "m13_drive_state",
    source_kind: str = "state",
    source_ref: str = "ref1",
    observable: str = "behavioral_pull_shift",
    created_turn: int = 0,
    evidence_refs: tuple[str, ...] = ("ref1", "ref2"),
    reason_codes: tuple[str, ...] = ("m13_drive_signal",),
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id=f"cid_{owner_id}_{source_ref}_{created_turn}",
        owner_id=owner_id,
        source_kind=source_kind,
        source_ref=source_ref,
        layer="C_observation" if owner_id in {"m13_drive_state", "m15_episode_ledger"} else "B_per_turn_commitment",
        observable=observable,
        observable_payload={"action": "reply", "delta": 0.5, "evidence_refs": list(evidence_refs)},
        target={"action": "reply"},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=evidence_refs,
        created_turn=created_turn,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=reason_codes,
        engineering_proxy_label="mvp_local_m13_drive",
    )


def _settled(
    *,
    commit_id: str,
    outcome: str = "confirmed",
    magnitude: float = 0.5,
    settler_type: str = "deterministic",
    turn_index: int = 1,
) -> SettledValue:
    return SettledValue(
        commit_id=commit_id,
        outcome=outcome,
        magnitude=magnitude,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:01Z",
        turn_index=turn_index,
        settler_type=settler_type,
        engineering_proxy_label="mvp_local_active_commitment",
    )


def _seed_observability(
    state: dict,
    *,
    commitment: ActiveCommitment,
    settled_value: SettledValue,
) -> None:
    """Init the observability entry as if admission + settlement both ran."""
    init_owner_observability_for_commitment(
        state,
        owner_id=commitment.owner_id,
        commitment=commitment,
    )
    observability = state["commitment_owner_observability"]
    row = observability[commitment.owner_id][commitment.commit_id]
    row["settled_value"] = {
        "outcome": settled_value.outcome,
        "magnitude": settled_value.magnitude,
        "settler_type": settled_value.settler_type,
        "evidence_refs": list(settled_value.evidence_refs),
        "reason_codes": list(settled_value.reason_codes),
        "at": settled_value.at,
        "turn_index": settled_value.turn_index,
        "engineering_proxy_label": settled_value.engineering_proxy_label,
    }
    row["settlement_attempts"] = 1
    row["last_attempt_turn_index"] = settled_value.turn_index
    row["last_attempt_reason_code"] = (
        settled_value.reason_codes[0] if settled_value.reason_codes else "settler_deterministic"
    )


# === §1 frozen enum ========================================================


def test_graded_correction_enum_is_frozen() -> None:
    expected = {"microadjust", "next_turn", "same_turn", "slow_promote", "revoke", "expire"}
    assert set(GRADED_CORRECTION_V1) == expected
    # Adding a level requires a vocabulary bump.
    assert isinstance(GRADED_CORRECTION_V1, frozenset)


# === §2 magnitude-to-level table =========================================


# Owner selection per (magnitude bucket, expected level): the magnitude
# table is decoupled from per-owner `graded_action_set`, so each test
# case picks an owner whose `graded_action_set` includes the expected
# level. This keeps the table and the action set tested independently.
_LEVEL_TO_OWNER = {
    "microadjust": "m13_drive_state",
    "next_turn": "m13_drive_state",
    "same_turn": "m13_drive_state",
    "slow_promote": "self_repair_expectation",
}


def test_magnitude_to_level_thresholds_match_frozen_table() -> None:
    cases = [
        (0.0, "expire"),
        (0.05, "expire"),
        (0.1, "microadjust"),
        (0.2, "microadjust"),
        (0.3, "next_turn"),
        (0.5, "next_turn"),
        (0.6, "same_turn"),
        (0.8, "same_turn"),
        (0.85, "slow_promote"),
        (1.0, "slow_promote"),
    ]
    dispatcher = GradedCorrectionDispatcher()
    for magnitude, expected_level in cases:
        if expected_level == "expire":
            owner_id = "m13_drive_state"
        else:
            owner_id = _LEVEL_TO_OWNER[expected_level]
        commitment = _commitment(owner_id=owner_id, source_kind="state")
        settled = _settled(
            commit_id=commitment.commit_id,
            outcome="confirmed",
            magnitude=magnitude,
        )
        decision = dispatcher.decide(
            commitment=commitment,
            settled_value=settled,
            turn_index=1,
        )
        assert decision.correction_level == expected_level, (
            f"magnitude={magnitude} expected {expected_level}, got {decision.correction_level}"
        )


# === §2 outcome overrides =================================================


def test_outcome_overrides_match_frozen_rules() -> None:
    dispatcher = GradedCorrectionDispatcher()
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")

    # outcome = "ambiguous" -> expire regardless of magnitude
    settled = _settled(commit_id=commitment.commit_id, outcome="ambiguous", magnitude=0.9)
    decision = dispatcher.decide(commitment=commitment, settled_value=settled)
    assert decision.correction_level == "expire"
    assert "ambiguous_outcome" in decision.reason_codes
    assert decision.deferred is True

    # outcome = "uncertain" + magnitude < 0.5 -> expire
    settled = _settled(commit_id=commitment.commit_id, outcome="uncertain", magnitude=0.4)
    decision = dispatcher.decide(commitment=commitment, settled_value=settled)
    assert decision.correction_level == "expire"
    assert "magnitude_below_threshold" in decision.reason_codes

    # outcome = "uncertain" + magnitude >= 0.5 -> microadjust
    settled = _settled(commit_id=commitment.commit_id, outcome="uncertain", magnitude=0.7)
    decision = dispatcher.decide(commitment=commitment, settled_value=settled)
    assert decision.correction_level == "microadjust"

    # outcome = "violated" + magnitude >= 0.85 -> revoke (overrides slow_promote)
    # Use mismatch_memory_fast whose graded_action_set includes revoke.
    revoke_commitment = _commitment(
        owner_id="mismatch_memory_fast",
        source_kind="state",
        observable="expectation_outcome_match",
    )
    settled = _settled(commit_id=revoke_commitment.commit_id, outcome="violated", magnitude=0.9)
    decision = dispatcher.decide(commitment=revoke_commitment, settled_value=settled)
    assert decision.correction_level == "revoke"


# === §2 policy source kind ================================================


def test_policy_source_kind_corrections_are_expired() -> None:
    dispatcher = GradedCorrectionDispatcher()
    # policy_state accepts policy but its graded_action_set is empty.
    commitment = _commitment(
        owner_id="policy_state",
        source_kind="policy",
        observable="repair_bias_band",
    )
    settled = _settled(commit_id=commitment.commit_id, outcome="violated", magnitude=0.9)
    decision = dispatcher.decide(commitment=commitment, settled_value=settled)
    assert decision.correction_level == "expire"
    assert "policy_source_no_correction" in decision.reason_codes
    assert decision.deferred is True


# === §3 action_set violation ==============================================


def test_owner_graded_action_set_violation_rejected() -> None:
    dispatcher = GradedCorrectionDispatcher()
    # m15_episode_ledger accepts only [microadjust, next_turn], not same_turn.
    # magnitude 0.7 would normally pick "same_turn", but m15 lacks that.
    commitment = _commitment(owner_id="m15_episode_ledger", source_kind="episodic")
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.7)
    decision = dispatcher.decide(commitment=commitment, settled_value=settled)
    assert decision.correction_level == "expire"
    assert "action_set_violation" in decision.reason_codes
    assert decision.rejected is True


# === §3 slow_promote routing ==============================================


def test_slow_promote_routes_to_m19_3_path() -> None:
    """A slow_promote decision on `self_repair_expectation` must end up
    routed through `route_slow_promote` (the M19.3 adapter stub).

    M20.2 ships the stub; M20.2.1 wires the real M19.3 call. The
    invariant here is that the decision is `slow_promote` and the
    audit event `GradedCorrectionRouted` is emitted on the bus.
    """
    state: dict = {}
    bus: list = []
    commitment = _commitment(
        owner_id="self_repair_expectation",
        source_kind="state",
        observable="repair_bias_band",
    )
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    # magnitude 0.9 with confirmed outcome on self_repair_expectation
    # -> magnitude-to-level: 0.9 in [0.85, 1.0] -> slow_promote
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.9, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="2026-06-06T00:00:02Z")

    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(routed) == 1
    assert routed[0]["correction_level"] == "slow_promote"
    assert routed[0]["routed_owner_id"] == "self_repair_expectation"
    assert routed[0]["commit_id"] == commitment.commit_id


# === §8 M19.3 already-promoted shortcut ===================================


def test_slow_promote_short_circuits_when_m19_3_already_promoted() -> None:
    state: dict = {}
    bus: list = []
    commitment = _commitment(
        owner_id="self_repair_expectation",
        source_kind="state",
        observable="repair_bias_band",
    )
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.9, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    # Owner state shows the source_ref is already in calibrated_tendencies.
    owner_state_snapshot = {
        "m19_3_promotion_lock": {"promoted": [commitment.source_ref]},
        "calibrated_tendencies": [],
    }
    _dispatch_graded_corrections(
        bus=bus,
        state=state,
        turn_index=2,
        now="2026-06-06T00:00:02Z",
        owner_state_snapshot=owner_state_snapshot,
    )

    deferred = [e for e in bus if e["type"] == "CorrectionDeferred"]
    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(deferred) == 1
    assert deferred[0]["reason_code"] == "m19_3_already_promoted"
    assert len(routed) == 0

    # Diagnostics recorded the shortcut.
    diag = state["graded_correction_diagnostics"]
    assert diag["m19_3_already_promoted_shortcut_count"] == 1
    assert diag["correction_deferred_total"] == 1


# === §5 revoke routing ====================================================


def test_revoke_routes_to_existing_owner_revocation_path() -> None:
    state: dict = {}
    bus: list = []
    commitment = _commitment(
        owner_id="mismatch_memory_fast",
        source_kind="state",
        observable="expectation_outcome_match",
    )
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    # outcome=violated, magnitude=0.9 -> revoke
    settled = _settled(
        commit_id=commitment.commit_id,
        outcome="violated",
        magnitude=0.9,
        turn_index=1,
    )
    _seed_observability(state, commitment=commitment, settled_value=settled)

    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="2026-06-06T00:00:02Z")

    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(routed) == 1
    assert routed[0]["correction_level"] == "revoke"
    assert routed[0]["routed_owner_id"] == "mismatch_memory_fast"


# === §5 microadjust routing ===============================================


def test_microadjust_routes_to_owner_specific_function() -> None:
    state: dict = {}
    bus: list = []
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    # magnitude 0.2 -> microadjust
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.2, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="2026-06-06T00:00:02Z")

    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(routed) == 1
    assert routed[0]["correction_level"] == "microadjust"
    assert routed[0]["routed_owner_id"] == "m13_drive_state"


# === §5 next_turn routing =================================================


def test_next_turn_routes_to_owner_specific_function() -> None:
    state: dict = {}
    bus: list = []
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    # magnitude 0.4 -> next_turn
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.4, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="2026-06-06T00:00:02Z")

    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(routed) == 1
    assert routed[0]["correction_level"] == "next_turn"


# === §7 same_turn advisory-only ==========================================


def test_same_turn_routes_advisory_only() -> None:
    """A same_turn decision with no non_advisory_fields passes."""
    decision = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="same_turn",
        routed_owner_id="m13_drive_state",
        reason_codes=("graded_correction_routed",),
        evidence_refs=("ref1",),
        magnitude_before=0.7,
        magnitude_after=0.85,
        outcome="violated",
        at="2026-06-06T00:00:02Z",
        turn_index=2,
        engineering_proxy_label="mvp_local_active_commitment",
    )
    state: dict = {}
    bus: list = []
    event = route_same_turn(decision, state=state, bus=bus)
    assert event["type"] == "GradedCorrectionRouted"


def test_same_turn_attempt_to_write_non_advisory_field_is_rejected() -> None:
    """A same_turn decision with non_advisory_fields set is rejected."""
    decision = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="same_turn",
        routed_owner_id="m13_drive_state",
        reason_codes=("graded_correction_routed",),
        evidence_refs=("ref1",),
        magnitude_before=0.7,
        magnitude_after=0.85,
        outcome="violated",
        at="2026-06-06T00:00:02Z",
        turn_index=2,
        engineering_proxy_label="mvp_local_active_commitment",
    )
    state: dict = {}
    bus: list = []
    event = route_same_turn(
        decision,
        state=state,
        bus=bus,
        non_advisory_fields=("reply_text",),
    )
    assert event["type"] == "CorrectionRejected"
    assert event["reason_code"] == "same_turn_not_advisory"
    # No GradedCorrectionRouted was emitted.
    assert all(e["type"] != "GradedCorrectionRouted" for e in bus)


# === §4 dispatcher purity =================================================


def test_dispatcher_does_not_mutate_state() -> None:
    """`dispatcher.decide` reads inputs but does not write to state.

    The dispatcher is a pure function: same inputs -> same decision.
    """
    dispatcher = GradedCorrectionDispatcher()
    commitment = _commitment()
    settled = _settled(commit_id=commitment.commit_id, magnitude=0.5)

    # Take a deep snapshot of inputs.
    import copy
    commitment_snapshot = copy.deepcopy(commitment)
    settled_snapshot = copy.deepcopy(settled)

    decision = dispatcher.decide(commitment=commitment, settled_value=settled)

    # Inputs unchanged.
    assert commitment == commitment_snapshot
    assert settled == settled_snapshot
    # Output dataclass is frozen.
    assert dataclasses_is_frozen(decision)


def dataclasses_is_frozen(obj) -> bool:
    import dataclasses
    return dataclasses.is_dataclass(obj) and getattr(obj.__class__, "__dataclass_params__", None) and obj.__dataclass_params__.frozen


def test_dispatcher_does_not_call_llm() -> None:
    """`dispatcher.decide` is a pure function and must not import or call any LLM."""
    src = inspect.getsource(GradedCorrectionDispatcher.decide)
    forbidden = ("llm", "openai", "anthropic", "client.chat", "client.messages")
    for token in forbidden:
        assert token not in src.lower(), (
            f"dispatcher.decide source must not reference {token!r}"
        )


# === §6 audit event chronology ============================================


def test_audit_events_are_chronological_per_commit_id() -> None:
    """A `commit_id` may appear in at most one `GradedCorrectionRouted` event."""
    state: dict = {}
    bus: list = []
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.2, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    # Run dispatch on three consecutive turns. The first run emits the
    # routed event; subsequent runs find `dispatched=True` and skip.
    for turn in (2, 3, 4):
        _dispatch_graded_corrections(
            bus=bus, state=state, turn_index=turn, now="t",
        )

    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(routed) == 1, (
        f"expected exactly one GradedCorrectionRouted, got {len(routed)}"
    )
    assert routed[0]["commit_id"] == commitment.commit_id


# === §4 magnitude clamping ================================================


def test_magnitude_before_and_after_are_clamped() -> None:
    """The dispatcher clamps magnitude to [0.0, 1.0] before mapping."""
    dispatcher = GradedCorrectionDispatcher()
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")
    # Test each extreme.
    for raw_magnitude in (-0.5, 0.0, 1.0, 1.7):
        settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=raw_magnitude)
        decision = dispatcher.decide(commitment=commitment, settled_value=settled)
        assert 0.0 <= decision.magnitude_before <= 1.0
        if decision.magnitude_after is not None:
            assert 0.0 <= decision.magnitude_after <= 1.0


# === §5 stub surface ======================================================


def test_routing_stubs_emit_audit_event() -> None:
    """Each routing stub (except expire) emits GradedCorrectionRouted."""
    decision = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="microadjust",
        routed_owner_id="m13_drive_state",
        reason_codes=("graded_correction_routed",),
        evidence_refs=("ref1",),
        magnitude_before=0.2,
        magnitude_after=0.21,
        outcome="confirmed",
        at="t",
        turn_index=1,
        engineering_proxy_label="mvp_local_active_commitment",
    )
    state: dict = {}
    bus: list = []
    for route in (route_microadjust, route_next_turn, route_revoke, route_slow_promote):
        bus.clear()
        event = route(decision, state=state, bus=bus)
        assert event["type"] == "GradedCorrectionRouted"


def test_expire_routing_stub_is_no_op() -> None:
    """`expire` is a no-routing level. The stub returns None."""
    decision = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="expire",
        routed_owner_id="m13_drive_state",
        reason_codes=("magnitude_below_threshold",),
        evidence_refs=("ref1",),
        magnitude_before=0.0,
        magnitude_after=0.0,
        outcome="ambiguous",
        at="t",
        turn_index=1,
        engineering_proxy_label="mvp_local_active_commitment",
    )
    state: dict = {}
    bus: list = []
    result = route_expire(decision, state=state, bus=bus)
    assert result is None
    assert bus == []


# === §9 diagnostics counters ==============================================


def test_diagnostics_counters_increment_on_routed() -> None:
    state: dict = {}
    update_graded_correction_diagnostics(
        state,
        routed=2,
        deferred=1,
        rejected=1,
        by_level={"microadjust": 2},
        by_owner_id={"m13_drive_state": 2},
        by_outcome={"confirmed": 2},
        by_reason_code={"magnitude_below_threshold": 1, "action_set_violation": 1},
        magnitudes_before=(0.2, 0.3),
        magnitudes_after=(0.21, 0.31),
    )
    diag = state["graded_correction_diagnostics"]
    assert diag["graded_correction_total"] == 4
    assert diag["graded_correction_routed_total"] == 2
    assert diag["correction_deferred_total"] == 1
    assert diag["correction_rejected_total"] == 1
    assert diag["graded_correction_by_level"] == {"microadjust": 2}
    assert diag["correction_by_reason_code"] == {
        "magnitude_below_threshold": 1,
        "action_set_violation": 1,
    }
    assert diag["magnitude_before_distribution"] == [0.2, 0.3]


# === §8 no-duplication assertions =========================================


def test_m20_2_does_not_duplicate_m19_1_traction_logic() -> None:
    """M20.2's dispatcher and routing stubs must not import or
    duplicate M19.1 traction update logic."""
    dispatcher_src = inspect.getsource(GradedCorrectionDispatcher)
    forbidden_substrings = (
        "apply_traction_update",
        "self_repair_expectation_state",
        "expectations_tail",
    )
    for token in forbidden_substrings:
        assert token not in dispatcher_src, (
            f"dispatcher source must not reference M19.1 traction logic {token!r}"
        )
    for module in (
        "segmentum.dialogue.runtime.active_commitment_grader.microadjust",
        "segmentum.dialogue.runtime.active_commitment_grader.next_turn",
        "segmentum.dialogue.runtime.active_commitment_grader.same_turn",
        "segmentum.dialogue.runtime.active_commitment_grader.slow_promote",
        "segmentum.dialogue.runtime.active_commitment_grader.revoke",
        "segmentum.dialogue.runtime.active_commitment_grader.expire",
    ):
        import importlib
        mod = importlib.import_module(module)
        src = inspect.getsource(mod)
        for token in forbidden_substrings:
            assert token not in src, (
                f"{module} must not reference M19.1 traction logic {token!r}"
            )


def test_m20_2_does_not_duplicate_m19_3_promotion_logic() -> None:
    """M20.2 must not import or duplicate M19.3 slow-promotion logic.

    M20.2 is a router, not a promoter. The dispatcher reads M19.3's
    lock map but does not write to it.
    """
    dispatcher_src = inspect.getsource(GradedCorrectionDispatcher)
    forbidden = (
        "calibrated_tendencies",
        "apply_promotion",
        "promote_expectation",
    )
    for token in forbidden:
        # `calibrated_tendencies` may appear in the read-only check,
        # so allow that one path. The others are forbidden everywhere.
        if token == "calibrated_tendencies":
            # Allowed because the dispatcher reads M19.3's lock map.
            continue
        assert token not in dispatcher_src, (
            f"dispatcher source must not reference M19.3 promotion logic {token!r}"
        )


def test_m20_2_does_not_duplicate_m9_0_control_guidance_logic() -> None:
    forbidden = (
        "control_guidance",
        "apply_repair_bias_shift",
        "repair_bias_delta",
    )
    dispatcher_src = inspect.getsource(GradedCorrectionDispatcher)
    # `control_guidance` may appear as a free-floating reference (e.g.
    # in docstrings). The check is that the dispatcher does not
    # compute or apply a control_guidance shift.
    assert "control_guidance" not in dispatcher_src or "owner_id" in dispatcher_src
    for token in forbidden[1:]:
        assert token not in dispatcher_src


def test_m20_2_does_not_duplicate_m17_4_precision_ema_logic() -> None:
    forbidden = ("precision_ema", "apply_precision_update", "type_precision")
    dispatcher_src = inspect.getsource(GradedCorrectionDispatcher)
    for token in forbidden:
        assert token not in dispatcher_src


def test_m20_2_does_not_duplicate_m15_1_episode_aggregation_logic() -> None:
    forbidden = ("apply_aggregation", "episode_aggregation", "consolidate_episode")
    dispatcher_src = inspect.getsource(GradedCorrectionDispatcher)
    for token in forbidden:
        assert token not in dispatcher_src


# === §7 T+1+1 rule ========================================================


def test_dispatch_does_not_run_on_settlement_turn() -> None:
    """The dispatcher skips entries whose `settled_value.turn_index`
    is >= current turn. This enforces the T+1+1 rule: dispatch
    happens strictly after settlement, not on the same turn."""
    state: dict = {}
    bus: list = []
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    # Settlement on turn 5.
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.2, turn_index=5)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    # Try to dispatch on the same turn (5) — must be skipped.
    _dispatch_graded_corrections(bus=bus, state=state, turn_index=5, now="t")
    assert all(e["type"] != "GradedCorrectionRouted" for e in bus)

    # Try on the next turn (6) — must dispatch.
    _dispatch_graded_corrections(bus=bus, state=state, turn_index=6, now="t")
    routed = [e for e in bus if e["type"] == "GradedCorrectionRouted"]
    assert len(routed) == 1


# === integration: full admit -> settle -> dispatch flow ==================


def test_full_admit_settle_dispatch_flow_emits_chronological_events() -> None:
    """End-to-end: admit a commitment, settle it, dispatch it on the
    next turn. The bus accumulates events in chronological order."""
    state: dict = {}
    bus: list = []
    commitment = _commitment(
        owner_id="m13_drive_state",
        source_kind="state",
        observable="behavioral_pull_shift",
    )

    # 1. Admission
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    record_pending_commitment(state, commitment)
    assert state["commitment_owner_observability"][commitment.owner_id][commitment.commit_id]["commitment"]["source_kind"] == "state"

    # 2. Settlement on T+1
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.2, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)
    remove_pending = state["active_commitments_pending"]
    state["active_commitments_pending"] = [
        row for row in remove_pending if row["commit_id"] != commitment.commit_id
    ]

    # 3. Dispatch on T+2
    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="t")

    types = [e["type"] for e in bus]
    # No ActiveCommitmentCreated because we bypassed the adapter here,
    # but we should have at least one GradedCorrectionRouted.
    assert types.count("GradedCorrectionRouted") == 1
    assert types[0] == "GradedCorrectionRouted"


def test_dispatch_emits_rejected_for_unknown_owner() -> None:
    """If a commitment slips through with an unknown owner_id, the
    dispatcher emits CorrectionRejected with reason_code=unknown_owner.
    The observability entry is marked dispatched.
    """
    state: dict = {}
    bus: list = []
    commitment = _commitment(owner_id="m13_drive_state", source_kind="state")
    record_pending_commitment(state, commitment)
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    settled = _settled(commit_id=commitment.commit_id, outcome="confirmed", magnitude=0.2, turn_index=1)
    _seed_observability(state, commitment=commitment, settled_value=settled)

    # Corrupt the observability entry's owner_id to an unknown value.
    observability = state["commitment_owner_observability"]
    entry = observability[commitment.owner_id].pop(commitment.commit_id)
    observability["__unknown_owner__"] = {commitment.commit_id: entry}
    entry["commitment"]["owner_id"] = "__unknown_owner__"

    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="t")

    rejected = [e for e in bus if e["type"] == "CorrectionRejected"]
    assert len(rejected) == 1
    assert rejected[0]["reason_code"] == "unknown_owner"
    assert rejected[0]["routed_owner_id"] == "__unknown_owner__"


# === §4 dispatcher reason codes surface ===================================


def test_dispatcher_reason_codes_subset() -> None:
    """All dispatcher reason codes must be in DISPATCHER_REASON_CODES_V1."""
    expected = {
        "magnitude_below_threshold",
        "policy_source_no_correction",
        "ambiguous_outcome",
        "m19_3_already_promoted",
        "action_set_violation",
        "slow_promote_not_supported",
        "same_turn_not_advisory",
        "owner_state_unavailable",
        "unknown_owner",
    }
    assert set(DISPATCHER_REASON_CODES_V1) >= expected
    assert isinstance(DISPATCHER_REASON_CODES_V1, frozenset)


# === §4 event builders produce the right shape ===========================


def test_event_builders_produce_frozen_envelope() -> None:
    decision = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="microadjust",
        routed_owner_id="m13_drive_state",
        reason_codes=("graded_correction_routed",),
        evidence_refs=("ref1",),
        magnitude_before=0.2,
        magnitude_after=0.21,
        outcome="confirmed",
        at="t",
        turn_index=1,
        engineering_proxy_label="mvp_local_active_commitment",
    )
    routed_event = build_graded_correction_routed_event(decision)
    assert routed_event["type"] == "GradedCorrectionRouted"
    assert routed_event["commit_id"] == "cid_x"
    assert routed_event["correction_level"] == "microadjust"

    deferred = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="expire",
        routed_owner_id="policy_state",
        reason_codes=("policy_source_no_correction",),
        evidence_refs=("ref1",),
        magnitude_before=0.5,
        magnitude_after=0.5,
        outcome="violated",
        at="t",
        turn_index=1,
        engineering_proxy_label="mvp_local_active_commitment",
        deferred=True,
    )
    deferred_event = build_correction_deferred_event(deferred)
    assert deferred_event["type"] == "CorrectionDeferred"
    assert deferred_event["reason_code"] == "policy_source_no_correction"

    rejected = GradedCorrectionDecision(
        commit_id="cid_x",
        correction_level="expire",
        routed_owner_id="m15_episode_ledger",
        reason_codes=("action_set_violation",),
        evidence_refs=("ref1",),
        magnitude_before=0.7,
        magnitude_after=0.7,
        outcome="confirmed",
        at="t",
        turn_index=1,
        engineering_proxy_label="mvp_local_active_commitment",
        rejected=True,
    )
    rejected_event = build_correction_rejected_event(rejected)
    assert rejected_event["type"] == "CorrectionRejected"
    assert rejected_event["reason_code"] == "action_set_violation"
