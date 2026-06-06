"""Tests for M20.2.1 real write paths: m13_drive_state (3 levels) and
self_cognition_calibrated_tendencies (2 levels).

M20.2.1 v1 scope:

- `m13_drive_state` accepts `microadjust` / `next_turn` / `same_turn`
- `self_cognition_calibrated_tendencies` accepts `slow_promote` / `revoke`

The routing stubs call into `active_commitment_grader._write_paths`,
which mutates the actual owner state (a bounded traction float bump
for m13, a calibrated_tendencies row append for self_cognition) and
emits a per-owner audit event alongside the existing
`GradedCorrectionRouted` envelope.

Out-of-scope (level, owner_id) pairs must remain no-op for the write
path; the `GradedCorrectionRouted` event still fires.
"""

from __future__ import annotations

from types import MappingProxyType

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
    SettledValue,
    init_owner_observability_for_commitment,
    record_pending_commitment,
)
from segmentum.dialogue.runtime.active_commitment_grader import (
    route_microadjust,
    route_next_turn,
    route_revoke,
    route_same_turn,
    route_slow_promote,
)
from segmentum.dialogue.runtime.active_commitment_grader._write_paths import (
    apply_m13_microadjust,
    apply_m13_next_turn,
    apply_m13_same_turn,
    apply_self_cognition_revoke,
    apply_self_cognition_slow_promote,
    run_m20_2_1_write_path,
)
from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state


# === fixtures =============================================================


def _m13_commitment(
    *,
    source_ref: str = "src1",
    user_id: str = "u1",
    action: str = "answer",
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id=f"cid_{source_ref}",
        owner_id="m13_drive_state",
        source_kind="state",
        source_ref=source_ref,
        layer="C_observation",
        observable="behavioral_pull_shift",
        observable_payload={
            "action": action,
            "delta": 0.5,
            "user_id": user_id,
            "evidence_refs": ["ref1"],
        },
        target={"action": action},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref1",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m13_drive_signal",),
        engineering_proxy_label="mvp_local_m13_drive",
    )


def _self_cognition_commitment(
    *,
    source_ref: str = "mismatch_key_1",
    target_context: str = "short_casual_reply",
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id=f"cid_sc_{source_ref}",
        owner_id="self_cognition_calibrated_tendencies",
        source_kind="episodic",
        source_ref=source_ref,
        layer="A_long_term_prior",
        observable="expectation_outcome_match",
        observable_payload={
            "source_expectation_id": source_ref,
            "target_context": target_context,
            "outcome": "",
            "evidence_refs": ["ref1"],
        },
        target={"target_context": target_context},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref1",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("self_repair_bridge",),
        engineering_proxy_label="mvp_local_self_repair",
    )


def _decision(
    *,
    commit_id: str,
    owner_id: str,
    level: str,
    magnitude_before: float = 0.5,
    magnitude_after: float = 0.5,
    outcome: str = "confirmed",
    source_ref: str = "src1",
    observable_payload: dict | None = None,
    evidence_refs: tuple[str, ...] = ("ref1",),
) -> GradedCorrectionDecision:
    return GradedCorrectionDecision(
        commit_id=commit_id,
        correction_level=level,
        routed_owner_id=owner_id,
        reason_codes=("graded_correction_routed",),
        evidence_refs=evidence_refs,
        magnitude_before=magnitude_before,
        magnitude_after=magnitude_after,
        outcome=outcome,
        at="2026-06-06T00:00:01Z",
        turn_index=1,
        engineering_proxy_label="mvp_local_active_commitment",
    )


def _commitment_for_decision(decision: GradedCorrectionDecision) -> ActiveCommitment:
    """Build a minimal commitment matching the decision's metadata.

    The decision dataclass doesn't carry the commitment; the
    `_write_paths` functions read `decision.commitment` directly,
    so the test must construct a real commitment that aligns
    with the decision's commit_id / owner_id / source_ref.
    """
    return ActiveCommitment(
        commit_id=decision.commit_id,
        owner_id=decision.routed_owner_id,
        source_kind="state",
        source_ref="src1",
        layer="C_observation",
        observable="behavioral_pull_shift",
        observable_payload={
            "action": "answer",
            "delta": 0.5,
            "user_id": "u1",
            "evidence_refs": list(decision.evidence_refs),
        },
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=decision.evidence_refs,
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m13_drive_signal",),
        engineering_proxy_label="mvp_local_m13_drive",
    )


# === m13 microadjust =====================================================


def test_m13_microadjust_bumps_traction_bounded() -> None:
    state: dict = {"m13_drive_state": default_m13_drive_state()}
    bus: list = []
    commitment = _m13_commitment(source_ref="s1", user_id="alice", action="answer")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="microadjust",
        magnitude_before=0.5,
        magnitude_after=0.55,
    )

    apply_m13_microadjust(decision, commitment, state, bus)

    traction = state["m13_drive_state"]["traction_by_action"]
    assert traction["answer|alice"] == round(0.05 * 0.5, 6)  # 0.025

    micro_events = [e for e in bus if e["type"] == "M13TractionMicroadjust"]
    assert len(micro_events) == 1
    assert micro_events[0]["traction_after"] == round(0.05 * 0.5, 6)
    assert micro_events[0]["commit_id"] == commitment.commit_id


def test_m13_microadjust_caps_at_one() -> None:
    state: dict = {
        "m13_drive_state": {
            **default_m13_drive_state(),
            "traction_by_action": {"answer|alice": 0.999},
        }
    }
    bus: list = []
    commitment = _m13_commitment(user_id="alice", action="answer")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="microadjust",
        magnitude_before=1.0,
    )

    apply_m13_microadjust(decision, commitment, state, bus)

    assert state["m13_drive_state"]["traction_by_action"]["answer|alice"] == 1.0


def test_m13_microadjust_is_no_op_when_state_missing() -> None:
    state: dict = {}
    bus: list = []
    commitment = _m13_commitment()
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="microadjust",
    )
    apply_m13_microadjust(decision, commitment, state, bus)
    assert "m13_drive_state" not in state
    assert bus == []


# === m13 next_turn ========================================================


def test_m13_next_turn_bumps_traction_and_appends_pending() -> None:
    state: dict = {"m13_drive_state": default_m13_drive_state()}
    bus: list = []
    commitment = _m13_commitment(user_id="bob", action="empathize")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="next_turn",
        magnitude_before=0.4,
        magnitude_after=0.5,
    )

    apply_m13_next_turn(decision, commitment, state, bus)

    traction = state["m13_drive_state"]["traction_by_action"]
    assert traction["empathize|bob"] == round(0.10 * 0.4, 6)  # 0.04

    pending = state["m13_drive_state"]["pending_settlements"]
    assert len(pending) == 1
    assert pending[0]["commit_id"] == commitment.commit_id
    assert pending[0]["action"] == "empathize"

    events = [e for e in bus if e["type"] == "M13TractionNextTurn"]
    assert len(events) == 1
    assert events[0]["commit_id"] == commitment.commit_id


# === m13 same_turn (advisory) ============================================


def test_m13_same_turn_nudges_trace_does_not_mutate_traction() -> None:
    state: dict = {"m13_drive_state": default_m13_drive_state()}
    bus: list = []
    commitment = _m13_commitment(user_id="carol", action="clarify")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="same_turn",
        magnitude_before=0.7,
        magnitude_after=0.85,
    )

    apply_m13_same_turn(decision, commitment, state, bus)

    # Advisory: tractation MUST NOT be mutated.
    assert "traction_by_action" not in state["m13_drive_state"] or (
        state["m13_drive_state"]["traction_by_action"] == {}
    )
    # recent_action_trace gets the nudge row.
    trace = state["m13_drive_state"]["recent_action_trace"]
    assert len(trace) == 1
    assert trace[0]["kind"] == "m20_2_1_pull_nudge"
    assert trace[0]["action"] == "clarify"
    assert trace[0]["priority_boost"] == round(0.15 * 0.85, 6)

    events = [e for e in bus if e["type"] == "M13PullNudge"]
    assert len(events) == 1
    assert events[0]["advisory"] is True


# === self_cognition slow_promote ========================================


def test_self_cognition_slow_promote_appends_calibrated_tendency() -> None:
    state: dict = {
        "self_cognition": {
            "calibrated_tendencies": [],
            "repair_priors": [],
        }
    }
    bus: list = []
    commitment = _self_cognition_commitment(
        source_ref="mkey_42",
        target_context="short_casual_reply",
    )
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="slow_promote",
        magnitude_after=0.8,
        evidence_refs=("ref_x", "ref_y"),
    )

    apply_self_cognition_slow_promote(decision, commitment, state, bus)

    tendencies = state["self_cognition"]["calibrated_tendencies"]
    assert len(tendencies) == 1
    assert tendencies[0]["source_mismatch_key"] == "mkey_42"
    assert tendencies[0]["status"] == "active"
    assert tendencies[0]["confidence"] == 0.8
    assert tendencies[0]["target_context"] == "short_casual_reply"
    assert tendencies[0]["evidence_refs"] == ["ref_x", "ref_y"]

    priors = state["self_cognition"]["repair_priors"]
    assert len(priors) == 1
    assert priors[0]["source_mismatch_key"] == "mkey_42"
    assert priors[0]["status"] == "active"

    events = [e for e in bus if e["type"] == "M19_3TendencyPromoted"]
    assert len(events) == 1
    assert events[0]["confidence"] == 0.8
    assert events[0]["tendency_id"] == tendencies[0]["id"]
    assert events[0]["repair_prior_id"] == priors[0]["id"]


def test_self_cognition_slow_promote_skips_when_already_active() -> None:
    state: dict = {
        "self_cognition": {
            "calibrated_tendencies": [
                {
                    "id": "existing_id",
                    "source_mismatch_key": "mkey_42",
                    "status": "active",
                    "target_context": "ctx",
                }
            ],
            "repair_priors": [],
        }
    }
    bus: list = []
    commitment = _self_cognition_commitment(source_ref="mkey_42")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="slow_promote",
        magnitude_after=0.8,
    )

    apply_self_cognition_slow_promote(decision, commitment, state, bus)

    # No new entry was appended; the existing one is untouched.
    assert len(state["self_cognition"]["calibrated_tendencies"]) == 1
    assert state["self_cognition"]["calibrated_tendencies"][0]["id"] == "existing_id"

    events = [e for e in bus if e["type"] == "M19_3PromotionAlreadyActive"]
    assert len(events) == 1
    assert events[0]["tendency_id"] == "existing_id"

    # No M19_3TendencyPromoted event was emitted.
    assert not any(e["type"] == "M19_3TendencyPromoted" for e in bus)


# === self_cognition revoke ===============================================


def test_self_cognition_revoke_marks_rows_revoked() -> None:
    state: dict = {
        "self_cognition": {
            "calibrated_tendencies": [
                {
                    "id": "t1",
                    "source_mismatch_key": "mkey_42",
                    "status": "active",
                    "target_context": "ctx",
                }
            ],
            "repair_priors": [
                {
                    "id": "p1",
                    "source_mismatch_key": "mkey_42",
                    "status": "active",
                }
            ],
        }
    }
    bus: list = []
    commitment = _self_cognition_commitment(source_ref="mkey_42")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="revoke",
    )

    apply_self_cognition_revoke(decision, commitment, state, bus)

    assert state["self_cognition"]["calibrated_tendencies"][0]["status"] == "revoked"
    assert state["self_cognition"]["repair_priors"][0]["status"] == "revoked"
    assert (
        state["self_cognition"]["calibrated_tendencies"][0]["revoked_at_turn"]
        == decision.turn_index
    )

    events = [e for e in bus if e["type"] == "M19_3TendencyRevoked"]
    assert len(events) == 1
    assert events[0]["revoked_tendency_ids"] == ["t1"]
    assert events[0]["revoked_prior_ids"] == ["p1"]


def test_self_cognition_revoke_is_idempotent() -> None:
    """A second revoke on already-revoked rows must be a no-op for the rows."""
    state: dict = {
        "self_cognition": {
            "calibrated_tendencies": [
                {
                    "id": "t1",
                    "source_mismatch_key": "mkey_42",
                    "status": "revoked",
                    "revoked_at_turn": 0,
                }
            ],
            "repair_priors": [
                {
                    "id": "p1",
                    "source_mismatch_key": "mkey_42",
                    "status": "revoked",
                    "revoked_at_turn": 0,
                }
            ],
        }
    }
    bus: list = []
    commitment = _self_cognition_commitment(source_ref="mkey_42")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="revoke",
    )

    apply_self_cognition_revoke(decision, commitment, state, bus)

    # revoked_at_turn is NOT overwritten.
    assert state["self_cognition"]["calibrated_tendencies"][0]["revoked_at_turn"] == 0

    events = [e for e in bus if e["type"] == "M19_3TendencyRevoked"]
    assert len(events) == 1
    assert events[0]["revoked_tendency_ids"] == []
    assert events[0]["revoked_prior_ids"] == []


# === dispatcher routing ==================================================


def test_run_m20_2_1_write_path_handles_v1_scope() -> None:
    """Each (level, owner_id) in v1 scope returns True and mutates state."""
    state: dict = {
        "m13_drive_state": default_m13_drive_state(),
        "self_cognition": {
            "calibrated_tendencies": [],
            "repair_priors": [],
        },
    }
    bus: list = []
    commitment = _m13_commitment(user_id="u1", action="answer")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="microadjust",
        magnitude_before=0.5,
    )

    handled = run_m20_2_1_write_path(
        level="microadjust",
        owner_id="m13_drive_state",
        decision=decision,
        commitment=commitment,
        state=state,
        bus=bus,
    )
    assert handled is True
    assert "answer|u1" in state["m13_drive_state"]["traction_by_action"]


def test_run_m20_2_1_write_path_returns_false_for_out_of_scope() -> None:
    state: dict = {"m15_episode_ledger": {}}
    bus: list = []
    decision = _decision(
        commit_id="cid_oos",
        owner_id="m15_episode_ledger",
        level="microadjust",
    )
    commitment = _commitment_for_decision(decision)
    # m15 is out of v1 scope; the dispatcher returns False.
    handled = run_m20_2_1_write_path(
        level="microadjust",
        owner_id="m15_episode_ledger",
        decision=decision,
        commitment=commitment,
        state=state,
        bus=bus,
    )
    assert handled is False
    assert bus == []


# === end-to-end via routing stubs =========================================


def test_routing_stub_microadjust_for_m13_actually_writes() -> None:
    state: dict = {"m13_drive_state": default_m13_drive_state()}
    bus: list = []
    commitment = _m13_commitment(user_id="dave", action="answer")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="microadjust",
        magnitude_before=0.5,
    )

    event = route_microadjust(decision, state=state, bus=bus, commitment=commitment)

    # The audit envelope still fires.
    assert event["type"] == "GradedCorrectionRouted"
    # AND the v1-scope write path also fired.
    assert "answer|dave" in state["m13_drive_state"]["traction_by_action"]
    assert any(e["type"] == "M13TractionMicroadjust" for e in bus)


def test_routing_stub_slow_promote_for_self_cognition_actually_writes() -> None:
    state: dict = {
        "self_cognition": {
            "calibrated_tendencies": [],
            "repair_priors": [],
        }
    }
    bus: list = []
    commitment = _self_cognition_commitment(source_ref="mkey_99")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="slow_promote",
        magnitude_after=0.7,
    )

    event = route_slow_promote(decision, state=state, bus=bus, commitment=commitment)

    assert event["type"] == "GradedCorrectionRouted"
    assert len(state["self_cognition"]["calibrated_tendencies"]) == 1
    assert any(e["type"] == "M19_3TendencyPromoted" for e in bus)


def test_routing_stub_revoke_for_self_cognition_actually_writes() -> None:
    state: dict = {
        "self_cognition": {
            "calibrated_tendencies": [
                {
                    "id": "t1",
                    "source_mismatch_key": "mkey_99",
                    "status": "active",
                }
            ],
            "repair_priors": [],
        }
    }
    bus: list = []
    commitment = _self_cognition_commitment(source_ref="mkey_99")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="revoke",
    )

    event = route_revoke(decision, state=state, bus=bus, commitment=commitment)

    assert event["type"] == "GradedCorrectionRouted"
    assert state["self_cognition"]["calibrated_tendencies"][0]["status"] == "revoked"
    assert any(e["type"] == "M19_3TendencyRevoked" for e in bus)


def test_routing_stub_out_of_scope_owner_does_not_mutate_state() -> None:
    """An out-of-scope (level, owner_id) pair emits the audit envelope
    but does NOT mutate the owner state."""
    state: dict = {"m15_episode_ledger": {}}
    bus: list = []
    decision = _decision(
        commit_id="cid_oos",
        owner_id="m15_episode_ledger",
        level="microadjust",
    )
    commitment = _commitment_for_decision(decision)

    event = route_microadjust(decision, state=state, bus=bus, commitment=commitment)

    # Audit envelope still fires.
    assert event["type"] == "GradedCorrectionRouted"
    # No per-owner mutation event.
    assert not any(e["type"].startswith("M13Traction") for e in bus)
    assert not any(e["type"].startswith("M15") for e in bus)


# === write-path purity assertions ========================================


def test_write_path_does_not_call_llm() -> None:
    """The write-path functions must not import or call any LLM."""
    import inspect

    src = inspect.getsource(apply_m13_microadjust) + \
          inspect.getsource(apply_m13_next_turn) + \
          inspect.getsource(apply_m13_same_turn) + \
          inspect.getsource(apply_self_cognition_slow_promote) + \
          inspect.getsource(apply_self_cognition_revoke)
    forbidden = ("llm.", "openai", "anthropic", "client.chat", "client.messages")
    for token in forbidden:
        assert token not in src, f"write paths must not reference {token!r}"


def test_write_path_does_not_invent_new_state_bucket() -> None:
    """The write paths only mutate keys already in the existing state shape."""
    state: dict = {
        "m13_drive_state": default_m13_drive_state(),
        "self_cognition": {
            "calibrated_tendencies": [],
            "repair_priors": [],
        },
    }
    before_keys = set(state.keys())
    bus: list = []

    commitment = _m13_commitment(user_id="u1", action="answer")
    decision = _decision(
        commit_id=commitment.commit_id,
        owner_id="m13_drive_state",
        level="microadjust",
    )
    apply_m13_microadjust(decision, commitment, state, bus)

    sc_commitment = _self_cognition_commitment(source_ref="mkey_1")
    sc_decision = _decision(
        commit_id=sc_commitment.commit_id,
        owner_id="self_cognition_calibrated_tendencies",
        level="slow_promote",
        magnitude_after=0.7,
    )
    apply_self_cognition_slow_promote(sc_decision, sc_commitment, state, bus)

    # No new top-level state keys.
    assert set(state.keys()) == before_keys
    # No new top-level keys inside m13_drive_state or self_cognition
    # beyond what default_m13_drive_state already declares.
    from segmentum.dialogue.runtime.m13_drive import default_m13_drive_state as _default
    assert set(state["m13_drive_state"].keys()) <= set(_default().keys()) | {"traction_by_action"}
    assert set(state["self_cognition"].keys()) <= {
        "calibrated_tendencies",
        "repair_priors",
        "summary",
        "current_self_view",
        "identity_tensions",
        "stable_values",
        "known_limits",
        "patch_history",
    }


# === full mvp_loop integration: admit -> settle -> dispatch ==============


def test_mvp_loop_dispatch_writes_to_m13_and_self_cognition() -> None:
    """End-to-end: mvp_loop's _dispatch_graded_corrections fires the
    write path functions, and the bus accumulates the routed event
    AND the per-owner audit event."""
    from segmentum.dialogue.runtime.mvp_loop import _dispatch_graded_corrections

    state: dict = {
        "m13_drive_state": default_m13_drive_state(),
        "self_cognition": {
            "calibrated_tendencies": [],
            "repair_priors": [],
        },
    }
    bus: list = []
    commitment = _m13_commitment(user_id="eve", action="answer")
    init_owner_observability_for_commitment(
        state, owner_id=commitment.owner_id, commitment=commitment,
    )
    settled = SettledValue(
        commit_id=commitment.commit_id,
        outcome="confirmed",
        magnitude=0.2,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="t",
        turn_index=1,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_active_commitment",
    )
    state["commitment_owner_observability"][commitment.owner_id][commitment.commit_id]["settled_value"] = {
        "outcome": "confirmed",
        "magnitude": 0.2,
        "settler_type": "deterministic",
        "evidence_refs": ["ref1"],
        "reason_codes": ["settler_deterministic"],
        "at": "t",
        "turn_index": 1,
        "engineering_proxy_label": "mvp_local_active_commitment",
    }

    _dispatch_graded_corrections(bus=bus, state=state, turn_index=2, now="t2")

    types = [e["type"] for e in bus]
    assert "GradedCorrectionRouted" in types
    assert "M13TractionMicroadjust" in types
    # m13 traction is bumped.
    assert "answer|eve" in state["m13_drive_state"]["traction_by_action"]
