"""Tests for M20.4 v1 §4 write path on `group_addressee_graph`.

The M20.2.1 write path was extended in M20.4 v1 to
support `group_addressee_graph.microadjust` (was no-op in
M20.3) and `group_addressee_graph.revoke` (clears the row).

The path appends an attribution row to
`state["addressee_graph"][m18_7_commit_id]` and emits a
`GroupAddresseeGraphUpdated` audit event.
"""

from __future__ import annotations

from types import SimpleNamespace

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    GradedCorrectionDecision,
)
from segmentum.dialogue.runtime.active_commitment_grader._write_paths import (
    run_m20_2_1_write_path,
)
from segmentum.dialogue.runtime.m20_4_attribution import (
    M20_4_ENGINEERING_PROXY_LABEL,
    REASON_GRAPH_MICROADJUST,
    REASON_GRAPH_REVOKE,
    clear_addressee_graph_row,
    write_addressee_graph_microadjust,
)


def _commitment(
    *,
    observable: str = "addressee_target_match",
    m18_7_commit_id: str = "abcd" * 10,
    attribution_turn_id: str = "",
) -> ActiveCommitment:
    payload = {
        "hypothesis": {
            "addressed_to_assistant": True,
            "confidence": 0.9,
        } if observable == "addressee_target_match" else {
            "is_about_assistant_claim": True,
            "reaction_to_turn_id": "turn_0",
            "confidence": 0.7,
        },
        "hypothesis_commit_id": m18_7_commit_id,
        "current_turn_id": "0",
        "inbound_bounded_excerpt": "hi",
        "attributed_turn_id": attribution_turn_id or "",
        "attributed_bounded_excerpt": "old turn",
        "ambiguity_band": "high",
    }
    return ActiveCommitment(
        commit_id="cid_w",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref=f"m18_7_{m18_7_commit_id[:8]}",
        layer="B_per_turn_commitment",
        observable=observable,
        observable_payload=payload,
        target={"m18_7_commit_id": m18_7_commit_id},
        due_at={"kind": "next_turn"},
        priority=0.9,
        confidence=0.9,
        evidence_refs=("turn_0_user_utterance",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m20_4_attribution",),
        engineering_proxy_label="mvp_local_group_attribution",
        horizon="next_turn",
    )


def _decision(*, level: str = "microadjust") -> GradedCorrectionDecision:
    return GradedCorrectionDecision(
        commit_id="cid_w",
        correction_level=level,
        routed_owner_id="group_addressee_graph",
        reason_codes=("graded_correction_routed",),
        evidence_refs=("turn_0_user_utterance",),
        magnitude_before=1.0,
        magnitude_after=1.0,
        outcome="confirmed",
        at="2026-06-06T00:00:00Z",
        turn_index=0,
        engineering_proxy_label="mvp_local_group_attribution",
    )


# === write path: microadjust ==========================================


def test_group_addressee_graph_microadjust_appends_attribution_row() -> None:
    state: dict = {}
    bus: list = []
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
        at="2026-06-06T00:00:00Z",
    )
    assert event is not None
    assert state["addressee_graph"]["abcd" * 10]["settled_outcome"] == "confirmed"


def test_group_addressee_graph_microadjust_emits_GroupAddresseeGraphUpdated_event() -> None:
    state: dict = {}
    bus: list = []
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
        at="2026-06-06T00:00:00Z",
    )
    assert event["type"] == "GroupAddresseeGraphUpdated"
    assert event["owner_id"] == "group_addressee_graph"
    assert event["settled_outcome"] == "confirmed"
    assert event["m18_7_commit_id"] == "abcd" * 10


def test_group_addressee_graph_microadjust_caps_state_at_256_entries() -> None:
    state: dict = {}
    for i in range(300):
        commitment = _commitment(m18_7_commit_id=f"cid_{i:04d}_" + "0" * 32)
        write_addressee_graph_microadjust(
            state=state,
            decision=_decision(),
            commitment=commitment,
            at="2026-06-06T00:00:00Z",
        )
    assert len(state["addressee_graph"]) == 256


def test_group_addressee_graph_microadjust_via_run_m20_2_1_write_path() -> None:
    """End-to-end: the M20.2.1 dispatcher routes
    (microadjust, group_addressee_graph) to the real write path.
    """
    state: dict = {}
    bus: list = []
    commitment = _commitment(m18_7_commit_id="abcd" * 10)
    handled = run_m20_2_1_write_path(
        level="microadjust",
        owner_id="group_addressee_graph",
        decision=_decision(),
        commitment=commitment,
        state=state,
        bus=bus,
    )
    assert handled is True
    assert "abcd" * 10 in state["addressee_graph"]
    assert any(e["type"] == "GroupAddresseeGraphUpdated" for e in bus)


# === write path: revoke ==============================================


def test_group_addressee_graph_revoke_clears_attribution_row() -> None:
    state: dict = {
        "addressee_graph": {
            "abcd" * 10: {"settled_outcome": "confirmed"}
        }
    }
    event = clear_addressee_graph_row(
        state=state,
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
    )
    assert "abcd" * 10 not in state["addressee_graph"]
    assert event["settled_outcome"] == "revoked"


def test_group_addressee_graph_revoke_clears_nonexistent_row_silently() -> None:
    """No-op when the row does not exist (revoke is idempotent)."""
    state: dict = {}
    event = clear_addressee_graph_row(
        state=state,
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
    )
    assert event is not None
    assert event["settled_outcome"] == "revoked"


def test_group_addressee_graph_revoke_via_run_m20_2_1_write_path() -> None:
    state: dict = {
        "addressee_graph": {
            "abcd" * 10: {"settled_outcome": "confirmed"}
        }
    }
    bus: list = []
    handled = run_m20_2_1_write_path(
        level="revoke",
        owner_id="group_addressee_graph",
        decision=GradedCorrectionDecision(
            commit_id="cid_w",
            correction_level="revoke",
            routed_owner_id="group_addressee_graph",
            reason_codes=("graded_correction_routed",),
            evidence_refs=(),
            magnitude_before=0.0,
            magnitude_after=0.0,
            outcome="violated",
            at="2026-06-06T00:00:00Z",
            turn_index=0,
            engineering_proxy_label="mvp_local_group_attribution",
        ),
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
        state=state,
        bus=bus,
    )
    assert handled is True
    assert "abcd" * 10 not in state["addressee_graph"]


# === write path: other levels = no-op =================================


def test_group_addressee_graph_next_turn_is_not_handled_in_v1() -> None:
    """M20.4 v1 only wires `microadjust` and `revoke` for
    `group_addressee_graph`. `next_turn` and `expire` remain
    no-op (out of v1 scope; future M20.4.1 / M20.4.2 may
    expand).
    """
    state: dict = {}
    bus: list = []
    handled = run_m20_2_1_write_path(
        level="next_turn",
        owner_id="group_addressee_graph",
        decision=_decision(level="next_turn"),
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
        state=state,
        bus=bus,
    )
    assert handled is False
    # No state surface written (the no-op path does not
    # initialize the addressee_graph key).
    assert "addressee_graph" not in state


def test_group_addressee_graph_expire_is_not_handled_in_v1() -> None:
    state: dict = {}
    bus: list = []
    handled = run_m20_2_1_write_path(
        level="expire",
        owner_id="group_addressee_graph",
        decision=_decision(level="expire"),
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
        state=state,
        bus=bus,
    )
    assert handled is False


def test_run_m20_2_1_write_path_unknown_owner_still_returns_false() -> None:
    """Other owner_id / level pairs are not in v1 scope."""
    state: dict = {}
    bus: list = []
    handled = run_m20_2_1_write_path(
        level="microadjust",
        owner_id="some_other_owner",
        decision=_decision(),
        commitment=_commitment(m18_7_commit_id="abcd" * 10),
        state=state,
        bus=bus,
    )
    assert handled is False
