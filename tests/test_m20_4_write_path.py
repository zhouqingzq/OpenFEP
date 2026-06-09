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
    M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE,
    REASON_GRAPH_MICROADJUST,
    REASON_GRAPH_REVOKE,
    REASON_GRAPH_SKIP_ADDRESSEE_DIRECTED_LOW_CONFIDENCE,
    _should_skip_addressee_directed_write,
    clear_addressee_graph_row,
    write_addressee_graph_microadjust,
)


def _commitment(
    *,
    observable: str = "addressee_target_match",
    m18_7_commit_id: str = "abcd" * 10,
    attribution_turn_id: str = "",
    # P0-5 (2026-06-09): default to conf strictly > 0.9
    # so the existing "write path appends a row" tests
    # continue to pin the v1 admit path. The
    # `addressed_to_assistant=True` default pins the
    # "addressed" sub-class; P0-5's write-path filter
    # (0.9 boundary, strict `<=`) only fires for conf
    # strictly below or equal to 0.9. New P0-5 tests
    # use lower conf to exercise the filter.
    hypothesis_confidence: float = 0.95,
) -> ActiveCommitment:
    payload = {
        "hypothesis": {
            "addressed_to_assistant": True,
            "confidence": hypothesis_confidence,
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


# === P0-5: write-path filter for `addressed_to_assistant=True` =========
#
# P0-5 (2026-06-09) introduces
# M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE = 0.9
# as a write-path safety net for the "addressed" sub-class
# (P1: recall_on_addressed = 0.0 in bqxsmofri). The filter
# is strictly additive: the v1 admit path (settler +
# tie-breaker + persistent graph) is preserved when
# conf > 0.9, and skipped (returning None) when conf <= 0.9.
# The producer admit threshold (P0-4, 0.7) is upstream and
# unchanged. The "not addressed" sub-class and the reaction
# observable are unchanged.


def test_p0_5_skip_threshold_constant_is_frozen() -> None:
    """The 0.9 boundary is the v1 → P0-5 contract."""
    assert M20_4_WRITE_PATH_SKIP_ADDRESSEE_DIRECTED_BELOW_CONFIDENCE == 0.9


def test_p0_5_skip_helper_returns_true_at_or_below_threshold() -> None:
    """The skip helper is strict `<=` (consistent with the
    M20.4 v1 tie-breaker style).
    """
    # At threshold (0.9) — skip.
    assert _should_skip_addressee_directed_write(confidence=0.9) is True
    # Below threshold — skip.
    assert _should_skip_addressee_directed_write(confidence=0.5) is True
    assert _should_skip_addressee_directed_write(confidence=0.0) is True


def test_p0_5_skip_helper_returns_false_strictly_above_threshold() -> None:
    """Strictly above 0.9 — write proceeds."""
    assert _should_skip_addressee_directed_write(confidence=0.91) is False
    assert _should_skip_addressee_directed_write(confidence=0.95) is False
    assert _should_skip_addressee_directed_write(confidence=1.0) is False


def test_p0_5_skip_helper_handles_invalid_inputs() -> None:
    """NaN / non-numeric inputs are treated as 'skip'
    (conservative). This prevents a malformed hypothesis
    from bypassing the filter.
    """
    # NaN — skip.
    assert (
        _should_skip_addressee_directed_write(confidence=float("nan"))
        is True
    )
    # Non-numeric — skip.
    assert _should_skip_addressee_directed_write(confidence="0.95") is True
    assert _should_skip_addressee_directed_write(confidence=None) is True


def test_p0_5_addressee_directed_write_skipped_at_0_9_boundary() -> None:
    """P0-5: at the 0.9 boundary, the write returns None
    and the diagnostic counter is bumped. The persistent
    `state["addressee_graph"]` is NOT mutated.
    """
    state: dict = {}
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(
            m18_7_commit_id="abcd" * 10,
            hypothesis_confidence=0.9,
        ),
        at="2026-06-06T00:00:00Z",
    )
    assert event is None
    assert "addressee_graph" not in state
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get("write_path_skip_addressee_directed_low_confidence_total")
        == 1
    )


def test_p0_5_addressee_directed_write_skipped_below_0_9() -> None:
    """P0-5: at conf 0.5, the write is skipped. This is
    the typical bqxsmofri P1 'addressed' emit (recall 0.0).
    """
    state: dict = {}
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(
            m18_7_commit_id="abcd" * 10,
            hypothesis_confidence=0.5,
        ),
        at="2026-06-06T00:00:00Z",
    )
    assert event is None
    assert "addressee_graph" not in state
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get("write_path_skip_addressee_directed_low_confidence_total")
        == 1
    )


def test_p0_5_addressee_directed_write_proceeds_strictly_above_0_9() -> None:
    """P0-5: at conf 0.95, the write proceeds. The
    v1 admit path is preserved for the actionable
    high-confidence 'addressed' signal.
    """
    state: dict = {}
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(
            m18_7_commit_id="abcd" * 10,
            hypothesis_confidence=0.95,
        ),
        at="2026-06-06T00:00:00Z",
    )
    assert event is not None
    assert state["addressee_graph"]["abcd" * 10]["confidence"] == 0.95
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get(
            "write_path_skip_addressee_directed_low_confidence_total", 0
        )
        == 0
    )


def test_p0_5_reaction_observable_unaffected_by_filter() -> None:
    """P0-5 does NOT touch the reaction observable. The
    filter is observable == "addressee_target_match" only.
    """
    state: dict = {}
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(
            m18_7_commit_id="abcd" * 10,
            observable="reaction_attribution_match",
        ),
        at="2026-06-06T00:00:00Z",
    )
    # reaction_attribution_match write proceeds at conf 0.7
    # (the v1 default for reaction).
    assert event is not None
    assert state["addressee_graph"]["abcd" * 10]["confidence"] == 0.7
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get(
            "write_path_skip_addressee_directed_low_confidence_total", 0
        )
        == 0
    )


def test_p0_5_addressee_not_directed_unaffected_by_filter() -> None:
    """P0-5 only filters the `addressed_to_assistant == True`
    sub-class. The "not addressed" sub-class (P1 precision
    1.0) keeps the v1 admit path at the v1 0.4 producer
    threshold.
    """
    # Build a commitment with `addressed_to_assistant=False`
    # directly so the filter does not engage.
    payload = {
        "hypothesis": {
            "addressed_to_assistant": False,
            "confidence": 0.5,
        },
        "hypothesis_commit_id": "abcd" * 10,
        "current_turn_id": "0",
        "inbound_bounded_excerpt": "hi",
        "attributed_turn_id": "",
        "attributed_bounded_excerpt": "old turn",
        "ambiguity_band": "high",
    }
    commitment = ActiveCommitment(
        commit_id="cid_w",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_abcdabcd",
        layer="B_per_turn_commitment",
        observable="addressee_target_match",
        observable_payload=payload,
        target={"m18_7_commit_id": "abcd" * 10},
        due_at={"kind": "next_turn"},
        priority=0.5,
        confidence=0.5,
        evidence_refs=("turn_0_user_utterance",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m20_4_attribution",),
        engineering_proxy_label="mvp_local_group_attribution",
        horizon="next_turn",
    )
    state: dict = {}
    event = write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=commitment,
        at="2026-06-06T00:00:00Z",
    )
    # "not addressed" sub-class — filter does not engage
    # even at conf 0.5 (the filter is observable_match
    # only on `addressed_to_assistant == True`).
    assert event is not None
    assert state["addressee_graph"]["abcd" * 10]["confidence"] == 0.5
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get(
            "write_path_skip_addressee_directed_low_confidence_total", 0
        )
        == 0
    )


def test_p0_5_skip_counter_accumulates_across_calls() -> None:
    """The diagnostic counter accumulates across multiple
    skipped writes. This is the M20.4 owner-facing signal
    for the addressee_graph safety net.
    """
    state: dict = {}
    for i in range(3):
        write_addressee_graph_microadjust(
            state=state,
            decision=_decision(),
            commitment=_commitment(
                m18_7_commit_id=f"cid_{i}_" + "0" * 32,
                hypothesis_confidence=0.5,
            ),
            at="2026-06-06T00:00:00Z",
        )
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get("write_path_skip_addressee_directed_low_confidence_total")
        == 3
    )
    assert "addressee_graph" not in state


def test_p0_5_skip_reason_code_exists() -> None:
    """The skip reason code is exported and string-stable.
    M20.4 owner and any audit reader can match on it.
    """
    assert (
        REASON_GRAPH_SKIP_ADDRESSEE_DIRECTED_LOW_CONFIDENCE
        == "m20_4_addressee_graph_skip_addressee_directed_low_confidence"
    )


def test_p0_5_mixed_batch_skips_directed_low_confidence_writes_only() -> None:
    """Mixed batch: 1 'addressed' @ 0.5 (skip), 1 'addressed'
    @ 0.95 (admit), 1 'not addressed' @ 0.5 (admit, P0-5
    does not touch this sub-class). Result: 2 graph rows
    written, 1 skip, the 'not addressed' row is in the
    graph (P0-5 does not filter it).
    """
    state: dict = {}
    # 1. addressed @ 0.5 — skip.
    write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(
            m18_7_commit_id="cid_skip_" + "0" * 32,
            hypothesis_confidence=0.5,
        ),
        at="2026-06-06T00:00:00Z",
    )
    # 2. addressed @ 0.95 — admit.
    write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=_commitment(
            m18_7_commit_id="cid_admit_" + "0" * 32,
            hypothesis_confidence=0.95,
        ),
        at="2026-06-06T00:00:00Z",
    )
    # 3. not addressed @ 0.5 — admit (P0-5 not engaged).
    payload = {
        "hypothesis": {
            "addressed_to_assistant": False,
            "confidence": 0.5,
        },
        "hypothesis_commit_id": "cid_notaddr_" + "0" * 31,
        "current_turn_id": "0",
        "inbound_bounded_excerpt": "hi",
        "attributed_turn_id": "",
        "attributed_bounded_excerpt": "old turn",
        "ambiguity_band": "high",
    }
    commitment_not = ActiveCommitment(
        commit_id="cid_w3",
        owner_id="group_addressee_graph",
        source_kind="state",
        source_ref="m18_7_cidnot",
        layer="B_per_turn_commitment",
        observable="addressee_target_match",
        observable_payload=payload,
        target={"m18_7_commit_id": "cid_notaddr_" + "0" * 31},
        due_at={"kind": "next_turn"},
        priority=0.5,
        confidence=0.5,
        evidence_refs=("turn_0_user_utterance",),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m20_4_attribution",),
        engineering_proxy_label="mvp_local_group_attribution",
        horizon="next_turn",
    )
    write_addressee_graph_microadjust(
        state=state,
        decision=_decision(),
        commitment=commitment_not,
        at="2026-06-06T00:00:00Z",
    )
    # 2 graph rows; 1 skip.
    assert len(state["addressee_graph"]) == 2
    assert "cid_skip_" + "0" * 32 not in state["addressee_graph"]
    assert "cid_admit_" + "0" * 32 in state["addressee_graph"]
    assert "cid_notaddr_" + "0" * 31 in state["addressee_graph"]
    diag = state.get("m20_4_attribution_diagnostics", {})
    assert (
        diag.get("write_path_skip_addressee_directed_low_confidence_total")
        == 1
    )
