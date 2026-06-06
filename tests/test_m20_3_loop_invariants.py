"""Tests for M20.3 §4 LoopInvariants (Tier A.3).

M20.3 freezes the rule matrix for the fast_chat minimum loop
invariant. The invariant is audit-only: it emits
`MinimumLoopCoverageMissed` but does NOT block the turn. The
counters increment on miss, decrement on fix.
"""

from __future__ import annotations

from types import MappingProxyType

from segmentum.dialogue.runtime.active_commitment import ActiveCommitment
from segmentum.dialogue.runtime.loop_invariants import (
    LoopInvariants,
    LoopCoverageVerdict,
    RULE_POLICY_SOURCE_REQUIRED,
    RULE_RUNTIME_MODE_STATE_REQUIRED,
    build_minimum_loop_coverage_missed_event,
)


def _commitment(
    *,
    commit_id: str = "cid1",
    owner_id: str = "runtime_mode_state",
    source_kind: str = "policy",
    observable: str = "runtime_mode_state",
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id=commit_id,
        owner_id=owner_id,
        source_kind=source_kind,
        source_ref=commit_id,
        layer="B_per_turn_commitment",
        observable=observable,
        observable_payload={},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=(commit_id,),
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_admission",
        horizon="same_turn_surface",
    )


# === rule matrix ========================================================


def test_rule_a_misses_when_no_policy_commitment() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    assert RULE_POLICY_SOURCE_REQUIRED in rules
    assert not verdict.passed


def test_rule_a_passes_when_policy_commitment_present() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[_commitment(source_kind="policy")],
        surface_intent="",
        is_external_turn=True,
    )
    assert verdict.missed == ()


def test_rule_b_misses_for_chat_surface_without_runtime_mode_state() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[_commitment(owner_id="outreach_intent_registry")],
        surface_intent="chat",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    assert RULE_RUNTIME_MODE_STATE_REQUIRED in rules


def test_rule_b_misses_for_bot_surface_without_runtime_mode_state() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[_commitment(owner_id="outreach_intent_registry")],
        surface_intent="bot",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    assert RULE_RUNTIME_MODE_STATE_REQUIRED in rules


def test_rule_b_passes_for_chat_surface_with_runtime_mode_state() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[_commitment(owner_id="runtime_mode_state")],
        surface_intent="chat",
        is_external_turn=True,
    )
    assert verdict.missed == ()


def test_rule_b_does_not_trigger_for_abstain_surface() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="abstain",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    assert RULE_RUNTIME_MODE_STATE_REQUIRED not in rules
    # Rule A still triggers.
    assert RULE_POLICY_SOURCE_REQUIRED in rules


def test_rule_b_does_not_trigger_when_surface_empty() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    assert RULE_RUNTIME_MODE_STATE_REQUIRED not in rules


# === invariant does not block the turn ==================================


def test_invariant_module_does_not_block_turn() -> None:
    """Audit-only: the invariant is computed and returned, but the
    caller decides whether to continue. The verdict itself does not
    raise or refuse to compute.
    """
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    assert isinstance(verdict, LoopCoverageVerdict)
    assert not verdict.passed


# === invariant module counter increments and decrements =================


def test_invariant_module_counter_increments_and_decrements() -> None:
    invariants = LoopInvariants()
    # Turn 0: miss rule A
    invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 1
    assert invariants.last_miss_turn[RULE_POLICY_SOURCE_REQUIRED] == 0

    # Turn 1: still miss rule A
    invariants.enforce_minimum_loop_coverage(
        turn_index=1,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 2

    # Turn 2: pass rule A
    invariants.enforce_minimum_loop_coverage(
        turn_index=2,
        proposed_commitments=[_commitment(source_kind="policy")],
        surface_intent="",
        is_external_turn=True,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 1

    # Turn 3: pass again — counter stays at 0
    invariants.enforce_minimum_loop_coverage(
        turn_index=3,
        proposed_commitments=[_commitment(source_kind="policy")],
        surface_intent="",
        is_external_turn=True,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 0


def test_invariant_module_counter_does_not_decrement_below_zero() -> None:
    invariants = LoopInvariants()
    invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[_commitment(source_kind="policy")],
        surface_intent="",
        is_external_turn=True,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 0


# === invariant module is called after conscious loop (rule order) ======


def test_invariant_module_handles_multiple_commitments() -> None:
    invariants = LoopInvariants()
    commitments = [
        _commitment(commit_id="a", source_kind="policy"),
        _commitment(commit_id="b", source_kind="state"),
        _commitment(commit_id="c", source_kind="episodic"),
    ]
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=commitments,
        surface_intent="chat",
        is_external_turn=True,
    )
    # rule A passes (>= 1 policy), rule B passes (>= 1 rms).
    assert verdict.missed == ()


# === audit envelope =====================================================


def test_build_minimum_loop_coverage_missed_event_shape() -> None:
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=42,
        proposed_commitments=[],
        surface_intent="chat",
        is_external_turn=True,
    )
    assert not verdict.passed
    event = build_minimum_loop_coverage_missed_event(verdict)
    event["at"] = "2026-06-06T00:00:42Z"
    assert event["type"] == "MinimumLoopCoverageMissed"
    assert event["turn_index"] == 42
    rules = {row["rule"] for row in event["missing"]}
    assert RULE_POLICY_SOURCE_REQUIRED in rules
    assert RULE_RUNTIME_MODE_STATE_REQUIRED in rules
    assert event["engineering_proxy_label"] == "mvp_local_minimum_loop"


# === idle turn handling =================================================


def test_invariant_module_does_not_tick_on_idle_turn() -> None:
    invariants = LoopInvariants()
    # First miss on an external turn.
    invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 1

    # Idle turn should NOT affect counters.
    invariants.enforce_minimum_loop_coverage(
        turn_index=1,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=False,
    )
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 1


# === fast_chat path enforcement =========================================


def test_fast_chat_must_admit_at_least_one_policy_commitment() -> None:
    """M20.3 §4: a fast_chat turn (chat surface) must yield >= 1
    policy commitment AND >= 1 runtime_mode_state commitment.
    """
    invariants = LoopInvariants()
    # No commitments → both rules miss.
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="chat",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    assert RULE_POLICY_SOURCE_REQUIRED in rules
    assert RULE_RUNTIME_MODE_STATE_REQUIRED in rules


def test_fast_chat_must_admit_runtime_mode_state_for_chat_or_bot() -> None:
    """M20.3 §4: bot surface must yield >= 1 runtime_mode_state commitment."""
    invariants = LoopInvariants()
    # Only a state commitment, no runtime_mode_state.
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[_commitment(source_kind="state", owner_id="m13_drive_state")],
        surface_intent="bot",
        is_external_turn=True,
    )
    rules = {row["rule"] for row in verdict.missed}
    # Rule A passes (>= 1 policy? NO — state is not policy). Rule B fails.
    assert RULE_POLICY_SOURCE_REQUIRED in rules
    assert RULE_RUNTIME_MODE_STATE_REQUIRED in rules


# === unknown proposal shapes ============================================


def test_invariant_module_ignores_non_commitment_items() -> None:
    """The invariant should not crash on non-ActiveCommitment items."""
    invariants = LoopInvariants()
    verdict = invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[
            _commitment(source_kind="policy"),
            "not a commitment",
            {"some": "dict"},
            None,
        ],
        surface_intent="chat",
        is_external_turn=True,
    )
    # The valid commitment satisfies both rules.
    assert verdict.missed == ()


# === frozen engineering proxy label =====================================


def test_invariant_module_uses_frozen_proxy_label() -> None:
    invariants = LoopInvariants()
    assert invariants.ENGINEERING_PROXY_LABEL == "mvp_local_minimum_loop"


# === read-only miss_counters property ===================================


def test_miss_counters_returns_a_copy() -> None:
    invariants = LoopInvariants()
    invariants.enforce_minimum_loop_coverage(
        turn_index=0,
        proposed_commitments=[],
        surface_intent="",
        is_external_turn=True,
    )
    snapshot = invariants.miss_counters
    snapshot[RULE_POLICY_SOURCE_REQUIRED] = 999  # mutate the snapshot
    # The internal counter must be unchanged.
    assert invariants.miss_counters[RULE_POLICY_SOURCE_REQUIRED] == 1
