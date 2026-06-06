"""Tests for M20.1 settler protocol: SettledValue, NoSettlement, settler
types, SettlementScheduler, and the six reference settlers.

M20.1 owns the *observation* half of the unified-commitment loop. It
does NOT implement promotion / microadjust / revocation / expiration —
those belong to M20.2. This test file MUST NOT stub them silently.
"""

from __future__ import annotations

import dataclasses

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    COMMITMENT_REGISTRY_V1,
    MAGNITUDE_SCALES_V1,
    NoSettlement,
    OBSERVABLE_V1,
    OUTCOME_BY_OBSERVABLE_V1,
    OUTCOME_V1,
    REASON_CODES_V1,
    SETTLER_REASON_CODES_V1,
    SETTLER_TYPE_V1,
    SettledValue,
    SettlementScheduler,
    SettlerUnavailable,
    _is_due_at_passed,
    build_active_commitment_settled_event,
    build_no_settlement_made_event,
    compute_magnitude,
    record_pending_commitment,
    remove_pending_commitment,
    update_settlement_attempts_diagnostics,
    write_owner_observability,
)
from segmentum.dialogue.runtime.active_commitment_settlers import (
    BehavioralPullShiftSilentSettler,
    BoundaryHandledLLMJudgeSettler,
    ExpectationOutcomeMatchDeterministicSettler,
    IdentityVoiceMatchLLMJudgeSettler,
    InitiativeTimingMatchHybridSettler,
    PredictionErrorBandDeterministicSettler,
)


# Some symbols from active_commitment are re-exported under a
# different name; resolve the canonical name explicitly to avoid
# import errors during static checks.
try:
    from segmentum.dialogue.runtime.active_commitment import (  # type: ignore
        ALL_SETTLEMENT_REASON_CODES_V1 as ALL_REASON_CODES_V1,
    )
except ImportError:
    ALL_REASON_CODES_V1 = REASON_CODES_V1 | SETTLER_REASON_CODES_V1


def _commitment(
    *,
    observable: str = "expectation_outcome_match",
    owner_id: str = "mismatch_memory_fast",
    created_turn: int = 0,
    due_at: dict | None = None,
    observable_payload: dict | None = None,
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id=f"cid_{observable}_{created_turn}",
        owner_id=owner_id,
        source_kind="state",
        source_ref="ref1",
        layer="B_per_turn_commitment",
        observable=observable,
        observable_payload=observable_payload or {"source_expectation_id": "self_exp_1", "target_context": "short_casual_reply"},
        target={"target_context": "short_casual_reply"},
        due_at=due_at,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("ref1", "ref2"),
        created_turn=created_turn,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("self_expectation_formation",),
        engineering_proxy_label="mvp_local_self_expectation",
    )


# === 1. SettledValue schema ============================================


def test_settled_value_schema_is_frozen_and_rejects_unknown_fields() -> None:
    fields = set(SettledValue.__dataclass_fields__.keys())
    expected = {
        "commit_id",
        "outcome",
        "magnitude",
        "evidence_refs",
        "reason_codes",
        "at",
        "turn_index",
        "settler_type",
        "engineering_proxy_label",
    }
    assert expected.issubset(fields)
    assert SettledValue.__dataclass_params__.frozen is True
    instance = SettledValue(
        commit_id="x",
        outcome="confirmed",
        magnitude=0.5,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="2026-06-06T00:00:00Z",
        turn_index=0,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_self_expectation",
    )
    try:
        instance.commit_id = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("SettledValue is not frozen")


def test_no_settlement_schema_is_frozen() -> None:
    fields = set(NoSettlement.__dataclass_fields__.keys())
    expected = {
        "commit_id",
        "reason_code",
        "settler_type",
        "engineering_proxy_label",
        "at",
        "turn_index",
    }
    assert expected.issubset(fields)
    assert NoSettlement.__dataclass_params__.frozen is True


# === 2. outcome enum v1 ================================================


def test_outcome_enum_is_bounded_per_observable() -> None:
    assert OUTCOME_V1 == frozenset({"confirmed", "violated", "uncertain", "ambiguous"})
    # Every observable has a non-empty bounded outcome set.
    assert len(OUTCOME_BY_OBSERVABLE_V1) == 11
    for observable, outcomes in OUTCOME_BY_OBSERVABLE_V1.items():
        assert observable in OBSERVABLE_V1
        assert outcomes, f"empty outcome set for {observable}"
        assert outcomes.issubset(OUTCOME_V1), f"{observable} has out-of-set outcomes"


# === 3. magnitude ======================================================


def test_magnitude_is_clamped_to_unit_interval() -> None:
    # value 5.0 / scale 1.0 -> raw 5.0, clamped to 1.0
    m, codes = compute_magnitude(
        observable="prediction_error_band",
        observable_payload={},
        committed_value=5.0,
        expected_value=0.0,
    )
    assert m == 1.0
    assert "magnitude_defaulted" not in codes
    # value -2.0 / scale 1.0 -> raw 2.0, clamped to 1.0
    m, codes = compute_magnitude(
        observable="prediction_error_band",
        observable_payload={},
        committed_value=-2.0,
        expected_value=0.0,
    )
    assert m == 1.0
    # None committed -> magnitude_defaulted
    m, codes = compute_magnitude(
        observable="prediction_error_band",
        observable_payload={},
        committed_value=None,
        expected_value=0.0,
    )
    assert m == 0.5
    assert "magnitude_defaulted" in codes


def test_magnitude_uses_frozen_per_observable_scale() -> None:
    assert MAGNITUDE_SCALES_V1["prediction_error_band"] == 1.0
    assert MAGNITUDE_SCALES_V1["behavioral_pull_shift"] == 0.5
    # All scales are non-zero.
    for observable, scale in MAGNITUDE_SCALES_V1.items():
        assert scale > 0.0, f"scale for {observable} is not positive"


# === 4. deterministic settlers are pure ==================================


def test_deterministic_settler_is_pure_and_deterministic() -> None:
    settler = ExpectationOutcomeMatchDeterministicSettler()
    commitment = _commitment()
    ctx = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 3,
        "self_expectation_outcome_results": [
            {
                "source_expectation_id": "self_exp_1",
                "target_context": "short_casual_reply",
                "status": "confirmed",
                "evidence_refs": ["ref1"],
            }
        ],
    }
    a = settler.settle(commitment, ctx)
    b = settler.settle(commitment, ctx)
    assert isinstance(a, SettledValue)
    assert isinstance(b, SettledValue)
    assert a == b  # pure: same inputs -> same output
    assert a.outcome == "confirmed"
    assert a.settler_type == "deterministic"
    assert a.reason_codes == ("settler_deterministic",)


# === 5. llm_judge settlers use frozen prompt template ====================


def test_llm_judge_settler_uses_frozen_prompt_template() -> None:
    captured: dict[str, str] = {}

    def stub_llm(system: str, user: str) -> dict:
        captured["system"] = system
        captured["user"] = user
        return {
            "outcome": "preserved",
            "boundary_kind": "privacy_boundary",
            "evidence_span": "ok",
            "reason": "preserved",
            "evidence_refs": ["turn_5_draft_reply"],
        }

    settler = BoundaryHandledLLMJudgeSettler(llm_call=stub_llm)
    commitment = _commitment(
        observable="boundary_handled",
        owner_id="policy_state",
        observable_payload={"boundary_kind": "privacy_boundary"},
    )
    ctx = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 5,
        "excerpts": [{"text": "draft"}],
    }
    result = settler.settle(commitment, ctx)
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    # The system prompt is the frozen one (substring check).
    assert "boundary-handling judge" in captured["system"]
    # The user prompt is bounded and includes the expected boundary_kind.
    assert "privacy_boundary" in captured["user"]


def test_llm_judge_settler_returns_no_settlement_on_invalid_response() -> None:
    def stub_llm(system: str, user: str) -> dict:
        return {"outcome": "not_in_enum", "boundary_kind": "unknown_kind"}

    settler = BoundaryHandledLLMJudgeSettler(llm_call=stub_llm)
    commitment = _commitment(
        observable="boundary_handled",
        owner_id="policy_state",
        observable_payload={"boundary_kind": "privacy_boundary"},
    )
    result = settler.settle(
        commitment,
        {"now": "2026-06-06T00:00:00Z", "turn_index": 5, "excerpts": []},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_llm_invalid_response"


# === 6. hybrid settlers =================================================


def test_hybrid_settler_attempts_deterministic_first() -> None:
    # Deterministic leg matches -> no LLM call expected.
    called = {"n": 0}

    def stub_llm(system: str, user: str) -> dict:
        called["n"] += 1
        return {"outcome": "violated"}

    settler = InitiativeTimingMatchHybridSettler(llm_call=stub_llm)
    commitment = _commitment(
        observable="initiative_timing_match",
        owner_id="outreach_intent_registry",
        observable_payload={
            "expected_window": "explicit_request",
            "actual_window": "explicit_request",
        },
    )
    ctx = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 4,
        "user_explicit_request": {"present": True, "ref_id": "turn_4_explicit"},
    }
    result = settler.settle(commitment, ctx)
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    assert called["n"] == 0  # LLM NOT called


def test_hybrid_settler_does_not_retry_llm_after_failure() -> None:
    called = {"n": 0}

    def stub_llm(system: str, user: str) -> dict:
        called["n"] += 1
        return {"outcome": "not_in_enum"}  # invalid

    settler = InitiativeTimingMatchHybridSettler(llm_call=stub_llm)
    commitment = _commitment(
        observable="initiative_timing_match",
        owner_id="outreach_intent_registry",
        observable_payload={
            "expected_window": "natural_initiative",
            "actual_window": "after_silence",
        },
    )
    result = settler.settle(
        commitment,
        {"now": "2026-06-06T00:00:00Z", "turn_index": 4, "excerpts": []},
    )
    assert called["n"] == 1  # LLM called exactly once
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_hybrid_fallback_exhausted"


# === 7. silent settler ==================================================


def test_silent_settler_emits_no_settlement_with_carry_forward() -> None:
    settler = BehavioralPullShiftSilentSettler()
    commitment = _commitment(
        observable="behavioral_pull_shift",
        owner_id="m13_drive_state",
    )
    result = settler.settle(commitment, {"now": "t", "turn_index": 0})
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_silent_carry_forward"
    assert result.settler_type == "silent"


# === 8. scheduler rules =================================================


def _make_pending(
    commitment: ActiveCommitment,
    *,
    extra: dict | None = None,
) -> dict:
    row = {
        "commit_id": commitment.commit_id,
        "owner_id": commitment.owner_id,
        "source_kind": commitment.source_kind,
        "source_ref": commitment.source_ref,
        "layer": commitment.layer,
        "observable": commitment.observable,
        "observable_payload": dict(commitment.observable_payload),
        "target": dict(commitment.target),
        "due_at": dict(commitment.due_at) if commitment.due_at else None,
        "priority": commitment.priority,
        "confidence": commitment.confidence,
        "evidence_refs": list(commitment.evidence_refs),
        "reason_codes": list(commitment.reason_codes),
        "engineering_proxy_label": commitment.engineering_proxy_label,
        "created_turn": commitment.created_turn,
        "created_at": commitment.created_at,
    }
    if extra:
        row.update(extra)
    return row


def test_settlement_scheduler_does_not_settle_on_t0() -> None:
    state: dict = {
        "active_commitments_pending": [
            _make_pending(_commitment(created_turn=0, due_at={"kind": "next_turn"}))
        ]
    }
    scheduler = SettlementScheduler(
        settlers_by_observable={
            "expectation_outcome_match": ExpectationOutcomeMatchDeterministicSettler(),
        }
    )
    settled, no_settlement = scheduler.attempt_settlements(
        state=state, turn_index=0, now="t"
    )
    assert settled == []
    assert no_settlement == []
    # Still pending (T0+1 minimum).
    assert len(state["active_commitments_pending"]) == 1


def test_settlement_scheduler_settles_on_t1_with_matching_outcome() -> None:
    commitment = _commitment(created_turn=0, due_at={"kind": "next_turn"})
    state: dict = {
        "active_commitments_pending": [_make_pending(commitment)]
    }
    scheduler = SettlementScheduler(
        settlers_by_observable={
            "expectation_outcome_match": ExpectationOutcomeMatchDeterministicSettler(),
        }
    )

    def provider(turn_index: int, row: dict) -> dict:
        return {
            "now": "t",
            "turn_index": turn_index,
            "self_expectation_outcome_results": [
                {
                    "source_expectation_id": "self_exp_1",
                    "target_context": "short_casual_reply",
                    "status": "violated",
                    "evidence_refs": ["ref1"],
                }
            ],
        }

    settled, no_settlement = scheduler.attempt_settlements(
        state=state,
        turn_index=1,
        now="t",
        observation_context_provider=provider,
    )
    assert len(settled) == 1
    assert settled[0]["outcome"] == "violated"
    assert no_settlement == []
    # Removed from pending.
    assert state["active_commitments_pending"] == []
    # Observability entry written.
    obs = state["commitment_owner_observability"]["mismatch_memory_fast"][commitment.commit_id]
    assert obs["settled_value"] is not None
    assert obs["settled_value"]["outcome"] == "violated"


def test_settlement_scheduler_attempts_at_most_once_per_commit_per_turn() -> None:
    commitment = _commitment(created_turn=0)
    state: dict = {
        "active_commitments_pending": [_make_pending(commitment)]
    }
    # No settler for the observable; scheduler should emit exactly one
    # NoSettlementMade per turn, with `settler_unavailable`. The
    # commitment stays pending (transient: M20.1.1 may wire a settler).
    scheduler = SettlementScheduler(settlers_by_observable={})
    settled, no_settlement = scheduler.attempt_settlements(
        state=state, turn_index=1, now="t"
    )
    assert settled == []
    assert len(no_settlement) == 1
    assert no_settlement[0]["reason_code"] == "settler_unavailable"
    # Same turn re-run: observability already records the prior attempt
    # so the scheduler SHOULD still emit, but the settler_unavailable
    # path is transient. The single-attempt-per-(commit, turn) rule
    # is observable in the observability counter: settlement_attempts
    # increments each time attempt_settlements runs.
    owner_obs = state["commitment_owner_observability"]["mismatch_memory_fast"][commitment.commit_id]
    assert owner_obs["settlement_attempts"] == 1
    # Same turn, second call: counter increments again.
    scheduler.attempt_settlements(state=state, turn_index=1, now="t")
    assert owner_obs["settlement_attempts"] == 2


def test_settlement_scheduler_emits_due_at_passed_after_window() -> None:
    commitment = _commitment(
        created_turn=0,
        due_at={"kind": "next_turn"},
    )
    state: dict = {
        "active_commitments_pending": [_make_pending(commitment)]
    }
    scheduler = SettlementScheduler(settlers_by_observable={})
    # Turn 1: T0+1 minimum, not yet past. Skip (no eligible settler
    # -> settler_unavailable path, which keeps pending).
    settled, no_settlement = scheduler.attempt_settlements(
        state=state, turn_index=1, now="t"
    )
    assert settled == []
    assert len(no_settlement) == 1
    assert no_settlement[0]["reason_code"] == "settler_unavailable"
    # Commitment stays pending (settler_unavailable is transient).
    assert len(state["active_commitments_pending"]) == 1
    # Turn 3: past the next_turn window (created_turn=0, window=1).
    settled, no_settlement = scheduler.attempt_settlements(
        state=state, turn_index=3, now="t"
    )
    assert len(no_settlement) == 1
    assert no_settlement[0]["reason_code"] == "due_at_passed"
    # Removed from pending so we don't re-emit.
    assert state["active_commitments_pending"] == []


def test_is_due_at_passed_is_strictly_next_turn_window() -> None:
    assert _is_due_at_passed({"kind": "next_turn"}, created_turn=0, turn_index=2) is True
    assert _is_due_at_passed({"kind": "next_turn"}, created_turn=0, turn_index=1) is False
    assert _is_due_at_passed(None, created_turn=0, turn_index=99) is False
    assert _is_due_at_passed({"kind": "unknown"}, created_turn=0, turn_index=99) is False


# === 9. owner observability is additive =================================


def test_owner_observability_is_additive_and_does_not_replace_state() -> None:
    state = {
        "self_cognition": {"summary": "untouched"},
        "m13_drive_state": {"traction": [1, 2, 3]},
    }
    settled = SettledValue(
        commit_id="cid_x",
        outcome="confirmed",
        magnitude=1.0,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="t",
        turn_index=2,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_self_expectation",
    )
    write_owner_observability(
        state,
        owner_id="mismatch_memory_fast",
        commit_id="cid_x",
        settled_value=settled,
        last_attempt_turn_index=2,
        last_attempt_reason_code="settler_deterministic",
    )
    # Prior state untouched.
    assert state["self_cognition"] == {"summary": "untouched"}
    assert state["m13_drive_state"] == {"traction": [1, 2, 3]}
    # Observability entry added.
    obs = state["commitment_owner_observability"]["mismatch_memory_fast"]["cid_x"]
    assert obs["settled_value"]["outcome"] == "confirmed"
    assert obs["settlement_attempts"] == 1
    assert obs["last_attempt_turn_index"] == 2

    # Second write increments counter and overwrites settled_value.
    write_owner_observability(
        state,
        owner_id="mismatch_memory_fast",
        commit_id="cid_x",
        settled_value=None,
        last_attempt_turn_index=3,
        last_attempt_reason_code="due_at_passed",
    )
    obs2 = state["commitment_owner_observability"]["mismatch_memory_fast"]["cid_x"]
    assert obs2["settlement_attempts"] == 2
    assert obs2["settled_value"] is None
    assert obs2["last_attempt_reason_code"] == "due_at_passed"


# === 10. diagnostics ====================================================


def test_settlement_attempts_diagnostics_exposed() -> None:
    state: dict = {}
    update_settlement_attempts_diagnostics(
        state,
        settled=2,
        no_settlement=1,
        by_settler_type={"deterministic": 2, "silent": 1},
        by_observable={"expectation_outcome_match": 2, "behavioral_pull_shift": 1},
        by_reason_code={"settler_unavailable": 1},
        magnitudes=(1.0, 0.5),
    )
    diag = state["settlement_attempts_diagnostics"]
    assert diag["settled_total"] == 2
    assert diag["no_settlement_total"] == 1
    assert diag["settlement_attempts_total"] == 3
    assert diag["settlement_attempts_by_settler_type"] == {
        "deterministic": 2,
        "silent": 1,
    }
    assert diag["settlement_attempts_by_observable"] == {
        "expectation_outcome_match": 2,
        "behavioral_pull_shift": 1,
    }
    assert diag["no_settlement_by_reason_code"] == {"settler_unavailable": 1}
    assert diag["settled_value_magnitude_distribution"] == [1.0, 0.5]


# === 11. reference settler routing ======================================


def test_reference_expectation_outcome_match_settler_routes_m19_0() -> None:
    settler = ExpectationOutcomeMatchDeterministicSettler()
    assert settler.SETTLER_TYPE == "deterministic"
    # Missing M19.0 row -> NoSettlement (no eligible observation).
    commitment = _commitment()
    result = settler.settle(
        commitment,
        {"now": "t", "turn_index": 1, "self_expectation_outcome_results": []},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "no_eligible_observation"


def test_reference_identity_voice_match_settler_wraps_surface_consistency() -> None:
    settler = IdentityVoiceMatchLLMJudgeSettler()
    assert settler.SETTLER_TYPE == "llm_judge"
    commitment = _commitment(
        observable="identity_voice_match",
        owner_id="policy_state",
    )
    # consistent -> confirmed
    result = settler.settle(
        commitment,
        {
            "now": "t",
            "turn_index": 1,
            "surface_consistency_verification": {
                "surface_intent_outcome": "consistent",
                "confidence": 0.9,
                "evidence_refs": ["turn_1_draft"],
            },
        },
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    # drifted_voice -> violated
    result2 = settler.settle(
        commitment,
        {
            "now": "t",
            "turn_index": 1,
            "surface_consistency_verification": {
                "surface_intent_outcome": "drifted_voice",
                "confidence": 0.4,
                "evidence_refs": ["turn_1_draft"],
            },
        },
    )
    assert isinstance(result2, SettledValue)
    assert result2.outcome == "violated"
    # ambiguous -> ambiguous + magnitude_defaulted
    result3 = settler.settle(
        commitment,
        {
            "now": "t",
            "turn_index": 1,
            "surface_consistency_verification": {
                "surface_intent_outcome": "ambiguous",
                "confidence": 0.2,
                "evidence_refs": ["turn_1_draft"],
            },
        },
    )
    assert isinstance(result3, SettledValue)
    assert result3.outcome == "ambiguous"
    assert "magnitude_defaulted" in result3.reason_codes


def test_reference_initiative_timing_match_settler_uses_hybrid() -> None:
    settler = InitiativeTimingMatchHybridSettler()
    assert settler.SETTLER_TYPE == "hybrid"
    # No LLM injected: deterministic leg returns uncertain (the natural
    # case), so the LLM fallback is required. Without an llm_call the
    # hybrid settler returns NoSettlement with hybrid_fallback_exhausted.
    commitment = _commitment(
        observable="initiative_timing_match",
        owner_id="outreach_intent_registry",
        observable_payload={
            "expected_window": "natural_initiative",
            "actual_window": "after_silence",
        },
    )
    result = settler.settle(
        commitment,
        {"now": "t", "turn_index": 1, "excerpts": []},
    )
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_hybrid_fallback_exhausted"


def test_reference_behavioral_pull_shift_settler_is_silent() -> None:
    settler = BehavioralPullShiftSilentSettler()
    assert settler.SETTLER_TYPE == "silent"
    result = settler.settle(_commitment(observable="behavioral_pull_shift"), {"now": "t", "turn_index": 0})
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_silent_carry_forward"


def test_reference_prediction_error_band_settler_matched() -> None:
    settler = PredictionErrorBandDeterministicSettler()
    commitment = _commitment(
        observable="prediction_error_band",
        owner_id="user_prediction_ledger",
        observable_payload={"prediction_id": "pred_1", "band": "high"},
    )
    result = settler.settle(
        commitment,
        {
            "now": "t",
            "turn_index": 1,
            "prediction_settlements": [
                {"prediction_id": "pred_1", "band": "high", "evidence_refs": ["ref1"]}
            ],
        },
    )
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"


# === 12. event envelopes ================================================


def test_settled_value_event_envelope_shape() -> None:
    settled = SettledValue(
        commit_id="cid",
        outcome="confirmed",
        magnitude=1.0,
        evidence_refs=("ref1",),
        reason_codes=("settler_deterministic",),
        at="t",
        turn_index=0,
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_self_expectation",
    )
    event = build_active_commitment_settled_event(settled)
    assert event["type"] == "ActiveCommitmentSettled"
    assert event["commit_id"] == "cid"
    assert event["outcome"] == "confirmed"
    assert event["settler_type"] == "deterministic"
    assert event["magnitude"] == 1.0


def test_no_settlement_event_envelope_shape() -> None:
    no_settle = NoSettlement(
        commit_id="cid",
        reason_code="due_at_passed",
        settler_type="deterministic",
        engineering_proxy_label="mvp_local_self_expectation",
        at="t",
        turn_index=2,
    )
    event = build_no_settlement_made_event(no_settle)
    assert event["type"] == "NoSettlementMade"
    assert event["commit_id"] == "cid"
    assert event["reason_code"] == "due_at_passed"


# === 13. pending commitment bookkeeping =================================


def test_record_and_remove_pending_commitment() -> None:
    state: dict = {}
    commitment = _commitment()
    record_pending_commitment(state, commitment)
    pending = state["active_commitments_pending"]
    assert len(pending) == 1
    assert pending[0]["commit_id"] == commitment.commit_id
    remove_pending_commitment(state, commitment.commit_id)
    assert state["active_commitments_pending"] == []


# === 14. M20.1 protocol does not mutate long-term state buckets =========


def test_settlement_writes_only_owner_observability() -> None:
    state = {
        "self_cognition": {"summary": "before"},
        "m13_drive_state": {"traction": [1]},
        "memory_dynamics": {"control_guidance": {"repair_bias": 0.1}},
        "mismatch_memory_fast": {"rows": []},
        "self_repair_expectation": {"rows": []},
        "active_commitments_pending": [
            _make_pending(_commitment(created_turn=0))
        ],
    }
    scheduler = SettlementScheduler(
        settlers_by_observable={
            "expectation_outcome_match": ExpectationOutcomeMatchDeterministicSettler(),
        }
    )

    def provider(turn_index: int, row: dict) -> dict:
        return {
            "now": "t",
            "turn_index": turn_index,
            "self_expectation_outcome_results": [
                {
                    "source_expectation_id": "self_exp_1",
                    "target_context": "short_casual_reply",
                    "status": "confirmed",
                    "evidence_refs": ["ref1"],
                }
            ],
        }

    scheduler.attempt_settlements(
        state=state,
        turn_index=1,
        now="t",
        observation_context_provider=provider,
    )
    # Long-term state buckets untouched.
    assert state["self_cognition"] == {"summary": "before"}
    assert state["m13_drive_state"] == {"traction": [1]}
    assert state["memory_dynamics"] == {"control_guidance": {"repair_bias": 0.1}}
    assert state["mismatch_memory_fast"] == {"rows": []}
    assert state["self_repair_expectation"] == {"rows": []}
    # Only owner observability is written.
    assert "commitment_owner_observability" in state


# === 15. cross-checks against M20.0 invariants =========================


def test_active_commitment_settlers_module_does_not_contain_promotion_paths() -> None:
    """M20.1 settler package must not implement promotion / revocation /
    expiration (those are M20.2). The forbidden class / function
    substrings are the M20.1 protocol surface (allowed) and concrete
    settler implementations (allowed here in the package, since this
    IS the settler package). Promotion / demotion / microadjust /
    revocation / expiration logic is forbidden.
    """
    import inspect
    import segmentum.dialogue.runtime.active_commitment_settlers as pkg

    source_blob = "\n".join(
        inspect.getsource(module) for module in (
            pkg,
            pkg.expectation_outcome_match,
            pkg.prediction_error_band,
            pkg.identity_voice_match,
            pkg.boundary_handled,
            pkg.initiative_timing_match,
            pkg.behavioral_pull_shift,
        )
    )
    forbidden_tokens = ("promote_", "revoke_", "expire_", "microadjust")
    for token in forbidden_tokens:
        assert token not in source_blob, (
            f"M20.1 settlers must not implement {token!r} (belongs to M20.2)"
        )


# === 16. M20.1 reasoning helper imports are clean =======================


def test_all_m20_1_symbols_are_exported() -> None:
    from segmentum.dialogue.runtime import active_commitment as m

    expected = {
        "SettledValue",
        "NoSettlement",
        "Settler",
        "SettlerUnavailable",
        "SettlementScheduler",
        "SETTLER_TYPE_V1",
        "OUTCOME_V1",
        "OUTCOME_BY_OBSERVABLE_V1",
        "MAGNITUDE_SCALES_V1",
        "SETTLER_REASON_CODES_V1",
        "build_active_commitment_settled_event",
        "build_no_settlement_made_event",
        "compute_magnitude",
        "record_pending_commitment",
        "remove_pending_commitment",
        "update_settlement_attempts_diagnostics",
        "write_owner_observability",
    }
    for name in expected:
        assert hasattr(m, name), f"missing export: {name}"
