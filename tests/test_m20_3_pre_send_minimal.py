"""Tests for M20.3 pre-send minimal verify (P0-1).

The full `surface_consistency_verification` LLM call (~3.4KB M19.x
prompt) is SKIPPED in `latency_mode == "fast_chat"`. That skip breaks
the M20.3 §3.2 pre-send gate for `runtime_mode_state` commitments with
`expected_mode` set: the gate sees `audit_absent` and falls through
to `ambiguous` / `advisory_guidance` (never `block`). For the
Sophia 短句纠正 scenario — a 短句纠正 to the bot while
`expected_mode = "bot_system"` is committed same-turn — the gate
cannot block the same turn.

P0-1 adds a small bounded minimal LLM call in fast_chat that runs
ONLY when a `runtime_mode_state` horizon commitment with
`expected_mode` is present. The call returns a 4-key JSON
(`surface_intent_outcome`, `confidence`, `evidence_span`,
`committed_surface_intent`) that gets written to
`reply_contract["surface_consistency_verification"]`. The pre-send
gate then sees a real audit row.

Scope of this file:
- Pure-function tests for the prompt builder + normalizer.
- Pure-function test for the horizon commitment inspector.
- Unit tests for the extracted call site
  `_run_fast_chat_pre_send_minimal` (no `run_turn` / state hydration).
- Integration tests that:
  - Confirm the helper returns the audit row + verified event
    when the LLM call succeeds.
  - Confirm the helper returns the empty verification + degraded
    event on LLM failure.
  - Confirm the helper returns empty verification + no events
    when no blockable commitment is present.
  - Confirm the audit row is read correctly by the pre-send gate.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    compute_commit_id,
)
from segmentum.dialogue.runtime.mvp_loop import (
    _AUXILIARY_LLM_STAGES,
    _build_m20_3_pre_send_minimal_degraded_event,
    _build_m20_3_pre_send_minimal_verified_event,
    _has_runtime_mode_state_horizon_with_expected_mode,
    _run_fast_chat_pre_send_minimal,
    build_m20_3_pre_send_minimal_prompt,
    normalize_m20_3_pre_send_minimal,
    normalize_surface_consistency_verification,
)
from segmentum.dialogue.runtime.same_turn_surface import (
    _SURFACE_TO_M20,
    _actual_mode,
    _surface_intent_outcome,
)


# === Fixtures / helpers =================================================


def _make_runtime_mode_state_commitment(
    *,
    expected_mode: str = "bot_system",
    turn_index: int = 0,
    horizon: str = "same_turn_surface",
) -> ActiveCommitment:
    """Build a `runtime_mode_state` commitment with a non-empty
    `expected_mode` payload (the P0-1 trigger)."""
    return ActiveCommitment(
        commit_id=compute_commit_id(
            owner_id="runtime_mode_state",
            source_ref="policy_command_mode",
            layer="A_turn_entry",
            observable="runtime_mode_state",
            created_turn=turn_index,
        ),
        owner_id="runtime_mode_state",
        source_kind="state",
        source_ref="policy_command_mode",
        layer="A_turn_entry",
        observable="runtime_mode_state",
        observable_payload={"expected_mode": expected_mode},
        target={"mode": expected_mode},
        due_at=None,
        priority=0.7,
        confidence=0.7,
        evidence_refs=("policy_command_mode",),
        created_turn=turn_index,
        created_at="2026-06-08T00:00:00Z",
        reason_codes=("policy_prior",),
        engineering_proxy_label="mvp_local_policy_producer",
        horizon=horizon,
    )


def _make_boom_complete(stage: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """A `complete_json_stage` callable that raises — used to test
    the helper's try/except fallback."""
    raise RuntimeError("simulated LLM failure for P0-1 test")


def _make_canned_complete(payload: Mapping[str, Any]) -> Callable[..., dict[str, Any]]:
    """A `complete_json_stage` callable that returns a fixed payload
    and records calls."""
    recorded: list[tuple[str, str, str]] = []

    def _complete(stage: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        recorded.append((stage, system_prompt, user_prompt))
        return dict(payload)

    _complete.recorded = recorded  # type: ignore[attr-defined]
    return _complete


# === Pure-function tests: prompt builder =================================


def test_build_m20_3_pre_send_minimal_prompt_is_small() -> None:
    """The P0-1 prompt must be small enough to keep the fast_chat
    latency budget bounded (~1-2KB system+user)."""
    sys_p, user_p = build_m20_3_pre_send_minimal_prompt(
        user_text="在吗？",
        draft_reply="在线，路由正常，待命中。",
        surface_commitment={
            "surface_intent": "chat",
            "self_identification": "",
            "persona_should_apply": False,
        },
        expected_mode="bot_system",
        turn_index=0,
    )
    total = len(sys_p) + len(user_p)
    # 2.5KB is a generous ceiling (M18.7.2 uses ~2KB; P0-1 is
    # narrower in scope so should be smaller in practice).
    assert total < 2500, f"prompt too large: {total} chars"
    assert total > 200, f"prompt suspiciously small: {total} chars"


def test_build_m20_3_pre_send_minimal_prompt_includes_required_signals() -> None:
    """The prompt must surface the inputs the LLM needs to grade
    voice match."""
    sys_p, user_p = build_m20_3_pre_send_minimal_prompt(
        user_text="在吗？",
        draft_reply="在线，路由正常，待命中。",
        surface_commitment={"surface_intent": "bot_system"},
        expected_mode="bot_system",
        turn_index=42,
    )
    assert "runtime_mode_state" in sys_p
    assert "expected_mode" in sys_p
    assert "consistent" in sys_p
    assert "drifted_intent" in sys_p
    assert "drifted_self_id" in sys_p
    assert "ambiguous" in sys_p
    # user prompt must include all 4 inputs
    assert "在吗？" in user_p
    assert "在线，路由正常，待命中。" in user_p
    assert "bot_system" in user_p
    assert "42" in user_p


def test_build_m20_3_pre_send_minimal_prompt_json_spec_has_four_keys() -> None:
    """The output spec must be exactly 4 keys — no extras that would
    tempt the LLM to over-elaborate or be parsed by ad hoc code."""
    _sys_p, user_p = build_m20_3_pre_send_minimal_prompt(
        user_text="x",
        draft_reply="y",
        surface_commitment={"surface_intent": "chat"},
        expected_mode="bot_system",
        turn_index=0,
    )
    assert '"surface_intent_outcome"' in user_p
    assert '"confidence"' in user_p
    assert '"evidence_span"' in user_p
    assert '"committed_surface_intent"' in user_p
    # JSON spec must NOT include conscious-loop / M13 / M19 fields
    # that the M18.7.2 minimal prompt explicitly strips.
    assert "control_guidance" not in user_p
    assert "self_expectation" not in user_p
    assert "thought_type" not in user_p


# === Pure-function tests: normalizer =====================================


def test_normalize_m20_3_pre_send_minimal_passes_bot_system_through() -> None:
    """`committed_surface_intent` MUST NOT be filtered through
    `ALLOWED_SURFACE_INTENTS = {"bot", "chat", "abstain"}` — the
    minimal prompt's vocabulary mirrors `expected_mode` (e.g.
    `bot_system`), not the conscious-loop's `surface_intent`."""
    v = normalize_m20_3_pre_send_minimal(
        {
            "surface_intent_outcome": "consistent",
            "confidence": 0.9,
            "evidence_span": "在线",
            "committed_surface_intent": "bot_system",
        }
    )
    assert v["surface_intent_outcome"] == "consistent"
    assert v["confidence"] == 0.9
    assert v["committed_surface_intent"] == "bot_system"


def test_normalize_m20_3_pre_send_minimal_folds_drifted_voice() -> None:
    """`drifted_voice` is a v1 nuance the minimal call does not
    grade. Folding it into `drifted_intent` keeps the pre-send
    gate's `_SURFACE_TO_M20["drifted_intent"] = "violated"` mapping
    intact."""
    v = normalize_m20_3_pre_send_minimal(
        {"surface_intent_outcome": "drifted_voice", "confidence": 0.8}
    )
    assert v["surface_intent_outcome"] == "drifted_intent"


def test_normalize_m20_3_pre_send_minimal_unknown_outcome_falls_back_to_ambiguous() -> None:
    """Out-of-enum outcomes fall back to `ambiguous` so the gate
    treats them as `audit_absent` (defensive)."""
    v = normalize_m20_3_pre_send_minimal({"surface_intent_outcome": "alien"})
    assert v["surface_intent_outcome"] == "ambiguous"


def test_normalize_m20_3_pre_send_minimal_clamps_confidence() -> None:
    v = normalize_m20_3_pre_send_minimal({"confidence": 1.7})
    assert v["confidence"] == 1.0
    v = normalize_m20_3_pre_send_minimal({"confidence": -0.4})
    assert v["confidence"] == 0.0


def test_normalize_m20_3_pre_send_minimal_empty_input_yields_audit_absent() -> None:
    """Empty input must yield the same `ambiguous` / `""` /
    `confidence=0` shape so the pre-send gate's
    `audit_absent` branch is preserved on the LLM-failure path."""
    v = normalize_m20_3_pre_send_minimal({})
    assert v["surface_intent_outcome"] == "ambiguous"
    assert v["confidence"] == 0.0
    assert v["evidence_span"] == ""
    assert v["committed_surface_intent"] == ""


def test_normalize_m20_3_pre_send_minimal_truncates_evidence_span() -> None:
    """`evidence_span` is bounded by `MAX_SURFACE_EVIDENCE_SPAN_CHARS`
    (= 120)."""
    v = normalize_m20_3_pre_send_minimal({"evidence_span": "x" * 500})
    assert len(v["evidence_span"]) == 120


def test_normalize_m20_3_pre_send_minimal_drops_overlong_committed_intent() -> None:
    v = normalize_m20_3_pre_send_minimal(
        {"committed_surface_intent": "a" * 60}
    )
    assert v["committed_surface_intent"] == ""


# === Pure-function tests: horizon inspector ==============================


def test_inspector_finds_runtime_mode_state_horizon_with_expected_mode() -> None:
    state: dict[str, Any] = {
        "m20_3_horizon_commitments": [
            _make_runtime_mode_state_commitment(expected_mode="bot_system")
        ]
    }
    found, expected = _has_runtime_mode_state_horizon_with_expected_mode(state)
    assert found is True
    assert expected == "bot_system"


def test_inspector_returns_false_when_no_commitments() -> None:
    state: dict[str, Any] = {"m20_3_horizon_commitments": []}
    found, expected = _has_runtime_mode_state_horizon_with_expected_mode(state)
    assert found is False
    assert expected == ""


def test_inspector_returns_false_when_other_observable() -> None:
    c = ActiveCommitment(
        commit_id="x",
        owner_id="mismatch_memory_fast",
        source_kind="state",
        source_ref="x",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        observable_payload={"expected_mode": "bot_system"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("x",),
        created_turn=0,
        created_at="2026-06-08T00:00:00Z",
        reason_codes=("test",),
        engineering_proxy_label="test",
        horizon="same_turn_surface",
    )
    state: dict[str, Any] = {"m20_3_horizon_commitments": [c]}
    found, _ = _has_runtime_mode_state_horizon_with_expected_mode(state)
    assert found is False


def test_inspector_returns_false_when_horizon_not_same_turn_surface() -> None:
    c = _make_runtime_mode_state_commitment(
        expected_mode="bot_system", horizon="next_turn"
    )
    state: dict[str, Any] = {"m20_3_horizon_commitments": [c]}
    found, _ = _has_runtime_mode_state_horizon_with_expected_mode(state)
    assert found is False


def test_inspector_returns_false_when_expected_mode_empty() -> None:
    c = _make_runtime_mode_state_commitment(expected_mode="")
    state: dict[str, Any] = {"m20_3_horizon_commitments": [c]}
    found, _ = _has_runtime_mode_state_horizon_with_expected_mode(state)
    assert found is False


def test_inspector_handles_missing_state_field() -> None:
    state: dict[str, Any] = {}
    found, expected = _has_runtime_mode_state_horizon_with_expected_mode(state)
    assert found is False
    assert expected == ""


# === Audit event builder tests ===========================================


def test_build_m20_3_pre_send_minimal_verified_event_shape() -> None:
    event = _build_m20_3_pre_send_minimal_verified_event(
        turn_index=7,
        verification={
            "surface_intent_outcome": "consistent",
            "confidence": 0.91,
            "evidence_span": "在线，路由正常",
            "committed_surface_intent": "bot_system",
        },
        commitment={"self_identification": "Bot"},
        expected_mode="bot_system",
    )
    assert event["type"] == "M20_3_PreSendMinimalVerifiedEvent"
    assert event["turn_index"] == 7
    assert event["surface_intent_outcome"] == "consistent"
    assert event["committed_surface_intent"] == "bot_system"
    assert event["expected_mode"] == "bot_system"
    assert event["confidence"] == 0.91
    assert event["engineering_proxy_label"] == "mvp_local_pre_send_minimal_audit"


def test_build_m20_3_pre_send_minimal_degraded_event_shape() -> None:
    event = _build_m20_3_pre_send_minimal_degraded_event(
        turn_index=7, reason="llm_error:Timeout"
    )
    assert event["type"] == "M20_3_PreSendMinimalDegradedEvent"
    assert event["turn_index"] == 7
    assert event["reason_code"] == "llm_error:Timeout"
    assert event["engineering_proxy_label"] == "mvp_local_pre_send_minimal_audit"


# === Auxiliary stage registration ========================================


def test_m20_3_pre_send_minimal_stage_is_registered_as_auxiliary() -> None:
    """The minimal call uses the auxiliary profile (12s / 0 retries)
    so the call stays cheap and bounded in fast_chat."""
    assert "m20_3_pre_send_minimal" in _AUXILIARY_LLM_STAGES
    # The full M19.x surface verify is also in the set; the minimal
    # path reuses the same latency budget.
    assert "surface_consistency_verification" in _AUXILIARY_LLM_STAGES


# === Unit tests: extracted call site _run_fast_chat_pre_send_minimal =====


def test_run_fast_chat_pre_send_minimal_returns_audit_row_on_success() -> None:
    """When a `runtime_mode_state` commitment is present and the
    LLM call succeeds, the helper returns:
    - a verification dict with `committed_surface_intent` set (so
      the pre-send gate's `_actual_mode` can read it),
    - a single `M20_3_PreSendMinimalVerifiedEvent` audit event.
    """
    state: dict[str, Any] = {
        "m20_3_horizon_commitments": [
            _make_runtime_mode_state_commitment(expected_mode="bot_system")
        ]
    }
    complete = _make_canned_complete(
        {
            "surface_intent_outcome": "consistent",
            "confidence": 0.91,
            "evidence_span": "在线，路由正常",
            "committed_surface_intent": "bot_system",
        }
    )
    surface_commitment = {
        "surface_intent": "chat",
        "self_identification": "",
        "persona_should_apply": False,
    }
    verification, events = _run_fast_chat_pre_send_minimal(
        state=state,
        surface_commitment=surface_commitment,
        raw_reply="在线，路由正常，待命中。",
        user_text="在吗？",
        turn_index=0,
        complete_json_stage=complete,
    )
    # Stage was called with the right marker.
    assert len(complete.recorded) == 1  # type: ignore[attr-defined]
    assert complete.recorded[0][0] == "m20_3_pre_send_minimal"  # type: ignore[attr-defined]
    # Verification row shape.
    assert verification["surface_intent_outcome"] == "consistent"
    assert verification["confidence"] == 0.91
    assert verification["committed_surface_intent"] == "bot_system"
    # One verified event, no degraded event.
    assert len(events) == 1
    assert events[0]["type"] == "M20_3_PreSendMinimalVerifiedEvent"
    assert events[0]["expected_mode"] == "bot_system"
    assert events[0]["committed_surface_intent"] == "bot_system"


def test_run_fast_chat_pre_send_minimal_emits_degraded_event_on_llm_failure() -> None:
    """When the LLM call raises, the helper returns:
    - an empty verification (the gate sees `audit_absent` →
      `ambiguous`, the prior fast_chat behavior is preserved),
    - a single `M20_3_PreSendMinimalDegradedEvent` audit event,
    - NO `M20_3_PreSendMinimalVerifiedEvent`.
    """
    state: dict[str, Any] = {
        "m20_3_horizon_commitments": [
            _make_runtime_mode_state_commitment(expected_mode="bot_system")
        ]
    }
    surface_commitment = {"surface_intent": "chat", "self_identification": ""}
    verification, events = _run_fast_chat_pre_send_minimal(
        state=state,
        surface_commitment=surface_commitment,
        raw_reply="在线",
        user_text="在吗？",
        turn_index=0,
        complete_json_stage=_make_boom_complete,
    )
    # Empty verification (gate sees audit_absent).
    assert verification["surface_intent_outcome"] == "ambiguous"
    assert verification["confidence"] == 0.0
    assert verification.get("committed_surface_intent", "") == ""
    # One degraded event, no verified event.
    assert len(events) == 1
    assert events[0]["type"] == "M20_3_PreSendMinimalDegradedEvent"
    assert "RuntimeError" in events[0]["reason_code"]


def test_run_fast_chat_pre_send_minimal_no_op_without_commitment() -> None:
    """When no `runtime_mode_state` commitment is present, the
    helper is a no-op:
    - empty verification,
    - empty events list (the caller emits the prior
      `SurfaceConsistencyVerificationSkippedEvent`).
    """
    state: dict[str, Any] = {"m20_3_horizon_commitments": []}
    complete = _make_canned_complete(
        {
            "surface_intent_outcome": "consistent",
            "confidence": 0.91,
            "evidence_span": "x",
            "committed_surface_intent": "bot_system",
        }
    )
    verification, events = _run_fast_chat_pre_send_minimal(
        state=state,
        surface_commitment={"surface_intent": "chat"},
        raw_reply="x",
        user_text="x",
        turn_index=0,
        complete_json_stage=complete,
    )
    # LLM was NOT called.
    assert len(complete.recorded) == 0  # type: ignore[attr-defined]
    # Empty verification + no events.
    assert verification["surface_intent_outcome"] == "ambiguous"
    assert verification.get("committed_surface_intent", "") == ""
    assert events == []


def test_run_fast_chat_pre_send_minimal_ignores_other_observables() -> None:
    """When a same-turn-surface commitment exists for a non-`runtime_mode_state`
    observable (e.g. `identity_voice_match`), the helper is a no-op.
    Only `runtime_mode_state` is blockable in v2 (the only owner with
    `accepts_same_turn_block = true`)."""
    c = ActiveCommitment(
        commit_id="x",
        owner_id="identity_voice_match",
        source_kind="state",
        source_ref="x",
        layer="B_per_turn_commitment",
        observable="identity_voice_match",
        observable_payload={"expected_mode": "bot_system"},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=("x",),
        created_turn=0,
        created_at="2026-06-08T00:00:00Z",
        reason_codes=("test",),
        engineering_proxy_label="test",
        horizon="same_turn_surface",
    )
    state: dict[str, Any] = {"m20_3_horizon_commitments": [c]}
    complete = _make_canned_complete({"surface_intent_outcome": "consistent"})
    verification, events = _run_fast_chat_pre_send_minimal(
        state=state,
        surface_commitment={"surface_intent": "chat"},
        raw_reply="x",
        user_text="x",
        turn_index=0,
        complete_json_stage=complete,
    )
    assert len(complete.recorded) == 0  # type: ignore[attr-defined]
    assert events == []
    assert verification["surface_intent_outcome"] == "ambiguous"


# === Pure-function integration: surface row on reply_contract ============


def test_minimal_verify_writes_committed_surface_intent_for_gate() -> None:
    """The verification dict written to `reply_contract` must
    include `committed_surface_intent` so the pre-send gate's
    `_actual_mode` can read it. (The full M19.x
    `normalize_surface_consistency_verification` does NOT carry that
    field through; P0-1's helper adds it explicitly.)"""
    minimal = normalize_m20_3_pre_send_minimal(
        {
            "surface_intent_outcome": "consistent",
            "confidence": 0.91,
            "evidence_span": "在线",
            "committed_surface_intent": "bot_system",
        }
    )
    # Simulate the call-site shape: normalize + add committed_surface_intent.
    row = normalize_surface_consistency_verification(minimal)
    row["committed_surface_intent"] = minimal.get("committed_surface_intent", "")
    assert row["committed_surface_intent"] == "bot_system"
    # And the pre-send gate can read it.
    actual = _actual_mode("在线，路由正常，待命中。", {"surface_consistency_verification": row})
    assert actual == "bot_system"


def test_minimal_drifted_intent_lets_pre_send_gate_detect_violation() -> None:
    """When the LLM reports a drift, the pre-send gate should see
    a `violated` outcome (which, combined with `expected != actual`,
    can `block`)."""
    minimal = normalize_m20_3_pre_send_minimal(
        {
            "surface_intent_outcome": "drifted_intent",
            "confidence": 0.88,
            "evidence_span": "嗯嗯，本堂主在",
            "committed_surface_intent": "chat",  # persona voice, not bot
        }
    )
    row = normalize_surface_consistency_verification(minimal)
    row["committed_surface_intent"] = minimal.get("committed_surface_intent", "")

    # The gate's _SURFACE_TO_M20 maps "drifted_intent" -> "violated".
    assert _SURFACE_TO_M20[minimal["surface_intent_outcome"]] == "violated"
    outcome = _surface_intent_outcome({"surface_consistency_verification": row})
    assert outcome == "drifted_intent"


def test_minimal_helper_audit_row_is_read_by_pre_send_gate() -> None:
    """End-to-end: the helper's verification row, when fed to the
    pre-send gate's `_actual_mode`, yields a real mode string
    (not empty / not `audit_absent`)."""
    state: dict[str, Any] = {
        "m20_3_horizon_commitments": [
            _make_runtime_mode_state_commitment(expected_mode="bot_system")
        ]
    }
    complete = _make_canned_complete(
        {
            "surface_intent_outcome": "consistent",
            "confidence": 0.91,
            "evidence_span": "在线，路由正常",
            "committed_surface_intent": "bot_system",
        }
    )
    verification, _ = _run_fast_chat_pre_send_minimal(
        state=state,
        surface_commitment={"surface_intent": "chat"},
        raw_reply="在线，路由正常，待命中。",
        user_text="在吗？",
        turn_index=0,
        complete_json_stage=complete,
    )
    # The gate can read the mode string.
    actual = _actual_mode("在线", {"surface_consistency_verification": verification})
    assert actual == "bot_system"
    # And the gate sees the outcome enum.
    outcome = _surface_intent_outcome({"surface_consistency_verification": verification})
    assert outcome == "consistent"

