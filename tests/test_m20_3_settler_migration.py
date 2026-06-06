"""Tests for M20.3 §5 M20.1.1 settler migration (Tier C).

M20.1.1 closes gap D: the existing per-loop settlers (M13.2 band
check, M15.0 episode aggregation) are wired onto the M20.1
runtime via thin adapters. Each adapter emits
`ActiveCommitmentSettled` alongside the existing owner audit
event; the agreement tests assert the two events agree on
outcome.
"""

from __future__ import annotations

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
    build_active_commitment_settled_event,
)
from segmentum.dialogue.runtime.m20_1_1_settler_migration import (
    M13BandCheckAdapter,
    M15EpisodeAggregationAdapter,
)


# === fixtures ============================================================


def _m13_commitment(
    *,
    pending_id: str = "m13_pending_1",
    evidence_refs: tuple[str, ...] = ("m13_evidence_1",),
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id="cid_m13",
        owner_id="m13_drive_state",
        source_kind="state",
        source_ref="m13_src",
        layer="C_observation",
        observable="traction_delta_band",
        observable_payload={"pending_id": pending_id},
        target={"action": "answer", "user_id": "u1"},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=evidence_refs,
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("m13_drive_signal",),
        engineering_proxy_label="mvp_local_m13_drive",
    )


def _m15_commitment(
    *,
    episode_id: str = "m15_episode_1",
    evidence_refs: tuple[str, ...] = ("m15_evidence_1",),
) -> ActiveCommitment:
    return ActiveCommitment(
        commit_id="cid_m15",
        owner_id="m15_episode_ledger",
        source_kind="episodic",
        source_ref="m15_src",
        layer="B_per_turn_commitment",
        observable="expectation_outcome_match",
        observable_payload={"episode_id": episode_id},
        target={},
        due_at=None,
        priority=0.5,
        confidence=0.5,
        evidence_refs=evidence_refs,
        created_turn=0,
        created_at="2026-06-06T00:00:00Z",
        reason_codes=("memory_dynamics_guidance",),
        engineering_proxy_label="mvp_local_m15_episode",
    )


# === M13.2 band check adapter ===========================================


def test_m13_2_band_check_thin_adapter_emits_active_commitment_settled() -> None:
    """M20.3 §5.1: the adapter emits a SettledValue (i.e. an
    `ActiveCommitmentSettled` event after the caller wraps it).
    """
    adapter = M13BandCheckAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "m13_pending_1",
                "prediction_error_proxy": 0.05,  # within band
                "outcome_band": "positive",
                "evidence_refs": ["m13_evidence_1"],
            }
        ],
    }
    result = adapter.settle(_m13_commitment(), observation)
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    event = build_active_commitment_settled_event(result)
    assert event["type"] == "ActiveCommitmentSettled"
    assert event["commit_id"] == "cid_m13"
    assert event["outcome"] == "confirmed"


def test_m13_2_band_check_thin_adapter_agrees_with_existing_event() -> None:
    """M20.3 §5.1: the adapter's outcome MUST agree with the
    existing M13.2 `outcome_band`. `positive` ↔ `confirmed`,
    `negative` ↔ `violated`, `uncertain` is rare: it implies the
    M13.2 system returned `outcome_band = "uncertain"` while the
    band check saw a within-band proxy, which the adapter
    surfaces as disagreement. The existing signal is treated as
    authoritative for the M20.1 outcome.
    """
    adapter = M13BandCheckAdapter()
    cases = [
        # (outcome_band, proxy_value, expected_outcome, agreement)
        # Agreement cases:
        ("positive", 0.0, "confirmed", True),
        ("negative", 0.3, "violated", True),
        # Disagreement cases (existing signal is authoritative):
        # existing "positive" → "confirmed", regardless of band check.
        ("positive", 0.3, "confirmed", False),
        # existing "negative" → "violated", regardless of band check.
        ("negative", 0.0, "violated", False),
    ]
    for outcome_band, proxy_value, expected, agreed in cases:
        observation = {
            "now": "2026-06-06T00:00:00Z",
            "turn_index": 0,
            "m13_reward_settlements": [
                {
                    "pending_id": "m13_pending_1",
                    "prediction_error_proxy": proxy_value,
                    "outcome_band": outcome_band,
                    "evidence_refs": ["m13_evidence_1"],
                }
            ],
        }
        result = adapter.settle(_m13_commitment(), observation)
        assert isinstance(result, SettledValue)
        assert result.outcome == expected, (
            f"outcome_band={outcome_band!r} expected {expected!r}, got {result.outcome!r}"
        )
        if agreed:
            assert "m13_2_agreement" in result.reason_codes
        else:
            assert "m13_2_disagreement" in result.reason_codes


def test_m13_2_band_check_thin_adapter_handles_no_eligible_observation() -> None:
    """No matching row → NoSettlement."""
    adapter = M13BandCheckAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "other_pending",
                "prediction_error_proxy": 0.0,
                "outcome_band": "positive",
            }
        ],
    }
    result = adapter.settle(_m13_commitment(pending_id="m13_pending_1"), observation)
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "no_eligible_observation"


def test_m13_2_band_check_thin_adapter_handles_invalid_response() -> None:
    """Unknown outcome_band → NoSettlement with settler_llm_invalid_response."""
    adapter = M13BandCheckAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "m13_pending_1",
                "prediction_error_proxy": 0.0,
                "outcome_band": "unknown_band",
            }
        ],
    }
    result = adapter.settle(_m13_commitment(), observation)
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "settler_llm_invalid_response"


# === M15.0 episode aggregation adapter ==================================


def test_m15_0_episode_aggregation_thin_adapter_emits_active_commitment_settled() -> None:
    adapter = M15EpisodeAggregationAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m15_episode_settlements": [
            {
                "episode_id": "m15_episode_1",
                "outcome_summary": "settled",
                "delta_fe_proxy": 0.1,
                "evidence_refs": ["m15_evidence_1"],
            }
        ],
    }
    result = adapter.settle(_m15_commitment(), observation)
    assert isinstance(result, SettledValue)
    assert result.outcome == "confirmed"
    event = build_active_commitment_settled_event(result)
    assert event["type"] == "ActiveCommitmentSettled"
    assert event["commit_id"] == "cid_m15"
    assert event["outcome"] == "confirmed"


def test_m15_0_episode_aggregation_thin_adapter_agrees_with_existing_event() -> None:
    """M20.3 §5.1: the adapter's outcome MUST agree with the
    existing M15.0 `outcome_summary`. `settled` ↔ `confirmed`,
    `violated` ↔ `violated`, `uncertain` / `ignored` ↔ `uncertain`.
    """
    adapter = M15EpisodeAggregationAdapter()
    cases = [
        ("settled", 0.1, "confirmed"),
        ("violated", -0.2, "violated"),
        ("uncertain", 0.0, "uncertain"),
        ("ignored", 0.0, "uncertain"),
    ]
    for outcome_summary, delta, expected in cases:
        observation = {
            "now": "2026-06-06T00:00:00Z",
            "turn_index": 0,
            "m15_episode_settlements": [
                {
                    "episode_id": "m15_episode_1",
                    "outcome_summary": outcome_summary,
                    "delta_fe_proxy": delta,
                    "evidence_refs": ["m15_evidence_1"],
                }
            ],
        }
        result = adapter.settle(_m15_commitment(), observation)
        assert isinstance(result, SettledValue)
        assert result.outcome == expected
        assert "m15_0_agreement" in result.reason_codes


def test_m15_0_episode_aggregation_thin_adapter_handles_no_eligible_observation() -> None:
    adapter = M15EpisodeAggregationAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m15_episode_settlements": [
            {
                "episode_id": "other_episode",
                "outcome_summary": "settled",
                "delta_fe_proxy": 0.1,
            }
        ],
    }
    result = adapter.settle(_m15_commitment(episode_id="m15_episode_1"), observation)
    assert isinstance(result, NoSettlement)
    assert result.reason_code == "no_eligible_observation"


# === migration does not alter observable behavior ======================


def test_migration_does_not_alter_observable_behavior() -> None:
    """M20.3 §5: the thin adapters are read-only bridges. They
    do not mutate any state bucket; they return a fresh
    `SettledValue` per call. Verify by running the same
    commitment through M13BandCheckAdapter twice and asserting
    the settler returns identical `SettledValue` shapes (no
    state drift).
    """
    adapter = M13BandCheckAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "m13_pending_1",
                "prediction_error_proxy": 0.0,
                "outcome_band": "positive",
                "evidence_refs": ["m13_evidence_1"],
            }
        ],
    }
    a = adapter.settle(_m13_commitment(), observation)
    b = adapter.settle(_m13_commitment(), observation)
    assert isinstance(a, SettledValue)
    assert isinstance(b, SettledValue)
    assert a.outcome == b.outcome
    assert a.magnitude == b.magnitude


# === migration emits both audit events in order ========================


def test_migration_emits_active_commitment_settled_envelope() -> None:
    """M20.3 §5.1: the adapter's SettledValue, when wrapped via
    `build_active_commitment_settled_event`, produces a valid
    `ActiveCommitmentSettled` audit envelope.
    """
    adapter = M13BandCheckAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "m13_pending_1",
                "prediction_error_proxy": 0.2,
                "outcome_band": "negative",
                "evidence_refs": ["m13_evidence_1"],
            }
        ],
    }
    result = adapter.settle(_m13_commitment(), observation)
    assert isinstance(result, SettledValue)
    event = build_active_commitment_settled_event(result)
    # Required fields per M20.1 §10.
    assert event["type"] == "ActiveCommitmentSettled"
    assert event["commit_id"] == result.commit_id
    assert event["outcome"] == result.outcome
    assert event["magnitude"] == result.magnitude
    assert event["settler_type"] == result.settler_type
    assert event["engineering_proxy_label"] == result.engineering_proxy_label
    assert event["at"] == result.at
    assert event["turn_index"] == result.turn_index


# === magnitude defaults =================================================


def test_m13_2_band_check_magnitude_clamped() -> None:
    """M13.2 `prediction_error_proxy` is unbounded; the adapter
    clamps magnitude to [0, 1].
    """
    adapter = M13BandCheckAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m13_reward_settlements": [
            {
                "pending_id": "m13_pending_1",
                "prediction_error_proxy": 5.0,  # very large
                "outcome_band": "negative",
                "evidence_refs": ["m13_evidence_1"],
            }
        ],
    }
    result = adapter.settle(_m13_commitment(), observation)
    assert isinstance(result, SettledValue)
    assert 0.0 <= result.magnitude <= 1.0


def test_m15_0_episode_aggregation_magnitude_clamped() -> None:
    adapter = M15EpisodeAggregationAdapter()
    observation = {
        "now": "2026-06-06T00:00:00Z",
        "turn_index": 0,
        "m15_episode_settlements": [
            {
                "episode_id": "m15_episode_1",
                "outcome_summary": "settled",
                "delta_fe_proxy": 2.0,  # very large
                "evidence_refs": ["m15_evidence_1"],
            }
        ],
    }
    result = adapter.settle(_m15_commitment(), observation)
    assert isinstance(result, SettledValue)
    assert 0.0 <= result.magnitude <= 1.0


# === settler type and engineering proxy label ===========================


def test_m13_2_band_check_uses_deterministic_settler_type() -> None:
    adapter = M13BandCheckAdapter()
    assert adapter.SETTLER_TYPE == "deterministic"
    assert adapter.ENGINEERING_PROXY_LABEL == "mvp_local_m13_drive"


def test_m15_0_episode_aggregation_uses_deterministic_settler_type() -> None:
    adapter = M15EpisodeAggregationAdapter()
    assert adapter.SETTLER_TYPE == "deterministic"
    assert adapter.ENGINEERING_PROXY_LABEL == "mvp_local_m15_episode"
