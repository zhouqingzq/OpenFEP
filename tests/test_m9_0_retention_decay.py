"""M9.0 R2: Value-Based Retention and Decay Integration tests."""

import pytest

from segmentum.memory_dynamics import (
    RetentionPressure,
    compute_decay_state,
    compute_retention_pressure,
)


# ── R2.1: Retention pressure formula ─────────────────────────────────────

def test_retention_pressure_all_positive_components():
    """High-value components produce high retention pressure."""
    rp = compute_retention_pressure(
        identity_continuity_value=0.8,
        relationship_continuity_value=0.7,
        future_prediction_value=0.6,
        affective_salience=0.5,
        user_emphasis=0.9,
        confidence=0.9,
    )
    assert rp.total_pressure > 0.5
    assert rp.identity_continuity_value == 0.8
    assert rp.relationship_continuity_value == 0.7


def test_retention_pressure_penalties_reduce_total():
    """Penalties (privacy, contradiction, low confidence) reduce pressure."""
    rp = compute_retention_pressure(
        identity_continuity_value=0.5,
        privacy_or_safety_penalty=0.8,
        contradiction_penalty=0.6,
        confidence=0.4,
    )
    assert rp.total_pressure < 0.2
    assert rp.privacy_or_safety_penalty == 0.8


def test_retention_pressure_auto_derives_low_confidence_penalty():
    """Low confidence auto-derives a penalty when not explicitly set."""
    rp = compute_retention_pressure(
        identity_continuity_value=0.3,
        confidence=0.2,
    )
    assert rp.low_confidence_penalty > 0.0
    assert rp.total_pressure < 0.3


def test_retention_pressure_conflict_auto_penalty():
    """Conflict flag auto-derives contradiction penalty."""
    rp_no_conflict = compute_retention_pressure(
        identity_continuity_value=0.5,
        has_conflict=False,
    )
    rp_conflict = compute_retention_pressure(
        identity_continuity_value=0.5,
        has_conflict=True,
    )
    assert rp_conflict.contradiction_penalty > rp_no_conflict.contradiction_penalty
    assert rp_conflict.total_pressure < rp_no_conflict.total_pressure


def test_retention_pressure_decay_reason_logs_protection():
    """Decay reason records protection for high-value memories."""
    rp = compute_retention_pressure(
        identity_continuity_value=0.8,
        relationship_continuity_value=0.7,
        confidence=0.9,
    )
    assert "protected" in rp.decay_reason


def test_retention_pressure_decay_reason_logs_risks():
    """Decay reason records risk factors for low-value memories."""
    rp = compute_retention_pressure(
        confidence=0.2,
        has_conflict=True,
        privacy_or_safety_penalty=0.5,
    )
    assert "decay_candidate" in rp.decay_reason
    assert "low_confidence" in rp.decay_reason


def test_retention_pressure_to_dict():
    """to_dict() includes all component fields."""
    rp = compute_retention_pressure(
        identity_continuity_value=0.6,
        user_emphasis=0.8,
        confidence=0.9,
    )
    d = rp.to_dict()
    assert d["identity_continuity_value"] == 0.6
    assert d["user_emphasis"] == 0.8
    assert "total_pressure" in d
    assert "decay_reason" in d


# ── R2.2: Decay state computation ───────────────────────────────────────

def test_high_value_memory_decays_slower_than_low_value_memory():
    """High retention pressure extends the fresh/active window significantly."""
    high_rp = RetentionPressure(total_pressure=0.7)
    low_rp = RetentionPressure(total_pressure=-0.4)

    # Same age of 30 cycles
    age = 30

    high_state = compute_decay_state(
        retention_pressure=high_rp,
        last_access_cycles_ago=age,
        cycle=age,
        created_at_cycle=0,
    )
    low_state = compute_decay_state(
        retention_pressure=low_rp,
        last_access_cycles_ago=age,
        cycle=age,
        created_at_cycle=0,
    )

    # High-value should be fresher than low-value
    decay_order = {"fresh": 0, "active": 1, "fading": 2, "dormant": 3, "pruned": 4}
    assert decay_order[high_state] < decay_order[low_state], (
        f"High-value decay={high_state}, low-value decay={low_state} "
        f"— high-value should be fresher"
    )


def test_fresh_memory_stays_fresh_with_high_pressure():
    """High retention pressure keeps memory in 'fresh' state longer."""
    rp = RetentionPressure(total_pressure=0.8)
    # Age 10 would normally be "active" but high pressure extends fresh
    state = compute_decay_state(
        retention_pressure=rp,
        cycle=10, created_at_cycle=0,
        max_cycles_fresh=5,
    )
    # With pressure 0.8, max_fresh = 5*3 = 15, so age 10 is still fresh
    assert state == "fresh"


def test_low_value_memory_prunes_early():
    """Negative retention pressure causes early pruning."""
    rp = RetentionPressure(total_pressure=-0.5)
    state = compute_decay_state(
        retention_pressure=rp,
        cycle=60, created_at_cycle=0,
        max_cycles_fading=50,
    )
    assert state == "pruned"


def test_high_value_memory_preserved_even_when_old():
    """High-value memory goes dormant instead of pruned when old."""
    high_rp = RetentionPressure(total_pressure=0.7)
    low_rp = RetentionPressure(total_pressure=0.0)

    # At age 200, high-value memory is dormant (extended fading window);
    # low-value memory is pruned.
    state_high = compute_decay_state(
        retention_pressure=high_rp,
        cycle=200, created_at_cycle=0,
        max_cycles_fading=50,
    )
    state_low = compute_decay_state(
        retention_pressure=low_rp,
        cycle=200, created_at_cycle=0,
        max_cycles_fading=50,
    )

    assert state_high == "dormant", f"Expected dormant, got {state_high}"
    assert state_low == "pruned", f"Expected pruned, got {state_low}"


def test_frequent_access_keeps_memory_fresh():
    """Frequently accessed memories stay fresh longer."""
    rp = RetentionPressure(total_pressure=0.1)

    # access_frequency=5 doubles max_fresh from 5 to 10; age=8 fits within
    frequent = compute_decay_state(
        retention_pressure=rp,
        cycle=8, created_at_cycle=0,
        access_frequency=5,
        max_cycles_fresh=5,
    )
    # access_frequency=1 gets normal max_fresh=5; age=8 > 5 so it moves past fresh
    infrequent = compute_decay_state(
        retention_pressure=rp,
        cycle=8, created_at_cycle=0,
        access_frequency=1,
        max_cycles_fresh=5,
    )

    assert frequent == "fresh", f"Expected fresh, got {frequent}"
    assert infrequent != "fresh", f"Expected not fresh, got {infrequent}"


def test_decay_state_progression_follows_expected_order():
    """As age increases, decay state progresses fresh -> active -> fading -> pruned."""
    rp = RetentionPressure(total_pressure=0.0)

    s0 = compute_decay_state(retention_pressure=rp, cycle=0, created_at_cycle=0)
    s1 = compute_decay_state(retention_pressure=rp, cycle=10, created_at_cycle=0,
                             max_cycles_fresh=5, max_cycles_active=15)
    s2 = compute_decay_state(retention_pressure=rp, cycle=30, created_at_cycle=0,
                             max_cycles_fresh=5, max_cycles_active=15, max_cycles_fading=40)
    s3 = compute_decay_state(retention_pressure=rp, cycle=60, created_at_cycle=0,
                             max_cycles_fresh=5, max_cycles_active=15, max_cycles_fading=40)

    assert s0 == "fresh"
    assert s1 == "active"
    assert s2 == "fading"
    assert s3 == "pruned"
