"""Tests for PrecisionManipulator (M2.7 Phase A)."""

from segmentum.precision_manipulation import (
    DEFAULT_SUPPRESS_INTENSITY,
    FreeEnergyCost,
    ManipulationType,
    PRECISION_DEBT_ACCUMULATION,
    PRECISION_DEBT_DECAY,
    PrecisionManipulator,
    PrecisionManipulationResult,
)


def test_initial_channel_precisions_are_one():
    pm = PrecisionManipulator()
    for ch, prec in pm.channel_precisions.items():
        assert prec == 1.0, f"{ch} should start at 1.0"


def test_personality_modulated_precisions():
    pm = PrecisionManipulator(neuroticism=0.9, openness=0.8, trust_prior=0.5)
    precs = pm.compute_channel_precisions()
    # High neuroticism → danger precision above baseline
    assert precs["danger"] > 1.0
    assert precs["threat"] > 1.0
    # High openness → novelty up
    assert precs["novelty"] > 1.0
    # Positive trust → social/attachment up
    assert precs["attachment"] > 1.0


def test_suppress_lowers_precision():
    pm = PrecisionManipulator()
    before = pm.channel_precisions["self_worth"]
    result = pm.apply_manipulation("suppress", "self_worth", 1.0)
    assert result.precision_after < before
    assert pm.channel_precisions["self_worth"] == result.precision_after


def test_suppress_accumulates_precision_debt():
    pm = PrecisionManipulator()
    assert pm.precision_debt == 0.0
    pm.apply_manipulation("suppress", "self_worth", 1.0)
    assert pm.precision_debt > 0.0
    first_debt = pm.precision_debt
    pm.apply_manipulation("suppress", "self_worth", 1.0)
    assert pm.precision_debt > first_debt


def test_precision_debt_grows_with_repeated_suppress():
    pm = PrecisionManipulator()
    for _ in range(10):
        pm.apply_manipulation("suppress", "self_worth", 0.8)
    assert pm.precision_debt > PRECISION_DEBT_ACCUMULATION * 5  # accumulated


def test_amplify_raises_precision():
    pm = PrecisionManipulator()
    before = pm.channel_precisions["danger"]
    result = pm.apply_manipulation("amplify", "danger", 0.8)
    assert result.precision_after > before


def test_redirect_transfers_precision():
    pm = PrecisionManipulator()
    before_source = pm.channel_precisions["attachment"]
    before_target = pm.channel_precisions["novelty"]
    result = pm.apply_manipulation(
        "redirect", "novelty", 0.8, source_channel="attachment"
    )
    assert pm.channel_precisions["attachment"] < before_source
    assert pm.channel_precisions["novelty"] > before_target


def test_reframe_slightly_lowers_precision():
    pm = PrecisionManipulator()
    before = pm.channel_precisions["self_worth"]
    result = pm.apply_manipulation("reframe", "self_worth", 0.5)
    assert result.precision_after < before
    # Reframe is gentler than suppress
    assert result.precision_after > pm.channel_precisions.get("self_worth", 0) - 0.2


def test_manipulation_cost_suppress_cheapest_short_term():
    pm = PrecisionManipulator()
    suppress_cost = pm.compute_manipulation_cost("suppress", 0.5)
    amplify_cost = pm.compute_manipulation_cost("amplify", 0.5)
    assert suppress_cost.short_term < amplify_cost.short_term


def test_manipulation_cost_suppress_most_expensive_long_term():
    pm = PrecisionManipulator()
    # Build up some debt first
    for _ in range(5):
        pm.apply_manipulation("suppress", "self_worth", 0.5)
    suppress_cost = pm.compute_manipulation_cost("suppress", 0.5)
    redirect_cost = pm.compute_manipulation_cost("redirect", 0.5)
    assert suppress_cost.long_term_projected > redirect_cost.long_term_projected


def test_precision_debt_decays():
    pm = PrecisionManipulator()
    pm.apply_manipulation("suppress", "self_worth", 1.0)
    debt_before = pm.precision_debt
    pm.decay_precision_debt()
    assert pm.precision_debt < debt_before
    assert pm.precision_debt == debt_before * PRECISION_DEBT_DECAY


def test_precision_floor():
    pm = PrecisionManipulator()
    # Many suppress calls should not drive precision below 0.05
    for _ in range(50):
        pm.apply_manipulation("suppress", "self_worth", 1.0)
    assert pm.channel_precisions["self_worth"] >= 0.05


def test_precision_ceiling():
    pm = PrecisionManipulator()
    for _ in range(50):
        pm.apply_manipulation("amplify", "danger", 1.0)
    assert pm.channel_precisions["danger"] <= 2.0


def test_manipulation_history_recorded():
    pm = PrecisionManipulator()
    pm.apply_manipulation("suppress", "social", 0.5, cycle=3)
    assert len(pm.manipulation_history) == 1
    assert pm.manipulation_history[0]["cycle"] == 3
    assert pm.manipulation_history[0]["type"] == "suppress"


def test_serialization_round_trip():
    pm = PrecisionManipulator(neuroticism=0.8, trust_prior=-0.3)
    pm.apply_manipulation("suppress", "self_worth", 0.7, cycle=1)
    pm.apply_manipulation("amplify", "danger", 0.4, cycle=2)

    d = pm.to_dict()
    pm2 = PrecisionManipulator.from_dict(d)

    assert pm2.neuroticism == pm.neuroticism
    assert pm2.trust_prior == pm.trust_prior
    assert abs(pm2.precision_debt - pm.precision_debt) < 1e-6
    assert pm2.channel_precisions["self_worth"] == pm.channel_precisions["self_worth"]
    assert len(pm2.manipulation_history) == len(pm.manipulation_history)


def test_update_personality():
    pm = PrecisionManipulator(neuroticism=0.5)
    pm.update_personality(neuroticism=0.9, trust_prior=-0.5)
    assert pm.neuroticism == 0.9
    assert pm.trust_prior == -0.5
