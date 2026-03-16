"""Tests for DefenseStrategySelector (M2.7 Phase A)."""

from segmentum.defense_strategy import (
    DefenseStrategy,
    DefenseStrategySelector,
    IdentityPE,
    StrategyEvaluation,
    StrategyOutcome,
)
from segmentum.precision_manipulation import PrecisionManipulator


def _make_selector(**kwargs):
    pm = PrecisionManipulator()
    return DefenseStrategySelector(pm, **kwargs)


def _make_pe(magnitude=0.5, channel="self_worth", valence=-0.3):
    return IdentityPE(
        source_channel=channel,
        magnitude=magnitude,
        valence=valence,
        current_belief_mean=0.25,
        current_belief_variance=0.04,
    )


def test_evaluate_returns_four_strategies():
    sel = _make_selector()
    evals = sel.evaluate_strategies(_make_pe())
    assert len(evals) == 4
    strategies = {e.strategy for e in evals}
    assert strategies == {"accommodate", "assimilate", "suppress", "redirect"}


def test_strategies_sorted_by_efe():
    sel = _make_selector()
    evals = sel.evaluate_strategies(_make_pe())
    effective = [e.efe_total - e.personality_bias for e in evals]
    assert effective == sorted(effective)


def test_high_neuroticism_favors_suppress():
    sel = _make_selector(neuroticism=0.9, openness=0.2)
    evals = sel.evaluate_strategies(_make_pe())
    strategy, _ = sel.select_strategy(evals)
    assert strategy is DefenseStrategy.SUPPRESS


def test_high_openness_favors_accommodate():
    sel = _make_selector(openness=0.95, neuroticism=0.1)
    evals = sel.evaluate_strategies(_make_pe(magnitude=0.3))
    strategy, _ = sel.select_strategy(evals)
    assert strategy is DefenseStrategy.ACCOMMODATE


def test_dissociation_raises_suppress_cost():
    sel = _make_selector(neuroticism=0.8)
    pe = _make_pe()

    evals_no_diss = sel.evaluate_strategies(pe, dissociation_level=0.0)
    evals_high_diss = sel.evaluate_strategies(pe, dissociation_level=0.8)

    suppress_efe_no = next(e for e in evals_no_diss if e.strategy == "suppress")
    suppress_efe_hi = next(e for e in evals_high_diss if e.strategy == "suppress")
    assert suppress_efe_hi.efe_short_term > suppress_efe_no.efe_short_term


def test_precision_debt_raises_suppress_long_term_cost():
    sel = _make_selector()
    pe = _make_pe()

    evals_no_debt = sel.evaluate_strategies(pe, precision_debt=0.0)
    evals_hi_debt = sel.evaluate_strategies(pe, precision_debt=2.0)

    supp_no = next(e for e in evals_no_debt if e.strategy == "suppress")
    supp_hi = next(e for e in evals_hi_debt if e.strategy == "suppress")
    assert supp_hi.efe_long_term > supp_no.efe_long_term


def test_execute_accommodate_shifts_belief():
    sel = _make_selector()
    pe = _make_pe(valence=0.5)
    outcome = sel.execute_strategy(DefenseStrategy.ACCOMMODATE, pe)
    assert "self_worth" in outcome.belief_changes
    assert outcome.belief_changes["self_worth"] != 0.0
    assert outcome.long_term_cost == 0.0


def test_execute_suppress_changes_precision():
    sel = _make_selector()
    pe = _make_pe()
    outcome = sel.execute_strategy(DefenseStrategy.SUPPRESS, pe)
    assert "self_worth" in outcome.precision_changes
    assert outcome.precision_changes["self_worth"] < 0
    assert outcome.manipulation_result is not None


def test_execute_redirect_uses_alternative_channel():
    sel = _make_selector()
    pe = _make_pe(channel="attachment")
    outcome = sel.execute_strategy(DefenseStrategy.REDIRECT, pe)
    assert "attachment" in outcome.precision_changes
    assert outcome.manipulation_result is not None


def test_strategy_history_recorded():
    sel = _make_selector()
    pe = _make_pe()
    sel.execute_strategy(DefenseStrategy.ACCOMMODATE, pe, cycle=5)
    assert len(sel.strategy_history) == 1
    assert sel.strategy_history[0]["cycle"] == 5
    assert sel.strategy_history[0]["strategy"] == "accommodate"


def test_serialization_round_trip():
    pm = PrecisionManipulator()
    sel = DefenseStrategySelector(pm, neuroticism=0.7, openness=0.3)
    sel.execute_strategy(DefenseStrategy.SUPPRESS, _make_pe(), cycle=1)

    d = sel.to_dict()
    sel2 = DefenseStrategySelector.from_dict(d, pm)

    assert sel2.neuroticism == sel.neuroticism
    assert sel2.openness == sel.openness
    assert len(sel2.strategy_history) == 1
