"""Tests for MetaCognitiveLayer (M2.7 Phase B)."""

from segmentum.metacognitive import (
    META_PE_THRESHOLD,
    PATTERN_DETECTION_MIN_FREQUENCY,
    MetaCognitiveLayer,
)


def _feed_suppress_records(meta, n=20, channel="self_worth"):
    """Feed n suppress manipulation records into the layer."""
    for i in range(n):
        meta.observe_precision_pattern({
            "cycle": i,
            "type": "suppress",
            "target": channel,
            "intensity": 0.5,
            "precision_debt": 0.1 * (i + 1),
        })
        meta.observe_strategy_pattern({
            "cycle": i,
            "strategy": "suppress",
            "channel": channel,
            "pe_magnitude": 0.4,
        })


def test_detects_repeated_suppress_pattern():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=15)
    obs = meta.observe_precision_pattern({
        "cycle": 15, "type": "suppress", "target": "self_worth",
        "intensity": 0.5, "precision_debt": 1.6,
    })
    assert obs.suppression_frequency >= PATTERN_DETECTION_MIN_FREQUENCY
    assert "self_worth" in obs.chronic_suppression_channels


def test_meta_pe_nonzero_after_suppress_pattern():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=20)
    meta_pe = meta.compute_meta_prediction_error()
    assert meta_pe.magnitude > 0.0
    assert meta_pe.dominant_maladaptive_pattern == "chronic_suppression"


def test_meta_pe_exceeds_threshold_triggers_dissociation():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=25)
    meta_pe = meta.compute_meta_prediction_error()
    assert meta_pe.magnitude >= META_PE_THRESHOLD
    signal = meta.generate_dissociation_signal(meta_pe)
    assert signal is not None
    assert signal.strength > 0
    assert signal.suppress_efe_penalty > 0
    assert signal.belief_variance_boost > 0


def test_dissociation_increases_suppress_efe_penalty():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=25)
    meta_pe = meta.compute_meta_prediction_error()
    signal = meta.generate_dissociation_signal(meta_pe)
    assert signal is not None
    assert signal.suppress_efe_penalty > 0


def test_dissociation_loosens_belief_variance():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=25)
    meta_pe = meta.compute_meta_prediction_error()
    signal = meta.generate_dissociation_signal(meta_pe)
    assert signal is not None
    assert signal.belief_variance_boost > 0


def test_no_dissociation_when_balanced():
    meta = MetaCognitiveLayer()
    # Feed balanced strategy mix
    strategies = ["accommodate", "assimilate", "suppress", "redirect"]
    for i in range(20):
        meta.observe_strategy_pattern({
            "cycle": i,
            "strategy": strategies[i % 4],
            "channel": "self_worth",
            "pe_magnitude": 0.3,
        })
    meta_pe = meta.compute_meta_prediction_error()
    signal = meta.generate_dissociation_signal(meta_pe)
    assert signal is None or signal.strength < 0.1


def test_meta_beliefs_updated():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=15)
    meta_pe = meta.compute_meta_prediction_error()
    meta.update_meta_beliefs(meta_pe=meta_pe)
    assert "meta_pe_magnitude" in meta.meta_beliefs
    assert "maladaptive_pattern" in meta.meta_beliefs
    assert meta.meta_beliefs["maladaptive_pattern"] == "chronic_suppression"


def test_observe_cycle_integration():
    meta = MetaCognitiveLayer()
    manip_records = [
        {"cycle": 0, "type": "suppress", "target": "self_worth",
         "intensity": 0.5, "precision_debt": 0.1}
    ]
    strat_records = [
        {"cycle": 0, "strategy": "suppress", "channel": "self_worth",
         "pe_magnitude": 0.4}
    ]
    prec_obs, strat_obs, meta_pe, diss = meta.observe_cycle(
        manip_records, strat_records
    )
    assert prec_obs is not None
    assert strat_obs is not None
    assert meta_pe is not None


def test_dissociation_decays_without_pattern():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=25)
    meta_pe = meta.compute_meta_prediction_error()
    meta.generate_dissociation_signal(meta_pe)
    initial_diss = meta.dissociation_level
    assert initial_diss > 0

    # Feed accommodate records to break pattern
    for i in range(30):
        meta.observe_strategy_pattern({
            "cycle": 25 + i,
            "strategy": "accommodate",
            "channel": "self_worth",
            "pe_magnitude": 0.3,
        })
    meta_pe = meta.compute_meta_prediction_error()
    meta.generate_dissociation_signal(meta_pe)
    # Dissociation should have decayed
    assert meta.dissociation_level < initial_diss


def test_serialization_round_trip():
    meta = MetaCognitiveLayer()
    _feed_suppress_records(meta, n=10)
    meta.compute_meta_prediction_error()
    meta.dissociation_level = 0.35

    d = meta.to_dict()
    meta2 = MetaCognitiveLayer.from_dict(d)

    assert meta2.dissociation_level == meta.dissociation_level
    assert len(meta2.strategy_history) == len(meta.strategy_history)
    assert len(meta2.precision_history) == len(meta.precision_history)
    assert meta2.enabled == meta.enabled
