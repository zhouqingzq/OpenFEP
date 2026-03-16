"""Tests for VIAProjection (M2.7 Phase A)."""

from segmentum.via_projection import VIA_STRENGTHS, VIAProfile, VIAProjection


def test_project_returns_24_strengths():
    proj = VIAProjection()
    profile = proj.project()
    assert len(profile.strengths) == len(VIA_STRENGTHS)
    for strength in VIA_STRENGTHS:
        assert strength in profile.strengths


def test_all_strengths_in_unit_range():
    proj = VIAProjection()
    profile = proj.project(openness=0.9, neuroticism=0.1, extraversion=0.8)
    for name, val in profile.strengths.items():
        assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"


def test_high_openness_boosts_curiosity():
    proj = VIAProjection()
    hi = proj.project(openness=0.95)
    lo = proj.project(openness=0.1)
    assert hi.strengths["curiosity"] > lo.strengths["curiosity"]
    assert hi.strengths["creativity"] > lo.strengths["creativity"]


def test_high_neuroticism_reduces_bravery():
    proj = VIAProjection()
    hi_n = proj.project(neuroticism=0.9)
    lo_n = proj.project(neuroticism=0.1)
    assert lo_n.strengths["bravery"] > hi_n.strengths["bravery"]


def test_high_agreeableness_boosts_kindness():
    proj = VIAProjection()
    hi_a = proj.project(agreeableness=0.9)
    lo_a = proj.project(agreeableness=0.1)
    assert hi_a.strengths["kindness"] > lo_a.strengths["kindness"]


def test_meaning_construction_boosts_spirituality():
    proj = VIAProjection()
    hi = proj.project(meaning_construction_tendency=0.9)
    lo = proj.project(meaning_construction_tendency=0.1)
    assert hi.strengths["spirituality"] > lo.strengths["spirituality"]


def test_emotional_regulation_boosts_humor():
    proj = VIAProjection()
    hi = proj.project(emotional_regulation_style=0.9)
    lo = proj.project(emotional_regulation_style=0.1)
    assert hi.strengths["humor"] > lo.strengths["humor"]


def test_trust_prior_affects_love():
    proj = VIAProjection()
    hi = proj.project(trust_prior=0.8)
    lo = proj.project(trust_prior=-0.8)
    assert hi.strengths["love"] > lo.strengths["love"]
    assert hi.strengths["gratitude"] > lo.strengths["gratitude"]


def test_top_strengths():
    proj = VIAProjection()
    profile = proj.project(openness=0.9, extraversion=0.9)
    top = profile.top_strengths(3)
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_to_dict():
    proj = VIAProjection()
    profile = proj.project()
    d = profile.to_dict()
    assert isinstance(d, dict)
    assert len(d) == 24
    for v in d.values():
        assert isinstance(v, float)


def test_neutral_personality_moderate_strengths():
    """Neutral personality should have moderate (not extreme) strengths."""
    proj = VIAProjection()
    profile = proj.project()
    for name, val in profile.strengths.items():
        assert 0.2 < val < 0.8, f"Neutral {name}={val} seems extreme"
