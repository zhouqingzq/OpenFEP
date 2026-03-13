from __future__ import annotations

import random

from segmentum.agent import SegmentAgent


def test_error_attribution_distinguishes_self_world_and_ambiguous() -> None:
    agent = SegmentAgent(rng=random.Random(17))
    agent.energy = 0.01
    traced = agent.prediction_error_trace(
        {"energy": 0.1, "threat": 0.9},
        {"energy": 0.7, "threat": 0.2},
    )

    assert traced["energy"]["error_source"] == "self"
    assert traced["threat"]["error_source"] == "world"

    agent.energy = 0.5
    ambiguous = agent.prediction_error_trace({"energy": 0.2}, {"energy": 0.4})
    assert ambiguous["energy"]["error_source"] == "ambiguous"
