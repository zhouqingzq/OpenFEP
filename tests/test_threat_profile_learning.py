from __future__ import annotations

import random

from segmentum.agent import SegmentAgent
from segmentum.self_model import ThreatProfile


def test_threat_profile_backward_compatibility_and_learning() -> None:
    legacy = ThreatProfile.from_dict({"energy": {"critical_low": 0.05}})
    assert legacy.get("energy", {})["critical_low"] == 0.05

    agent = SegmentAgent(rng=random.Random(11))
    agent.self_model.threat_profile.add_learned_threat(
        pattern="forage in cluster 0 -> resource_loss",
        risk_level=0.8,
        tick=12,
    )

    payload = agent.self_model.to_dict()
    restored = ThreatProfile.from_dict(payload["threat_profile"])

    assert restored.learned_threats
    assert restored.hard_limits["energy"]["critical_low"] == 0.05
