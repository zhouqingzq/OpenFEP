from __future__ import annotations

from segmentum.action_schema import ActionSchema
from segmentum.memory import Episode


def test_episode_parameterized_round_trip() -> None:
    episode = Episode(
        timestamp=7,
        state_vector={"obs_food": 0.2},
        action_taken=ActionSchema(
            name="web_fetch",
            params={"timeout": 30, "url": "https://example.com"},
        ),
        outcome_state={"free_energy_drop": 0.1},
        predicted_outcome="neutral",
        prediction_error=0.2,
        risk=0.3,
        value_score=0.4,
        total_surprise=0.5,
        embedding=[0.2],
        preferred_probability=0.7,
        preference_log_value=-0.1,
    )

    restored = Episode.from_dict(episode.to_dict())

    assert restored.action_taken.name == "web_fetch"
    assert restored.action_taken.params["url"] == "https://example.com"
