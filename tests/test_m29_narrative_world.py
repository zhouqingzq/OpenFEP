from __future__ import annotations

from pathlib import Path

from segmentum.narrative_types import NarrativeEpisode
from segmentum.narrative_world import NarrativeWorld, NarrativeWorldConfig
from segmentum.runtime import SegmentRuntime


WORLD_DIR = Path(__file__).resolve().parent.parent / "data" / "worlds"


def test_m29_world_config_round_trip_is_stable() -> None:
    payload = {
        "world_id": "round_trip_world",
        "seed": 9,
        "baseline_observation": {"food": 0.7, "danger": 0.2},
        "drift_profile": {"food": 0.05, "danger": 0.02},
        "event_schedule": [
            {
                "tick": 3,
                "event_type": "storm",
                "observation_delta": {"danger": 0.1},
                "narrative_text": "A storm crosses the ridge.",
                "tags": ["hazard"],
            }
        ],
        "resource_rules": {"forage_energy_gain": 0.2},
        "hazard_rules": {"hide_danger_relief": 0.1},
        "social_rules": {"contact_social_gain": 0.2},
    }

    config = NarrativeWorldConfig.from_dict(payload)
    restored = NarrativeWorldConfig.from_dict(config.to_dict())

    assert restored.to_dict() == config.to_dict()


def test_m29_scheduled_events_and_episode_schema_are_deterministic() -> None:
    first = SegmentRuntime.load_world(WORLD_DIR / "foraging_valley.json", seed=11)
    second = SegmentRuntime.load_world(WORLD_DIR / "foraging_valley.json", seed=11)

    obs_a = first.observe(8)
    obs_b = second.observe(8)
    episodes_a = first.narrative_episodes(8)
    episodes_b = second.narrative_episodes(8)

    assert obs_a == obs_b
    assert [episode.to_dict() for episode in episodes_a] == [
        episode.to_dict() for episode in episodes_b
    ]
    assert len(episodes_a) == 1
    assert isinstance(episodes_a[0], NarrativeEpisode)
    assert episodes_a[0].metadata["world_id"] == "foraging_valley"
    assert episodes_a[0].metadata["event_type"] == "berry_patch"


def test_m29_action_feedback_is_deterministic_and_observation_is_clamped() -> None:
    first = SegmentRuntime.load_world(WORLD_DIR / "predator_river.json", seed=17)
    second = SegmentRuntime.load_world(WORLD_DIR / "predator_river.json", seed=17)

    first.observe(0)
    second.observe(0)
    first_feedback = first.apply_action("hide", 0)
    second_feedback = second.apply_action("hide", 0)
    next_obs = first.observe(1)

    assert first_feedback == second_feedback
    assert 0.0 <= next_obs.danger <= 1.0
    assert 0.0 <= next_obs.shelter <= 1.0


def test_m29_runtime_run_world_episode_ingests_events() -> None:
    runtime = SegmentRuntime()
    world = SegmentRuntime.load_world(WORLD_DIR / "social_shelter.json", seed=23)

    summary = runtime.run_world_episode(world=world, cycles=24, ingest_events=True)

    assert summary["world_id"] == "social_shelter"
    assert summary["ticks"] == 24
    assert summary["event_count"] >= 1
    assert summary["narrative_ingestion_count"] >= 1
    assert summary["agent_state"]["narrative_priors"]


def test_m29_world_state_round_trip_restores_feedback_schedule() -> None:
    world = SegmentRuntime.load_world(WORLD_DIR / "predator_river.json", seed=17)
    world.observe(0)
    world.apply_action("forage", 0)

    restored = NarrativeWorld.from_dict(world.to_dict())
    original_next = world.observe(1)
    restored_next = restored.observe(1)

    assert original_next == restored_next
