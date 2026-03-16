from __future__ import annotations

from pathlib import Path

from segmentum.narrative_world import NarrativeWorld, NarrativeWorldConfig
from segmentum.runtime import SegmentRuntime


WORLD_DIR = Path(__file__).resolve().parent.parent / "data" / "worlds"


def test_world_config_load_and_round_trip() -> None:
    world = SegmentRuntime.load_world(WORLD_DIR / "predator_river.json", seed=17)
    payload = world.to_dict()
    restored = NarrativeWorld.from_dict(payload)

    assert restored.config.world_id == "predator_river"
    assert restored.to_dict()["config"]["baseline_observation"]["danger"] == payload["config"]["baseline_observation"]["danger"]


def test_world_observation_and_events_are_deterministic() -> None:
    first = SegmentRuntime.load_world(WORLD_DIR / "foraging_valley.json", seed=11)
    second = SegmentRuntime.load_world(WORLD_DIR / "foraging_valley.json", seed=11)

    obs_a = first.observe(8)
    obs_b = second.observe(8)
    episodes_a = [episode.to_dict() for episode in first.narrative_episodes(8)]
    episodes_b = [episode.to_dict() for episode in second.narrative_episodes(8)]

    assert obs_a == obs_b
    assert episodes_a == episodes_b
    assert len(episodes_a) == 1


def test_runtime_run_world_episode_produces_rollout_summary() -> None:
    runtime = SegmentRuntime()
    world = SegmentRuntime.load_world(WORLD_DIR / "social_shelter.json", seed=23)

    summary = runtime.run_world_episode(world=world, cycles=24, ingest_events=True)

    assert summary["world_id"] == "social_shelter"
    assert summary["ticks"] == 24
    assert summary["event_count"] >= 1
    assert "rest" in summary["action_distribution"] or "seek_contact" in summary["action_distribution"]
