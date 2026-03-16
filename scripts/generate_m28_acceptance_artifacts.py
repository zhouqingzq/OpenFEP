from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m28_benchmarks import load_world, run_personality_anova, run_transfer_benchmark, run_world
from segmentum.runtime import SegmentRuntime


def generate_artifacts(*, seed: int = 42) -> None:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    for index, world_name in enumerate(("foraging_valley", "predator_river", "social_shelter")):
        result = run_world(world_name=world_name, seed=seed + index, cycles=60)
        output_path = os.path.join(artifacts_dir, f"m28_world_rollout_{world_name}.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "world_id": result["world_id"],
                    "ticks": result["cycles"],
                    "event_count": result["event_count"],
                    "action_distribution": result["action_distribution"],
                    "mean_conditioned_prediction_error": result["mean_conditioned_prediction_error"],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    from scripts.run_m28_attention_benchmark import run_attention_benchmark

    attention = run_attention_benchmark(seed=seed, cycles=80)
    with open(os.path.join(artifacts_dir, "m28_attention_on_off.json"), "w", encoding="utf-8") as handle:
        json.dump(attention, handle, indent=2, ensure_ascii=False)

    personality = run_personality_anova(seed=seed, cycles=48, repeats=3)
    with open(os.path.join(artifacts_dir, "m28_personality_anova.json"), "w", encoding="utf-8") as handle:
        json.dump(personality, handle, indent=2, ensure_ascii=False)

    transfer_one = run_transfer_benchmark(
        seed=seed,
        train_world="predator_river",
        eval_worlds=["foraging_valley"],
        train_cycles=60,
        eval_cycles=40,
    )
    transfer_two = run_transfer_benchmark(
        seed=seed + 1,
        train_world="foraging_valley",
        eval_worlds=["social_shelter"],
        train_cycles=60,
        eval_cycles=40,
    )
    transfer_payload = {
        "benchmarks": [transfer_one, transfer_two],
    }
    with open(os.path.join(artifacts_dir, "m28_transfer_benchmark.json"), "w", encoding="utf-8") as handle:
        json.dump(transfer_payload, handle, indent=2, ensure_ascii=False)


def main(seed: int = 42) -> None:
    generate_artifacts(seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
