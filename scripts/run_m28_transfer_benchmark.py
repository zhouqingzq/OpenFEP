from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m28_benchmarks import run_transfer_benchmark


def main(
    seed: int = 42,
    train_world: str = "predator_river",
    eval_worlds: list[str] | None = None,
    train_cycles: int = 60,
    eval_cycles: int = 40,
) -> None:
    result = run_transfer_benchmark(
        seed=seed,
        train_world=train_world,
        eval_worlds=eval_worlds or ["foraging_valley", "social_shelter"],
        train_cycles=train_cycles,
        eval_cycles=eval_cycles,
    )
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "m28_transfer_benchmark.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-world", type=str, default="predator_river")
    parser.add_argument("--eval-worlds", nargs="*", default=["foraging_valley", "social_shelter"])
    parser.add_argument("--train-cycles", type=int, default=60)
    parser.add_argument("--eval-cycles", type=int, default=40)
    args = parser.parse_args()
    main(
        seed=args.seed,
        train_world=args.train_world,
        eval_worlds=list(args.eval_worlds),
        train_cycles=args.train_cycles,
        eval_cycles=args.eval_cycles,
    )
