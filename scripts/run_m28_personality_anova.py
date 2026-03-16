from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m28_benchmarks import run_personality_anova


def main(seed: int = 42, cycles: int = 48, repeats: int = 3) -> None:
    result = run_personality_anova(seed=seed, cycles=cycles, repeats=repeats)
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "m28_personality_anova.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycles", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    main(seed=args.seed, cycles=args.cycles, repeats=args.repeats)
