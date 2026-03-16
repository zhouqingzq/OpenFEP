from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m210_benchmarks import ARTIFACTS_DIR, run_personality_validation, write_json


def main(*, seed: int, cycles_per_world: int, repeats: int) -> None:
    payload = run_personality_validation(
        seed=seed,
        cycles_per_world=cycles_per_world,
        repeats=repeats,
    )
    write_json(ARTIFACTS_DIR / "m210_personality_anova.json", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--cycles-per-world", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=4)
    arguments = parser.parse_args()
    main(
        seed=arguments.seed,
        cycles_per_world=arguments.cycles_per_world,
        repeats=arguments.repeats,
    )
