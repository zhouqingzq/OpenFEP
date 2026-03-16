from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m210_benchmarks import ARTIFACTS_DIR, run_longitudinal_stability, write_json


def main(*, seed: int, cycles_per_world: int, repeats: int) -> None:
    payload = run_longitudinal_stability(
        seed=seed,
        cycles_per_world=cycles_per_world,
        repeats=repeats,
    )
    write_json(ARTIFACTS_DIR / "m210_longitudinal_stability.json", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--cycles-per-world", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    main(
        seed=arguments.seed,
        cycles_per_world=arguments.cycles_per_world,
        repeats=arguments.repeats,
    )
