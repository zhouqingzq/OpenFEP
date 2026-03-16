from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m29_benchmarks import write_m29_artifacts


def main(*, rollout_seed: int = 42, rollout_cycles: int = 60) -> None:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    written = write_m29_artifacts(
        artifacts_dir,
        rollout_seed=rollout_seed,
        rollout_cycles=rollout_cycles,
    )
    print(written["transfer_benchmark"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--rollout-cycles", type=int, default=60)
    args = parser.parse_args()
    main(
        rollout_seed=args.rollout_seed,
        rollout_cycles=args.rollout_cycles,
    )
