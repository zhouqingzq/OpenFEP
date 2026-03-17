from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m220_benchmarks import write_m220_acceptance_artifacts


def main() -> None:
    write_m220_acceptance_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
