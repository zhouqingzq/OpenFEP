from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from segmentum.predictive_coding import (
    PredictiveCodingHyperparameters,
    predictive_coding_profile,
    predictive_coding_profile_names,
)
from segmentum.runtime import SegmentRuntime


def _default_state_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "segment_v0_1_state.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Project Segmentum prototype.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=int(os.getenv("SEGMENTUM_CYCLES", "20")),
        help="Number of unified runtime cycles to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEGMENTUM_SEED", "17")),
        help="Seed for the toy world when creating a new state.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=_default_state_path(),
        help="JSON snapshot path for the persistent prototype state.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Ignore any existing prototype state snapshot and start fresh.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optional JSONL path for per-cycle structured trace output.",
    )
    parser.add_argument(
        "--tick-seconds",
        type=float,
        default=float(os.getenv("SEGMENTUM_TICK_SECONDS", "0.0")),
        help="Optional delay between runtime cycles.",
    )
    parser.add_argument(
        "--host-telemetry",
        action="store_true",
        help="Sample host telemetry and append inner speech during each runtime cycle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cycle prototype logs and print only the final summary.",
    )
    parser.add_argument(
        "--precision-profile",
        choices=predictive_coding_profile_names(),
        default=os.getenv("SEGMENTUM_PRECISION_PROFILE", "balanced"),
        help="Named predictive-coding hyperparameter profile.",
    )
    parser.add_argument(
        "--predictive-config",
        type=Path,
        default=None,
        help="Optional JSON file overriding layer precision and digestion hyperparameters.",
    )
    parser.add_argument(
        "--reset-predictive-precisions",
        action="store_true",
        help="When loading a snapshot, reset dynamic precisions to the chosen profile/config defaults.",
    )
    return parser


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Project Segmentum requires Python 3.11+.")

    args = _build_parser().parse_args()
    predictive_hyperparameters = predictive_coding_profile(args.precision_profile)
    if args.predictive_config is not None:
        payload = json.loads(args.predictive_config.read_text(encoding="utf-8"))
        predictive_hyperparameters = PredictiveCodingHyperparameters.from_dict(
            payload,
            default=predictive_hyperparameters,
        )
    runtime = SegmentRuntime.load_or_create(
        state_path=args.state_path,
        trace_path=args.trace_path,
        seed=args.seed,
        reset=args.reset_state,
        predictive_hyperparameters=predictive_hyperparameters,
        reset_predictive_precisions=args.reset_predictive_precisions,
    )
    runtime.run(
        cycles=args.cycles,
        verbose=not args.quiet,
        host_telemetry=args.host_telemetry,
        tick_interval_seconds=args.tick_seconds,
    )


if __name__ == "__main__":
    main()

