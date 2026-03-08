from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentum.predictive_coding import (
    PredictiveCodingHyperparameters,
    predictive_coding_profile,
    predictive_coding_profile_names,
)
from segmentum.runtime import SegmentRuntime


LAYER_KEYS = (
    ("interoceptive", "interoceptive_update"),
    ("sensorimotor", "sensorimotor_update"),
    ("strategic", "strategic_update"),
)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _channel_mean(values: dict[str, float]) -> float:
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


def _run_profile(
    *,
    label: str,
    hyperparameters: PredictiveCodingHyperparameters,
    cycles: int,
    seed: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / f"{label}_state.json"
        trace_path = Path(tmp_dir) / f"{label}_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
            predictive_hyperparameters=hyperparameters,
            reset_predictive_precisions=True,
        )
        summary = runtime.run(cycles=cycles, verbose=False)
        trace_records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]

    hierarchy_stats: dict[str, dict[str, float]] = {}
    for label_key, trace_key in LAYER_KEYS:
        precision_means: list[float] = []
        residual_means: list[float] = []
        propagation_means: list[float] = []
        propagated_cycles = 0
        for record in trace_records:
            update = record["hierarchy"][trace_key]
            precision_means.append(_channel_mean(update["error_precision"]))
            residual_means.append(
                _mean([abs(value) for value in update["residual_error"].values()])
            )
            propagated = [abs(value) for value in update["propagated_error"].values()]
            propagation_means.append(_mean(propagated))
            if any(value > 0.0 for value in propagated):
                propagated_cycles += 1
        hierarchy_stats[label_key] = {
            "avg_precision": _mean(precision_means),
            "avg_residual_error": _mean(residual_means),
            "avg_propagated_error": _mean(propagation_means),
            "propagation_rate": propagated_cycles / max(len(trace_records), 1),
        }

    return {
        "profile": label,
        "cycles": cycles,
        "seed": seed,
        "hyperparameters": hyperparameters.to_dict(),
        "summary": summary,
        "hierarchy": hierarchy_stats,
    }


def _print_table(results: list[dict[str, object]]) -> None:
    print(
        "profile           avg_fe   entropy  dom_share  intero_prop  sensor_prop  strat_prop"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    for result in results:
        summary = result["summary"]
        hierarchy = result["hierarchy"]
        assert isinstance(summary, dict)
        assert isinstance(hierarchy, dict)
        print(
            f"{str(result['profile']):16}"
            f" {float(summary['avg_free_energy_after']):7.3f}"
            f" {float(summary['action_entropy']):8.3f}"
            f" {float(summary['dominant_action_share']):10.3f}"
            f" {float(hierarchy['interoceptive']['propagation_rate']):12.3f}"
            f" {float(hierarchy['sensorimotor']['propagation_rate']):12.3f}"
            f" {float(hierarchy['strategic']['propagation_rate']):11.3f}"
        )
    print()
    for result in results:
        hierarchy = result["hierarchy"]
        assert isinstance(hierarchy, dict)
        print(f"[{result['profile']}]")
        for layer_name in ("interoceptive", "sensorimotor", "strategic"):
            stats = hierarchy[layer_name]
            print(
                f"  {layer_name:14}"
                f" avg_precision={float(stats['avg_precision']):.3f}"
                f" avg_residual={float(stats['avg_residual_error']):.3f}"
                f" avg_propagated={float(stats['avg_propagated_error']):.3f}"
                f" propagation_rate={float(stats['propagation_rate']):.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare predictive-coding precision schedules across profiles."
    )
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=predictive_coding_profile_names(),
        default=["balanced", "high_precision", "low_precision", "hair_trigger"],
        help="Named predictive-coding profiles to compare.",
    )
    parser.add_argument(
        "--custom-config",
        type=Path,
        default=None,
        help="Optional JSON config to append as an extra comparison run.",
    )
    parser.add_argument(
        "--custom-label",
        type=str,
        default="custom",
        help="Label to use with --custom-config.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    args = parser.parse_args()

    results = [
        _run_profile(
            label=profile_name,
            hyperparameters=predictive_coding_profile(profile_name),
            cycles=args.cycles,
            seed=args.seed,
        )
        for profile_name in args.profiles
    ]
    if args.custom_config is not None:
        payload = json.loads(args.custom_config.read_text(encoding="utf-8"))
        hyperparameters = PredictiveCodingHyperparameters.from_dict(
            payload,
            default=predictive_coding_profile("balanced"),
        )
        results.append(
            _run_profile(
                label=args.custom_label,
                hyperparameters=hyperparameters,
                cycles=args.cycles,
                seed=args.seed,
            )
        )

    if args.format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=True, sort_keys=True))
        return
    _print_table(results)


if __name__ == "__main__":
    main()
