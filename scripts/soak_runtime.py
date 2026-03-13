from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentum.runtime import SegmentRuntime


ACCEPTANCE_PROFILES: dict[str, dict[str, float | int]] = {
    "m0": {
        "min_unique_actions": 3,
        "min_action_entropy": 0.20,
        "max_dominant_action_share": 0.92,
        "max_action_streak": 48,
        "min_action_switch_count": 24,
    },
    "nightly": {
        "min_unique_actions": 4,
        "min_action_entropy": 0.45,
        "max_dominant_action_share": 0.90,
        "max_action_streak": 18,
        "min_action_switch_count": 80,
    },
}


def _first_snapshot_difference(
    first: object,
    second: object,
    *,
    path: str = "snapshot",
) -> str | None:
    if type(first) is not type(second):
        return (
            f"{path} type mismatch: "
            f"{type(first).__name__} != {type(second).__name__}"
        )

    if isinstance(first, dict):
        first_keys = set(first)
        second_keys = set(second)
        missing_from_first = sorted(second_keys - first_keys)
        if missing_from_first:
            return f"{path} missing keys in original snapshot: {missing_from_first!r}"
        missing_from_second = sorted(first_keys - second_keys)
        if missing_from_second:
            return f"{path} missing keys after reload: {missing_from_second!r}"
        for key in sorted(first):
            difference = _first_snapshot_difference(
                first[key],
                second[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None

    if isinstance(first, list):
        if len(first) != len(second):
            return f"{path} length mismatch: {len(first)} != {len(second)}"
        for index, (first_item, second_item) in enumerate(zip(first, second)):
            difference = _first_snapshot_difference(
                first_item,
                second_item,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None

    if first != second:
        return f"{path} value mismatch: {first!r} != {second!r}"

    return None


def _assert_close_summary(
    first: dict[str, object],
    second: dict[str, object],
    *,
    float_tolerance: float = 1e-12,
) -> None:
    if first.keys() != second.keys():
        raise AssertionError("summary keys differ across repeated runs")

    for key in first:
        first_value = first[key]
        second_value = second[key]
        if isinstance(first_value, float):
            if abs(first_value - second_value) > float_tolerance:
                raise AssertionError(
                    f"summary field {key!r} drifted: {first_value} != {second_value}"
                )
        elif first_value != second_value:
            raise AssertionError(
                f"summary field {key!r} drifted: {first_value!r} != {second_value!r}"
            )


def _assert_acceptance(
    summary: dict[str, object],
    *,
    cycles: int,
    min_unique_actions: int,
    min_action_entropy: float,
    max_dominant_action_share: float,
    max_action_streak: int,
    min_action_switch_count: int,
) -> None:
    if summary["cycles_completed"] != cycles:
        raise AssertionError("runtime did not complete the requested number of cycles")
    if summary["survival_ticks"] != cycles:
        raise AssertionError("agent did not survive all requested cycles")
    if summary["termination_reason"] != "cycles_exhausted":
        raise AssertionError("runtime terminated for an unexpected reason")
    if summary["persistence_error_count"] != 0:
        raise AssertionError("persistence errors were recorded during acceptance soak")
    if summary["unique_actions"] < min_unique_actions:
        raise AssertionError(
            f"behavior collapsed below unique action threshold: {summary['unique_actions']}"
        )
    if summary["action_entropy"] < min_action_entropy:
        raise AssertionError(
            f"behavior entropy fell below threshold: {summary['action_entropy']}"
        )
    if summary["dominant_action_share"] > max_dominant_action_share:
        raise AssertionError(
            "behavior collapsed into one dominant action: "
            f"{summary['dominant_action_share']}"
        )
    if summary["max_action_streak"] > max_action_streak:
        raise AssertionError(
            f"action streak exceeded threshold: {summary['max_action_streak']}"
        )
    if summary["action_switch_count"] < min_action_switch_count:
        raise AssertionError(
            "behavior did not switch actions often enough: "
            f"{summary['action_switch_count']}"
        )


def _run_once(cycles: int, seed: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        summary = runtime.run(cycles=cycles, verbose=False)
        snapshot = runtime.export_snapshot()

        reloaded = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed + 1,
        )
        reloaded_snapshot = reloaded.export_snapshot()
        trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
        trace_records = [json.loads(line) for line in trace_lines]

    if snapshot != reloaded_snapshot:
        difference = _first_snapshot_difference(snapshot, reloaded_snapshot)
        if difference is None:
            difference = "unknown snapshot difference"
        raise AssertionError(
            f"snapshot reload mismatch after soak run: {difference}"
        )
    if summary["cycles_completed"] != snapshot["metrics"]["cycles_completed"]:
        raise AssertionError("metrics summary and persisted snapshot diverged")
    if snapshot["agent"]["cycle"] != snapshot["metrics"]["cycles_completed"]:
        raise AssertionError("agent cycle count diverged from persisted metrics")
    if len(trace_records) != cycles:
        raise AssertionError("trace record count diverged from requested cycle count")
    if trace_records[-1]["running_metrics"]["cycles_completed"] != cycles:
        raise AssertionError("trace running metrics did not reach the final cycle count")

    return {
        "summary": summary,
        "snapshot_cycles": snapshot["metrics"]["cycles_completed"],
        "agent_cycle": snapshot["agent"]["cycle"],
        "termination_reason": summary["termination_reason"],
        "trace_records": len(trace_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a repeatable long-run soak check for SegmentRuntime."
    )
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--profile",
        choices=sorted(ACCEPTANCE_PROFILES),
        default="m0",
        help="Named acceptance threshold profile.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeat the same seeded soak run and compare summaries.",
    )
    parser.add_argument(
        "--min-unique-actions",
        type=int,
        default=None,
        help="Override the minimum number of unique actions required.",
    )
    parser.add_argument(
        "--min-action-entropy",
        type=float,
        default=None,
        help="Override the minimum action entropy required.",
    )
    parser.add_argument(
        "--max-dominant-action-share",
        type=float,
        default=None,
        help="Override the maximum share allowed for the dominant action.",
    )
    parser.add_argument(
        "--max-action-streak",
        type=int,
        default=None,
        help="Override the maximum allowed repeated action streak.",
    )
    parser.add_argument(
        "--min-action-switch-count",
        type=int,
        default=None,
        help="Override the minimum number of action switches required.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    thresholds = dict(ACCEPTANCE_PROFILES[args.profile])
    if args.min_unique_actions is not None:
        thresholds["min_unique_actions"] = args.min_unique_actions
    if args.min_action_entropy is not None:
        thresholds["min_action_entropy"] = args.min_action_entropy
    if args.max_dominant_action_share is not None:
        thresholds["max_dominant_action_share"] = args.max_dominant_action_share
    if args.max_action_streak is not None:
        thresholds["max_action_streak"] = args.max_action_streak
    if args.min_action_switch_count is not None:
        thresholds["min_action_switch_count"] = args.min_action_switch_count

    runs = [_run_once(args.cycles, args.seed) for _ in range(args.repeats)]
    baseline_summary = runs[0]["summary"]
    _assert_acceptance(
        baseline_summary,
        cycles=args.cycles,
        min_unique_actions=int(thresholds["min_unique_actions"]),
        min_action_entropy=float(thresholds["min_action_entropy"]),
        max_dominant_action_share=float(thresholds["max_dominant_action_share"]),
        max_action_streak=int(thresholds["max_action_streak"]),
        min_action_switch_count=int(thresholds["min_action_switch_count"]),
    )
    for run in runs[1:]:
        _assert_close_summary(baseline_summary, run["summary"])
        _assert_acceptance(
            run["summary"],
            cycles=args.cycles,
            min_unique_actions=int(thresholds["min_unique_actions"]),
            min_action_entropy=float(thresholds["min_action_entropy"]),
            max_dominant_action_share=float(thresholds["max_dominant_action_share"]),
            max_action_streak=int(thresholds["max_action_streak"]),
            min_action_switch_count=int(thresholds["min_action_switch_count"]),
        )

    report = {
        "cycles": args.cycles,
        "seed": args.seed,
        "repeats": args.repeats,
        "profile": args.profile,
        "baseline_summary": baseline_summary,
        "all_runs_consistent": True,
        "acceptance_thresholds": thresholds,
    }
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
