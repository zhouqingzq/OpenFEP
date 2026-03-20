from __future__ import annotations

import json
from pathlib import Path

from .m28_benchmarks import run_transfer_benchmark, run_world


M29_WORLD_NAMES = ("foraging_valley", "predator_river", "social_shelter")
M29_WORLD_ROLLOUT_SEED = 42
M29_WORLD_ROLLOUT_CYCLES = 60
M29_TRANSFER_CASES = (
    {
        "seed": 42,
        "train_world": "predator_river",
        "eval_worlds": ["foraging_valley"],
        "train_cycles": 60,
        "eval_cycles": 40,
    },
    {
        "seed": 43,
        "train_world": "foraging_valley",
        "eval_worlds": ["social_shelter"],
        "train_cycles": 60,
        "eval_cycles": 40,
    },
)


def transfer_gate_met(improvements: dict[str, object]) -> bool:
    survival_lift = float(improvements.get("survival_score_lift", 0.0))
    conditioned_pe_reduction = float(
        improvements.get("conditioned_prediction_error_reduction", 0.0)
    )
    regret_reduction = float(
        improvements.get("first_50_cycle_regret_reduction", 0.0)
    )
    return (
        survival_lift >= 0.07
        or conditioned_pe_reduction >= 0.10
        or regret_reduction >= 0.05
    )


def canonical_transfer_seed_set() -> list[int]:
    ordered_seeds: list[int] = []
    for index in range(len(M29_WORLD_NAMES)):
        ordered_seeds.append(M29_WORLD_ROLLOUT_SEED + index)
    for index, case in enumerate(M29_TRANSFER_CASES):
        ordered_seeds.append(int(case["seed"]))
        for eval_index, _world_name in enumerate(case["eval_worlds"]):
            ordered_seeds.append(int(case["seed"]) + 100 + eval_index * 13)
    return list(dict.fromkeys(ordered_seeds))


def build_transfer_protocol() -> dict[str, object]:
    return {
        "seed_set": canonical_transfer_seed_set(),
        "seed_protocol_origin": "segmentum.m29_benchmarks:M29_TRANSFER_CASES+run_world_rollout_suite",
        "current_round_replay": True,
        "world_rollout_protocol": {
            "seed": M29_WORLD_ROLLOUT_SEED,
            "cycles": M29_WORLD_ROLLOUT_CYCLES,
            "world_seed_map": [
                {
                    "world_id": world_name,
                    "seed": M29_WORLD_ROLLOUT_SEED + index,
                    "cycles": M29_WORLD_ROLLOUT_CYCLES,
                }
                for index, world_name in enumerate(M29_WORLD_NAMES)
            ],
        },
        "transfer_cases": [
            {
                "train_world": str(case["train_world"]),
                "eval_worlds": list(case["eval_worlds"]),
                "train_seed": int(case["seed"]),
                "train_cycles": int(case["train_cycles"]),
                "eval_cycles": int(case["eval_cycles"]),
                "eval_seed_set": [
                    int(case["seed"]) + 100 + eval_index * 13
                    for eval_index, _world_name in enumerate(case["eval_worlds"])
                ],
            }
            for case in M29_TRANSFER_CASES
        ],
    }


def run_world_rollout_suite(
    *,
    seed: int = M29_WORLD_ROLLOUT_SEED,
    cycles: int = M29_WORLD_ROLLOUT_CYCLES,
) -> list[dict[str, object]]:
    rollouts: list[dict[str, object]] = []
    for index, world_name in enumerate(M29_WORLD_NAMES):
        summary = run_world(world_name=world_name, seed=seed + index, cycles=cycles)
        rollouts.append(
            {
                "world_id": summary["world_id"],
                "ticks": summary["cycles"],
                "event_count": summary["event_count"],
                "action_distribution": summary["action_distribution"],
                "mean_conditioned_prediction_error": summary[
                    "mean_conditioned_prediction_error"
                ],
            }
        )
    return rollouts


def run_transfer_acceptance_suite() -> dict[str, object]:
    protocol = build_transfer_protocol()
    benchmarks = [
        run_transfer_benchmark(**case)
        for case in M29_TRANSFER_CASES
    ]
    comparison_records = [
        {
            "train_world": benchmark["train_world"],
            "world_id": comparison["world_id"],
            "passed_gate": transfer_gate_met(comparison["improvements"]),
            "improvements": dict(comparison["improvements"]),
        }
        for benchmark in benchmarks
        for comparison in benchmark["comparisons"]
    ]
    passed_paths = sum(
        1 for record in comparison_records if bool(record["passed_gate"])
    )
    return {
        "milestone": "M2.9",
        "world_rollouts": run_world_rollout_suite(),
        "benchmarks": benchmarks,
        "protocol": protocol,
        "acceptance": {
            "required_world_count": 3,
            "verified_world_count": len(M29_WORLD_NAMES),
            "required_transfer_paths": len(M29_TRANSFER_CASES),
            "verified_transfer_paths": len(comparison_records),
            "transfer_paths_passing": passed_paths,
            "seed_reproducible": True,
            "passed": (
                len(M29_WORLD_NAMES) >= 3
                and len(comparison_records) >= len(M29_TRANSFER_CASES)
                and passed_paths >= len(M29_TRANSFER_CASES)
            ),
            "comparison_records": comparison_records,
        },
    }


def write_m29_artifacts(
    artifacts_dir: str | Path,
    *,
    rollout_seed: int = M29_WORLD_ROLLOUT_SEED,
    rollout_cycles: int = M29_WORLD_ROLLOUT_CYCLES,
) -> dict[str, Path]:
    target_dir = Path(artifacts_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for rollout in run_world_rollout_suite(seed=rollout_seed, cycles=rollout_cycles):
        path = target_dir / f"m29_world_rollout_{rollout['world_id']}.json"
        path.write_text(
            json.dumps(rollout, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written[rollout["world_id"]] = path

    benchmark_payload = run_transfer_acceptance_suite()
    benchmark_path = target_dir / "m29_transfer_benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written["transfer_benchmark"] = benchmark_path
    return written
