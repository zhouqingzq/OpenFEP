from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.agent import SegmentAgent
from segmentum.environment import SimulatedWorld
from segmentum.memory import compute_prediction_error


def run_attention_benchmark(*, seed: int, cycles: int) -> dict[str, object]:
    conditions: dict[str, dict[str, object]] = {}

    for enabled in (False, True):
        label = "attention_on" if enabled else "attention_off"
        world = SimulatedWorld(seed=seed)
        agent = SegmentAgent(rng=world.rng)
        agent.configure_attention_bottleneck(enabled=enabled, capacity=3)

        conditioned_prediction_errors: list[float] = []
        free_energies: list[float] = []
        actions: list[str] = []
        selected_channels: Counter[str] = Counter()
        salient_ticks = 0
        salient_hits = 0
        survival_score = 0.0

        for _ in range(cycles):
            agent.cycle += 1
            observation = world.observe()
            decision = agent.decision_cycle(observation)
            prediction = decision["prediction"]
            filtered_observation = (
                agent.last_attention_filtered_observation
                if enabled and agent.last_attention_filtered_observation
                else decision["observed"]
            )
            conditioned_prediction_errors.append(
                compute_prediction_error(filtered_observation, prediction)
            )
            free_energies.append(float(decision["free_energy_before"]))
            action_name = decision["diagnostics"].chosen.choice
            actions.append(action_name)
            world_feedback = world.apply_action(action_name)
            agent.apply_action_feedback(world_feedback)

            survival_score += max(0.0, agent.energy) + max(0.0, 1.0 - agent.stress)
            if enabled and agent.last_attention_trace is not None:
                selected_channels.update(agent.last_attention_trace.allocation.selected_channels)
                obs = decision["observed"]
                expected = []
                if float(obs.get("danger", 0.0)) >= 0.65:
                    expected.append("danger")
                if float(obs.get("novelty", 0.0)) >= 0.65:
                    expected.append("novelty")
                if expected:
                    salient_ticks += 1
                    selected = set(agent.last_attention_trace.allocation.selected_channels)
                    if all(channel in selected for channel in expected):
                        salient_hits += 1

        conditions[label] = {
            "enabled": enabled,
            "cycles": cycles,
            "mean_conditioned_prediction_error": round(
                sum(conditioned_prediction_errors) / max(1, len(conditioned_prediction_errors)),
                6,
            ),
            "mean_free_energy_before": round(
                sum(free_energies) / max(1, len(free_energies)),
                6,
            ),
            "survival_score": round(survival_score / max(1, cycles), 6),
            "action_distribution": {
                key: value / max(1, cycles)
                for key, value in sorted(Counter(actions).items())
            },
            "selected_channel_statistics": dict(selected_channels),
            "salient_topk_hit_rate": round(
                salient_hits / max(1, salient_ticks),
                6,
            ),
        }

    off = conditions["attention_off"]
    on = conditions["attention_on"]
    pe_improvement = 0.0
    if float(off["mean_conditioned_prediction_error"]) > 0:
        pe_improvement = 1.0 - (
            float(on["mean_conditioned_prediction_error"])
            / float(off["mean_conditioned_prediction_error"])
        )
    survival_ratio = 0.0
    if float(off["survival_score"]) > 0:
        survival_ratio = float(on["survival_score"]) / float(off["survival_score"])

    return {
        "seed": seed,
        "cycles": cycles,
        "conditions": conditions,
        "evaluation": {
            "conditioned_prediction_error_improvement": round(pe_improvement, 6),
            "survival_ratio": round(survival_ratio, 6),
            "topk_hit_rate": float(on["salient_topk_hit_rate"]),
            "acceptance": {
                "topk_hit_rate_gte_0_80": float(on["salient_topk_hit_rate"]) >= 0.80,
                "prediction_error_improvement_gte_0_10": pe_improvement >= 0.10,
                "survival_ratio_gte_0_95": survival_ratio >= 0.95,
            },
        },
    }


def main(seed: int = 42, cycles: int = 80) -> None:
    result = run_attention_benchmark(seed=seed, cycles=cycles)
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "m28_attention_on_off.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["evaluation"], indent=2, ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycles", type=int, default=80)
    args = parser.parse_args()
    main(seed=args.seed, cycles=args.cycles)
