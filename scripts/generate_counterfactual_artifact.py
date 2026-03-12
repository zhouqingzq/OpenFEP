from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.agent import SegmentAgent
from segmentum.runtime import STATE_VERSION
from tests.test_counterfactual import (
    HARMFUL_BODY_STATE,
    HARMFUL_OUTCOME,
    OBS_DANGEROUS_DICT,
    PREDICTION,
    _errors,
)

SEED = 42


def _configure_agent(agent: SegmentAgent) -> None:
    agent.energy = 0.70
    agent.stress = 0.40
    agent.fatigue = 0.25
    agent.temperature = 0.46
    agent.long_term_memory.minimum_support = 1
    agent.long_term_memory.sleep_minimum_support = 1


def _populate_dangerous_episodes(agent: SegmentAgent, *, count: int = 5) -> None:
    errors = _errors()
    for cycle in range(1, count + 1):
        agent.cycle = cycle
        decision = agent.long_term_memory.maybe_store_episode(
            cycle=cycle,
            observation=OBS_DANGEROUS_DICT,
            prediction=PREDICTION,
            errors=errors,
            action="forage",
            outcome=HARMFUL_OUTCOME,
            body_state=HARMFUL_BODY_STATE,
        )
        if not decision.episode_created:
            agent.long_term_memory.store_episode(
                cycle=cycle,
                observation=OBS_DANGEROUS_DICT,
                prediction=PREDICTION,
                errors=errors,
                action="forage",
                outcome=HARMFUL_OUTCOME,
                body_state=HARMFUL_BODY_STATE,
            )


def main() -> None:
    state_path = ROOT / "data" / "segment_v0_5_counterfactual_state.json"
    trace_path = ROOT / "data" / "segment_v0_5_counterfactual_trace.jsonl"
    summary_path = ROOT / "artifacts" / "segment_v0_5_counterfactual_summary.json"

    agent = SegmentAgent(rng=random.Random(SEED))
    _configure_agent(agent)
    _populate_dangerous_episodes(agent, count=5)
    agent.long_term_memory.assign_clusters()

    cluster_id = int(agent.long_term_memory.episodes[0]["cluster_id"])
    forage_bias_before = float(agent.world_model.get_policy_bias(cluster_id, "forage"))
    hide_bias_before = float(agent.world_model.get_policy_bias(cluster_id, "hide"))
    episodes_before_sleep = len(agent.long_term_memory.episodes)

    agent.cycle = 20
    sleep_summary = agent.sleep()

    forage_bias_after = float(agent.world_model.get_policy_bias(cluster_id, "forage"))
    hide_bias_after = float(agent.world_model.get_policy_bias(cluster_id, "hide"))
    sandbox_entry = next(
        (
            entry
            for entry in sleep_summary.counterfactual_log
            if entry.get("type") == "virtual_sandbox_reasoning"
        ),
        {},
    )
    absorption_entry = next(
        (
            entry
            for entry in sleep_summary.counterfactual_log
            if entry.get("type") == "absorption"
        ),
        {},
    )

    policy_delta = float(absorption_entry.get("policy_delta", 0.0))
    forage_bias_pre_counterfactual = float(absorption_entry.get("new_orig_bias", forage_bias_after)) + policy_delta
    hide_bias_pre_counterfactual = float(absorption_entry.get("new_cf_bias", hide_bias_after)) - policy_delta

    summary_payload = {
        "scenario": "M2.4 counterfactual learning",
        "seed": SEED,
        "cluster": cluster_id,
        "episodes_before_sleep": episodes_before_sleep,
        "counterfactual_episodes_evaluated": sleep_summary.counterfactual_episodes_evaluated,
        "counterfactual_insights_generated": sleep_summary.counterfactual_insights_generated,
        "counterfactual_insights_absorbed": sleep_summary.counterfactual_insights_absorbed,
        "counterfactual_energy_spent": round(sleep_summary.counterfactual_energy_spent, 3),
        "forage_bias_pre_sleep": round(forage_bias_before, 3),
        "forage_bias_pre_counterfactual": round(forage_bias_pre_counterfactual, 3),
        "forage_bias_after": round(forage_bias_after, 3),
        "hide_bias_pre_sleep": round(hide_bias_before, 3),
        "hide_bias_pre_counterfactual": round(hide_bias_pre_counterfactual, 3),
        "hide_bias_after": round(hide_bias_after, 3),
        "policy_delta": round(policy_delta, 3),
        "sandbox_label": sandbox_entry.get("label", ""),
        "absorbed_counterfactual_action": absorption_entry.get("counterfactual_action", ""),
    }

    state_payload = {
        "state_version": STATE_VERSION,
        "artifact": summary_payload,
        "agent": agent.to_dict(),
    }
    trace_records = [
        {
            "event": "dangerous_episode_fixture",
            "seed": SEED,
            "observation": OBS_DANGEROUS_DICT,
            "prediction": PREDICTION,
            "outcome": HARMFUL_OUTCOME,
            "body_state": HARMFUL_BODY_STATE,
            "episodes_seeded": episodes_before_sleep,
        },
        {
            "event": "counterfactual_sleep_cycle",
            "cycle": agent.cycle,
            "counterfactual_summary": {
                "episodes_evaluated": sleep_summary.counterfactual_episodes_evaluated,
                "insights_generated": sleep_summary.counterfactual_insights_generated,
                "insights_absorbed": sleep_summary.counterfactual_insights_absorbed,
                "energy_spent": sleep_summary.counterfactual_energy_spent,
            },
            "counterfactual_log": sleep_summary.counterfactual_log,
            "policy_biases": agent.world_model.policy_biases,
        },
        {
            "event": "counterfactual_policy_shift",
            "cluster": cluster_id,
            "forage_bias_pre_sleep": forage_bias_before,
            "forage_bias_after": forage_bias_after,
            "hide_bias_pre_sleep": hide_bias_before,
            "hide_bias_after": hide_bias_after,
            "policy_delta": policy_delta,
        },
    ]

    state_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    state_path.write_text(
        json.dumps(state_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trace_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in trace_records),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

