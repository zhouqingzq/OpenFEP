from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation, SimulatedWorld
from segmentum.runtime import SegmentRuntime, STATE_VERSION
from tests.test_counterfactual import (
    HARMFUL_BODY_STATE,
    HARMFUL_OUTCOME,
    OBS_DANGEROUS_DICT,
    PREDICTION,
    _errors,
)


ARTIFACT_DIR = ROOT / "artifacts"
SUMMARY_PATH = ARTIFACT_DIR / "m2_targeted_repair_summary.json"
MEMORY_TRACE_PATH = ARTIFACT_DIR / "m2_memory_prediction_trace.jsonl"
FAILURE_TRACE_PATH = ARTIFACT_DIR / "m2_runtime_failure_trace.jsonl"
CLOSURE_PATH = ARTIFACT_DIR / "m2_sleep_counterfactual_closure.json"


class HTTPTimeout(RuntimeError):
    pass


class TokenLimitExceeded(RuntimeError):
    pass


class ExternalFailureWorld:
    def __init__(self) -> None:
        self.rng = random.Random(17)

    def observe(self):  # noqa: ANN001
        raise HTTPTimeout("external service timed out")


class InternalFailureWorld:
    def __init__(self) -> None:
        self.rng = random.Random(17)

    def observe(self):  # noqa: ANN001
        raise TokenLimitExceeded("context budget exhausted")


def _prime_dangerous_memory(agent: SegmentAgent, *, count: int = 5) -> None:
    for cycle in range(1, count + 1):
        decision = agent.long_term_memory.maybe_store_episode(
            cycle=cycle,
            observation=OBS_DANGEROUS_DICT,
            prediction=PREDICTION,
            errors=_errors(),
            action="forage",
            outcome=HARMFUL_OUTCOME,
            body_state=HARMFUL_BODY_STATE,
        )
        if not decision.episode_created:
            agent.long_term_memory.store_episode(
                cycle=cycle,
                observation=OBS_DANGEROUS_DICT,
                prediction=PREDICTION,
                errors=_errors(),
                action="forage",
                outcome=HARMFUL_OUTCOME,
                body_state=HARMFUL_BODY_STATE,
            )


def _run_memory_prediction_comparison() -> dict[str, object]:
    baseline = SegmentAgent(rng=random.Random(17))
    trained = SegmentAgent(rng=random.Random(17))
    _prime_dangerous_memory(trained, count=5)

    world = SimulatedWorld(seed=17)
    records: list[dict[str, object]] = []
    prediction_changed_cycles = 0
    action_changed_cycles = 0

    for cycle in range(1, 9):
        observation = world.observe()
        baseline_diag = baseline.decision_cycle(observation)["diagnostics"]
        trained_diag = trained.decision_cycle(observation)["diagnostics"]

        prediction_changed = (
            trained_diag.prediction_after_memory != baseline_diag.prediction_after_memory
        )
        action_changed = baseline_diag.chosen.choice != trained_diag.chosen.choice
        prediction_changed_cycles += int(prediction_changed)
        action_changed_cycles += int(action_changed)
        records.append(
            {
                "event": "memory_prediction_comparison",
                "cycle": cycle,
                "observation": asdict(observation),
                "without_memory": {
                    "choice": baseline_diag.chosen.choice,
                    "prediction": baseline_diag.prediction_after_memory,
                    "memory_hit": baseline_diag.memory_hit,
                },
                "with_memory": {
                    "choice": trained_diag.chosen.choice,
                    "memory_hit": trained_diag.memory_hit,
                    "retrieved_episode_ids": list(trained_diag.retrieved_episode_ids),
                    "memory_context_summary": trained_diag.memory_context_summary,
                    "prediction_before_memory": trained_diag.prediction_before_memory,
                    "prediction_after_memory": trained_diag.prediction_after_memory,
                    "prediction_delta": trained_diag.prediction_delta,
                },
                "prediction_changed": prediction_changed,
                "action_changed": action_changed,
            }
        )
        world.drift()

    MEMORY_TRACE_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "trace_path": str(MEMORY_TRACE_PATH),
        "cycles": len(records),
        "prediction_changed_cycles": prediction_changed_cycles,
        "action_changed_cycles": action_changed_cycles,
    }


def _run_failure_case(case_name: str, world, *, seed: int = 17) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = Path(tmp_dir) / f"{case_name}_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        runtime.world = world
        runtime.run(cycles=1, verbose=False)
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
    error_record = next(record for record in records if record.get("event") == "error")
    return {
        "case": case_name,
        "error_record": error_record,
    }


def _run_runtime_failure_evidence() -> dict[str, object]:
    cases = [
        _run_failure_case("self_context_budget", InternalFailureWorld()),
        _run_failure_case("world_timeout", ExternalFailureWorld()),
    ]
    FAILURE_TRACE_PATH.write_text(
        "".join(json.dumps(case, ensure_ascii=True, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    return {
        "trace_path": str(FAILURE_TRACE_PATH),
        "cases": [
            {
                "case": case["case"],
                "classification": case["error_record"]["error_attribution"]["classification"],
                "attribution": case["error_record"]["error_attribution"]["attribution"],
            }
            for case in cases
        ],
    }


def _run_sleep_counterfactual_closure() -> dict[str, object]:
    agent = SegmentAgent(rng=random.Random(42))
    agent.energy = 0.70
    agent.stress = 0.40
    agent.fatigue = 0.25
    agent.temperature = 0.46
    agent.long_term_memory.minimum_support = 1
    agent.long_term_memory.sleep_minimum_support = 1

    first_episode = agent.long_term_memory.maybe_store_episode(
        cycle=1,
        observation=OBS_DANGEROUS_DICT,
        prediction=PREDICTION,
        errors=_errors(),
        action="forage",
        outcome=HARMFUL_OUTCOME,
        body_state=HARMFUL_BODY_STATE,
    )
    _prime_dangerous_memory(agent, count=5)
    agent.long_term_memory.assign_clusters()
    cluster_id = int(agent.long_term_memory.episodes[0]["cluster_id"])

    forage_bias_before = float(agent.world_model.get_policy_bias(cluster_id, "forage"))
    hide_bias_before = float(agent.world_model.get_policy_bias(cluster_id, "hide"))
    threat_before = float(agent.world_model.get_threat_prior(cluster_id))
    penalty_before = float(agent.world_model.get_preference_penalty(cluster_id, "forage"))

    agent.cycle = 20
    sleep_summary = agent.sleep()
    wake_diagnostics = agent.decision_cycle(Observation(**OBS_DANGEROUS_DICT))["diagnostics"]
    absorption_entry = next(
        (
            entry
            for entry in sleep_summary.counterfactual_log
            if entry.get("type") == "absorption"
        ),
        {},
    )

    payload = {
        "state_version": STATE_VERSION,
        "seed": 42,
        "high_surprise_episode": {
            "action": "forage",
            "predicted_outcome": first_episode.predicted_outcome,
            "total_surprise": first_episode.total_surprise,
            "risk": first_episode.risk,
        },
        "sleep_rule_extraction": {
            "rules_extracted": sleep_summary.rules_extracted,
            "rule_ids": list(sleep_summary.rule_ids),
            "threat_updates": sleep_summary.threat_updates,
            "preference_updates": sleep_summary.preference_updates,
            "policy_bias_updates": sleep_summary.policy_bias_updates,
        },
        "slow_weight_update": {
            "cluster": cluster_id,
            "threat_prior_before": threat_before,
            "threat_prior_after": float(agent.world_model.get_threat_prior(cluster_id)),
            "preference_penalty_before": penalty_before,
            "preference_penalty_after": float(
                agent.world_model.get_preference_penalty(cluster_id, "forage")
            ),
        },
        "counterfactual": {
            "episodes_evaluated": sleep_summary.counterfactual_episodes_evaluated,
            "insights_generated": sleep_summary.counterfactual_insights_generated,
            "insights_absorbed": sleep_summary.counterfactual_insights_absorbed,
            "original_action": absorption_entry.get("original_action", "forage"),
            "better_action": absorption_entry.get("counterfactual_action", ""),
            "policy_delta": absorption_entry.get("policy_delta", 0.0),
        },
        "bias_comparison": {
            "forage_bias_before": forage_bias_before,
            "forage_bias_after": float(agent.world_model.get_policy_bias(cluster_id, "forage")),
            "hide_bias_before": hide_bias_before,
            "hide_bias_after": float(agent.world_model.get_policy_bias(cluster_id, "hide")),
        },
        "wake_probe": {
            "observation": dict(OBS_DANGEROUS_DICT),
            "chosen_action_before_episode": "forage",
            "chosen_action_after_sleep": wake_diagnostics.chosen.choice,
            "predicted_outcome_after_sleep": wake_diagnostics.chosen.predicted_outcome,
            "prediction_before_memory": wake_diagnostics.prediction_before_memory,
            "prediction_after_memory": wake_diagnostics.prediction_after_memory,
            "memory_context_summary": wake_diagnostics.memory_context_summary,
            "top_scores_after_sleep": [
                {
                    "choice": option.choice,
                    "policy_score": option.policy_score,
                    "risk": option.risk,
                    "policy_bias": option.policy_bias,
                }
                for option in wake_diagnostics.ranked_options[:3]
            ],
        },
    }
    CLOSURE_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "artifact_path": str(CLOSURE_PATH),
        "wake_action": wake_diagnostics.chosen.choice,
        "counterfactual_action": absorption_entry.get("counterfactual_action", ""),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    memory = _run_memory_prediction_comparison()
    failures = _run_runtime_failure_evidence()
    closure = _run_sleep_counterfactual_closure()
    summary = {
        "memory_prediction": memory,
        "runtime_failures": failures,
        "sleep_counterfactual_closure": closure,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
