from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.runtime import STATE_VERSION
from segmentum.types import SleepRule

CONFIDENCE_BOOST = 0.05
NARRATIVE_PREFIX = "LLM consolidated:"


class ArtifactMockLLMExtractor:
    """Deterministic LLM stub for artifact generation."""

    def __init__(self) -> None:
        self.rules_before: list[SleepRule] = []
        self.rules_after: list[SleepRule] = []

    def __call__(
        self,
        rules: list[SleepRule],
        episodes: list[dict[str, object]],
    ) -> list[SleepRule]:
        self.rules_before = list(rules)
        self.rules_after = [
            replace(
                rule,
                confidence=min(0.99, rule.confidence + CONFIDENCE_BOOST),
                narrative_insight=(
                    f"{NARRATIVE_PREFIX} action '{rule.action}' in cluster {rule.cluster} "
                    f"consistently leads to {rule.observed_outcome} "
                    f"(support={rule.support}). Confidence raised by {CONFIDENCE_BOOST}."
                ),
            )
            for rule in rules
        ]
        return self.rules_after


def _populate_episodes(
    agent: SegmentAgent,
    observation: dict[str, float],
    prediction: dict[str, float],
    errors: dict[str, float],
    harmful_outcome: dict[str, float],
    harmful_body_state: dict[str, float],
    *,
    count: int = 5,
):
    initial_memory = agent.long_term_memory.maybe_store_episode(
        cycle=1,
        observation=observation,
        prediction=prediction,
        errors=errors,
        action="forage",
        outcome=harmful_outcome,
        body_state=harmful_body_state,
    )
    for cycle in range(2, count + 1):
        agent.long_term_memory.store_episode(
            cycle=cycle,
            observation=observation,
            prediction=prediction,
            errors=errors,
            action="forage",
            outcome=harmful_outcome,
            body_state=harmful_body_state,
        )
    return initial_memory


def _configure_agent(agent: SegmentAgent) -> None:
    agent.energy = 0.22
    agent.stress = 0.30
    agent.fatigue = 0.18
    agent.temperature = 0.46
    agent.cycle = 6
    agent.long_term_memory.minimum_support = 1


def _first_rule_confidence(entries: list[dict[str, object]]) -> float:
    if not entries:
        return 0.0
    return float(entries[0].get("confidence", 0.0))


def _first_narrative(entries: list[dict[str, object]]) -> str:
    if not entries:
        return ""
    return str(entries[0].get("narrative_insight", ""))


def main() -> None:
    state_path = ROOT / "data" / "segment_v0_4_sleep_state.json"
    trace_path = ROOT / "data" / "segment_v0_4_sleep_trace.jsonl"

    agent = SegmentAgent(rng=random.Random(29))
    _configure_agent(agent)

    observation = {
        "food": 0.38,
        "danger": 0.58,
        "novelty": 0.22,
        "shelter": 0.18,
        "temperature": 0.46,
        "social": 0.18,
    }
    prediction = {
        "food": 0.72,
        "danger": 0.18,
        "novelty": 0.42,
        "shelter": 0.42,
        "temperature": 0.50,
        "social": 0.30,
    }
    errors = {key: observation[key] - prediction[key] for key in observation}
    harmful_outcome = {
        "energy_delta": -0.08,
        "stress_delta": 0.24,
        "fatigue_delta": 0.16,
        "temperature_delta": 0.02,
        "free_energy_drop": -0.42,
    }
    harmful_body_state = {
        "energy": 0.18,
        "stress": 0.82,
        "fatigue": 0.32,
        "temperature": 0.46,
    }

    initial_memory = _populate_episodes(
        agent,
        observation,
        prediction,
        errors,
        harmful_outcome,
        harmful_body_state,
        count=5,
    )

    episodes_before_sleep = len(agent.long_term_memory.episodes)
    sleep_summary = agent.sleep()

    diagnostics = agent.decision_cycle(Observation(**observation))["diagnostics"]
    chosen_action = diagnostics.chosen.choice
    repeated_outcome = (
        harmful_outcome
        if chosen_action == "forage"
        else {
            "energy_delta": -0.02,
            "stress_delta": -0.06,
            "fatigue_delta": -0.04,
            "temperature_delta": 0.0,
            "free_energy_drop": 0.06,
        }
    )
    repeat_memory = agent.long_term_memory.maybe_store_episode(
        cycle=100,
        observation=observation,
        prediction=prediction,
        errors=errors,
        action=chosen_action,
        outcome=repeated_outcome,
        body_state={
            "energy": 0.24,
            "stress": 0.38,
            "fatigue": 0.24,
            "temperature": 0.47,
        },
    )

    snapshot = {
        "state_version": STATE_VERSION,
        "artifact": {
            "scenario": "M2.3 sleep consolidation acceptance sample",
            "seed": 29,
            "episodes_before_sleep": episodes_before_sleep,
            "episodes_after_sleep": len(agent.long_term_memory.episodes),
            "before_sleep_surprise": initial_memory.total_surprise,
            "after_sleep_surprise": repeat_memory.total_surprise,
            "chosen_action_after_sleep": chosen_action,
        },
        "agent": agent.to_dict(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    records = [
        {
            "event": "anomaly_experience",
            "cycle": 1,
            "action": "forage",
            "observation": observation,
            "prediction": prediction,
            "outcome": harmful_outcome,
            "memory_decision": initial_memory.to_dict(),
        },
        {
            "event": "sleep_cycle",
            "cycle": agent.cycle,
            "sleep_summary": asdict(sleep_summary),
            "semantic_memory": [asdict(entry) for entry in agent.semantic_memory],
            "policy_biases": agent.world_model.policy_biases,
            "threat_priors": agent.world_model.threat_priors,
            "preference_penalties": agent.world_model.preference_penalties,
        },
        {
            "event": "repeat_encounter",
            "cycle": 100,
            "observation": observation,
            "chosen_action": chosen_action,
            "decision": {
                "predicted_outcome": diagnostics.chosen.predicted_outcome,
                "risk": diagnostics.chosen.risk,
                "policy_score": diagnostics.chosen.policy_score,
            },
            "memory_decision": repeat_memory.to_dict(),
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    # ---- LLM-enhanced artifact ----
    llm_trace_path = ROOT / "artifacts" / "segment_v0_4_sleep_llm_trace.jsonl"
    llm_mock = ArtifactMockLLMExtractor()
    llm_agent = SegmentAgent(rng=random.Random(29), sleep_llm_extractor=llm_mock)
    heuristic_agent = SegmentAgent(rng=random.Random(29))
    for candidate in (llm_agent, heuristic_agent):
        _configure_agent(candidate)
        candidate.long_term_memory.sleep_minimum_support = 3

    _populate_episodes(
        heuristic_agent,
        observation,
        prediction,
        errors,
        harmful_outcome,
        harmful_body_state,
        count=3,
    )
    heuristic_summary = heuristic_agent.sleep()

    _populate_episodes(
        llm_agent,
        observation,
        prediction,
        errors,
        harmful_outcome,
        harmful_body_state,
        count=3,
    )
    llm_summary = llm_agent.sleep()

    rules_before = [asdict(rule) for rule in llm_mock.rules_before]
    rules_after = [asdict(rule) for rule in llm_mock.rules_after]
    heuristic_semantic = [asdict(entry) for entry in heuristic_agent.semantic_memory]
    llm_semantic = [asdict(entry) for entry in llm_agent.semantic_memory]
    heuristic_confidence = _first_rule_confidence(heuristic_semantic)
    llm_confidence = _first_rule_confidence(rules_after)
    heuristic_threat = float(heuristic_agent.world_model.threat_priors.get("0", 0.0))
    llm_threat = float(llm_agent.world_model.threat_priors.get("0", 0.0))
    heuristic_penalty = float(heuristic_agent.world_model.preference_penalties.get("0", {}).get("forage", 0.0))
    llm_penalty = float(llm_agent.world_model.preference_penalties.get("0", {}).get("forage", 0.0))

    llm_records = [
        {
            "event": "llm_sleep_consolidation",
            "sleep_cycle_id": llm_summary.sleep_cycle_id,
            "llm_used": llm_summary.llm_used,
            "fixture": {
                "scenario": "non_saturated_llm_rule_refinement",
                "episodes": 3,
                "sleep_minimum_support": 3,
                "confidence_boost": CONFIDENCE_BOOST,
            },
            "rules_before_llm": rules_before,
            "rules_after_llm": rules_after,
            "heuristic_semantic_memory": heuristic_semantic,
            "semantic_memory": llm_semantic,
            "semantic_rules_written": llm_summary.semantic_entries_written,
            "rules_extracted": llm_summary.rules_extracted,
            "threat_updates": llm_summary.threat_updates,
            "preference_updates": llm_summary.preference_updates,
            "heuristic_summary": asdict(heuristic_summary),
            "llm_summary": asdict(llm_summary),
            "slow_weight_delta": {
                "heuristic_threat_prior": heuristic_threat,
                "llm_threat_prior": llm_threat,
                "threat_prior_gain": round(llm_threat - heuristic_threat, 6),
                "heuristic_preference_penalty": heuristic_penalty,
                "llm_preference_penalty": llm_penalty,
                "preference_penalty_gain": round(llm_penalty - heuristic_penalty, 6),
            },
            "rule_delta": {
                "heuristic_confidence": heuristic_confidence,
                "llm_confidence": llm_confidence,
                "confidence_gain": round(llm_confidence - heuristic_confidence, 6),
                "narrative_added": _first_narrative(rules_after),
            },
        },
    ]

    llm_trace_path.parent.mkdir(parents=True, exist_ok=True)
    llm_trace_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in llm_records
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

