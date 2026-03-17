from __future__ import annotations

import copy
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation
from segmentum.self_model import IdentityCommitment, IdentityNarrative

ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _exploration_narrative() -> IdentityNarrative:
    return IdentityNarrative(
        core_identity="I am an exploratory agent.",
        commitments=[
            IdentityCommitment(
                commitment_id="commitment-exploration-drive",
                commitment_type="behavioral_style",
                statement="When conditions are stable, reduce uncertainty through exploration.",
                target_actions=["scan"],
                discouraged_actions=["rest"],
                confidence=0.95,
                priority=0.95,
                source_claim_ids=["claim-explore"],
                source_chapter_ids=[1],
                evidence_ids=["ep-explore-001"],
                last_reaffirmed_tick=40,
            )
        ],
    )


def _survival_narrative() -> IdentityNarrative:
    return IdentityNarrative(
        core_identity="I am a risk-averse, survival-focused agent.",
        commitments=[
            IdentityCommitment(
                commitment_id="commitment-survival-priority",
                commitment_type="value_guardrail",
                statement="Protect survival and integrity before opportunistic gain.",
                target_actions=["hide", "rest", "exploit_shelter"],
                discouraged_actions=["forage"],
                confidence=0.95,
                priority=0.95,
                source_claim_ids=["claim-survival"],
                source_chapter_ids=[1],
                evidence_ids=["ep-survival-001"],
                last_reaffirmed_tick=40,
            )
        ],
    )


def _episode(
    tick: int,
    action: str,
    *,
    outcome: str = "neutral",
    surprise: float = 0.8,
    energy: float = 0.7,
    free_energy_drop: float = 0.05,
) -> dict[str, object]:
    return {
        "timestamp": tick,
        "action_taken": action,
        "predicted_outcome": outcome,
        "total_surprise": surprise,
        "risk": 0.4 if outcome == "neutral" else 4.0,
        "body_state": {
            "energy": energy,
            "stress": 0.3,
            "fatigue": 0.2,
            "temperature": 0.5,
        },
        "outcome_state": {"free_energy_drop": free_energy_drop},
        "identity_critical": outcome == "survival_threat",
    }


def _decision(tick: int, action: str, *, risk: float) -> dict[str, object]:
    return {
        "tick": tick,
        "action": action,
        "dominant_component": "goal_alignment",
        "risk": risk,
        "active_goal": "CONTROL",
        "goal_alignment": 0.4,
        "preferred_probability": 0.5,
        "policy_score": 0.2,
    }


def _refresh_phase(
    agent: SegmentAgent,
    *,
    decisions: list[dict[str, object]],
    episodes: list[dict[str, object]],
    tick: int,
    chapter_signal: str | None = None,
) -> None:
    agent.decision_history.extend(decisions)
    agent.long_term_memory.episodes.extend(episodes)
    agent.self_model.update_preferred_policies(agent.decision_history, current_tick=tick)
    agent.self_model.update_identity_narrative(
        episodic_memory=list(agent.long_term_memory.episodes),
        preference_labels=agent.long_term_memory.preference_model.legacy_value_hierarchy_dict(),
        current_tick=tick,
        decision_history=list(agent.decision_history),
        sleep_metrics={"policy_bias_updates": 1, "threat_updates": 1},
        conflict_history=list(agent.goal_stack.conflict_history),
        weight_adjustments=list(agent.goal_stack.weight_adjustments),
        chapter_signal=chapter_signal,
    )


def main() -> None:
    observation = Observation(
        food=0.25,
        danger=0.05,
        novelty=1.0,
        shelter=0.1,
        temperature=0.5,
        social=0.1,
    )
    baseline = SegmentAgent(rng=random.Random(7))
    committed = SegmentAgent(rng=random.Random(7))
    committed.self_model.identity_narrative = _exploration_narrative()
    baseline_diag = baseline.decision_cycle(observation)["diagnostics"]
    committed_diag = committed.decision_cycle(observation)["diagnostics"]
    commitment_trace = {
        "baseline_ranking": [
            {
                "choice": option.choice,
                "policy_score": round(option.policy_score, 6),
                "commitment_bias": round(option.commitment_bias, 6),
            }
            for option in baseline_diag.ranked_options
        ],
        "committed_ranking": [
            {
                "choice": option.choice,
                "policy_score": round(option.policy_score, 6),
                "commitment_bias": round(option.commitment_bias, 6),
            }
            for option in committed_diag.ranked_options
        ],
        "scan_overtook_rest": committed_diag.policy_scores["scan"]
        > committed_diag.policy_scores["rest"],
    }
    _write_json(
        ARTIFACTS_DIR / "m215_identity_commitment_trace.json",
        commitment_trace,
    )

    survival_agent = SegmentAgent(rng=random.Random(11))
    survival_agent.self_model.identity_narrative = _survival_narrative()
    assessment = survival_agent.policy_evaluator.commitment_assessment(
        action="forage",
        projected_state={"danger": 0.95, "novelty": 0.1, "shelter": 0.05},
    )
    decision = survival_agent.decision_cycle(
        Observation(
            food=0.85,
            danger=0.82,
            novelty=0.15,
            shelter=0.2,
            temperature=0.5,
            social=0.1,
        )
    )
    observed = dict(decision["observed"])
    prediction = dict(decision["prediction"])
    observed["danger"] = 1.0
    observed["shelter"] = 0.0
    prediction["danger"] = 0.0
    prediction["shelter"] = 1.0
    errors = {
        key: abs(observed.get(key, 0.0) - prediction.get(key, 0.0))
        for key in observed
    }
    survival_agent.integrate_outcome(
        choice=decision["diagnostics"].chosen.choice,
        observed=observed,
        prediction=prediction,
        errors=errors,
        free_energy_before=float(decision["free_energy_before"]),
        free_energy_after=float(decision["free_energy_before"]) + 0.5,
    )
    protected = copy.deepcopy(survival_agent.long_term_memory.episodes[-1])
    for index in range(3):
        duplicate = copy.deepcopy(protected)
        duplicate["episode_id"] = f"duplicate-{index}"
        duplicate["identity_critical"] = False
        duplicate["identity_commitment_reason"] = ""
        duplicate["identity_commitment_ids"] = []
        duplicate["lifecycle_stage"] = "validated_episode"
        survival_agent.long_term_memory.episodes.append(duplicate)
    removed = survival_agent.long_term_memory.compress_episodes()
    repair_artifact = {
        "violation_assessment": assessment,
        "identity_tension_history": list(survival_agent.identity_tension_history),
        "compression_removed": removed,
        "post_compression_episodes": [
            {
                "episode_id": payload.get("episode_id"),
                "identity_critical": bool(payload.get("identity_critical", False)),
                "identity_commitment_reason": payload.get("identity_commitment_reason", ""),
                "identity_commitment_ids": list(payload.get("identity_commitment_ids", [])),
                "lifecycle_stage": payload.get("lifecycle_stage", ""),
                "compressed_count": int(payload.get("compressed_count", 1)),
            }
            for payload in survival_agent.long_term_memory.episodes
        ],
    }
    _write_json(
        ARTIFACTS_DIR / "m215_self_inconsistency_repair.json",
        repair_artifact,
    )

    def build_chapter_sequence() -> list[str]:
        agent = SegmentAgent(rng=random.Random(5))
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 81)],
            episodes=[_episode(40, "scan", outcome="neutral", surprise=0.9, energy=0.74)],
            tick=80,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "hide", risk=0.2) for tick in range(81, 161)],
            episodes=[
                _episode(
                    120,
                    "hide",
                    outcome="survival_threat",
                    surprise=4.8,
                    energy=0.16,
                    free_energy_drop=-0.5,
                )
            ],
            tick=160,
        )
        _refresh_phase(
            agent,
            decisions=[_decision(tick, "rest", risk=0.2) for tick in range(161, 241)],
            episodes=[_episode(200, "rest", outcome="neutral", surprise=0.6, energy=0.6)],
            tick=240,
        )
        narrative = agent.self_model.identity_narrative
        assert narrative is not None
        return [chapter.dominant_theme for chapter in narrative.chapters]

    sequence_a = build_chapter_sequence()
    sequence_b = build_chapter_sequence()
    narrative_agent = SegmentAgent(rng=random.Random(3))
    _refresh_phase(
        narrative_agent,
        decisions=[_decision(tick, "scan", risk=3.0) for tick in range(1, 101)],
        episodes=[_episode(40, "scan", outcome="neutral", surprise=0.9, energy=0.75)],
        tick=100,
    )
    _refresh_phase(
        narrative_agent,
        decisions=[_decision(tick, "hide", risk=0.2) for tick in range(101, 201)],
        episodes=[
            _episode(
                150,
                "hide",
                outcome="survival_threat",
                surprise=5.0,
                energy=0.15,
                free_energy_drop=-0.45,
            )
        ],
        tick=200,
        chapter_signal="Goal priority shifted: SURVIVAL overtook EXPLORATION at tick 150",
    )
    narrative = narrative_agent.self_model.identity_narrative
    assert narrative is not None

    status = "PASS"
    if not commitment_trace["scan_overtook_rest"]:
        status = "FAIL"
    if float(assessment["tension"]) <= 0.0:
        status = "FAIL"
    if sequence_a != sequence_b:
        status = "FAIL"
    if not any(
        payload["identity_critical"] and payload["identity_commitment_ids"]
        for payload in repair_artifact["post_compression_episodes"]
    ):
        status = "FAIL"
    if not all(commitment.evidence_ids for commitment in narrative.commitments):
        status = "FAIL"

    report = {
        "milestone": "M2.15",
        "status": status,
        "gates": {
            "identity_commitments_alter_action_ranking": commitment_trace["scan_overtook_rest"],
            "self_inconsistency_events_traced": float(assessment["tension"]) > 0.0,
            "deterministic_chapter_progression": sequence_a == sequence_b,
            "evidence_backed_narrative_commitments": all(
                commitment.evidence_ids and commitment.source_claim_ids
                for commitment in narrative.commitments
            ),
            "identity_critical_memories_preserved_after_compaction": any(
                payload["identity_critical"] and payload["identity_commitment_ids"]
                for payload in repair_artifact["post_compression_episodes"]
            ),
        },
        "summary": {
            "commitment_focus": committed_diag.commitment_focus,
            "violated_commitments": decision["diagnostics"].violated_commitments,
            "chapter_transition_evidence_count": len(narrative.chapter_transition_evidence),
            "commitment_count": len(narrative.commitments),
        },
    }
    _write_json(REPORTS_DIR / "m215_acceptance_report.json", report)


if __name__ == "__main__":
    main()
