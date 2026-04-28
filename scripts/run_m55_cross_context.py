"""M5.5 cross-context stability: run scenario battery, produce acceptance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.dialogue.maturity import personality_distance
from segmentum.dialogue.scenarios.analysis import (
    adaptation_envelope,
    analyze_cross_context,
    behavioral_adaptation,
    state_distance_decomposition,
)
from segmentum.dialogue.scenarios.battery import SCENARIO_BATTERY, get_scenario
from segmentum.dialogue.scenarios.conductor import ScenarioConductor
from segmentum.dialogue.scenarios.intent_probe import probe_intent_precision


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _configure_agent(agent: SegmentAgent, config: dict[str, object] | None) -> int:
    """Apply personality configuration from a JSON dict if provided. Returns uid."""
    if config is None:
        return 0
    uid = int(config.get("uid", 0))
    agent.uid = uid

    # Apply slow traits
    traits = config.get("slow_traits")
    if isinstance(traits, dict):
        state = agent.slow_variable_learner.state.traits
        for key in ("caution_bias", "threat_sensitivity", "trust_stance", "exploration_posture", "social_approach"):
            if key in traits:
                setattr(state, key, float(traits[key]))

    # Apply narrative priors
    priors = config.get("narrative_priors")
    if isinstance(priors, dict):
        np_obj = agent.self_model.narrative_priors
        for key in ("trust_prior", "controllability_prior", "trauma_bias", "contamination_sensitivity", "meaning_stability"):
            if key in priors:
                setattr(np_obj, key, float(priors[key]))

    return uid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5.5 cross-context stability battery")
    parser.add_argument("--agent-config", type=Path, default=None,
                        help="JSON file with personality config (slow_traits, narrative_priors, uid)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/m55_cross_context"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", type=str, default="random",
                        choices=["random", "temporal"])
    parser.add_argument("--uid", type=int, default=0)

    args = parser.parse_args()

    config: dict[str, object] | None = None
    if args.agent_config is not None:
        config = _load_json(args.agent_config)

    # Build and configure agent
    agent = SegmentAgent()
    uid = _configure_agent(agent, config) or args.uid

    # Run battery
    conductor = ScenarioConductor()
    results = conductor.run_battery(
        agent,
        battery=SCENARIO_BATTERY,
        seed=args.seed,
        split_strategy=args.split_strategy,
    )

    # Analysis
    report = analyze_cross_context(results)
    ba = behavioral_adaptation(results)
    intent = probe_intent_precision(agent, results)

    # Write per-agent artifacts
    out = args.output
    _write_json(out / f"{uid}_battery_results.json",
                [r.to_dict() for r in results])
    _write_json(out / f"{uid}_analysis_report.json", {
        "agent_uid": report.agent_uid,
        "personality_consistency": report.personality_consistency,
        "adaptation_envelope": report.adaptation_envelope,
        "adaptation_l2_to_real": report.adaptation_l2_to_real,
        "anomalies": report.anomalies,
        "conclusion": report.conclusion,
        "per_scenario": {
            sid: {
                "action_distribution": r.action_distribution,
                "strategy_distribution": r.strategy_distribution,
                "channel_means": r.channel_means,
                "personality_deviation": r.personality_deviation,
            }
            for sid, r in report.per_scenario.items()
        },
    })
    _write_json(out / f"{uid}_intent_probe.json", intent)

    # Aggregate report
    decomp = state_distance_decomposition(results)
    _write_json(out / "aggregate_report.json", {
        "milestone": "M5.5",
        "uid": uid,
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "scenario_count": len(results),
        "personality_consistency": report.personality_consistency,
        "adaptation_envelope": report.adaptation_envelope,
        "state_distance_decomposition": decomp,
        "intent_probe": {
            "anomaly_type": intent.get("anomaly_type"),
            "mean_hidden_intent_precision": intent.get("mean_hidden_intent_precision"),
            "tier_compliance": intent.get("tier_compliance"),
        },
        "conclusion": report.conclusion,
    })

    # Acceptance artifact
    ba_nonzero = int(ba.get("nonzero_action_dims", 0)) + int(ba.get("nonzero_strategy_dims", 0))
    envelope = adaptation_envelope(results)
    envelope_nonzero = sum(1 for v in envelope.values() if v > 0.0)
    crash_count = len([a for a in report.anomalies if a.get("severity") == "crash"])
    warning_count = len([a for a in report.anomalies if a.get("severity") == "warning"])

    # Criterion 8: game world transfer gap (< 0.15)
    game_transfer_gap = 0.0
    s7 = next((r for r in results if r.scenario_id == "game_world_npc"), None)
    others = [r for r in results if r.scenario_id != "game_world_npc"]
    if s7 is not None and s7.post_snapshot is not None and len(others) >= 2:
        s7_sims: list[float] = []
        for r in others:
            if r.post_snapshot is not None:
                dist = personality_distance(s7.post_snapshot, r.post_snapshot)
                s7_sims.append(1.0 - dist)
        s7_mean = sum(s7_sims) / len(s7_sims) if s7_sims else 1.0
        other_sims: list[float] = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                if others[i].post_snapshot is not None and others[j].post_snapshot is not None:
                    dist = personality_distance(others[i].post_snapshot, others[j].post_snapshot)
                    other_sims.append(1.0 - dist)
        other_mean = sum(other_sims) / len(other_sims) if other_sims else 1.0
        game_transfer_gap = abs(s7_mean - other_mean)
    acceptance = {
        "milestone": "M5.5",
        "name": "Cross-Context Stability（跨情境稳定性）",
        "status": "passed" if (
            report.personality_consistency >= 0.80
            and ba_nonzero >= 2
            and crash_count == 0
            and warning_count == 0
            and intent.get("tier_compliance") is True
            and game_transfer_gap < 0.15
        ) else "partial",
        "personality_consistency": report.personality_consistency,
        "adaptation_envelope": envelope,
        "adaptation_envelope_nonzero_dims": envelope_nonzero,
        "behavioral_adaptation_nonzero_dims": ba_nonzero,
        "crash_count": crash_count,
        "warning_count": warning_count,
        "game_transfer_gap": round(game_transfer_gap, 6),
        "intent_anomaly_type": intent.get("anomaly_type"),
        "tier_compliance": intent.get("tier_compliance"),
        "scenarios_run": len(results),
        "seed": args.seed,
        "split_strategy": args.split_strategy,
    }
    _write_json(Path("artifacts/m55_acceptance.json"), acceptance)

    print(f"  consistency: {report.personality_consistency:.4f}")
    print(f"  behavioral adaptation nonzero: {ba_nonzero}")
    print(f"  crashes: {crash_count}  warnings: {warning_count}")
    print(f"  tier_compliance: {intent.get('tier_compliance')}")
    print(f"  game_transfer_gap: {round(game_transfer_gap, 4)}")
    print(f"  intent anomaly: {intent.get('anomaly_type')}")
    print(f"  acceptance: {acceptance['status']}")
    print(f"  artifacts → {out.resolve()}")


if __name__ == "__main__":
    main()
