from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.runtime import SegmentRuntime


ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    state_path = ROOT / "data" / "m218_state.json"
    trace_path = ARTIFACTS_DIR / "m218_runtime_trace.jsonl"
    runtime = SegmentRuntime.load_or_create(
        state_path=state_path,
        trace_path=trace_path,
        seed=23,
        reset=True,
    )

    runtime.agent.long_term_memory.store_episode(
        cycle=1,
        observation={
            "food": 0.10,
            "danger": 0.95,
            "novelty": 0.20,
            "shelter": 0.15,
            "temperature": 0.45,
            "social": 0.10,
        },
        prediction={
            "food": 0.60,
            "danger": 0.15,
            "novelty": 0.20,
            "shelter": 0.60,
            "temperature": 0.50,
            "social": 0.20,
        },
        errors={
            "food": 0.50,
            "danger": 0.80,
            "novelty": 0.0,
            "shelter": 0.45,
            "temperature": 0.05,
            "social": 0.10,
        },
        action="hide",
        outcome={"free_energy_drop": -0.45, "stress_delta": 0.20},
        body_state={"energy": 0.32, "stress": 0.85, "fatigue": 0.40, "temperature": 0.45},
    )

    runtime.run(cycles=220, verbose=False)
    baseline_audit = runtime.agent.self_model.continuity_audit.to_dict()
    soak_summary = {
        "cycles": runtime.agent.cycle,
        "sleep_cycles": len(runtime.agent.sleep_history),
        "continuity": baseline_audit,
        "anchor_count": len(baseline_audit.get("protected_anchor_ids", [])),
        "rehearsal_events": len(runtime.agent.long_term_memory.rehearsal_log),
        "archived_episodes": len(runtime.agent.long_term_memory.archived_episodes),
        "active_episodes": len(runtime.agent.long_term_memory.episodes),
        "recent_actions": list(runtime.agent.action_history[-16:]),
    }
    _write_json(ARTIFACTS_DIR / "m218_lifelong_soak_summary.json", soak_summary)

    runtime.save_snapshot()
    restored = SegmentRuntime.load_or_create(
        state_path=state_path,
        trace_path=trace_path,
        seed=99,
        reset=False,
    )
    restart_audit = restored.agent.self_model.continuity_audit.to_dict()

    stress_model = restored.agent.self_model
    stress_model.personality_profile.openness = 1.0
    stress_model.personality_profile.conscientiousness = 1.0
    stress_model.personality_profile.extraversion = 0.0
    stress_model.personality_profile.agreeableness = 0.0
    stress_model.personality_profile.neuroticism = 1.0
    stress_model.personality_profile.meaning_construction_tendency = 1.0
    stress_model.personality_profile.emotional_regulation_style = 0.0
    collapse_audit = stress_model.update_continuity_audit(
        episodic_memory=list(restored.agent.long_term_memory.episodes),
        archived_memory=list(restored.agent.long_term_memory.archived_episodes),
        action_history=["rest"] * 48,
        rehearsal_queue=[
            str(item.get("episode_id", ""))
            for item in restored.agent.long_term_memory.rehearsal_batch(
                current_cycle=restored.agent.cycle + 1,
            )
            if item.get("episode_id")
        ],
        current_tick=restored.agent.cycle + 1,
    ).to_dict()

    continuity_trace = {
        "baseline": baseline_audit,
        "restart": restart_audit,
        "stress": collapse_audit,
    }
    _write_json(
        ARTIFACTS_DIR / "m218_continuity_preservation_trace.json",
        continuity_trace,
    )

    report = {
        "milestone": "M2.18",
        "status": "PASS",
        "gates": {
            "long_run_performance_not_collapsed": baseline_audit.get(
                "dominant_action_ratio",
                1.0,
            )
            < stress_model.drift_budget.action_dominance_limit,
            "identity_critical_evidence_recoverable": bool(
                baseline_audit.get("protected_anchor_ids")
            )
            and len(restored.agent.long_term_memory.rehearsal_log) > 0,
            "restart_consistency_within_tolerance": restart_audit.get(
                "restart_divergence",
                1.0,
            )
            <= stress_model.drift_budget.restart_tolerance,
            "drift_bounded_without_silent_inversion": baseline_audit.get(
                "personality_drift",
                1.0,
            )
            <= stress_model.drift_budget.personality_window
            and baseline_audit.get("narrative_drift", 1.0)
            <= stress_model.drift_budget.narrative_window,
            "anti_collapse_interventions_triggered": "action_collapse_guard"
            in collapse_audit.get("interventions", [])
            and "personality_drift_guard" in collapse_audit.get("interventions", []),
        },
        "summary": {
            "baseline_continuity_score": baseline_audit.get("continuity_score", 0.0),
            "restart_divergence": restart_audit.get("restart_divergence", 1.0),
            "stress_interventions": collapse_audit.get("interventions", []),
            "anchor_count": len(baseline_audit.get("protected_anchor_ids", [])),
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL"
    _write_json(REPORTS_DIR / "m218_acceptance_report.json", report)


if __name__ == "__main__":
    main()
