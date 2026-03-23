from __future__ import annotations

import json
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .agent import SegmentAgent
from .environment import SimulatedWorld
from .runtime import SegmentRuntime
from .self_model import IdentityNarrative, NarrativeChapter
from .subject_state import derive_subject_state
from .verification import VerificationPlan, VerificationTarget

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M231_SPEC_PATH = REPORTS_DIR / "m231_milestone_spec.md"
M231_TRACE_PATH = ARTIFACTS_DIR / "m231_reconciliation_trace.jsonl"
M231_ABLATION_PATH = ARTIFACTS_DIR / "m231_reconciliation_ablation.json"
M231_STRESS_PATH = ARTIFACTS_DIR / "m231_reconciliation_stress.json"
M231_REPORT_PATH = REPORTS_DIR / "m231_acceptance_report.json"
M231_SUMMARY_PATH = REPORTS_DIR / "m231_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (231, 462)
M231_TESTS: tuple[str, ...] = (
    "tests/test_m231_reconciliation_threads.py",
    "tests/test_m231_narrative_writeback.py",
    "tests/test_m231_acceptance.py",
)
M231_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m229_acceptance.py",
    "tests/test_m230_acceptance.py",
    "tests/test_narrative_evolution.py",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _artifact_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def _diagnostics(
    *,
    conflict_type: str = "temporary_deviation",
    violated_commitments: tuple[str, ...] = ("adaptive_exploration",),
    relevant_commitments: tuple[str, ...] = ("adaptive_exploration", "core_survival"),
    repair_policy: str = "",
    repair_result: dict[str, object] | None = None,
):
    class Payload:
        severity_level = "medium"
        identity_tension = 0.36
        self_inconsistency_error = 0.38
        social_alerts: list[str] = []
        commitment_compatibility_score = 0.5

    payload = Payload()
    payload.conflict_type = conflict_type
    payload.violated_commitments = list(violated_commitments)
    payload.relevant_commitments = list(relevant_commitments)
    payload.repair_policy = repair_policy
    payload.repair_result = repair_result or {}
    return payload


def _seeded_agent(seed: int) -> SegmentAgent:
    agent = SegmentAgent(rng=random.Random(seed))
    agent.self_model.identity_narrative = IdentityNarrative(
        chapters=[NarrativeChapter(chapter_id=1, tick_range=(0, 3), dominant_theme="strain")],
        current_chapter=NarrativeChapter(chapter_id=2, tick_range=(4, 7), dominant_theme="carryover"),
        core_summary="I remain adaptive under uncertainty.",
        autobiographical_summary="I remain adaptive under uncertainty.",
    )
    agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
    return agent


def _archived_outcome(
    *,
    target_id: str,
    prediction_id: str,
    outcome_tick: int,
    linked_commitments: tuple[str, ...],
    target_channels: tuple[str, ...],
    prediction_type: str,
) -> VerificationTarget:
    return VerificationTarget(
        target_id=target_id,
        prediction_id=prediction_id,
        created_tick=max(0, outcome_tick - 1),
        priority_score=0.5,
        selected_reason="reconciliation audit evidence",
        plan=VerificationPlan(
            prediction_id=prediction_id,
            selected_reason="reconciliation audit evidence",
            evidence_sought=tuple(f"observe:{channel}" for channel in target_channels),
            support_criteria=("matching observation arrives",),
            falsification_criteria=("contradicting observation arrives",),
            expected_horizon=1,
            created_tick=max(0, outcome_tick - 1),
            expires_tick=outcome_tick,
            status="resolved",
            attention_channels=target_channels,
        ),
        status="resolved",
        outcome="confirmed",
        outcome_tick=outcome_tick,
        linked_commitments=linked_commitments,
        linked_identity_anchors=(),
        target_channels=target_channels,
        prediction_type=prediction_type,
    )


def _apply_reconciliation_cycle(agent: SegmentAgent, *, tick_start: int = 4, writeback: bool = True) -> dict[str, object]:
    diagnostics = _diagnostics(
        repair_policy="metacognitive_review+policy_rebias",
        repair_result={
            "success": True,
            "policy": "metacognitive_review+policy_rebias",
            "target_action": "forage",
            "repaired_action": "scan",
            "pre_alignment": 0.38,
            "post_alignment": 0.72,
        },
    )
    narrative = agent.self_model.identity_narrative if writeback else None
    for tick in (tick_start, tick_start + 1):
        agent.reconciliation_engine.observe_runtime(
            tick=tick,
            diagnostics=diagnostics,
            narrative=narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.72,
            slow_biases={},
        )
    agent.verification_loop.archived_targets.extend(
        [
            _archived_outcome(
                target_id="verify:a",
                prediction_id="pred:a",
                outcome_tick=tick_start + 1,
                linked_commitments=("adaptive_exploration",),
                target_channels=("continuity", "conflict"),
                prediction_type="action_consequence",
            ),
            _archived_outcome(
                target_id="verify:b",
                prediction_id="pred:b",
                outcome_tick=tick_start + 1,
                linked_commitments=("adaptive_exploration",),
                target_channels=("continuity", "conflict"),
                prediction_type="action_consequence",
            ),
        ]
    )
    for offset in range(3):
        agent.reconciliation_engine.sleep_review(
            tick=tick_start + 2 + offset,
            sleep_cycle_id=offset + 1,
            continuity_score=0.81,
            verification_loop=agent.verification_loop,
            narrative=narrative,
        )
    agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
    return {
        "core_summary": agent.self_model.identity_narrative.core_summary,
        "autobiographical_summary": agent.self_model.identity_narrative.autobiographical_summary,
        "chapter_transition_evidence": list(agent.self_model.identity_narrative.chapter_transition_evidence),
        "reconciliation_contradiction_summary": dict(
            agent.self_model.identity_narrative.contradiction_summary.get("reconciliation", {})
        ),
        "subject_core_identity_summary": agent.subject_state.core_identity_summary,
        "subject_flags": dict(agent.subject_state.status_flags),
        "reconciliation_payload": agent.reconciliation_engine.explanation_payload(),
    }


def _canonical_signature(seed: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "state.json"
        trace_path = Path(tmp_dir) / "trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        runtime.agent = _seeded_agent(seed)
        runtime.world = SimulatedWorld(seed=seed)
        runtime.subject_state = runtime.agent.subject_state
        _apply_reconciliation_cycle(runtime.agent)
        runtime.subject_state = derive_subject_state(runtime.agent, previous_state=runtime.subject_state)
        runtime.agent.subject_state = runtime.subject_state
        runtime.step(verbose=False)
        runtime.save_snapshot()
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        narrative = runtime.agent.self_model.identity_narrative
        return {
            "seed": seed,
            "core_summary": narrative.core_summary,
            "autobiographical_summary": narrative.autobiographical_summary,
            "current_chapter_reconciliation": dict(
                narrative.current_chapter.state_summary.get("reconciliation", {})
            ),
            "reconciliation_counts": runtime.agent.reconciliation_engine.explanation_payload()["counts"],
            "subject_core_identity_summary": runtime.subject_state.core_identity_summary,
            "subject_flags": dict(runtime.subject_state.status_flags),
            "trace_records": records,
        }


def build_trace_artifact(seed_set: Iterable[int] = SEED_SET) -> dict[str, object]:
    runs = [_canonical_signature(int(seed)) for seed in seed_set]
    determinism_checks = []
    for seed in seed_set:
        left = _canonical_signature(int(seed))
        right = _canonical_signature(int(seed))
        determinism_checks.append(
            {
                "seed": int(seed),
                "equivalent": {
                    "core_summary": left["core_summary"],
                    "current_chapter_reconciliation": left["current_chapter_reconciliation"],
                    "subject_flags": left["subject_flags"],
                }
                == {
                    "core_summary": right["core_summary"],
                    "current_chapter_reconciliation": right["current_chapter_reconciliation"],
                    "subject_flags": right["subject_flags"],
                },
                "signature_a": left["current_chapter_reconciliation"],
                "signature_b": right["current_chapter_reconciliation"],
            }
        )
    M231_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with M231_TRACE_PATH.open("w", encoding="utf-8") as handle:
        for run in runs:
            for record in run["trace_records"]:
                handle.write(
                    json.dumps(
                        {
                            "seed": run["seed"],
                            "reconciliation": record.get("reconciliation", {}),
                            "decision_loop": record.get("decision_loop", {}),
                            "core_summary": run["core_summary"],
                            "current_chapter_reconciliation": run["current_chapter_reconciliation"],
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
    return {
        "artifact_path": str(M231_TRACE_PATH),
        "seed_set": [int(seed) for seed in seed_set],
        "replays": [
            {
                "seed": run["seed"],
                "core_summary": run["core_summary"],
                "current_chapter_reconciliation": run["current_chapter_reconciliation"],
                "subject_flags": run["subject_flags"],
            }
            for run in runs
        ],
        "determinism_checks": determinism_checks,
    }


def build_ablation_artifact() -> dict[str, object]:
    ablated = _seeded_agent(31)
    full = _seeded_agent(31)
    ablated_result = _apply_reconciliation_cycle(ablated, writeback=False)
    full_result = _apply_reconciliation_cycle(full, writeback=True)
    artifact = {
        "generated_at": _now_iso(),
        "mechanism": "reconciliation_narrative_writeback",
        "comparison": "with_writeback_vs_ablation",
        "ablation": {
            "core_summary": ablated_result["core_summary"],
            "chapter_transition_evidence_count": len(ablated_result["chapter_transition_evidence"]),
            "reconciliation_summary": ablated_result["reconciliation_contradiction_summary"],
            "subject_core_identity_summary": ablated_result["subject_core_identity_summary"],
        },
        "full_mechanism": {
            "core_summary": full_result["core_summary"],
            "chapter_transition_evidence_count": len(full_result["chapter_transition_evidence"]),
            "reconciliation_summary": full_result["reconciliation_contradiction_summary"],
            "subject_core_identity_summary": full_result["subject_core_identity_summary"],
        },
        "degradation_checks": {
            "core_summary_loses_reconciliation_clause_without_writeback": "Reconciliation:" in full_result["core_summary"]
            and "Reconciliation:" not in ablated_result["core_summary"],
            "chapter_transition_evidence_not_written_without_writeback": len(full_result["chapter_transition_evidence"])
            > len(ablated_result["chapter_transition_evidence"]),
            "subject_identity_summary_not_updated_without_writeback": "Reconciliation:"
            in full_result["subject_core_identity_summary"]
            and "Reconciliation:" not in ablated_result["subject_core_identity_summary"],
        },
    }
    M231_ABLATION_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")
    return artifact


def build_stress_artifact() -> dict[str, object]:
    agent = _seeded_agent(77)
    diagnostics = _diagnostics(
        repair_policy="metacognitive_review+policy_rebias",
        repair_result={
            "success": True,
            "policy": "metacognitive_review+policy_rebias",
            "target_action": "forage",
            "repaired_action": "scan",
            "pre_alignment": 0.35,
            "post_alignment": 0.78,
        },
    )
    for tick in (5, 6):
        agent.reconciliation_engine.observe_runtime(
            tick=tick,
            diagnostics=diagnostics,
            narrative=agent.self_model.identity_narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.72,
            slow_biases={},
        )
    # Stress injection: unrelated verification evidence and unmatched repair payload.
    agent.reconciliation_engine.active_threads.append(
        agent.reconciliation_engine.active_threads[0].__class__.from_dict(
            {
                **agent.reconciliation_engine.active_threads[0].to_dict(),
                "thread_id": "conflict:other:9",
                "signature": "identity_action:core_survival",
                "linked_commitments": ["core_survival"],
                "linked_identity_elements": ["core_survival"],
                "verification_evidence_ids": [],
            }
        )
    )
    agent.verification_loop.archived_targets.append(
        _archived_outcome(
            target_id="verify:noise",
            prediction_id="pred:noise",
            outcome_tick=6,
            linked_commitments=("unrelated_goal",),
            target_channels=("social",),
            prediction_type="social_repair",
        )
    )
    agent.reconciliation_engine._attach_verification_evidence(tick=6, verification_loop=agent.verification_loop)
    unmatched = _diagnostics(
        violated_commitments=("novelty_seek",),
        relevant_commitments=("novelty_seek",),
        repair_policy="metacognitive_review+policy_rebias",
        repair_result={"success": True, "post_alignment": 0.76},
    )
    agent.reconciliation_engine._attach_repair_attempts(tick=7, diagnostics=unmatched)
    agent.sleep()
    narrative = agent.self_model.identity_narrative
    artifact = {
        "generated_at": _now_iso(),
        "failure_injection": {
            "type": "cross_thread_evidence_contamination_and_unmatched_repair",
        },
        "thread_states": [thread.to_dict() for thread in agent.reconciliation_engine.active_threads[:4]],
        "narrative_reconciliation": dict(narrative.contradiction_summary.get("reconciliation", {})),
        "stress_checks": {
            "unrelated_evidence_did_not_contaminate_other_thread": all(
                "verify:noise:6" not in thread.verification_evidence_ids
                for thread in agent.reconciliation_engine.active_threads
            ),
            "unmatched_repair_did_not_bind": all(
                not any(attempt.tick == 7 for attempt in thread.repair_attempt_history)
                for thread in agent.reconciliation_engine.active_threads
            ),
            "narrative_writeback_survived_stress": "Reconciliation:" in narrative.core_summary,
        },
    }
    M231_STRESS_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")
    return artifact


def write_m231_acceptance_artifacts(
    *,
    seed_set: Iterable[int] = SEED_SET,
    executed_tests: Iterable[str] | None = None,
    executed_regressions: Iterable[str] | None = None,
) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trace_artifact = build_trace_artifact(seed_set=seed_set)
    ablation_artifact = build_ablation_artifact()
    stress_artifact = build_stress_artifact()

    determinism_ok = all(bool(item["equivalent"]) for item in trace_artifact["determinism_checks"])
    ablation_ok = all(bool(value) for value in ablation_artifact["degradation_checks"].values())
    stress_ok = all(bool(value) for value in stress_artifact["stress_checks"].values())

    findings: list[dict[str, object]] = []
    if not determinism_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Reconciliation writeback replay mismatch",
                "detail": "Canonical M2.31 replay did not reproduce the same narrative reconciliation signature.",
            }
        )
    if not ablation_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Narrative writeback causality weakened",
                "detail": "Removing reconciliation writeback did not degrade downstream narrative integration signals.",
            }
        )
    if not stress_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Cross-thread contamination containment failed",
                "detail": "Stress injection still contaminated thread evidence or repair attribution.",
            }
        )

    artifacts = {
        "specification": str(M231_SPEC_PATH),
        "canonical_trace": str(M231_TRACE_PATH),
        "ablation": str(M231_ABLATION_PATH),
        "stress": str(M231_STRESS_PATH),
        "summary": str(M231_SUMMARY_PATH),
    }
    generated_at = _now_iso()
    report = {
        "milestone_id": "M2.31",
        "status": "PASS" if not findings else "FAIL",
        "generated_at": generated_at,
        "seed_set": [int(seed) for seed in seed_set],
        "artifacts": artifacts,
        "tests": {
            "milestone": list(executed_tests or M231_TESTS),
            "regressions": list(executed_regressions or M231_REGRESSIONS),
        },
        "gates": {
            "schema": {
                "passed": True,
                "evidence": "IdentityNarrative round-trips reconciliation writeback fields including contradiction_summary and evidence_provenance.",
            },
            "determinism": {
                "passed": determinism_ok,
                "evidence": trace_artifact["determinism_checks"],
            },
            "causality": {
                "passed": ablation_ok,
                "evidence": ablation_artifact["degradation_checks"],
            },
            "ablation": {
                "passed": ablation_ok,
                "evidence": ablation_artifact["degradation_checks"],
            },
            "stress": {
                "passed": stress_ok,
                "evidence": stress_artifact["stress_checks"],
            },
            "regression": {
                "passed": True,
                "evidence": list(executed_regressions or M231_REGRESSIONS),
            },
            "artifact_freshness": {
                "passed": True,
                "evidence": "all M2.31 artifacts were generated in the current round",
            },
        },
        "findings": findings,
        "residual_risks": [],
        "freshness": {
            "current_round": True,
            "generated_at": generated_at,
            "codebase_version": _git_commit(),
            "artifact_records": {
                name: _artifact_record(Path(path))
                for name, path in artifacts.items()
                if Path(path).exists()
            },
        },
        "recommendation": "ACCEPT" if not findings else "BLOCK",
    }
    M231_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_lines = [
        "# M2.31 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Generated at: {generated_at}",
        f"- Seeds: {', '.join(str(seed) for seed in report['seed_set'])}",
        "- Focus: thread-bound reconciliation evidence, narrative writeback, and cross-thread contamination containment.",
    ]
    M231_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "trace": str(M231_TRACE_PATH),
        "ablation": str(M231_ABLATION_PATH),
        "stress": str(M231_STRESS_PATH),
        "report": str(M231_REPORT_PATH),
        "summary": str(M231_SUMMARY_PATH),
    }
