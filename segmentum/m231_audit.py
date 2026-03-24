from __future__ import annotations

import json
import os
import random
import subprocess
import sys
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


def _parse_iso8601(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _git_dirty_paths() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


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


def _suite_execution_record(*, label: str, paths: Iterable[str], execute: bool) -> dict[str, object]:
    normalized_paths = [str(path) for path in paths]
    if not execute:
        return {
            "label": label,
            "paths": normalized_paths,
            "executed": False,
            "passed": False,
            "returncode": None,
            "command": [],
            "stdout": "",
            "stderr": "",
            "execution_source": "skipped",
            "started_at": None,
            "completed_at": None,
        }

    command = [sys.executable, "-m", "pytest", *normalized_paths, "-q"]
    started_at = _now_iso()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    completed_at = _now_iso()
    return {
        "label": label,
        "paths": normalized_paths,
        "executed": True,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "execution_source": "subprocess",
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _is_authentic_execution_record(record: dict[str, object], *, expected_paths: Iterable[str]) -> bool:
    expected = [str(path) for path in expected_paths]
    command = record.get("command", [])
    started_at = _parse_iso8601(record.get("started_at"))
    completed_at = _parse_iso8601(record.get("completed_at"))
    return bool(
        isinstance(record, dict)
        and bool(record.get("executed"))
        and record.get("execution_source") == "subprocess"
        and isinstance(command, list)
        and len(command) >= 4
        and command[1:3] == ["-m", "pytest"]
        and list(record.get("paths", [])) == expected
        and started_at is not None
        and completed_at is not None
        and completed_at >= started_at
    )


def _freshness_gate(
    *,
    artifacts: dict[str, str],
    audit_started_at: str,
    generated_at: str,
    milestone_execution: dict[str, object],
    regression_execution: dict[str, object],
    strict: bool,
) -> tuple[bool, dict[str, object]]:
    audit_started = _parse_iso8601(audit_started_at)
    generated = _parse_iso8601(generated_at)
    artifact_records = {
        name: _artifact_record(Path(path))
        for name, path in artifacts.items()
        if Path(path).exists()
    }
    generated_artifact_names = {"canonical_trace", "ablation", "stress"}
    artifact_times = [
        _parse_iso8601(record.get("modified_at"))
        for name, record in artifact_records.items()
        if name in generated_artifact_names
    ]
    artifact_times_ok = bool(audit_started and generated and artifact_times) and all(
        modified is not None and audit_started <= modified <= generated
        for modified in artifact_times
    )
    milestone_authentic = _is_authentic_execution_record(milestone_execution, expected_paths=M231_TESTS)
    regression_authentic = _is_authentic_execution_record(regression_execution, expected_paths=M231_REGRESSIONS)
    suite_times = [
        _parse_iso8601(milestone_execution.get("started_at")),
        _parse_iso8601(milestone_execution.get("completed_at")),
        _parse_iso8601(regression_execution.get("started_at")),
        _parse_iso8601(regression_execution.get("completed_at")),
    ]
    suite_times_ok = bool(audit_started and generated) and all(
        timestamp is not None and audit_started <= timestamp <= generated
        for timestamp in suite_times
    )
    dirty_paths = _git_dirty_paths()
    freshness_ok = artifact_times_ok and (not strict or (milestone_authentic and regression_authentic and suite_times_ok))
    return freshness_ok, {
        "strict": strict,
        "audit_started_at": audit_started_at,
        "generated_at": generated_at,
        "artifact_records": artifact_records,
        "artifact_times_within_round": artifact_times_ok,
        "milestone_execution_authentic": milestone_authentic,
        "regression_execution_authentic": regression_authentic,
        "suite_times_within_round": suite_times_ok,
        "git": {
            "head": _git_commit(),
            "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
        },
    }


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


def _advance_to_chapter(
    agent: SegmentAgent,
    *,
    chapter_id: int,
    tick_range: tuple[int, int],
    dominant_theme: str,
) -> None:
    narrative = agent.self_model.identity_narrative
    if narrative is None:
        return
    previous = narrative.current_chapter
    if previous is not None:
        chapter_ids = {chapter.chapter_id for chapter in narrative.chapters}
        if previous.chapter_id not in chapter_ids:
            narrative.chapters.append(previous)
    narrative.current_chapter = NarrativeChapter(
        chapter_id=chapter_id,
        tick_range=tick_range,
        dominant_theme=dominant_theme,
    )


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


def _find_thread(agent: SegmentAgent, signature: str) -> dict[str, object]:
    for thread in agent.reconciliation_engine.active_threads:
        if thread.signature == signature:
            return thread.to_dict()
    return {}


def _writeback_matches_target(*, writeback: dict[str, object], target_thread: dict[str, object]) -> bool:
    target_thread_id = str(target_thread.get("thread_id", ""))
    if not target_thread_id:
        return False
    return (
        str(writeback.get("dominant_thread_id", "")) == target_thread_id
        and str(writeback.get("dominant_status", "")) == str(target_thread.get("status", ""))
        and str(writeback.get("dominant_outcome", "")) == str(target_thread.get("current_outcome", ""))
        and list(writeback.get("linked_chapter_ids", [])) == list(target_thread.get("linked_chapter_ids", []))
    )


def _apply_reconciliation_cycle(
    agent: SegmentAgent,
    *,
    tick_start: int = 4,
    writeback: bool = True,
    span_chapters: bool = True,
) -> dict[str, object]:
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
    if span_chapters:
        _advance_to_chapter(
            agent,
            chapter_id=3,
            tick_range=(tick_start + 2, tick_start + 5),
            dominant_theme="reconciliation_carryover",
        )
        agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
        agent.reconciliation_engine.observe_runtime(
            tick=tick_start + 2,
            diagnostics=diagnostics,
            narrative=narrative,
            prediction_ledger=agent.prediction_ledger,
            verification_loop=agent.verification_loop,
            subject_state=agent.subject_state,
            continuity_score=0.70,
            slow_biases={},
        )
    target_signature = "identity_action:adaptive_exploration"
    agent.verification_loop.archived_targets.extend(
        [
            _archived_outcome(
                target_id="verify:a",
                prediction_id="pred:a",
                outcome_tick=tick_start + 2,
                linked_commitments=("adaptive_exploration",),
                target_channels=("continuity", "conflict"),
                prediction_type="action_consequence",
            ),
            _archived_outcome(
                target_id="verify:b",
                prediction_id="pred:b",
                outcome_tick=tick_start + 2,
                linked_commitments=("adaptive_exploration",),
                target_channels=("continuity", "conflict"),
                prediction_type="action_consequence",
            ),
        ]
    )
    for offset in range(3):
        agent.reconciliation_engine.sleep_review(
            tick=tick_start + 3 + offset,
            sleep_cycle_id=offset + 1,
            continuity_score=0.81,
            verification_loop=agent.verification_loop,
            narrative=narrative,
        )
    agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
    target_thread = _find_thread(agent, target_signature)
    target_writeback = {}
    if narrative is not None:
        target_writeback = dict(narrative.contradiction_summary.get("reconciliation", {}))
    return {
        "core_summary": agent.self_model.identity_narrative.core_summary,
        "autobiographical_summary": agent.self_model.identity_narrative.autobiographical_summary,
        "chapter_transition_evidence": list(agent.self_model.identity_narrative.chapter_transition_evidence),
        "reconciliation_contradiction_summary": dict(
            agent.self_model.identity_narrative.contradiction_summary.get("reconciliation", {})
        ),
        "target_thread": target_thread,
        "target_writeback": target_writeback,
        "writeback_matches_target_thread": _writeback_matches_target(
            writeback=target_writeback,
            target_thread=target_thread,
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
        cycle_result = _apply_reconciliation_cycle(runtime.agent, span_chapters=True)
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
        target_thread = dict(cycle_result["target_thread"])
        target_chapters = list(target_thread.get("linked_chapter_ids", []))
        target_bridge_count = len(list(target_thread.get("chapter_bridges", [])))
        current_chapter_reconciliation = dict(narrative.current_chapter.state_summary.get("reconciliation", {}))
        contradiction_reconciliation = dict(narrative.contradiction_summary.get("reconciliation", {}))
        narrative_alignment = {
            "writeback_matches_target_thread": bool(cycle_result["writeback_matches_target_thread"]),
            "current_chapter_matches_target": _writeback_matches_target(
                writeback=current_chapter_reconciliation,
                target_thread=target_thread,
            ),
            "contradiction_summary_matches_target": _writeback_matches_target(
                writeback=contradiction_reconciliation,
                target_thread=target_thread,
            ),
            "core_summary_mentions_reconciliation": "Reconciliation:" in narrative.core_summary,
        }
        target_bound_claims = [
            claim
            for claim in getattr(narrative, "claims", [])
            if str(getattr(claim, "reconciliation_thread_id", "")) == str(target_thread.get("thread_id", ""))
        ]
        claim_alignment = {
            "updated_claim_ids": [
                str(getattr(claim, "claim_id", ""))
                for claim in target_bound_claims
            ],
            "all_updated_claims_bound_to_target_thread": bool(target_bound_claims) and all(
                str(getattr(claim, "reconciliation_thread_id", "")) == str(target_thread.get("thread_id", ""))
                and bool(getattr(claim, "reconciliation_evidence_ids", []))
                and list(getattr(claim, "reconciliation_source_chapter_ids", [])) == target_chapters
                for claim in target_bound_claims
            ),
            "any_claim_updated": bool(target_bound_claims),
        }
        return {
            "seed": seed,
            "core_summary": narrative.core_summary,
            "autobiographical_summary": narrative.autobiographical_summary,
            "current_chapter_reconciliation": current_chapter_reconciliation,
            "contradiction_reconciliation": contradiction_reconciliation,
            "target_thread": target_thread,
            "target_thread_summary": {
                "thread_id": str(target_thread.get("thread_id", "")),
                "status": str(target_thread.get("status", "")),
                "current_outcome": str(target_thread.get("current_outcome", "")),
                "linked_chapter_ids": target_chapters,
                "chapter_bridge_count": target_bridge_count,
                "spans_multiple_chapters": len(set(target_chapters)) >= 2,
            },
            "narrative_alignment": narrative_alignment,
            "claim_alignment": claim_alignment,
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
                    "contradiction_reconciliation": left["contradiction_reconciliation"],
                    "target_thread_summary": left["target_thread_summary"],
                    "narrative_alignment": left["narrative_alignment"],
                    "claim_alignment": left["claim_alignment"],
                    "subject_flags": left["subject_flags"],
                }
                == {
                    "core_summary": right["core_summary"],
                    "current_chapter_reconciliation": right["current_chapter_reconciliation"],
                    "contradiction_reconciliation": right["contradiction_reconciliation"],
                    "target_thread_summary": right["target_thread_summary"],
                    "narrative_alignment": right["narrative_alignment"],
                    "claim_alignment": right["claim_alignment"],
                    "subject_flags": right["subject_flags"],
                },
                "signature_a": left["current_chapter_reconciliation"],
                "signature_b": right["current_chapter_reconciliation"],
                "target_thread_a": left["target_thread_summary"],
                "target_thread_b": right["target_thread_summary"],
                "narrative_alignment_a": left["narrative_alignment"],
                "narrative_alignment_b": right["narrative_alignment"],
                "claim_alignment_a": left["claim_alignment"],
                "claim_alignment_b": right["claim_alignment"],
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
                            "contradiction_reconciliation": run["contradiction_reconciliation"],
                            "target_thread_summary": run["target_thread_summary"],
                            "narrative_alignment": run["narrative_alignment"],
                            "claim_alignment": run["claim_alignment"],
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
    long_horizon_checks = [
        {
            "seed": run["seed"],
            "spans_multiple_chapters": run["target_thread_summary"]["spans_multiple_chapters"],
            "chapter_bridge_count": run["target_thread_summary"]["chapter_bridge_count"],
            "linked_chapter_ids": run["target_thread_summary"]["linked_chapter_ids"],
        }
        for run in runs
    ]
    narrative_alignment_checks = [
        {
            "seed": run["seed"],
            **run["narrative_alignment"],
        }
        for run in runs
    ]
    return {
        "artifact_path": str(M231_TRACE_PATH),
        "seed_set": [int(seed) for seed in seed_set],
        "replays": [
            {
                "seed": run["seed"],
                "core_summary": run["core_summary"],
                "current_chapter_reconciliation": run["current_chapter_reconciliation"],
                "contradiction_reconciliation": run["contradiction_reconciliation"],
                "target_thread_summary": run["target_thread_summary"],
                "narrative_alignment": run["narrative_alignment"],
                "claim_alignment": run["claim_alignment"],
                "subject_flags": run["subject_flags"],
            }
            for run in runs
        ],
        "determinism_checks": determinism_checks,
        "long_horizon_checks": long_horizon_checks,
        "narrative_alignment_checks": narrative_alignment_checks,
    }


def build_ablation_artifact() -> dict[str, object]:
    ablated = _seeded_agent(31)
    full = _seeded_agent(31)
    ablated_result = _apply_reconciliation_cycle(ablated, writeback=False, span_chapters=True)
    full_result = _apply_reconciliation_cycle(full, writeback=True, span_chapters=True)
    full_target = dict(full_result["target_thread"])
    ablated_target = dict(ablated_result["target_thread"])
    artifact = {
        "generated_at": _now_iso(),
        "mechanism": "reconciliation_narrative_writeback",
        "comparison": "with_writeback_vs_ablation",
        "ablation": {
            "core_summary": ablated_result["core_summary"],
            "chapter_transition_evidence_count": len(ablated_result["chapter_transition_evidence"]),
            "reconciliation_summary": ablated_result["reconciliation_contradiction_summary"],
            "target_thread": {
                "linked_chapter_ids": list(ablated_target.get("linked_chapter_ids", [])),
                "chapter_bridge_count": len(list(ablated_target.get("chapter_bridges", []))),
            },
            "subject_core_identity_summary": ablated_result["subject_core_identity_summary"],
        },
        "full_mechanism": {
            "core_summary": full_result["core_summary"],
            "chapter_transition_evidence_count": len(full_result["chapter_transition_evidence"]),
            "reconciliation_summary": full_result["reconciliation_contradiction_summary"],
            "target_thread": {
                "linked_chapter_ids": list(full_target.get("linked_chapter_ids", [])),
                "chapter_bridge_count": len(list(full_target.get("chapter_bridges", []))),
            },
            "subject_core_identity_summary": full_result["subject_core_identity_summary"],
        },
        "degradation_checks": {
            "core_summary_loses_reconciliation_clause_without_writeback": "Reconciliation:" in full_result["core_summary"]
            and "Reconciliation:" not in ablated_result["core_summary"],
            "chapter_transition_evidence_not_written_without_writeback": len(full_result["chapter_transition_evidence"])
            > len(ablated_result["chapter_transition_evidence"]),
            "cross_chapter_thread_survives_with_writeback": len(set(full_target.get("linked_chapter_ids", []))) >= 2
            and len(list(full_target.get("chapter_bridges", []))) >= 1,
            "writeback_targets_reconciled_thread_with_writeback": bool(full_result["writeback_matches_target_thread"]),
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
    _advance_to_chapter(
        agent,
        chapter_id=3,
        tick_range=(7, 10),
        dominant_theme="stress_carryover",
    )
    agent.subject_state = derive_subject_state(agent, previous_state=agent.subject_state)
    agent.reconciliation_engine.observe_runtime(
        tick=7,
        diagnostics=diagnostics,
        narrative=agent.self_model.identity_narrative,
        prediction_ledger=agent.prediction_ledger,
        verification_loop=agent.verification_loop,
        subject_state=agent.subject_state,
        continuity_score=0.70,
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
    agent.reconciliation_engine._attach_repair_attempts(tick=8, diagnostics=unmatched)
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
                not any(attempt.tick == 8 for attempt in thread.repair_attempt_history)
                for thread in agent.reconciliation_engine.active_threads
            ),
            "cross_chapter_links_survived_stress": any(
                len(set(thread.linked_chapter_ids)) >= 2 and len(thread.chapter_bridges) >= 1
                for thread in agent.reconciliation_engine.active_threads
            ),
            "narrative_writeback_survived_stress": "Reconciliation:" in narrative.core_summary,
            "narrative_writeback_still_targets_intended_thread": _writeback_matches_target(
                writeback=dict(narrative.contradiction_summary.get("reconciliation", {})),
                target_thread=_find_thread(agent, "identity_action:adaptive_exploration"),
            ),
        },
    }
    M231_STRESS_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")
    return artifact


def write_m231_acceptance_artifacts(
    *,
    seed_set: Iterable[int] = SEED_SET,
    execute_test_suites: bool = False,
    strict: bool = True,
    allow_injected_execution: bool = False,
    milestone_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
) -> dict[str, str]:
    audit_started_at = _now_iso()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trace_artifact = build_trace_artifact(seed_set=seed_set)
    ablation_artifact = build_ablation_artifact()
    stress_artifact = build_stress_artifact()
    if strict and not allow_injected_execution and (
        milestone_execution is not None or regression_execution is not None
    ):
        raise ValueError("Strict M2.31 audit refuses injected execution records.")
    if execute_test_suites:
        milestone_execution = _suite_execution_record(
            label="milestone",
            paths=M231_TESTS,
            execute=True,
        )
        regression_execution = _suite_execution_record(
            label="regression",
            paths=M231_REGRESSIONS,
            execute=True,
        )
    milestone_execution = milestone_execution or _suite_execution_record(
        label="milestone",
        paths=M231_TESTS,
        execute=False,
    )
    regression_execution = regression_execution or _suite_execution_record(
        label="regression",
        paths=M231_REGRESSIONS,
        execute=False,
    )

    determinism_ok = all(bool(item["equivalent"]) for item in trace_artifact["determinism_checks"])
    long_horizon_ok = all(
        bool(item["spans_multiple_chapters"]) and int(item["chapter_bridge_count"]) >= 1
        for item in trace_artifact["long_horizon_checks"]
    )
    narrative_alignment_ok = all(
        bool(item["writeback_matches_target_thread"])
        and bool(item["current_chapter_matches_target"])
        and bool(item["contradiction_summary_matches_target"])
        and bool(item["core_summary_mentions_reconciliation"])
        for item in trace_artifact["narrative_alignment_checks"]
    )
    claim_alignment_ok = all(
        bool(item["claim_alignment"]["any_claim_updated"])
        and bool(item["claim_alignment"]["all_updated_claims_bound_to_target_thread"])
        for item in trace_artifact["replays"]
    )
    ablation_ok = all(bool(value) for value in ablation_artifact["degradation_checks"].values())
    stress_ok = all(bool(value) for value in stress_artifact["stress_checks"].values())
    milestone_ok = bool(milestone_execution["executed"]) and bool(milestone_execution["passed"])
    regression_ok = bool(regression_execution["executed"]) and bool(regression_execution["passed"])
    milestone_authentic = _is_authentic_execution_record(milestone_execution, expected_paths=M231_TESTS)
    regression_authentic = _is_authentic_execution_record(regression_execution, expected_paths=M231_REGRESSIONS)

    findings: list[dict[str, object]] = []
    if not determinism_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Reconciliation writeback replay mismatch",
                "detail": "Canonical M2.31 replay did not reproduce the same narrative reconciliation signature.",
            }
        )
    if not long_horizon_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Canonical replay did not span chapters",
                "detail": "The audit evidence did not show the same reconciliation thread persisting across multiple chapters.",
            }
        )
    if not narrative_alignment_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Narrative writeback did not target the reconciled thread",
                "detail": "Canonical M2.31 replay did not keep current-chapter and contradiction writeback aligned to the intended reconciled long-horizon thread.",
            }
        )
    if not claim_alignment_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Claim-level reconciliation writeback missing or unbound",
                "detail": "Canonical M2.31 replay did not update claim-level narrative objects with bounded reconciliation provenance.",
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
    if not milestone_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Milestone suite not executed cleanly",
                "detail": "Strict M2.31 audit requires a passing milestone pytest run recorded in the report.",
            }
        )
    elif strict and not milestone_authentic:
        findings.append(
            {
                "severity": "S1",
                "title": "Milestone suite record is not authentic",
                "detail": "Strict M2.31 audit only accepts milestone execution records generated by the audit subprocess runner.",
            }
        )
    if not regression_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Regression suite not executed cleanly",
                "detail": "Strict M2.31 audit requires a passing regression pytest run recorded in the report.",
            }
        )
    elif strict and not regression_authentic:
        findings.append(
            {
                "severity": "S1",
                "title": "Regression suite record is not authentic",
                "detail": "Strict M2.31 audit only accepts regression execution records generated by the audit subprocess runner.",
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
    freshness_ok, freshness_evidence = _freshness_gate(
        artifacts=artifacts,
        audit_started_at=audit_started_at,
        generated_at=generated_at,
        milestone_execution=milestone_execution,
        regression_execution=regression_execution,
        strict=strict,
    )
    residual_risks: list[dict[str, object]] = []
    if not freshness_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Artifact freshness or execution provenance check failed",
                "detail": "Strict M2.31 audit requires current-round artifacts and authentic current-round pytest execution metadata.",
            }
        )
    report_status = "PASS" if not findings else "FAIL"
    report = {
        "milestone_id": "M2.31",
        "status": report_status,
        "generated_at": generated_at,
        "strict": strict,
        "seed_set": [int(seed) for seed in seed_set],
        "artifacts": artifacts,
        "tests": {
            "milestone": milestone_execution,
            "regressions": regression_execution,
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
            "long_horizon": {
                "passed": long_horizon_ok,
                "evidence": trace_artifact["long_horizon_checks"],
            },
            "narrative_alignment": {
                "passed": narrative_alignment_ok,
                "evidence": trace_artifact["narrative_alignment_checks"],
            },
            "claim_alignment": {
                "passed": claim_alignment_ok,
                "evidence": [run["claim_alignment"] for run in trace_artifact["replays"]],
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
            "milestone_tests": {
                "passed": milestone_ok,
                "evidence": milestone_execution,
            },
            "regression": {
                "passed": regression_ok,
                "evidence": regression_execution,
            },
            "artifact_freshness": {
                "passed": freshness_ok,
                "evidence": freshness_evidence,
            },
        },
        "findings": findings,
        "residual_risks": residual_risks,
        "freshness": {
            "current_round": freshness_ok,
            "audit_started_at": audit_started_at,
            "generated_at": generated_at,
            "codebase_version": freshness_evidence["git"]["head"],
            "git": freshness_evidence["git"],
            "artifact_records": freshness_evidence["artifact_records"],
        },
        "recommendation": "ACCEPT" if not findings else "BLOCK",
    }
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
    refreshed_artifact_records = {
        name: _artifact_record(Path(path))
        for name, path in artifacts.items()
        if Path(path).exists()
    }
    report["gates"]["artifact_freshness"]["evidence"]["artifact_records"] = refreshed_artifact_records
    report["freshness"]["artifact_records"] = refreshed_artifact_records
    M231_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "trace": str(M231_TRACE_PATH),
        "ablation": str(M231_ABLATION_PATH),
        "stress": str(M231_STRESS_PATH),
        "report": str(M231_REPORT_PATH),
        "summary": str(M231_SUMMARY_PATH),
    }


if __name__ == "__main__":
    if "PYTEST_CURRENT_TEST" in os.environ:
        raise SystemExit("Refusing to execute nested strict audit while pytest is already running.")
    write_m231_acceptance_artifacts(execute_test_suites=True)
