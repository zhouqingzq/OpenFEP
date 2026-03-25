from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .inquiry_scheduler import (
    InquiryBudgetScheduler,
    InquiryBudgetState,
    InquirySchedulingDecision,
    apply_scheduler_to_experiment_design,
)
from .narrative_experiment import ExperimentDesignResult, ExperimentPlan
from .narrative_uncertainty import (
    DecisionRelevanceMap,
    NarrativeUnknown,
    UncertaintyDecompositionResult,
)
from .prediction_ledger import PredictionHypothesis, PredictionLedger
from .runtime import SegmentRuntime
from .subject_state import SubjectState
from .verification import VerificationLoop, VerificationOutcome, VerificationPlan, VerificationTarget

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
SCHEMA_VERSION = "m235_audit_v1"

M235_SPEC_PATH = REPORTS_DIR / "m235_milestone_spec.md"
M235_PREPARATION_PATH = REPORTS_DIR / "m235_strict_audit_preparation.md"
M235_TRACE_PATH = ARTIFACTS_DIR / "m235_inquiry_scheduler_trace.jsonl"
M235_ABLATION_PATH = ARTIFACTS_DIR / "m235_inquiry_scheduler_ablation.json"
M235_STRESS_PATH = ARTIFACTS_DIR / "m235_inquiry_scheduler_stress.json"
M235_REPORT_PATH = REPORTS_DIR / "m235_acceptance_report.json"
M235_SUMMARY_PATH = REPORTS_DIR / "m235_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (235, 470)
M235_TESTS: tuple[str, ...] = (
    "tests/test_m235_inquiry_scheduler.py",
    "tests/test_m235_acceptance.py",
    "tests/test_m235_audit_preparation.py",
)
M235_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m234_experiment_design.py",
    "tests/test_m228_prediction_ledger.py",
    "tests/test_m229_verification_loop.py",
    "tests/test_runtime.py",
)
M235_GATES: tuple[str, ...] = (
    "cross_surface_ranking",
    "verification_budgeting",
    "workspace_allocation",
    "action_biasing",
    "downstream_causality",
    "snapshot_roundtrip",
    "regression",
    "artifact_freshness",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso8601(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _artifact_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def _same_contents(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


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


def _git_status_path(entry: str) -> str:
    parts = str(entry).strip().split(maxsplit=1)
    if len(parts) == 1:
        return parts[0]
    payload = parts[1].strip()
    if " -> " in payload:
        payload = payload.split(" -> ", 1)[1].strip()
    return payload


def _is_generated_m235_artifact(path: str) -> bool:
    normalized = str(path).replace("\\", "/")
    allowed_prefixes = (
        "artifacts/m235_",
        "reports/m235_acceptance_",
    )
    return normalized.startswith(allowed_prefixes) or normalized.startswith(".pytest_m235_")


def _strict_dirty_findings(dirty_paths: Iterable[str]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    blocking_paths = [path for path in (_git_status_path(item) for item in dirty_paths) if path and not _is_generated_m235_artifact(path)]
    if blocking_paths:
        findings.append(
            {
                "severity": "S1",
                "title": "Strict audit baseline is not frozen",
                "detail": (
                    "Strict M2.35 acceptance cannot rely on a dirty code or specification baseline. "
                    "Freeze or commit non-artifact changes before claiming strict PASS."
                ),
                "paths": blocking_paths,
            }
        )
    return findings


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
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
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
    evidence_times = [
        _parse_iso8601(record.get("modified_at"))
        for name, record in artifact_records.items()
        if name in {"canonical_trace", "ablation", "stress"}
    ]
    evidence_ok = bool(audit_started and generated and evidence_times) and all(
        modified is not None and audit_started <= modified <= generated
        for modified in evidence_times
    )
    report_times = [
        _parse_iso8601(record.get("modified_at"))
        for name, record in artifact_records.items()
        if name in {"report", "summary"}
    ]
    report_ok = bool(audit_started and report_times) and all(
        modified is not None and audit_started <= modified
        for modified in report_times
    )
    milestone_auth = _is_authentic_execution_record(milestone_execution, expected_paths=M235_TESTS)
    regression_auth = _is_authentic_execution_record(regression_execution, expected_paths=M235_REGRESSIONS)
    suite_times = [
        _parse_iso8601(milestone_execution.get("started_at")),
        _parse_iso8601(milestone_execution.get("completed_at")),
        _parse_iso8601(regression_execution.get("started_at")),
        _parse_iso8601(regression_execution.get("completed_at")),
    ]
    suite_ok = bool(audit_started and generated) and all(
        timestamp is not None and audit_started <= timestamp <= generated
        for timestamp in suite_times
    )
    dirty_paths = _git_dirty_paths()
    strict_dirty_findings = _strict_dirty_findings(dirty_paths)
    baseline_frozen = not strict_dirty_findings
    freshness_ok = evidence_ok and report_ok and (
        not strict or (milestone_auth and regression_auth and suite_ok and baseline_frozen)
    )
    return freshness_ok, {
        "strict": strict,
        "audit_started_at": audit_started_at,
        "generated_at": generated_at,
        "artifact_records": artifact_records,
        "evidence_times_within_round": evidence_ok,
        "report_times_within_round": report_ok,
        "milestone_execution_authentic": milestone_auth,
        "regression_execution_authentic": regression_auth,
        "suite_times_within_round": suite_ok,
        "baseline_frozen": baseline_frozen,
        "strict_dirty_findings": strict_dirty_findings,
        "current_round": freshness_ok,
        "git": {
            "head": _git_commit(),
            "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
        },
    }


def preparation_manifest() -> dict[str, object]:
    return {
        "milestone_id": "M2.35",
        "title": "Inquiry Budget Scheduler",
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARATION_READY",
        "assumption_source": str(M235_SPEC_PATH),
        "seed_set": list(SEED_SET),
        "artifacts": {
            "specification": str(M235_SPEC_PATH),
            "preparation": str(M235_PREPARATION_PATH),
            "canonical_trace": str(M235_TRACE_PATH),
            "ablation": str(M235_ABLATION_PATH),
            "stress": str(M235_STRESS_PATH),
            "report": str(M235_REPORT_PATH),
            "summary": str(M235_SUMMARY_PATH),
        },
        "tests": {
            "milestone": list(M235_TESTS),
            "regressions": list(M235_REGRESSIONS),
        },
        "gates": list(M235_GATES),
    }


def _unknown(
    unknown_id: str,
    *,
    unknown_type: str,
    uncertainty: float,
    total_score: float,
    verification_urgency: float,
    continuity_impact: float = 0.0,
    risk_level: float = 0.0,
) -> NarrativeUnknown:
    return NarrativeUnknown(
        unknown_id=unknown_id,
        unknown_type=unknown_type,
        source_episode_id="ep:m235",
        source_span="span",
        unresolved_reason=f"{unknown_type} unresolved",
        uncertainty_level=uncertainty,
        action_relevant=True,
        decision_relevance=DecisionRelevanceMap(
            verification_urgency=verification_urgency,
            continuity_impact=continuity_impact,
            risk_level=risk_level,
            total_score=total_score,
        ),
        competing_hypothesis_ids=(f"hyp:{unknown_id}",),
        promotion_reason="retained for bounded inquiry",
    )


def _plan(
    plan_id: str,
    *,
    action: str,
    target_unknown_id: str = "unk:danger",
    status: str = "queued_experiment",
    score: float = 0.8,
    informative_value: float = 0.8,
    inconclusive_count: int = 0,
) -> ExperimentPlan:
    return ExperimentPlan(
        plan_id=plan_id,
        candidate_id=f"cand:{plan_id}",
        target_unknown_id=target_unknown_id,
        target_hypothesis_ids=(f"hyp:{target_unknown_id}",),
        selected_action=action,
        selected_reason="high information gain and decision relevance",
        evidence_sought=("observe:danger",),
        outcome_differences=("danger persists",),
        fallback_behavior="rest",
        expected_horizon=1,
        status=status,
        score=score,
        informative_value=informative_value,
        inconclusive_count=inconclusive_count,
    )


def _prediction(
    prediction_id: str,
    *,
    plan_id: str,
    confidence: float = 0.75,
    decision_relevance: float = 0.8,
    attempts: int = 0,
) -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id=prediction_id,
        created_tick=1,
        last_updated_tick=1,
        source_module="narrative_experiment",
        prediction_type="danger_probe",
        target_channels=("danger",),
        expected_state={"danger": 0.8},
        confidence=confidence,
        expected_horizon=2,
        linked_unknown_ids=("unk:danger",),
        linked_hypothesis_ids=("hyp:unk:danger",),
        linked_experiment_plan_id=plan_id,
        decision_relevance=decision_relevance,
        verification_attempts=attempts,
    )


def _archived_deferred_target(prediction_id: str, suffix: str) -> VerificationTarget:
    plan = VerificationPlan(
        prediction_id=prediction_id,
        selected_reason="prior probe",
        evidence_sought=("observe:danger",),
        support_criteria=("danger high",),
        falsification_criteria=("danger low",),
        expected_horizon=1,
        created_tick=1,
        expires_tick=2,
        attention_channels=("danger",),
    )
    return VerificationTarget(
        target_id=f"vt:{prediction_id}:{suffix}",
        prediction_id=prediction_id,
        created_tick=1,
        priority_score=0.5,
        selected_reason="prior probe",
        plan=plan,
        outcome=VerificationOutcome.DEFERRED.value,
        status="deferred",
    )


def _candidate_bundle() -> tuple[UncertaintyDecompositionResult, ExperimentDesignResult, PredictionLedger, VerificationLoop, SubjectState]:
    uncertainty = UncertaintyDecompositionResult(
        unknowns=(
            _unknown(
                "unk:danger",
                unknown_type="threat_persistence",
                uncertainty=0.74,
                total_score=0.82,
                verification_urgency=0.86,
                continuity_impact=0.72,
            ),
            _unknown(
                "unk:low-value",
                unknown_type="general",
                uncertainty=0.98,
                total_score=0.08,
                verification_urgency=0.06,
            ),
        )
    )
    experiment = ExperimentDesignResult(
        plans=(
            _plan("plan:high", action="scan", target_unknown_id="unk:danger", score=0.84, informative_value=0.88),
            _plan("plan:social", action="seek_contact", target_unknown_id="unk:low-value", score=0.34, informative_value=0.24),
        )
    )
    ledger = PredictionLedger(
        predictions=[
            _prediction("pred:high", plan_id="plan:high", confidence=0.84, decision_relevance=0.88, attempts=0),
            _prediction("pred:cooldown", plan_id="plan:high", confidence=0.82, decision_relevance=0.9, attempts=2),
        ]
    )
    verification = VerificationLoop(
        archived_targets=[
            _archived_deferred_target("pred:cooldown", "a"),
            _archived_deferred_target("pred:cooldown", "b"),
        ]
    )
    subject_state = SubjectState()
    return uncertainty, experiment, ledger, verification, subject_state


def _scheduler_signature(seed: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / f"m235_state_{seed}.json"
        trace_path = Path(tmp_dir) / f"m235_trace_{seed}.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        uncertainty, experiment, ledger, verification, subject_state = _candidate_bundle()
        runtime.agent.cycle = 7
        runtime.agent.latest_narrative_uncertainty = uncertainty
        runtime.agent.latest_narrative_experiment = experiment
        runtime.agent.prediction_ledger = ledger
        runtime.agent.verification_loop = verification
        runtime.agent.subject_state = subject_state
        runtime.subject_state = subject_state
        runtime.agent._refresh_inquiry_budget()
        state = runtime.agent.inquiry_budget_scheduler.state
        return {
            "seed": seed,
            "active_candidate_ids": list(state.active_candidate_ids),
            "verification_assignments": [item.to_dict() for item in state.verification_assignments],
            "workspace_allocations": [item.to_dict() for item in state.workspace_allocations],
            "action_allocations": [item.to_dict() for item in state.action_allocations],
            "decisions": [item.to_dict() for item in state.decisions],
        }


def build_m235_runtime_evidence() -> dict[str, object]:
    scheduler = InquiryBudgetScheduler(max_verification_slots=1)
    uncertainty, experiment, ledger, verification, subject_state = _candidate_bundle()
    state = scheduler.schedule(
        tick=5,
        narrative_uncertainty=uncertainty,
        experiment_design=experiment,
        prediction_ledger=ledger,
        verification_loop=verification,
        subject_state=subject_state,
    )
    high_plan = state.decision_for_plan("plan:high")
    social_plan = state.decision_for_plan("plan:social")
    cooldown_decision = state.decision_for_prediction("pred:cooldown")
    active_prediction_decision = state.decision_for_prediction("pred:high")
    workspace_focus = state.workspace_focus()
    scan_bias = state.action_bias("scan")
    forage_bias = state.action_bias("forage")

    verification_with_scheduler = VerificationLoop(max_active_targets=2)
    ledger_with_scheduler = PredictionLedger(predictions=[_prediction("pred:high", plan_id="plan:high")])
    verification_update = verification_with_scheduler.refresh_targets(
        tick=7,
        ledger=ledger_with_scheduler,
        inquiry_state=state,
    )
    verification_without_scheduler = VerificationLoop(max_active_targets=2)
    ledger_without_scheduler = PredictionLedger(predictions=[_prediction("pred:high", plan_id="plan:high")])
    ablated_state = InquiryBudgetState()
    verification_update_without = verification_without_scheduler.refresh_targets(
        tick=7,
        ledger=ledger_without_scheduler,
        inquiry_state=ablated_state,
    )

    scheduled_experiment = apply_scheduler_to_experiment_design(experiment, state)
    unscheduled_experiment = apply_scheduler_to_experiment_design(experiment, ablated_state)

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=SEED_SET[0],
            reset=True,
        )
        runtime.agent.cycle = 7
        runtime.agent.latest_narrative_uncertainty = uncertainty
        runtime.agent.latest_narrative_experiment = experiment
        runtime.agent.prediction_ledger = PredictionLedger(predictions=[_prediction("pred:trace", plan_id="plan:high")])
        runtime.agent.subject_state = subject_state
        runtime.subject_state = subject_state
        runtime.step(verbose=False)
        trace_records_raw = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        restored = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=SEED_SET[0],
            reset=False,
        )

    replay_signatures = [_scheduler_signature(seed) for seed in SEED_SET]
    replay_repeat_signature = _scheduler_signature(SEED_SET[0])
    canonical_signature = dict(replay_signatures[0]) if replay_signatures else {}
    if canonical_signature:
        canonical_signature.pop("seed", None)
    equivalent_signatures = []
    for signature in replay_signatures:
        comparable = dict(signature)
        comparable.pop("seed", None)
        equivalent_signatures.append(comparable)

    runtime_trace = trace_records_raw[-1] if trace_records_raw else {}
    scheduled_plan_statuses = {item.plan_id: item.status for item in scheduled_experiment.plans}
    unscheduled_plan_statuses = {item.plan_id: item.status for item in unscheduled_experiment.plans}

    trace_records = [
        {
            "schema_version": SCHEMA_VERSION,
            "event": "scheduler_candidate_ranking",
            "active_candidate_ids": list(state.active_candidate_ids),
            "top_decisions": [item.to_dict() for item in state.decisions[:4]],
            "workspace_allocations": [item.to_dict() for item in state.workspace_allocations[:4]],
            "verification_assignments": [item.to_dict() for item in state.verification_assignments[:4]],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "event": "budgeted_verification",
            "created_targets_with_scheduler": list(verification_update.created_targets),
            "created_targets_without_scheduler": list(verification_update_without.created_targets),
            "cooled_prediction_decision": cooldown_decision.decision if cooldown_decision is not None else "",
            "action_bias": {"scan": scan_bias, "forage": forage_bias},
        },
        {
            "schema_version": SCHEMA_VERSION,
            "event": "runtime_consumption",
            "workspace_focus": workspace_focus,
            "trace_has_scheduler_payload": "inquiry_scheduler_payload" in runtime_trace.get("decision_loop", {}),
            "runtime_active_candidate_ids": list(
                runtime_trace.get("inquiry_scheduler", {}).get("state", {}).get("active_candidate_ids", [])
            ),
            "restored_active_candidate_ids": list(restored.agent.inquiry_budget_scheduler.state.active_candidate_ids),
        },
    ]

    ablation = {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": "M2.35",
        "with_scheduler": {
            "active_candidate_count": len(state.active_candidate_ids),
            "verification_created_target_count": len(verification_update.created_targets),
            "workspace_focus": workspace_focus,
            "scan_bias": scan_bias,
            "scheduled_plan_statuses": scheduled_plan_statuses,
        },
        "without_scheduler": {
            "active_candidate_count": len(ablated_state.active_candidate_ids),
            "verification_created_target_count": len(verification_update_without.created_targets),
            "workspace_focus": ablated_state.workspace_focus(),
            "scan_bias": ablated_state.action_bias("scan"),
            "scheduled_plan_statuses": unscheduled_plan_statuses,
        },
        "degradation_checks": {
            "verification_loses_scheduler_targeting": len(verification_update.created_targets)
            > len(verification_update_without.created_targets),
            "workspace_loses_focus": bool(workspace_focus) and not ablated_state.workspace_focus(),
            "action_scoring_loses_scheduler_bias": scan_bias > ablated_state.action_bias("scan"),
            "experiment_plans_lose_scheduler_status_changes": scheduled_plan_statuses != unscheduled_plan_statuses,
        },
    }

    stress = {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": "M2.35",
        "stress_checks": {
            "high_value_outranks_low_value": bool(high_plan and social_plan and high_plan.priority_score > social_plan.priority_score),
            "cooldown_low_yield_prediction": cooldown_decision is not None
            and cooldown_decision.decision == InquirySchedulingDecision.COOLDOWN.value,
            "verification_slots_bounded": len(state.verification_assignments) <= scheduler.max_verification_slots,
            "workspace_slots_bounded": len({item.channel for item in state.workspace_allocations}) <= scheduler.max_workspace_slots,
            "snapshot_roundtrip_preserves_state": restored.agent.inquiry_budget_scheduler.state.to_dict()
            == runtime.agent.inquiry_budget_scheduler.state.to_dict(),
            "trace_contains_scheduler_payload": "inquiry_scheduler_payload" in runtime_trace.get("decision_loop", {}),
            "replay_same_seed_equivalent": _same_contents(
                {key: value for key, value in replay_repeat_signature.items() if key != "seed"},
                canonical_signature,
            ),
            "replay_multi_seed_equivalent": bool(equivalent_signatures)
            and all(_same_contents(signature, canonical_signature) for signature in equivalent_signatures),
        },
        "details": {
            "determinism_replay_signatures": replay_signatures,
            "scheduled_plan_statuses": scheduled_plan_statuses,
            "runtime_trace_scheduler_summary": runtime_trace.get("decision_loop", {}).get("inquiry_scheduler_summary", ""),
            "restored_active_candidate_ids": list(restored.agent.inquiry_budget_scheduler.state.active_candidate_ids),
        },
    }

    gates = {
        "cross_surface_ranking": {
            "passed": bool(high_plan and social_plan)
            and high_plan.decision in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            }
            and social_plan.priority_score < high_plan.priority_score,
            "evidence": ["plan:high outranks plan:social", "shared scheduler priority surface"],
        },
        "verification_budgeting": {
            "passed": len(state.verification_assignments) <= scheduler.max_verification_slots
            and cooldown_decision is not None
            and cooldown_decision.decision == InquirySchedulingDecision.COOLDOWN.value
            and active_prediction_decision is not None
            and active_prediction_decision.decision in {
                InquirySchedulingDecision.PROMOTE.value,
                InquirySchedulingDecision.KEEP_ACTIVE.value,
                InquirySchedulingDecision.ESCALATE.value,
            },
            "evidence": ["verification slot limit", "cooldown", "active prediction retained"],
        },
        "workspace_allocation": {
            "passed": "danger" in workspace_focus and bool(state.workspace_allocations),
            "evidence": ["workspace_focus[danger]", "workspace_allocations"],
        },
        "action_biasing": {
            "passed": scan_bias > forage_bias and scan_bias > 0.0,
            "evidence": ["scan bias exceeds forage bias", "scheduler action bias positive"],
        },
        "downstream_causality": {
            "passed": all(ablation["degradation_checks"].values()),
            "evidence": list(ablation["degradation_checks"].keys()),
        },
        "snapshot_roundtrip": {
            "passed": stress["stress_checks"]["snapshot_roundtrip_preserves_state"]
            and stress["stress_checks"]["trace_contains_scheduler_payload"],
            "evidence": ["snapshot_roundtrip_preserves_state", "trace_contains_scheduler_payload"],
        },
    }

    return {
        "trace_records": trace_records,
        "ablation": ablation,
        "stress": stress,
        "gates": gates,
    }


def _write_trace(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _summary_markdown(report: dict[str, object]) -> str:
    gates = report["gates"]
    lines = [
        "# M2.35 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Cross-surface ranking gate: {'PASS' if gates['cross_surface_ranking']['passed'] else 'FAIL'}",
        f"- Verification budgeting gate: {'PASS' if gates['verification_budgeting']['passed'] else 'FAIL'}",
        f"- Workspace allocation gate: {'PASS' if gates['workspace_allocation']['passed'] else 'FAIL'}",
        f"- Action biasing gate: {'PASS' if gates['action_biasing']['passed'] else 'FAIL'}",
        f"- Downstream causality gate: {'PASS' if gates['downstream_causality']['passed'] else 'FAIL'}",
        f"- Snapshot gate: {'PASS' if gates['snapshot_roundtrip']['passed'] else 'FAIL'}",
        f"- Regression gate: {'PASS' if gates['regression']['passed'] else 'FAIL'}",
        f"- Freshness gate: {'PASS' if gates['artifact_freshness']['passed'] else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


def write_m235_acceptance_artifacts(
    *,
    strict: bool = True,
    execute_test_suites: bool = True,
    milestone_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
) -> dict[str, str]:
    if strict:
        for injected, label in ((milestone_execution, "milestone"), (regression_execution, "regression")):
            if injected is not None and injected.get("execution_source") != "subprocess":
                raise ValueError(
                    f"strict M2.35 audit refuses injected execution records for {label} tests"
                )

    audit_started_at = _now_iso()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    runtime_evidence = build_m235_runtime_evidence()
    _write_trace(M235_TRACE_PATH, runtime_evidence["trace_records"])
    _write_json(M235_ABLATION_PATH, runtime_evidence["ablation"])
    _write_json(M235_STRESS_PATH, runtime_evidence["stress"])

    milestone_execution = milestone_execution or _suite_execution_record(
        label="m235-milestone",
        paths=M235_TESTS,
        execute=execute_test_suites,
    )
    regression_execution = regression_execution or _suite_execution_record(
        label="m235-regression",
        paths=M235_REGRESSIONS,
        execute=execute_test_suites,
    )

    artifacts = {
        "specification": str(M235_SPEC_PATH),
        "preparation": str(M235_PREPARATION_PATH),
        "canonical_trace": str(M235_TRACE_PATH),
        "ablation": str(M235_ABLATION_PATH),
        "stress": str(M235_STRESS_PATH),
        "report": str(M235_REPORT_PATH),
        "summary": str(M235_SUMMARY_PATH),
    }

    def _assemble(generated_at_value: str, freshness_ok_value: bool, freshness_payload_value: dict[str, object]) -> dict[str, object]:
        gates = {
            **runtime_evidence["gates"],
            "regression": {
                "passed": bool(regression_execution.get("executed")) and bool(regression_execution.get("passed")),
                "execution": regression_execution,
            },
            "artifact_freshness": {
                "passed": freshness_ok_value,
                "details": freshness_payload_value,
            },
        }
        milestone_ok = bool(milestone_execution.get("executed")) and bool(milestone_execution.get("passed"))
        for gate_name in (
            "cross_surface_ranking",
            "verification_budgeting",
            "workspace_allocation",
            "action_biasing",
            "downstream_causality",
            "snapshot_roundtrip",
        ):
            gates[gate_name]["passed"] = bool(gates[gate_name]["passed"]) and milestone_ok
        all_passed = all(bool(item["passed"]) for item in gates.values())
        findings = list(freshness_payload_value.get("strict_dirty_findings", []))
        recommendation = "ACCEPT" if all_passed else "BLOCK"
        status = "PASS" if all_passed else "FAIL"
        residual_risks: list[str] = []
        if findings and not all_passed and not strict:
            recommendation = "ACCEPT_WITH_RESIDUAL_RISK"
            status = "PASS_WITH_RESIDUAL_RISK"
            residual_risks.append("working tree was dirty during audit generation")
        elif not all_passed:
            residual_risks.append("runtime evidence or freshness gate missing")
        return {
            "milestone_id": "M2.35",
            "title": "Inquiry Budget Scheduler",
            "schema_version": SCHEMA_VERSION,
            "strict": strict,
            "status": status,
            "recommendation": recommendation,
            "generated_at": generated_at_value,
            "seed_set": list(SEED_SET),
            "artifacts": artifacts,
            "tests": {
                "milestone": milestone_execution,
                "regressions": regression_execution,
            },
            "gates": gates,
            "findings": findings,
            "residual_risks": residual_risks,
            "freshness": freshness_payload_value,
        }

    provisional_generated_at = _now_iso()
    provisional_report = _assemble(
        provisional_generated_at,
        False,
        {
            "strict": strict,
            "audit_started_at": audit_started_at,
            "generated_at": provisional_generated_at,
            "artifact_records": {},
            "evidence_times_within_round": False,
            "report_times_within_round": False,
            "milestone_execution_authentic": False,
            "regression_execution_authentic": False,
            "suite_times_within_round": False,
            "baseline_frozen": False,
            "strict_dirty_findings": [],
            "current_round": False,
            "git": {
                "head": _git_commit(),
                "dirty": bool(_git_dirty_paths()),
                "dirty_paths": _git_dirty_paths(),
            },
        },
    )
    M235_REPORT_PATH.write_text(json.dumps(provisional_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M235_SUMMARY_PATH.write_text(_summary_markdown(provisional_report), encoding="utf-8")

    final_generated_at = _now_iso()
    freshness_ok, freshness_payload = _freshness_gate(
        artifacts=artifacts,
        audit_started_at=audit_started_at,
        generated_at=final_generated_at,
        milestone_execution=milestone_execution,
        regression_execution=regression_execution,
        strict=strict,
    )
    final_report = _assemble(final_generated_at, freshness_ok, freshness_payload)
    M235_REPORT_PATH.write_text(json.dumps(final_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M235_SUMMARY_PATH.write_text(_summary_markdown(final_report), encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    write_m235_acceptance_artifacts(strict=True, execute_test_suites=True)
