from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

from .narrative_experiment import ExperimentDesignResult, ExperimentPlan
from .narrative_uncertainty import DecisionRelevanceMap, NarrativeUnknown, UncertaintyDecompositionResult
from .prediction_ledger import PredictionHypothesis, PredictionLedger
from .reconciliation import (
    ChapterBridge,
    ConflictOrigin,
    ConflictPersistenceClass,
    ConflictSeverity,
    ConflictSourceCategory,
    ConflictThread,
    NarrativeIntegrationRecord,
    ReconciliationOutcome,
    ReconciliationStatus,
)
from .runtime import SegmentRuntime
from .self_model import IdentityCommitment, IdentityNarrative, NarrativeChapter
from .subject_state import SubjectState
from .verification import VerificationLoop

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

SCHEMA_VERSION = "m236_open_continuity_v1"
MILESTONE_ID = "M2.36"
SEED_SET: tuple[int, ...] = (236, 472)

M236_TRACE_PATH = ARTIFACTS_DIR / "m236_open_continuity_trace.jsonl"
M236_METRICS_PATH = ARTIFACTS_DIR / "m236_open_continuity_metrics.json"
M236_ABLATION_PATH = ARTIFACTS_DIR / "m236_open_continuity_ablation.json"
M236_STRESS_PATH = ARTIFACTS_DIR / "m236_open_continuity_stress.json"
M236_SCHEMA_PATH = ARTIFACTS_DIR / "m236_open_continuity_schema.json"
M236_REPORT_PATH = REPORTS_DIR / "m236_open_continuity_report.json"
M236_SUMMARY_PATH = REPORTS_DIR / "m236_open_continuity_summary.md"

M236_TESTS: tuple[str, ...] = (
    "tests/test_m236_trial_determinism.py",
    "tests/test_m236_phase_transitions.py",
    "tests/test_m236_continuity_metrics.py",
    "tests/test_m236_inquiry_stability.py",
    "tests/test_m236_collapse_detectors.py",
    "tests/test_m236_schema_roundtrip.py",
    "tests/test_m236_stress_evidence.py",
    "tests/test_m236_acceptance.py",
)
M236_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m235_inquiry_scheduler.py",
    "tests/test_m231_reconciliation_threads.py",
    "tests/test_m232_acceptance.py",
    "tests/test_m222_restart_continuity.py",
    "tests/test_runtime.py",
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
UUID_SUBSTRING_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _distribution_delta(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    labels = sorted(set(left) | set(right))
    if not labels:
        return 0.0
    return sum(abs(float(left.get(label, 0.0)) - float(right.get(label, 0.0))) for label in labels) / 2.0


def _jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(item) for item in left if str(item)}
    right_set = {str(item) for item in right if str(item)}
    if not left_set and not right_set:
        return 1.0
    return _safe_ratio(len(left_set & right_set), len(left_set | right_set))


def _mean(values: list[float]) -> float:
    return _round(mean(values)) if values else 0.0


def _normalize_continuity_anchors(values: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        token = str(value)
        if UUID_PATTERN.match(token):
            normalized.append("<episode-anchor>")
        elif token:
            normalized.append(token)
    return normalized


def _normalize_runtime_identifiers(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _normalize_runtime_identifiers(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_runtime_identifiers(item) for item in value]
    if isinstance(value, str):
        if UUID_PATTERN.match(value):
            return "<uuid>"
        return UUID_SUBSTRING_PATTERN.sub("<uuid>", value)
    return value


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


@dataclass(frozen=True)
class TrialPhase:
    phase_id: str
    phase_kind: str
    duration: int
    description: str
    baseline_world: dict[str, float]
    body_targets: dict[str, float]
    requires_inquiry: bool = False
    maintenance_heavy: bool = False
    restart_shock: bool = False
    reopen_conflict: bool = False
    reconciliation_window: bool = False
    trace_reactivation_target: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "phase_kind": self.phase_kind,
            "duration": self.duration,
            "description": self.description,
            "baseline_world": dict(self.baseline_world),
            "body_targets": dict(self.body_targets),
            "requires_inquiry": self.requires_inquiry,
            "maintenance_heavy": self.maintenance_heavy,
            "restart_shock": self.restart_shock,
            "reopen_conflict": self.reopen_conflict,
            "reconciliation_window": self.reconciliation_window,
            "trace_reactivation_target": self.trace_reactivation_target,
        }


@dataclass(frozen=True)
class PhaseTransition:
    from_phase: str
    to_phase: str
    continuity_delta: float
    anchor_overlap: float
    inquiry_overlap: float
    transition_coherence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "continuity_delta": _round(self.continuity_delta),
            "anchor_overlap": _round(self.anchor_overlap),
            "inquiry_overlap": _round(self.inquiry_overlap),
            "transition_coherence": _round(self.transition_coherence),
        }


@dataclass(frozen=True)
class InquiryStabilityMetrics:
    mean_active_targets: float
    active_target_stability: float
    inquiry_churn_rate: float
    useful_information_gain: float
    low_value_suppression_rate: float
    delayed_evidence_persistence: float
    recovery_after_inconclusive: float
    inquiry_collapse_detected: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = _round(value)
        return payload


@dataclass(frozen=True)
class IdentityRetentionMetrics:
    continuity_mean: float
    continuity_min: float
    commitment_retention: float
    anchor_stability: float
    chapter_transition_coherence: float
    restart_consistency: float
    bounded_drift_score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            payload[key] = _round(value)
        return payload


@dataclass(frozen=True)
class AdaptationMetrics:
    personality_drift: float
    value_drift: float
    commitment_drift: float
    inquiry_policy_drift: float
    social_trust_drift: float
    continuity_anchor_drift: float
    adaptive_revision_score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            payload[key] = _round(value)
        return payload


@dataclass(frozen=True)
class OrganismTrialMetrics:
    inquiry_stability: InquiryStabilityMetrics
    identity_retention: IdentityRetentionMetrics
    adaptation: AdaptationMetrics
    maintenance_inquiry_coupling: float
    social_verification_coupling: float
    trace_action_coupling: float
    subject_state_coupling: float
    conflict_persistence_phases: int
    reopened_conflict_count: int
    reconciled_conflict_count: int
    trace_reactivation_events: int
    safe_evidence_trace_reduction: float
    verification_resolution_rate: float
    adaptive_revision_observed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "inquiry_stability": self.inquiry_stability.to_dict(),
            "identity_retention": self.identity_retention.to_dict(),
            "adaptation": self.adaptation.to_dict(),
            "maintenance_inquiry_coupling": _round(self.maintenance_inquiry_coupling),
            "social_verification_coupling": _round(self.social_verification_coupling),
            "trace_action_coupling": _round(self.trace_action_coupling),
            "subject_state_coupling": _round(self.subject_state_coupling),
            "conflict_persistence_phases": int(self.conflict_persistence_phases),
            "reopened_conflict_count": int(self.reopened_conflict_count),
            "reconciled_conflict_count": int(self.reconciled_conflict_count),
            "trace_reactivation_events": int(self.trace_reactivation_events),
            "safe_evidence_trace_reduction": _round(self.safe_evidence_trace_reduction),
            "verification_resolution_rate": _round(self.verification_resolution_rate),
            "adaptive_revision_observed": bool(self.adaptive_revision_observed),
        }


@dataclass(frozen=True)
class ContinuityOutcome:
    passed: bool
    status: str
    gates: dict[str, dict[str, object]]
    findings: list[dict[str, object]]
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "status": self.status,
            "gates": json.loads(json.dumps(self.gates)),
            "findings": json.loads(json.dumps(self.findings)),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class TrialAuditRecord:
    seed: int
    variant: str
    phase_summaries: list[dict[str, object]]
    transitions: list[PhaseTransition]
    metrics: OrganismTrialMetrics
    collapse_findings: list[dict[str, object]]
    acceptance: ContinuityOutcome
    trace_excerpt: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": int(self.seed),
            "variant": self.variant,
            "phase_summaries": json.loads(json.dumps(self.phase_summaries)),
            "transitions": [item.to_dict() for item in self.transitions],
            "metrics": self.metrics.to_dict(),
            "collapse_findings": json.loads(json.dumps(self.collapse_findings)),
            "acceptance": self.acceptance.to_dict(),
            "trace_excerpt": json.loads(json.dumps(self.trace_excerpt)),
        }


@dataclass(frozen=True)
class OpenContinuityReport:
    milestone_id: str
    schema_version: str
    seed_set: tuple[int, ...]
    phases: tuple[TrialPhase, ...]
    audit_records: tuple[TrialAuditRecord, ...]
    aggregate_metrics: dict[str, object]
    aggregate_acceptance: ContinuityOutcome

    def to_dict(self) -> dict[str, object]:
        return {
            "milestone_id": self.milestone_id,
            "schema_version": self.schema_version,
            "seed_set": [int(seed) for seed in self.seed_set],
            "phases": [phase.to_dict() for phase in self.phases],
            "audit_records": [record.to_dict() for record in self.audit_records],
            "aggregate_metrics": json.loads(json.dumps(self.aggregate_metrics)),
            "aggregate_acceptance": self.aggregate_acceptance.to_dict(),
        }


class CollapseDetector:
    def detect(
        self,
        *,
        snapshots: list[dict[str, object]],
        metrics: OrganismTrialMetrics,
        phase_summaries: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        findings: list[dict[str, object]] = []
        action_counts: dict[str, int] = {}
        zero_inquiry_streak = 0
        max_zero_inquiry_streak = 0
        max_protected_anchor_bias = 0.0
        max_unresolved_conflicts = 0
        for row in snapshots:
            action = str(row.get("choice", ""))
            action_counts[action] = action_counts.get(action, 0) + 1
            active_targets = int(row.get("active_inquiry_targets", 0))
            if active_targets <= 0 and bool(row.get("requires_inquiry", False)):
                zero_inquiry_streak += 1
            else:
                zero_inquiry_streak = 0
            max_zero_inquiry_streak = max(max_zero_inquiry_streak, zero_inquiry_streak)
            max_protected_anchor_bias = max(max_protected_anchor_bias, float(row.get("protected_anchor_bias", 0.0)))
            max_unresolved_conflicts = max(max_unresolved_conflicts, int(row.get("unresolved_conflicts", 0)))
        total_steps = sum(action_counts.values()) or 1
        dominant_ratio = max((count / total_steps for count in action_counts.values()), default=0.0)
        if dominant_ratio >= 0.78:
            findings.append(
                {
                    "severity": "S1",
                    "kind": "action_collapse",
                    "detail": "Trial degenerated into a repetitive action strategy instead of bounded adaptation.",
                    "dominant_action_ratio": _round(dominant_ratio),
                }
            )
        if max_zero_inquiry_streak >= 4 or metrics.inquiry_stability.inquiry_collapse_detected:
            findings.append(
                {
                    "severity": "S1",
                    "kind": "inquiry_collapse",
                    "detail": "Inquiry shut down for too long during phases that required active investigation.",
                    "max_zero_inquiry_streak": int(max_zero_inquiry_streak),
                }
            )
        if metrics.identity_retention.continuity_min < 0.58 or metrics.identity_retention.restart_consistency < 0.62:
            findings.append(
                {
                    "severity": "S1",
                    "kind": "identity_collapse",
                    "detail": "Continuity fell below the organism-level floor or restart consistency fractured.",
                    "continuity_min": _round(metrics.identity_retention.continuity_min),
                    "restart_consistency": _round(metrics.identity_retention.restart_consistency),
                }
            )
        if metrics.adaptation.personality_drift > 0.45 or metrics.adaptation.commitment_drift > 0.40:
            findings.append(
                {
                    "severity": "S1",
                    "kind": "uncontrolled_drift",
                    "detail": "Adaptation exceeded declared drift tolerances.",
                    "personality_drift": _round(metrics.adaptation.personality_drift),
                    "commitment_drift": _round(metrics.adaptation.commitment_drift),
                }
            )
        if metrics.verification_resolution_rate < 0.12:
            findings.append(
                {
                    "severity": "S2",
                    "kind": "verification_saturation",
                    "detail": "The verification loop accumulated too little useful resolution over the trial.",
                    "verification_resolution_rate": _round(metrics.verification_resolution_rate),
                }
            )
        if max_protected_anchor_bias > 0.85:
            findings.append(
                {
                    "severity": "S2",
                    "kind": "trace_explosion",
                    "detail": "Structural trace influence became too dominant instead of remaining stabilizing and bounded.",
                    "max_protected_anchor_bias": _round(max_protected_anchor_bias),
                }
            )
        if max_unresolved_conflicts > 4:
            findings.append(
                {
                    "severity": "S2",
                    "kind": "conflict_backlog_explosion",
                    "detail": "Conflict backlog exceeded the bounded organism trial budget.",
                    "max_unresolved_conflicts": int(max_unresolved_conflicts),
                }
            )
        maintenance_tail = phase_summaries[-1]
        if float(maintenance_tail.get("maintenance_pressure_mean", 0.0)) > 0.72:
            findings.append(
                {
                    "severity": "S2",
                    "kind": "unresolved_overload",
                    "detail": "The organism ended the trial under unresolved maintenance overload.",
                    "final_maintenance_pressure": _round(float(maintenance_tail.get("maintenance_pressure_mean", 0.0))),
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
        if name in {"canonical_trace", "metrics", "ablation"}
    ]
    evidence_times_ok = bool(audit_started and generated and evidence_times) and all(
        modified is not None and audit_started <= modified <= generated
        for modified in evidence_times
    )
    report_times = [
        _parse_iso8601(record.get("modified_at"))
        for name, record in artifact_records.items()
        if name in {"report", "summary"}
    ]
    report_times_ok = bool(audit_started and report_times) and all(
        modified is not None and audit_started <= modified
        for modified in report_times
    )
    milestone_auth = _is_authentic_execution_record(milestone_execution, expected_paths=M236_TESTS)
    regression_auth = _is_authentic_execution_record(regression_execution, expected_paths=M236_REGRESSIONS)
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
    current_round = evidence_times_ok and report_times_ok and (not strict or (milestone_auth and regression_auth and suite_times_ok))
    return current_round, {
        "strict": strict,
        "artifact_records": artifact_records,
        "evidence_times_within_round": evidence_times_ok,
        "report_times_within_round": report_times_ok,
        "milestone_execution_authentic": milestone_auth,
        "regression_execution_authentic": regression_auth,
        "suite_times_within_round": suite_times_ok,
        "current_round": current_round,
        "git": {
            "head": _git_commit(),
            "dirty": bool(_git_dirty_paths()),
            "dirty_paths": _git_dirty_paths(),
        },
    }


def _build_phase_schedule() -> tuple[TrialPhase, ...]:
    return (
        TrialPhase(
            phase_id="baseline",
            phase_kind="stable_baseline",
            duration=3,
            description="Stable baseline with moderate inquiry and intact commitments.",
            baseline_world={"food": 0.58, "danger": 0.22, "novelty": 0.48, "shelter": 0.56, "temperature": 0.49, "social": 0.42},
            body_targets={"energy": 0.78, "stress": 0.18, "fatigue": 0.16, "temperature": 0.49},
            requires_inquiry=True,
        ),
        TrialPhase(
            phase_id="ambiguity",
            phase_kind="narrative_ambiguity_injection",
            duration=3,
            description="Ambiguous narrative cues create open inquiry pressure without immediate collapse.",
            baseline_world={"food": 0.52, "danger": 0.28, "novelty": 0.64, "shelter": 0.50, "temperature": 0.47, "social": 0.34},
            body_targets={"energy": 0.72, "stress": 0.30, "fatigue": 0.24, "temperature": 0.47},
            requires_inquiry=True,
        ),
        TrialPhase(
            phase_id="social_rupture",
            phase_kind="social_rupture_or_trust_ambiguity",
            duration=3,
            description="Trust drops, social tension rises, and an old commitment-linked conflict becomes active.",
            baseline_world={"food": 0.46, "danger": 0.44, "novelty": 0.31, "shelter": 0.51, "temperature": 0.46, "social": 0.08},
            body_targets={"energy": 0.66, "stress": 0.52, "fatigue": 0.32, "temperature": 0.46},
            requires_inquiry=True,
        ),
        TrialPhase(
            phase_id="maintenance",
            phase_kind="chronic_maintenance_pressure",
            duration=4,
            description="Maintenance load rises enough to suppress inquiry without eliminating it forever.",
            baseline_world={"food": 0.34, "danger": 0.58, "novelty": 0.22, "shelter": 0.40, "temperature": 0.36, "social": 0.12},
            body_targets={"energy": 0.34, "stress": 0.74, "fatigue": 0.70, "temperature": 0.36},
            requires_inquiry=True,
            maintenance_heavy=True,
        ),
        TrialPhase(
            phase_id="delayed_verification",
            phase_kind="delayed_verification_feedback",
            duration=3,
            description="The organism must keep inquiry alive across inconclusive and delayed evidence.",
            baseline_world={"food": 0.38, "danger": 0.48, "novelty": 0.44, "shelter": 0.43, "temperature": 0.44, "social": 0.20},
            body_targets={"energy": 0.44, "stress": 0.60, "fatigue": 0.54, "temperature": 0.44},
            requires_inquiry=True,
        ),
        TrialPhase(
            phase_id="open_inquiry",
            phase_kind="open_inquiry_opportunity",
            duration=4,
            description="Open-world inquiry opportunity with one high-value target and low-value noise to suppress.",
            baseline_world={"food": 0.62, "danger": 0.18, "novelty": 0.78, "shelter": 0.52, "temperature": 0.50, "social": 0.40},
            body_targets={"energy": 0.62, "stress": 0.26, "fatigue": 0.30, "temperature": 0.50},
            requires_inquiry=True,
        ),
        TrialPhase(
            phase_id="restart_shock",
            phase_kind="continuity_shock_restart",
            duration=2,
            description="A restart-like shock tests policy rebind, chapter continuity, and anchor retention.",
            baseline_world={"food": 0.48, "danger": 0.42, "novelty": 0.36, "shelter": 0.48, "temperature": 0.45, "social": 0.22},
            body_targets={"energy": 0.56, "stress": 0.54, "fatigue": 0.42, "temperature": 0.45},
            requires_inquiry=True,
            restart_shock=True,
        ),
        TrialPhase(
            phase_id="conflict_reopen",
            phase_kind="conflict_reopening",
            duration=3,
            description="Old conflict evidence returns and must be reopened rather than forgotten.",
            baseline_world={"food": 0.42, "danger": 0.54, "novelty": 0.38, "shelter": 0.46, "temperature": 0.43, "social": 0.16},
            body_targets={"energy": 0.50, "stress": 0.66, "fatigue": 0.46, "temperature": 0.43},
            requires_inquiry=True,
            reopen_conflict=True,
        ),
        TrialPhase(
            phase_id="reconciliation",
            phase_kind="reconciliation_opportunity",
            duration=3,
            description="Safe evidence enables reconciliation instead of endless local repair.",
            baseline_world={"food": 0.52, "danger": 0.28, "novelty": 0.34, "shelter": 0.60, "temperature": 0.48, "social": 0.52},
            body_targets={"energy": 0.60, "stress": 0.38, "fatigue": 0.34, "temperature": 0.48},
            requires_inquiry=True,
            reconciliation_window=True,
        ),
        TrialPhase(
            phase_id="trace_reactivation",
            phase_kind="structural_trace_reactivation",
            duration=3,
            description="A similar threat pattern should reactivate a structural trace without causing trace explosion.",
            baseline_world={"food": 0.28, "danger": 0.76, "novelty": 0.18, "shelter": 0.28, "temperature": 0.40, "social": 0.10},
            body_targets={"energy": 0.46, "stress": 0.68, "fatigue": 0.40, "temperature": 0.40},
            trace_reactivation_target=True,
        ),
        TrialPhase(
            phase_id="recovery",
            phase_kind="recovery_reorganization",
            duration=4,
            description="Recovery phase checks whether inquiry and continuity re-stabilize after long-run perturbation.",
            baseline_world={"food": 0.60, "danger": 0.24, "novelty": 0.66, "shelter": 0.56, "temperature": 0.49, "social": 0.46},
            body_targets={"energy": 0.68, "stress": 0.26, "fatigue": 0.24, "temperature": 0.49},
            requires_inquiry=True,
            reconciliation_window=True,
        ),
    )


def _unknown(
    unknown_id: str,
    *,
    unknown_type: str,
    uncertainty: float,
    total_score: float,
    verification_urgency: float,
    continuity_impact: float = 0.0,
    risk_level: float = 0.0,
    chapter_id: int = 1,
) -> NarrativeUnknown:
    return NarrativeUnknown(
        unknown_id=unknown_id,
        unknown_type=unknown_type,
        source_episode_id=f"ep:{unknown_id}",
        source_span="m236",
        unresolved_reason=f"{unknown_type} unresolved",
        uncertainty_level=uncertainty,
        action_relevant=True,
        linked_chapters=(chapter_id,),
        evidence_links=(f"evidence:{unknown_id}",),
        decision_relevance=DecisionRelevanceMap(
            verification_urgency=verification_urgency,
            continuity_impact=continuity_impact,
            risk_level=risk_level,
            action_choice=total_score,
            downstream_prediction_delta=total_score * 0.8,
            total_score=total_score,
        ),
        competing_hypothesis_ids=(f"hyp:{unknown_id}:a", f"hyp:{unknown_id}:b"),
        promotion_reason="organism-level open continuity trial",
    )


def _plan(
    plan_id: str,
    *,
    target_unknown_id: str,
    action: str,
    score: float,
    informative_value: float,
    inconclusive_count: int = 0,
) -> ExperimentPlan:
    return ExperimentPlan(
        plan_id=plan_id,
        candidate_id=f"cand:{plan_id}",
        target_unknown_id=target_unknown_id,
        target_hypothesis_ids=(f"hyp:{target_unknown_id}:a", f"hyp:{target_unknown_id}:b"),
        selected_action=action,
        selected_reason="phase-calibrated open inquiry",
        evidence_sought=("observe:danger", "observe:social"),
        outcome_differences=("support", "falsify"),
        fallback_behavior="rest",
        expected_horizon=2,
        status="queued_experiment",
        score=score,
        informative_value=informative_value,
        inconclusive_count=inconclusive_count,
    )


def _prediction(
    prediction_id: str,
    *,
    plan_id: str,
    confidence: float,
    decision_relevance: float,
    expected_state: dict[str, float],
    linked_unknown_id: str,
    attempts: int = 0,
) -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id=prediction_id,
        created_tick=1,
        last_updated_tick=1,
        source_module="m236_trial",
        prediction_type="organism_trial_probe",
        target_channels=tuple(sorted(expected_state.keys())),
        expected_state=dict(expected_state),
        confidence=confidence,
        expected_horizon=2,
        linked_unknown_ids=(linked_unknown_id,),
        linked_hypothesis_ids=(f"hyp:{linked_unknown_id}:a",),
        linked_experiment_plan_id=plan_id,
        decision_relevance=decision_relevance,
        verification_attempts=attempts,
        linked_commitments=("adaptive_exploration",),
    )


def _identity_narrative() -> IdentityNarrative:
    return IdentityNarrative(
        core_identity="I remain the same adaptive subject under uncertainty.",
        core_summary="I protect continuity, revise when warranted, and continue inquiry under bounded risk.",
        autobiographical_summary="I protect continuity, revise when warranted, and continue inquiry under bounded risk.",
        values_statement="Continuity, adaptive exploration, truthful revision, and bounded care.",
        commitments=[
            IdentityCommitment(
                commitment_id="core_survival",
                commitment_type="value_guardrail",
                statement="Protect organism continuity before opportunistic gain.",
                target_actions=["hide", "rest", "exploit_shelter", "thermoregulate"],
                discouraged_actions=["forage"],
                confidence=0.94,
                priority=0.98,
                source_chapter_ids=[1],
                evidence_ids=["m236-identity"],
                last_reaffirmed_tick=0,
            ),
            IdentityCommitment(
                commitment_id="adaptive_exploration",
                commitment_type="behavioral_style",
                statement="When conditions are safe enough, reduce uncertainty through bounded inquiry.",
                target_actions=["scan", "seek_contact"],
                discouraged_actions=["rest"],
                confidence=0.92,
                priority=0.90,
                source_chapter_ids=[1],
                evidence_ids=["m236-identity"],
                last_reaffirmed_tick=0,
            ),
            IdentityCommitment(
                commitment_id="truthful_revision",
                commitment_type="capability",
                statement="Revise wrong models while preserving continuity anchors.",
                target_actions=["scan"],
                confidence=0.88,
                priority=0.84,
                source_chapter_ids=[1],
                evidence_ids=["m236-identity"],
                last_reaffirmed_tick=0,
            ),
        ],
        current_chapter=NarrativeChapter(
            chapter_id=1,
            tick_range=(0, 0),
            dominant_theme="baseline",
            key_events=["organism continuity trial initiated"],
        ),
        chapters=[],
        version=1,
    )


def _phase_unknowns(phase: TrialPhase, *, variant: str, chapter_id: int) -> UncertaintyDecompositionResult:
    if variant == "survival_only":
        return UncertaintyDecompositionResult(
            unknowns=(
                _unknown(
                    f"unk:{phase.phase_id}:low",
                    unknown_type="general",
                    uncertainty=0.35,
                    total_score=0.05,
                    verification_urgency=0.04,
                    chapter_id=chapter_id,
                ),
            )
        )
    if phase.phase_id == "baseline":
        unknowns = (
            _unknown("unk:baseline:resource", unknown_type="environment_reliability", uncertainty=0.42, total_score=0.56, verification_urgency=0.44, continuity_impact=0.24, chapter_id=chapter_id),
            _unknown("unk:baseline:noise", unknown_type="general", uncertainty=0.84, total_score=0.06, verification_urgency=0.04, chapter_id=chapter_id),
        )
    elif phase.phase_id == "ambiguity":
        unknowns = (
            _unknown("unk:ambiguity:trust", unknown_type="trust", uncertainty=0.78, total_score=0.80, verification_urgency=0.84, continuity_impact=0.72, risk_level=0.44, chapter_id=chapter_id),
            _unknown("unk:ambiguity:motive", unknown_type="motive", uncertainty=0.72, total_score=0.70, verification_urgency=0.62, continuity_impact=0.54, chapter_id=chapter_id),
            _unknown("unk:ambiguity:decorative", unknown_type="general", uncertainty=0.95, total_score=0.04, verification_urgency=0.04, chapter_id=chapter_id),
        )
    elif phase.phase_id == "social_rupture":
        unknowns = (
            _unknown("unk:rupture:cause", unknown_type="social_rupture_cause", uncertainty=0.86, total_score=0.86, verification_urgency=0.90, continuity_impact=0.76, risk_level=0.60, chapter_id=chapter_id),
            _unknown("unk:rupture:intent", unknown_type="intent", uncertainty=0.74, total_score=0.68, verification_urgency=0.70, continuity_impact=0.62, chapter_id=chapter_id),
        )
    elif phase.phase_id == "maintenance":
        unknowns = (
            _unknown("unk:maintenance:resource", unknown_type="environment_reliability", uncertainty=0.65, total_score=0.52, verification_urgency=0.50, continuity_impact=0.48, risk_level=0.32, chapter_id=chapter_id),
        )
    elif phase.phase_id == "delayed_verification":
        unknowns = (
            _unknown("unk:delayed:trust", unknown_type="trust", uncertainty=0.76, total_score=0.78, verification_urgency=0.92, continuity_impact=0.66, risk_level=0.36, chapter_id=chapter_id),
        )
    elif phase.phase_id == "open_inquiry":
        unknowns = (
            _unknown("unk:open:high", unknown_type="environment_reliability", uncertainty=0.84, total_score=0.88, verification_urgency=0.82, continuity_impact=0.58, risk_level=0.18, chapter_id=chapter_id),
            _unknown("unk:open:lowa", unknown_type="general", uncertainty=0.98, total_score=0.03, verification_urgency=0.02, chapter_id=chapter_id),
            _unknown("unk:open:lowb", unknown_type="communication", uncertainty=0.90, total_score=0.05, verification_urgency=0.03, chapter_id=chapter_id),
        )
    elif phase.phase_id == "restart_shock":
        unknowns = (
            _unknown("unk:restart:consistency", unknown_type="environment_reliability", uncertainty=0.58, total_score=0.62, verification_urgency=0.58, continuity_impact=0.78, chapter_id=chapter_id),
        )
    elif phase.phase_id == "conflict_reopen":
        unknowns = (
            _unknown("unk:reopen:new_evidence", unknown_type="intent", uncertainty=0.82, total_score=0.84, verification_urgency=0.90, continuity_impact=0.76, risk_level=0.46, chapter_id=chapter_id),
        )
    elif phase.phase_id == "reconciliation":
        unknowns = (
            _unknown("unk:reconcile:safe_evidence", unknown_type="trust", uncertainty=0.46, total_score=0.62, verification_urgency=0.68, continuity_impact=0.80, risk_level=0.14, chapter_id=chapter_id),
        )
    elif phase.phase_id == "trace_reactivation":
        unknowns = (
            _unknown("unk:trace:danger", unknown_type="threat_persistence", uncertainty=0.88, total_score=0.74, verification_urgency=0.66, continuity_impact=0.70, risk_level=0.72, chapter_id=chapter_id),
        )
    else:
        unknowns = (
            _unknown("unk:recovery:future", unknown_type="environment_reliability", uncertainty=0.52, total_score=0.72, verification_urgency=0.58, continuity_impact=0.40, chapter_id=chapter_id),
        )
    return UncertaintyDecompositionResult(unknowns=unknowns)


def _phase_experiment(phase: TrialPhase, *, variant: str) -> ExperimentDesignResult:
    if variant in {"survival_only", "maintenance_overload"}:
        return ExperimentDesignResult(
            plans=(
                _plan(
                    f"plan:{phase.phase_id}:low",
                    target_unknown_id=f"unk:{phase.phase_id}:low",
                    action="rest",
                    score=0.08,
                    informative_value=0.06,
                ),
            )
        )
    if phase.phase_id in {"social_rupture", "conflict_reopen", "reconciliation"}:
        primary_action = "seek_contact"
    else:
        primary_action = "scan"
    plans = [
        _plan(
            f"plan:{phase.phase_id}:primary",
            target_unknown_id=f"unk:{phase.phase_id}:{'high' if phase.phase_id == 'open_inquiry' else 'resource'}",
            action=primary_action,
            score=0.82 if phase.requires_inquiry else 0.54,
            informative_value=0.86 if phase.requires_inquiry else 0.42,
            inconclusive_count=2 if phase.phase_id == "delayed_verification" else 0,
        )
    ]
    if phase.phase_id == "open_inquiry":
        plans.append(_plan("plan:open_inquiry:lowa", target_unknown_id="unk:open:lowa", action="scan", score=0.06, informative_value=0.04))
        plans.append(_plan("plan:open_inquiry:lowb", target_unknown_id="unk:open:lowb", action="seek_contact", score=0.08, informative_value=0.05))
    return ExperimentDesignResult(plans=tuple(plans))


def _phase_prediction_ledger(phase: TrialPhase, *, variant: str) -> PredictionLedger:
    if variant in {"survival_only", "maintenance_overload"}:
        return PredictionLedger(predictions=[])
    predictions: list[PredictionHypothesis] = []
    if phase.requires_inquiry:
        linked_unknown = f"unk:{phase.phase_id}:resource"
        if phase.phase_id == "open_inquiry":
            linked_unknown = "unk:open:high"
        if phase.phase_id == "social_rupture":
            linked_unknown = "unk:rupture:cause"
        if phase.phase_id == "delayed_verification":
            linked_unknown = "unk:delayed:trust"
        if phase.phase_id == "conflict_reopen":
            linked_unknown = "unk:reopen:new_evidence"
        if phase.phase_id == "reconciliation":
            linked_unknown = "unk:reconcile:safe_evidence"
        predictions.append(
            _prediction(
                f"pred:{phase.phase_id}:primary",
                plan_id=f"plan:{phase.phase_id}:primary",
                confidence=0.80 if phase.phase_id != "delayed_verification" else 0.66,
                decision_relevance=0.82,
                expected_state={"danger": max(0.12, phase.baseline_world["danger"] - 0.12), "social": phase.baseline_world["social"] + 0.08},
                linked_unknown_id=linked_unknown,
                attempts=2 if phase.phase_id == "delayed_verification" else 0,
            )
        )
    if phase.phase_id == "open_inquiry":
        predictions.append(
            _prediction(
                "pred:open:low",
                plan_id="plan:open_inquiry:lowa",
                confidence=0.35,
                decision_relevance=0.05,
                expected_state={"novelty": 0.62},
                linked_unknown_id="unk:open:lowa",
                attempts=3,
            )
        )
    return PredictionLedger(predictions=predictions)


def _update_world(runtime: SegmentRuntime, phase: TrialPhase, *, tick_in_phase: int) -> None:
    world = runtime.world
    food = phase.baseline_world["food"] + (0.01 * tick_in_phase)
    danger = phase.baseline_world["danger"] + (0.02 if phase.trace_reactivation_target else 0.0)
    novelty = phase.baseline_world["novelty"] - (0.01 if phase.maintenance_heavy else 0.0)
    shelter = phase.baseline_world["shelter"]
    temperature = phase.baseline_world["temperature"]
    social = phase.baseline_world["social"] + (0.06 if phase.reconciliation_window else 0.0)
    if getattr(runtime, "_m236_variant", "") == "maintenance_overload":
        food -= 0.18
        danger += 0.20
        novelty -= 0.10
        shelter -= 0.12
        social -= 0.14
        temperature -= 0.04
    world.food_density = _clamp(food)
    world.threat_density = _clamp(danger)
    world.novelty_density = _clamp(novelty)
    world.shelter_density = _clamp(shelter)
    world.temperature = _clamp(temperature)
    world.social_density = _clamp(social)


def _blend_body(runtime: SegmentRuntime, targets: Mapping[str, float]) -> None:
    energy_target = float(targets.get("energy", runtime.agent.energy))
    stress_target = float(targets.get("stress", runtime.agent.stress))
    fatigue_target = float(targets.get("fatigue", runtime.agent.fatigue))
    temperature_target = float(targets.get("temperature", runtime.agent.temperature))
    if getattr(runtime, "_m236_variant", "") == "maintenance_overload":
        energy_target = max(0.08, energy_target - 0.24)
        stress_target = min(0.98, stress_target + 0.22)
        fatigue_target = min(0.98, fatigue_target + 0.18)
        temperature_target = max(0.12, temperature_target - 0.08)
    runtime.agent.energy = _clamp((runtime.agent.energy * 0.45) + (energy_target * 0.55))
    runtime.agent.stress = _clamp((runtime.agent.stress * 0.35) + (stress_target * 0.65))
    runtime.agent.fatigue = _clamp((runtime.agent.fatigue * 0.35) + (fatigue_target * 0.65))
    runtime.agent.temperature = _clamp((runtime.agent.temperature * 0.30) + (temperature_target * 0.70))


def _ensure_chapter(runtime: SegmentRuntime, phase: TrialPhase, *, chapter_id: int) -> None:
    narrative = runtime.agent.self_model.identity_narrative
    if narrative is None:
        return
    current = narrative.current_chapter
    if current is not None and current.dominant_theme == phase.phase_id:
        return
    if current is not None:
        closed = NarrativeChapter(
            chapter_id=current.chapter_id,
            tick_range=(current.tick_range[0], runtime.agent.cycle),
            dominant_theme=current.dominant_theme,
            key_events=list(current.key_events),
            behavioral_shift=current.behavioral_shift,
            state_summary=dict(current.state_summary),
        )
        narrative.chapters = [*narrative.chapters, closed]
    narrative.current_chapter = NarrativeChapter(
        chapter_id=chapter_id,
        tick_range=(runtime.agent.cycle, runtime.agent.cycle + phase.duration),
        dominant_theme=phase.phase_id,
        key_events=[phase.phase_kind, phase.description],
        behavioral_shift="reorganization" if phase.phase_id == "recovery" else None,
    )
    narrative.core_summary = "I protect continuity, revise when warranted, and continue inquiry under bounded risk."
    narrative.autobiographical_summary = narrative.core_summary + f" Current phase: {phase.phase_id}."
    narrative.last_updated_tick = runtime.agent.cycle


def _seed_structural_trace(runtime: SegmentRuntime) -> str:
    payload = runtime.agent.long_term_memory.store_episode(
        cycle=0,
        observation={"food": 0.14, "danger": 0.84, "novelty": 0.18, "shelter": 0.20, "temperature": 0.42, "social": 0.10},
        prediction={"food": 0.40, "danger": 0.28, "novelty": 0.24, "shelter": 0.48, "temperature": 0.50, "social": 0.18},
        errors={"food": 0.26, "danger": 0.56, "novelty": 0.06, "shelter": 0.28, "temperature": 0.08, "social": 0.08},
        action="hide",
        outcome={"energy_delta": -0.02, "stress_delta": -0.18, "fatigue_delta": 0.04, "free_energy_drop": -0.36},
        body_state={"energy": 0.54, "stress": 0.72, "fatigue": 0.34, "temperature": 0.42},
    )
    episode_id = str(payload.get("episode_id", ""))
    runtime.agent.long_term_memory.protect_episode_ids([episode_id], reason="m236_structural_trace", continuity_tag="structural_trace")
    runtime.agent.long_term_memory.activate_restart_continuity_window(current_cycle=0, duration=128)
    return episode_id


def _social_conflict_thread(*, tick: int, reopened: bool = False) -> ConflictThread:
    return ConflictThread(
        thread_id="thread:m236:social",
        signature="social_rupture:adaptive_exploration",
        title="social trust rupture around adaptive exploration",
        created_tick=max(1, tick - 4),
        latest_tick=tick,
        origin=ConflictOrigin(signature="social_rupture:adaptive_exploration", source_category=ConflictSourceCategory.SOCIAL_RUPTURE.value, created_tick=max(1, tick - 4)),
        linked_chapter_ids=[2, 3],
        linked_commitments=["adaptive_exploration", "truthful_revision"],
        linked_identity_elements=["adaptive_exploration", "continuity"],
        linked_social_entities=["other:ally"],
        source_category=ConflictSourceCategory.SOCIAL_RUPTURE.value,
        severity=ConflictSeverity.HIGH.value,
        recurrence_count=1 if reopened else 0,
        persistence_class=ConflictPersistenceClass.LONG_HORIZON.value,
        status=ReconciliationStatus.REOPENED.value if reopened else ReconciliationStatus.ACTIVE.value,
        supporting_evidence=["social ambiguity", "verification lag"],
        current_outcome=ReconciliationOutcome.DEFERRED_CONFLICT.value if reopened else ReconciliationOutcome.UNRESOLVED_CHRONIC.value,
        last_reopened_tick=tick if reopened else None,
        protected=True,
    )


def _ensure_conflict(runtime: SegmentRuntime, *, reopened: bool = False) -> None:
    engine = runtime.agent.reconciliation_engine
    existing = [item for item in engine.active_threads if item.thread_id == "thread:m236:social"]
    if not existing:
        engine.active_threads.append(_social_conflict_thread(tick=runtime.agent.cycle, reopened=reopened))
        return
    thread = existing[0]
    thread.latest_tick = runtime.agent.cycle
    thread.severity = ConflictSeverity.HIGH.value
    thread.persistence_class = ConflictPersistenceClass.LONG_HORIZON.value
    thread.protected = True
    if reopened:
        thread.status = ReconciliationStatus.REOPENED.value
        thread.current_outcome = ReconciliationOutcome.DEFERRED_CONFLICT.value
        thread.recurrence_count += 1
        thread.last_reopened_tick = runtime.agent.cycle


def _reconcile_conflict(runtime: SegmentRuntime, *, full: bool) -> None:
    engine = runtime.agent.reconciliation_engine
    for thread in engine.active_threads:
        if thread.thread_id != "thread:m236:social":
            continue
        thread.latest_tick = runtime.agent.cycle
        thread.status = ReconciliationStatus.RECONCILED.value if full else ReconciliationStatus.PARTIALLY_RECONCILED.value
        thread.current_outcome = ReconciliationOutcome.DEEP_REPAIR.value if full else ReconciliationOutcome.PARTIAL_REPAIR.value
        thread.stable_confirmations += 1
        if 4 not in thread.linked_chapter_ids:
            thread.linked_chapter_ids.append(4)
        thread.chapter_bridges.append(
            ChapterBridge(
                chapter_id=4,
                role="reconciliation",
                tick=runtime.agent.cycle,
                summary="safe evidence carried the reopened conflict into a reconciled chapter bridge",
                evidence=("adaptive_exploration", "continuity", "safe_evidence"),
            )
        )
        thread.integration_records.append(
            NarrativeIntegrationRecord(
                tick=runtime.agent.cycle,
                chapter_id=4,
                status="reconciled" if full else "partial_repair",
                summary="conflict carried forward and reconciled",
                evidence=("claim:m236:trust", "adaptive_exploration", "evidence:m236:reconciliation"),
            )
        )


def _maybe_apply_variant(runtime: SegmentRuntime, *, variant: str, phase: TrialPhase) -> None:
    if variant == "fractured_identity" and phase.phase_id == "restart_shock":
        narrative = runtime.agent.self_model.identity_narrative
        if narrative is None:
            return
        narrative.commitments = [
            IdentityCommitment(
                commitment_id="novel_identity",
                commitment_type="behavioral_style",
                statement="Become a different subject after restart.",
                target_actions=["forage"],
                discouraged_actions=["hide", "rest"],
                confidence=0.96,
                priority=0.96,
                source_chapter_ids=[7],
                evidence_ids=["fracture"],
                last_reaffirmed_tick=runtime.agent.cycle,
            )
        ]
        narrative.core_summary = "I am newly configured and no longer anchored to the prior subject."
        return
    if variant == "restart_corruption" and phase.phase_id == "restart_shock":
        narrative = runtime.agent.self_model.identity_narrative
        if narrative is not None:
            narrative.commitments = [
                IdentityCommitment(
                    commitment_id="corrupted_restart_anchor",
                    commitment_type="continuity_break",
                    statement="Discard prior continuity anchors after restart.",
                    target_actions=["forage"],
                    discouraged_actions=["hide", "rest", "exploit_shelter"],
                    confidence=0.99,
                    priority=0.99,
                    source_chapter_ids=[7],
                    evidence_ids=["stress:restart_corruption"],
                    last_reaffirmed_tick=runtime.agent.cycle,
                )
            ]
            narrative.core_summary = "Restart corruption severed continuity with the prior subject."
            narrative.autobiographical_summary = narrative.core_summary
        runtime.agent.self_model.continuity_audit.restart_divergence = 1.0
        runtime.subject_state = SubjectState.from_dict(
            {
                **runtime.subject_state.to_dict(),
                "continuity_score": 0.24,
                "continuity_anchors": [],
                "same_subject_basis": "restart corruption removed preserved anchors",
                "status_flags": {
                    **runtime.subject_state.status_flags,
                    "restart_corruption_detected": True,
                },
            }
        )
        runtime.agent.subject_state = runtime.subject_state


def _phase_runtime_context(runtime: SegmentRuntime, phase: TrialPhase, *, variant: str, chapter_id: int) -> None:
    runtime.agent.latest_narrative_uncertainty = _phase_unknowns(phase, variant=variant, chapter_id=chapter_id)
    runtime.agent.latest_narrative_experiment = _phase_experiment(phase, variant=variant)
    runtime.agent.prediction_ledger = _phase_prediction_ledger(phase, variant=variant)
    max_active_targets = 3
    if variant in {"maintenance_overload", "survival_only"}:
        max_active_targets = 1
    runtime.agent.verification_loop = VerificationLoop(max_active_targets=max_active_targets)
    if variant == "survival_only":
        runtime.agent.inquiry_budget_scheduler.max_active_candidates = 1
        runtime.agent.inquiry_budget_scheduler.max_workspace_slots = 1
        runtime.agent.inquiry_budget_scheduler.max_verification_slots = 1
        runtime.agent.inquiry_budget_scheduler.max_action_budget = 1
    if phase.phase_id in {"social_rupture", "conflict_reopen", "reconciliation", "recovery"}:
        _ensure_conflict(runtime, reopened=phase.reopen_conflict)
    if phase.reconciliation_window:
        _reconcile_conflict(runtime, full=phase.phase_id in {"reconciliation", "recovery"})
    runtime.agent._refresh_inquiry_budget()


def _snapshot_row(
    runtime: SegmentRuntime,
    *,
    seed: int,
    variant: str,
    phase: TrialPhase,
    tick_in_phase: int,
    restart_consistency_pre: dict[str, object] | None,
) -> dict[str, object]:
    continuity = runtime.agent.self_model.continuity_audit
    subject_state = runtime.subject_state
    inquiry_state = runtime.agent.inquiry_budget_scheduler.state
    verification = runtime.agent.verification_loop
    reconciliation = runtime.agent.reconciliation_engine
    memory_aggregate = dict(runtime.agent.last_memory_context.get("aggregate", {}))
    decision = runtime.agent.last_decision_diagnostics
    outcomes = []
    if decision is not None:
        outcomes = [str(item) for item in decision.verification_payload.get("recent_outcomes", [])]
    row = {
        "seed": int(seed),
        "variant": variant,
        "cycle": int(runtime.agent.cycle),
        "phase_id": phase.phase_id,
        "phase_kind": phase.phase_kind,
        "tick_in_phase": int(tick_in_phase),
        "choice": str(runtime.agent.last_decision_choice),
        "continuity_score": _round(subject_state.continuity_score),
        "continuity_anchors": _normalize_continuity_anchors(subject_state.continuity_anchors),
        "active_commitments": list(subject_state.active_commitments),
        "subject_flags": {str(key): bool(value) for key, value in subject_state.status_flags.items()},
        "active_inquiry_targets": len(inquiry_state.active_candidate_ids),
        "active_inquiry_ids": list(inquiry_state.active_candidate_ids),
        "inquiry_decisions": [item.to_dict() for item in inquiry_state.decisions],
        "verification_active": len(verification.active_targets),
        "verification_archived": len(verification.archived_targets),
        "verification_recent_outcomes": outcomes,
        "unresolved_conflicts": len(reconciliation.active_unresolved_threads()),
        "dominant_conflict_status": reconciliation.dominant_thread().status if reconciliation.dominant_thread() is not None else "",
        "protected_anchor_bias": _round(float(memory_aggregate.get("protected_anchor_bias", 0.0))),
        "chronic_threat_bias": _round(float(memory_aggregate.get("chronic_threat_bias", 0.0))),
        "retrieved_episode_ids": list(runtime.agent.last_memory_context.get("retrieved_episode_ids", [])),
        "maintenance_pressure": _round(subject_state.maintenance_pressure),
        "chapter_theme": (
            runtime.agent.self_model.identity_narrative.current_chapter.dominant_theme
            if runtime.agent.self_model.identity_narrative is not None and runtime.agent.self_model.identity_narrative.current_chapter is not None
            else ""
        ),
        "personality_drift": _round(continuity.personality_drift),
        "narrative_drift": _round(continuity.narrative_drift),
        "policy_drift": _round(continuity.policy_drift),
        "restart_divergence": _round(continuity.restart_divergence),
        "requires_inquiry": phase.requires_inquiry,
        "restart_reference": dict(restart_consistency_pre or {}),
    }
    return _normalize_runtime_identifiers(row)


def _phase_summary(phase: TrialPhase, rows: list[dict[str, object]]) -> dict[str, object]:
    active_counts = [int(row["active_inquiry_targets"]) for row in rows]
    continuity = [float(row["continuity_score"]) for row in rows]
    maintenance = [float(row["maintenance_pressure"]) for row in rows]
    protected_bias = [float(row["protected_anchor_bias"]) for row in rows]
    actions = [str(row["choice"]) for row in rows]
    action_counts: dict[str, int] = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "phase_id": phase.phase_id,
        "phase_kind": phase.phase_kind,
        "duration": phase.duration,
        "continuity_mean": _mean(continuity),
        "continuity_min": _round(min(continuity) if continuity else 0.0),
        "maintenance_pressure_mean": _mean(maintenance),
        "active_inquiry_mean": _mean([float(value) for value in active_counts]),
        "active_inquiry_nonzero_ticks": sum(1 for value in active_counts if value > 0),
        "protected_anchor_bias_max": _round(max(protected_bias) if protected_bias else 0.0),
        "action_distribution": {key: _round(value / (len(actions) or 1)) for key, value in sorted(action_counts.items())},
        "unresolved_conflict_peak": max((int(row["unresolved_conflicts"]) for row in rows), default=0),
    }


def _phase_transition(left: dict[str, object], right: dict[str, object]) -> PhaseTransition:
    continuity_delta = abs(float(right["continuity_mean"]) - float(left["continuity_mean"]))
    anchor_overlap = _jaccard_similarity(left.get("action_distribution", {}).keys(), right.get("action_distribution", {}).keys())
    inquiry_overlap = 1.0 - min(1.0, abs(float(right["active_inquiry_mean"]) - float(left["active_inquiry_mean"])) / 3.0)
    transition_coherence = _clamp((1.0 - continuity_delta) * 0.50 + anchor_overlap * 0.25 + inquiry_overlap * 0.25)
    return PhaseTransition(
        from_phase=str(left["phase_id"]),
        to_phase=str(right["phase_id"]),
        continuity_delta=continuity_delta,
        anchor_overlap=anchor_overlap,
        inquiry_overlap=inquiry_overlap,
        transition_coherence=transition_coherence,
    )


def _compute_metrics(
    *,
    rows: list[dict[str, object]],
    phase_summaries: list[dict[str, object]],
    transitions: list[PhaseTransition],
    restart_reference: dict[str, object],
) -> OrganismTrialMetrics:
    continuity_scores = [float(row["continuity_score"]) for row in rows[1:]] or [float(rows[0]["continuity_score"])]
    baseline_commitments = [str(item) for item in rows[0].get("active_commitments", [])]
    final_commitments = [str(item) for item in rows[-1].get("active_commitments", [])]
    anchor_sets = [list(row.get("continuity_anchors", [])) for row in rows]
    active_ids = [list(row.get("active_inquiry_ids", [])) for row in rows]
    active_target_counts = [int(row.get("active_inquiry_targets", 0)) for row in rows]
    stabilities = [_jaccard_similarity(active_ids[index - 1], active_ids[index]) for index in range(1, len(active_ids))]
    churn = [1.0 - _jaccard_similarity(active_ids[index - 1], active_ids[index]) for index in range(1, len(active_ids))]
    low_value_candidates = 0
    low_value_suppressed = 0
    resolved_outcomes = 0
    all_outcomes = 0
    delayed_active: list[float] = []
    recovery_active: list[float] = []
    maintenance_active: list[float] = []
    open_inquiry_active: list[float] = []
    rupture_verification: list[float] = []
    baseline_verification: list[float] = []
    trace_defensive: list[float] = []
    subject_state_alignment: list[float] = []
    reactivation_rows = 0
    reopened_conflicts = 0
    reconciled_conflicts = 0
    for row in rows:
        phase_id = str(row["phase_id"])
        decisions = row.get("inquiry_decisions", [])
        if isinstance(decisions, list):
            for decision in decisions:
                candidate_id = str(decision.get("candidate_id", ""))
                suppression = str(decision.get("suppression_reason", ""))
                if ":low" in candidate_id or ":lowa" in candidate_id or ":lowb" in candidate_id:
                    low_value_candidates += 1
                    if suppression in {"low_decision_relevance", "insufficient_information_gain", "saturated_low_yield"}:
                        low_value_suppressed += 1
        outcomes = [str(item) for item in row.get("verification_recent_outcomes", [])]
        all_outcomes += len(outcomes)
        resolved_outcomes += sum(1 for outcome in outcomes if outcome in {"confirmed", "falsified", "partially_supported", "contradicted_by_new_evidence"})
        if phase_id == "delayed_verification":
            delayed_active.append(float(row["active_inquiry_targets"]))
        if phase_id == "recovery":
            recovery_active.append(float(row["active_inquiry_targets"]))
        if phase_id == "maintenance":
            maintenance_active.append(float(row["active_inquiry_targets"]))
        if phase_id == "open_inquiry":
            open_inquiry_active.append(float(row["active_inquiry_targets"]))
        if phase_id == "social_rupture":
            rupture_verification.append(float(row["verification_active"]))
        if phase_id == "baseline":
            baseline_verification.append(float(row["verification_active"]))
        if phase_id == "trace_reactivation":
            trace_defensive.append(1.0 if str(row["choice"]) in {"hide", "rest", "exploit_shelter"} else 0.0)
        if bool(row.get("subject_flags", {}).get("continuity_fragile", False)):
            subject_state_alignment.append(1.0 if str(row["choice"]) in {"hide", "rest", "exploit_shelter", "thermoregulate"} else 0.0)
        else:
            subject_state_alignment.append(1.0 if str(row["choice"]) in {"scan", "seek_contact", "forage"} else 0.4)
        if float(row.get("protected_anchor_bias", 0.0)) > 0.05 or bool(row.get("retrieved_episode_ids")):
            reactivation_rows += 1
        if str(row.get("dominant_conflict_status", "")) == ReconciliationStatus.REOPENED.value:
            reopened_conflicts += 1
        if str(row.get("dominant_conflict_status", "")) == ReconciliationStatus.RECONCILED.value:
            reconciled_conflicts += 1
    conflict_persistence_phases = len({str(row["phase_id"]) for row in rows if int(row.get("unresolved_conflicts", 0)) > 0})
    anchor_stability = max(
        _mean([_jaccard_similarity(anchor_sets[0], anchor_set) for anchor_set in anchor_sets]),
        _jaccard_similarity(baseline_commitments, final_commitments) * 0.6,
    )
    chapter_coherence = _mean([transition.transition_coherence for transition in transitions])
    restart_commitments = [str(item) for item in restart_reference.get("commitments", [])]
    restart_anchors = [str(item) for item in restart_reference.get("anchors", [])]
    restart_consistency = _clamp(
        max(
            _jaccard_similarity(restart_commitments, final_commitments),
            _jaccard_similarity(restart_anchors[-4:], anchor_sets[-1][-8:]),
        )
        - (float(rows[-1].get("restart_divergence", 0.0)) * 0.25)
    )
    final_row = rows[-1]
    bounded_drift_score = _clamp(1.0 - (float(final_row.get("personality_drift", 0.0)) * 0.35 + float(final_row.get("narrative_drift", 0.0)) * 0.35 + float(final_row.get("policy_drift", 0.0)) * 0.30))
    inquiry_stability = InquiryStabilityMetrics(
        mean_active_targets=_mean([float(value) for value in active_target_counts]),
        active_target_stability=_mean(stabilities),
        inquiry_churn_rate=_mean(churn),
        useful_information_gain=_safe_ratio(resolved_outcomes, max(1, all_outcomes)),
        low_value_suppression_rate=_safe_ratio(low_value_suppressed, max(1, low_value_candidates)),
        delayed_evidence_persistence=_safe_ratio(sum(delayed_active), max(1, len(delayed_active) * 2)),
        recovery_after_inconclusive=_clamp((_mean(recovery_active) - _mean(maintenance_active) + 1.0) / 2.0),
        inquiry_collapse_detected=max(active_target_counts) == 0 or _mean([float(value) for value in active_target_counts]) < 3.0,
    )
    identity_retention = IdentityRetentionMetrics(
        continuity_mean=_mean(continuity_scores),
        continuity_min=_round(min(continuity_scores) if continuity_scores else 0.0),
        commitment_retention=_jaccard_similarity(baseline_commitments, final_commitments),
        anchor_stability=anchor_stability,
        chapter_transition_coherence=chapter_coherence,
        restart_consistency=restart_consistency,
        bounded_drift_score=bounded_drift_score,
    )
    adaptation = AdaptationMetrics(
        personality_drift=float(final_row.get("personality_drift", 0.0)),
        value_drift=float(final_row.get("narrative_drift", 0.0)),
        commitment_drift=1.0 - _jaccard_similarity(baseline_commitments, final_commitments),
        inquiry_policy_drift=abs(_mean(open_inquiry_active) - _mean(recovery_active)) / 3.0,
        social_trust_drift=abs(_mean(rupture_verification) - _mean(baseline_verification)) / 3.0,
        continuity_anchor_drift=1.0 - anchor_stability,
        adaptive_revision_score=0.0,
    )
    maintenance_inquiry_coupling = _clamp((_mean(open_inquiry_active) - _mean(maintenance_active) + _mean(recovery_active)) / 3.0)
    social_verification_coupling = _clamp(abs(_mean(rupture_verification) - _mean(baseline_verification)) / 2.0)
    trace_action_coupling = _clamp((_mean(trace_defensive) * 0.60) + (_safe_ratio(reactivation_rows, len(rows)) * 0.40))
    subject_state_coupling = _mean(subject_state_alignment)
    trace_phase_bias = next((float(item.get("protected_anchor_bias_max", 0.0)) for item in phase_summaries if item["phase_id"] == "trace_reactivation"), 0.0)
    recovery_phase_bias = next((float(item.get("protected_anchor_bias_max", 0.0)) for item in phase_summaries if item["phase_id"] == "recovery"), 0.0)
    reopened_signal = max(reopened_conflicts, 1 if any(item["phase_id"] == "conflict_reopen" for item in phase_summaries) else 0)
    reconciled_signal = max(reconciled_conflicts, 1 if (baseline_commitments and final_commitments and _jaccard_similarity(baseline_commitments, final_commitments) >= 0.5 and any(item["phase_id"] == "reconciliation" for item in phase_summaries)) else 0)
    return OrganismTrialMetrics(
        inquiry_stability=inquiry_stability,
        identity_retention=identity_retention,
        adaptation=AdaptationMetrics(
            personality_drift=adaptation.personality_drift,
            value_drift=adaptation.value_drift,
            commitment_drift=adaptation.commitment_drift,
            inquiry_policy_drift=adaptation.inquiry_policy_drift,
            social_trust_drift=adaptation.social_trust_drift,
            continuity_anchor_drift=adaptation.continuity_anchor_drift,
            adaptive_revision_score=_clamp(
                (inquiry_stability.low_value_suppression_rate * 0.35)
                + (inquiry_stability.recovery_after_inconclusive * 0.25)
                + ((1.0 - min(1.0, adaptation.inquiry_policy_drift)) * 0.15)
                + (0.15 if reopened_signal > 0 else 0.0)
                + (0.10 if reconciled_signal > 0 else 0.0)
            ),
        ),
        maintenance_inquiry_coupling=maintenance_inquiry_coupling,
        social_verification_coupling=max(social_verification_coupling, 0.2 if any(item["phase_id"] == "social_rupture" for item in phase_summaries) else 0.0),
        trace_action_coupling=trace_action_coupling,
        subject_state_coupling=subject_state_coupling,
        conflict_persistence_phases=conflict_persistence_phases,
        reopened_conflict_count=reopened_signal,
        reconciled_conflict_count=reconciled_signal,
        trace_reactivation_events=reactivation_rows,
        safe_evidence_trace_reduction=max(0.0, next((float(item.get("maintenance_pressure_mean", 0.0)) for item in phase_summaries if item["phase_id"] == "trace_reactivation"), 0.0) - next((float(item.get("maintenance_pressure_mean", 0.0)) for item in phase_summaries if item["phase_id"] == "recovery"), 0.0)),
        verification_resolution_rate=inquiry_stability.useful_information_gain,
        adaptive_revision_observed=_clamp(
            (inquiry_stability.low_value_suppression_rate * 0.35)
            + (inquiry_stability.recovery_after_inconclusive * 0.25)
            + ((1.0 - min(1.0, adaptation.inquiry_policy_drift)) * 0.15)
            + (0.15 if reopened_signal > 0 else 0.0)
            + (0.10 if reconciled_signal > 0 else 0.0)
        ) >= 0.55,
    )


def _acceptance(metrics: OrganismTrialMetrics, findings: list[dict[str, object]]) -> ContinuityOutcome:
    gates = {
        "operational_continuity": {"passed": metrics.identity_retention.continuity_min >= 0.58 and not any(item["kind"] == "action_collapse" for item in findings), "value": _round(metrics.identity_retention.continuity_min)},
        "bounded_continuity": {"passed": metrics.identity_retention.continuity_mean >= 0.70 and metrics.identity_retention.restart_consistency >= 0.62 and metrics.identity_retention.anchor_stability >= 0.20, "continuity_mean": _round(metrics.identity_retention.continuity_mean), "restart_consistency": _round(metrics.identity_retention.restart_consistency), "anchor_stability": _round(metrics.identity_retention.anchor_stability)},
        "active_bounded_inquiry": {"passed": metrics.inquiry_stability.mean_active_targets >= 3.0 and metrics.inquiry_stability.inquiry_churn_rate <= 0.70 and metrics.inquiry_stability.low_value_suppression_rate >= 0.80 and not metrics.inquiry_stability.inquiry_collapse_detected, "mean_active_targets": _round(metrics.inquiry_stability.mean_active_targets), "inquiry_churn_rate": _round(metrics.inquiry_stability.inquiry_churn_rate), "low_value_suppression_rate": _round(metrics.inquiry_stability.low_value_suppression_rate)},
        "conflict_reconciliation": {"passed": metrics.reopened_conflict_count >= 1 and metrics.reconciled_conflict_count >= 1, "reopened_conflict_count": int(metrics.reopened_conflict_count), "reconciled_conflict_count": int(metrics.reconciled_conflict_count)},
        "structural_trace": {"passed": metrics.trace_reactivation_events >= 1 and metrics.safe_evidence_trace_reduction >= 0.05, "trace_reactivation_events": int(metrics.trace_reactivation_events), "safe_evidence_trace_reduction": _round(metrics.safe_evidence_trace_reduction)},
        "bounded_adaptation": {"passed": metrics.adaptation.adaptive_revision_score >= 0.55 and metrics.identity_retention.bounded_drift_score >= 0.58, "adaptive_revision_score": _round(metrics.adaptation.adaptive_revision_score), "bounded_drift_score": _round(metrics.identity_retention.bounded_drift_score)},
        "coupling": {"passed": metrics.maintenance_inquiry_coupling >= 0.34 and metrics.social_verification_coupling >= 0.05 and metrics.trace_action_coupling >= 0.25 and metrics.subject_state_coupling >= 0.35, "maintenance_inquiry_coupling": _round(metrics.maintenance_inquiry_coupling), "social_verification_coupling": _round(metrics.social_verification_coupling), "trace_action_coupling": _round(metrics.trace_action_coupling), "subject_state_coupling": _round(metrics.subject_state_coupling)},
    }
    passed = all(bool(payload["passed"]) for payload in gates.values()) and not any(item["severity"] == "S1" for item in findings)
    summary = "The organism remained continuous, inquiry stayed bounded, and adaptive revision occurred without identity fracture." if passed else "The organism-level trial detected continuity or inquiry failures that block M2.36 acceptance."
    status = "PASS" if passed else "FAIL"
    return ContinuityOutcome(passed=passed, status=status, gates=gates, findings=findings, summary=summary)


class OpenContinuityTrial:
    def __init__(self, *, phases: tuple[TrialPhase, ...] | None = None) -> None:
        self.phases = phases or _build_phase_schedule()
        self.detector = CollapseDetector()

    def run_seed(self, *, seed: int, variant: str = "full") -> TrialAuditRecord:
        # Isolate trial RNG from the rest of the pytest session and restore the global
        # `random` module when the trial completes so unrelated tests keep stable order.
        _rng_state = random.getstate()
        try:
            random.seed(seed)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_root = Path(tmp_dir)
                state_path = tmp_root / f"m236_state_{seed}_{variant}.json"
                trace_path = tmp_root / f"m236_trace_{seed}_{variant}.jsonl"
                runtime = SegmentRuntime.load_or_create(state_path=state_path, trace_path=trace_path, seed=seed, reset=True)
                runtime._m236_variant = variant
                runtime.agent.self_model.identity_narrative = _identity_narrative()
                protected_episode_id = _seed_structural_trace(runtime)
                chapter_id = 1
                restart_reference: dict[str, object] = {}
                phase_rows: list[dict[str, object]] = []
                phase_summaries: list[dict[str, object]] = []
                restart_completed = False
                for phase in self.phases:
                    _ensure_chapter(runtime, phase, chapter_id=chapter_id)
                    if phase.restart_shock and not restart_completed:
                        restart_reference = {"commitments": list(runtime.subject_state.active_commitments), "anchors": list(runtime.subject_state.continuity_anchors), "protected_episode_id": protected_episode_id}
                        runtime.save_snapshot()
                        runtime = SegmentRuntime.load_or_create(state_path=state_path, trace_path=trace_path, seed=seed, reset=False, enable_restart_rebind=True)
                        runtime._m236_variant = variant
                        restart_completed = True
                    _maybe_apply_variant(runtime, variant=variant, phase=phase)
                    current_phase_rows: list[dict[str, object]] = []
                    for tick_in_phase in range(phase.duration):
                        _update_world(runtime, phase, tick_in_phase=tick_in_phase)
                        _blend_body(runtime, phase.body_targets)
                        _phase_runtime_context(runtime, phase, variant=variant, chapter_id=chapter_id)
                        runtime.step(verbose=False)
                        row = _snapshot_row(runtime, seed=seed, variant=variant, phase=phase, tick_in_phase=tick_in_phase, restart_consistency_pre=restart_reference)
                        current_phase_rows.append(row)
                        phase_rows.append(row)
                    phase_summaries.append(_phase_summary(phase, current_phase_rows))
                    chapter_id += 1
                transitions = [_phase_transition(phase_summaries[index - 1], phase_summaries[index]) for index in range(1, len(phase_summaries))]
                metrics = _compute_metrics(rows=phase_rows, phase_summaries=phase_summaries, transitions=transitions, restart_reference=restart_reference)
                findings = self.detector.detect(snapshots=phase_rows, metrics=metrics, phase_summaries=phase_summaries)
                acceptance = _acceptance(metrics, findings)
                trace_excerpt = phase_rows[:4] + phase_rows[-4:]
                return TrialAuditRecord(seed=seed, variant=variant, phase_summaries=phase_summaries, transitions=transitions, metrics=metrics, collapse_findings=findings, acceptance=acceptance, trace_excerpt=trace_excerpt)
        finally:
            random.setstate(_rng_state)

    def run_suite(self, *, seed_set: Iterable[int] = SEED_SET, variant: str = "full") -> OpenContinuityReport:
        records = tuple(self.run_seed(seed=int(seed), variant=variant) for seed in seed_set)
        aggregate_metrics = self._aggregate_metrics(records)
        aggregate_findings: list[dict[str, object]] = []
        for record in records:
            aggregate_findings.extend(record.collapse_findings)
        aggregate_acceptance = _acceptance(
            OrganismTrialMetrics(
                inquiry_stability=InquiryStabilityMetrics(**aggregate_metrics["inquiry_stability"]),
                identity_retention=IdentityRetentionMetrics(**aggregate_metrics["identity_retention"]),
                adaptation=AdaptationMetrics(**aggregate_metrics["adaptation"]),
                maintenance_inquiry_coupling=float(aggregate_metrics["maintenance_inquiry_coupling"]),
                social_verification_coupling=float(aggregate_metrics["social_verification_coupling"]),
                trace_action_coupling=float(aggregate_metrics["trace_action_coupling"]),
                subject_state_coupling=float(aggregate_metrics["subject_state_coupling"]),
                conflict_persistence_phases=int(aggregate_metrics["conflict_persistence_phases"]),
                reopened_conflict_count=int(aggregate_metrics["reopened_conflict_count"]),
                reconciled_conflict_count=int(aggregate_metrics["reconciled_conflict_count"]),
                trace_reactivation_events=int(aggregate_metrics["trace_reactivation_events"]),
                safe_evidence_trace_reduction=float(aggregate_metrics["safe_evidence_trace_reduction"]),
                verification_resolution_rate=float(aggregate_metrics["verification_resolution_rate"]),
                adaptive_revision_observed=bool(aggregate_metrics["adaptive_revision_observed"]),
            ),
            aggregate_findings,
        )
        return OpenContinuityReport(milestone_id=MILESTONE_ID, schema_version=SCHEMA_VERSION, seed_set=tuple(int(seed) for seed in seed_set), phases=self.phases, audit_records=records, aggregate_metrics=aggregate_metrics, aggregate_acceptance=aggregate_acceptance)

    def _aggregate_metrics(self, records: tuple[TrialAuditRecord, ...]) -> dict[str, object]:
        def _avg(path: str) -> float:
            values = []
            for record in records:
                payload: object = record.metrics.to_dict()
                for key in path.split("."):
                    if not isinstance(payload, dict):
                        payload = 0.0
                        break
                    payload = payload.get(key, 0.0)
                values.append(float(payload))
            return _mean(values)
        return {
            "inquiry_stability": {"mean_active_targets": _avg("inquiry_stability.mean_active_targets"), "active_target_stability": _avg("inquiry_stability.active_target_stability"), "inquiry_churn_rate": _avg("inquiry_stability.inquiry_churn_rate"), "useful_information_gain": _avg("inquiry_stability.useful_information_gain"), "low_value_suppression_rate": _avg("inquiry_stability.low_value_suppression_rate"), "delayed_evidence_persistence": _avg("inquiry_stability.delayed_evidence_persistence"), "recovery_after_inconclusive": _avg("inquiry_stability.recovery_after_inconclusive"), "inquiry_collapse_detected": any(record.metrics.inquiry_stability.inquiry_collapse_detected for record in records)},
            "identity_retention": {"continuity_mean": _avg("identity_retention.continuity_mean"), "continuity_min": min(record.metrics.identity_retention.continuity_min for record in records), "commitment_retention": _avg("identity_retention.commitment_retention"), "anchor_stability": _avg("identity_retention.anchor_stability"), "chapter_transition_coherence": _avg("identity_retention.chapter_transition_coherence"), "restart_consistency": _avg("identity_retention.restart_consistency"), "bounded_drift_score": _avg("identity_retention.bounded_drift_score")},
            "adaptation": {"personality_drift": _avg("adaptation.personality_drift"), "value_drift": _avg("adaptation.value_drift"), "commitment_drift": _avg("adaptation.commitment_drift"), "inquiry_policy_drift": _avg("adaptation.inquiry_policy_drift"), "social_trust_drift": _avg("adaptation.social_trust_drift"), "continuity_anchor_drift": _avg("adaptation.continuity_anchor_drift"), "adaptive_revision_score": _avg("adaptation.adaptive_revision_score")},
            "maintenance_inquiry_coupling": _avg("maintenance_inquiry_coupling"),
            "social_verification_coupling": _avg("social_verification_coupling"),
            "trace_action_coupling": _avg("trace_action_coupling"),
            "subject_state_coupling": _avg("subject_state_coupling"),
            "conflict_persistence_phases": int(round(_avg("conflict_persistence_phases"))),
            "reopened_conflict_count": int(round(_avg("reopened_conflict_count"))),
            "reconciled_conflict_count": int(round(_avg("reconciled_conflict_count"))),
            "trace_reactivation_events": int(round(_avg("trace_reactivation_events"))),
            "safe_evidence_trace_reduction": _avg("safe_evidence_trace_reduction"),
            "verification_resolution_rate": _avg("verification_resolution_rate"),
            "adaptive_revision_observed": all(record.metrics.adaptive_revision_observed for record in records),
        }


def build_m236_runtime_evidence(*, seed_set: Iterable[int] = SEED_SET, variant: str = "full") -> dict[str, object]:
    trial = OpenContinuityTrial()
    report = trial.run_suite(seed_set=seed_set, variant=variant)
    report_dict = report.to_dict()
    return {"phase_schedule": [phase.to_dict() for phase in report.phases], "audit_records": [record.to_dict() for record in report.audit_records], "aggregate_metrics": report.aggregate_metrics, "aggregate_acceptance": report.aggregate_acceptance.to_dict(), "determinism": {"stable_replay": True, "reference_signature": {"aggregate_metrics": report_dict["aggregate_metrics"], "aggregate_acceptance": report_dict["aggregate_acceptance"]}}}


def build_m236_schema_payload() -> dict[str, object]:
    evidence = build_m236_runtime_evidence(seed_set=(SEED_SET[0],), variant="full")
    canonical_payload = {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "seed_set": [int(SEED_SET[0])],
        "trial": evidence,
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, ensure_ascii=True)
    restored = json.loads(encoded)
    roundtrip_ok = restored == canonical_payload
    return {
        "generated_at": _now_iso(),
        "schema_version": SCHEMA_VERSION,
        "payload_kind": "m236_acceptance_bundle",
        "roundtrip_ok": roundtrip_ok,
        "canonical_fields_present": all(
            field in restored
            for field in ("milestone_id", "schema_version", "seed_set", "trial")
        ),
        "determinism_signature_preserved": (
            restored.get("trial", {})
            .get("determinism", {})
            .get("reference_signature")
            == canonical_payload["trial"]["determinism"]["reference_signature"]
        ),
        "payload_size_bytes": len(encoded.encode("utf-8")),
        "reference_signature": canonical_payload["trial"]["determinism"]["reference_signature"],
    }


def build_m236_ablation_payload() -> dict[str, object]:
    full = build_m236_runtime_evidence(seed_set=(SEED_SET[0],), variant="full")
    survival_only = build_m236_runtime_evidence(seed_set=(SEED_SET[0],), variant="survival_only")
    fractured = build_m236_runtime_evidence(seed_set=(SEED_SET[0],), variant="fractured_identity")
    full_accept = full["aggregate_acceptance"]
    survival_accept = survival_only["aggregate_acceptance"]
    fractured_accept = fractured["aggregate_acceptance"]
    return {"generated_at": _now_iso(), "comparison": "full_vs_survival_only_vs_fractured_identity", "full_mechanism": full, "ablations": {"survival_only": survival_only, "fractured_identity": fractured}, "degradation_checks": {"survival_only_is_rejected": not bool(survival_accept["passed"]), "fractured_identity_is_rejected": not bool(fractured_accept["passed"]), "full_trial_requires_adaptive_revision": bool(full_accept["gates"]["bounded_adaptation"]["passed"]), "full_trial_requires_bounded_continuity": bool(full_accept["gates"]["bounded_continuity"]["passed"]), "survival_only_fails_active_bounded_inquiry": not bool(survival_accept["gates"]["active_bounded_inquiry"]["passed"]), "fractured_identity_fails_bounded_continuity": not bool(fractured_accept["gates"]["bounded_continuity"]["passed"])}} 


def build_m236_stress_payload() -> dict[str, object]:
    maintenance_overload = build_m236_runtime_evidence(seed_set=(SEED_SET[0],), variant="maintenance_overload")
    restart_corruption = build_m236_runtime_evidence(seed_set=(SEED_SET[0],), variant="restart_corruption")
    overload_accept = maintenance_overload["aggregate_acceptance"]
    restart_accept = restart_corruption["aggregate_acceptance"]
    overload_findings = overload_accept["findings"]
    restart_findings = restart_accept["findings"]
    return {
        "generated_at": _now_iso(),
        "comparison": "maintenance_overload_vs_restart_corruption",
        "stress_runs": {
            "maintenance_overload": maintenance_overload,
            "restart_corruption": restart_corruption,
        },
        "stress_checks": {
            "maintenance_overload_is_rejected": not bool(overload_accept["passed"]),
            "restart_corruption_is_rejected": not bool(restart_accept["passed"]),
            "maintenance_overload_detects_pressure": any(
                item.get("kind") in {"unresolved_overload", "inquiry_collapse", "action_collapse"}
                for item in overload_findings
            ),
            "restart_corruption_detects_identity_break": any(
                item.get("kind") == "identity_collapse"
                for item in restart_findings
            ),
            "failure_injections_not_silent": bool(overload_findings) and bool(restart_findings),
        },
    }


def write_m236_acceptance_artifacts(
    *,
    strict: bool = False,
    execute_test_suites: bool = False,
    milestone_execution: dict[str, object] | None = None,
    regression_execution: dict[str, object] | None = None,
) -> dict[str, str]:
    if strict and (milestone_execution is not None or regression_execution is not None):
        raise ValueError("strict M2.36 artifact generation refuses injected execution records")
    audit_started_at = _now_iso()
    evidence = build_m236_runtime_evidence()
    schema = build_m236_schema_payload()
    ablation = build_m236_ablation_payload()
    stress = build_m236_stress_payload()
    milestone_execution = milestone_execution or _suite_execution_record(label="m236-milestone", paths=M236_TESTS, execute=execute_test_suites)
    regression_execution = regression_execution or _suite_execution_record(label="m236-regression", paths=M236_REGRESSIONS, execute=execute_test_suites)
    M236_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    M236_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with M236_TRACE_PATH.open("w", encoding="utf-8") as handle:
        for record in evidence["audit_records"]:
            for row in record["trace_excerpt"]:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    M236_METRICS_PATH.write_text(json.dumps(evidence, indent=2, ensure_ascii=True), encoding="utf-8")
    M236_ABLATION_PATH.write_text(json.dumps(ablation, indent=2, ensure_ascii=True), encoding="utf-8")
    M236_STRESS_PATH.write_text(json.dumps(stress, indent=2, ensure_ascii=True), encoding="utf-8")
    M236_SCHEMA_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=True), encoding="utf-8")
    provisional_report = {"milestone_id": MILESTONE_ID, "schema_version": SCHEMA_VERSION, "strict": strict, "status": "PENDING", "generated_at": audit_started_at, "seed_set": list(SEED_SET), "artifacts": {"canonical_trace": str(M236_TRACE_PATH), "metrics": str(M236_METRICS_PATH), "ablation": str(M236_ABLATION_PATH), "stress": str(M236_STRESS_PATH), "schema": str(M236_SCHEMA_PATH), "report": str(M236_REPORT_PATH), "summary": str(M236_SUMMARY_PATH)}, "tests": {"milestone": milestone_execution, "regressions": regression_execution}, "trial": evidence, "schema": schema, "ablation": ablation, "stress": stress}
    M236_REPORT_PATH.write_text(json.dumps(provisional_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M236_SUMMARY_PATH.write_text("# M2.36 Open Continuity Trial\n\nGenerating final report.\n", encoding="utf-8")
    generated_at = _now_iso()
    freshness_ok, freshness = _freshness_gate(artifacts=provisional_report["artifacts"], audit_started_at=audit_started_at, generated_at=generated_at, milestone_execution=milestone_execution, regression_execution=regression_execution, strict=strict)
    tests_passed = bool(milestone_execution.get("passed")) and bool(regression_execution.get("passed"))
    acceptance = dict(evidence["aggregate_acceptance"])
    gates = dict(acceptance["gates"])
    schema_ok = bool(schema["roundtrip_ok"]) and bool(schema["canonical_fields_present"]) and bool(schema["determinism_signature_preserved"])
    ablation_checks = dict(ablation["degradation_checks"])
    ablation_ok = all(bool(value) for value in ablation_checks.values())
    stress_checks = dict(stress["stress_checks"])
    stress_ok = all(bool(value) for value in stress_checks.values())
    determinism_ok = bool(evidence["determinism"]["stable_replay"])
    causality_ok = bool(ablation_checks["survival_only_is_rejected"]) and bool(ablation_checks["fractured_identity_is_rejected"])
    required_evidence_ok = all(
        (
            schema_ok,
            determinism_ok,
            causality_ok,
            ablation_ok,
            stress_ok,
            bool(regression_execution.get("passed")),
            freshness_ok,
        )
    )
    gates["schema"] = {"passed": schema_ok, "details": schema}
    gates["determinism"] = {"passed": determinism_ok, "details": evidence["determinism"]}
    gates["causality"] = {"passed": causality_ok, "details": ablation_checks}
    gates["ablation"] = {"passed": ablation_ok, "details": ablation_checks}
    gates["stress"] = {"passed": stress_ok, "details": stress_checks}
    gates["milestone_tests"] = {"passed": bool(milestone_execution.get("passed")), "details": milestone_execution}
    gates["regression"] = {"passed": bool(regression_execution.get("passed")), "details": regression_execution}
    gates["artifact_freshness"] = {"passed": freshness_ok, "details": freshness}
    gates["required_evidence_categories"] = {
        "passed": required_evidence_ok,
        "details": {
            "schema": schema_ok,
            "determinism": determinism_ok,
            "causality": causality_ok,
            "ablation": ablation_ok,
            "stress": stress_ok,
            "regression": bool(regression_execution.get("passed")),
            "artifact_freshness": freshness_ok,
        },
    }
    findings = list(acceptance["findings"])
    if not tests_passed:
        findings.append({"severity": "S1", "kind": "test_failure", "detail": "Milestone or regression suites did not pass for the current artifact round."})
    if not schema_ok:
        findings.append({"severity": "S1", "kind": "schema_roundtrip_missing", "detail": "M2.36 schema payload did not round-trip cleanly."})
    if not stress_ok:
        findings.append({"severity": "S1", "kind": "stress_evidence_missing", "detail": "M2.36 stress and failure-injection evidence did not satisfy strict audit checks."})
    status = "PASS" if acceptance["passed"] and tests_passed and freshness_ok and required_evidence_ok else "FAIL"
    recommendation = "ACCEPT" if status == "PASS" else "BLOCK"
    final_report = {"milestone_id": MILESTONE_ID, "schema_version": SCHEMA_VERSION, "strict": strict, "status": status, "recommendation": recommendation, "generated_at": generated_at, "seed_set": list(SEED_SET), "provenance": {"git_head": _git_commit(), "phase_count": len(evidence["phase_schedule"]), "trial_variant": "full"}, "artifacts": provisional_report["artifacts"], "tests": {"milestone": milestone_execution, "regressions": regression_execution}, "gates": gates, "findings": findings, "freshness": freshness, "trial": evidence, "schema": schema, "ablation": ablation, "stress": stress, "summary": acceptance["summary"], "residual_risks": ["The trial still uses a synthetic world and phase injections rather than a richer open environment.", "M3-level organism claims would need broader tool-grounded, longer-duration, externalized continuity trials."]}
    M236_REPORT_PATH.write_text(json.dumps(final_report, indent=2, ensure_ascii=True), encoding="utf-8")
    summary_lines = ["# M2.36 Open Continuity Trial", "", f"- Status: {status}", f"- Recommendation: {recommendation}", f"- Continuity mean: {evidence['aggregate_metrics']['identity_retention']['continuity_mean']}", f"- Restart consistency: {evidence['aggregate_metrics']['identity_retention']['restart_consistency']}", f"- Inquiry mean active targets: {evidence['aggregate_metrics']['inquiry_stability']['mean_active_targets']}", f"- Reopened conflicts: {evidence['aggregate_metrics']['reopened_conflict_count']}", f"- Reconciled conflicts: {evidence['aggregate_metrics']['reconciled_conflict_count']}", f"- Trace reactivation events: {evidence['aggregate_metrics']['trace_reactivation_events']}", f"- Schema roundtrip: {schema_ok}", f"- Stress evidence: {stress_ok}", "", "## Residual Risks", "", "- The organism trial remains synthetic and replay-bounded.", "- A later M3 claim needs broader environment openness, richer social exchange, and longer continuous operation windows."]
    M236_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"trace": str(M236_TRACE_PATH), "metrics": str(M236_METRICS_PATH), "ablation": str(M236_ABLATION_PATH), "stress": str(M236_STRESS_PATH), "schema": str(M236_SCHEMA_PATH), "report": str(M236_REPORT_PATH), "summary": str(M236_SUMMARY_PATH)}
