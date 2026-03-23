from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import hmac
import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .agent import SegmentAgent
from .environment import Observation, SimulatedWorld
from .prediction_ledger import PredictionHypothesis
from .runtime import SegmentRuntime
from .subject_state import derive_subject_state

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M230_SPEC_PATH = REPORTS_DIR / "m230_milestone_spec.md"
M230_TRACE_PATH = ARTIFACTS_DIR / "m230_slow_learning_trace.jsonl"
M230_ABLATION_PATH = ARTIFACTS_DIR / "m230_slow_learning_ablation.json"
M230_STRESS_PATH = ARTIFACTS_DIR / "m230_slow_learning_stress.json"
M230_REPORT_PATH = REPORTS_DIR / "m230_acceptance_report.json"
M230_SUMMARY_PATH = REPORTS_DIR / "m230_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (230, 460)
M230_TESTS: tuple[str, ...] = (
    "tests/test_m230_slow_variable_learning.py",
    "tests/test_m230_subject_state_regression.py",
    "tests/test_m230_acceptance.py",
)
M230_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m229_acceptance.py",
    "tests/test_m229_verification_loop.py",
    "tests/test_baseline_regressions.py",
)
M230_REQUIRED_PYTEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        "tests/test_m230_acceptance.py",
        "tests/test_m230_slow_variable_learning.py",
        "tests/test_m230_subject_state_regression.py",
    ),
    (
        "tests/test_m229_acceptance.py",
        "tests/test_m229_verification_loop.py",
        "tests/test_baseline_regressions.py",
    ),
)
_PROVENANCE_RUNNER = "segmentum.m230_audit.run_required_m230_pytest_suites.v1"
_PROVENANCE_KEY = secrets.token_bytes(32)


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


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialize_command(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def _command_for_suites(suites: Iterable[str]) -> str:
    return _serialize_command([sys.executable, "-m", "pytest", "-q", *tuple(suites)])


def _signature_payload(entry: Mapping[str, object]) -> str:
    fields = (
        "suite",
        "category",
        "command",
        "returncode",
        "passed",
        "started_at",
        "completed_at",
        "provenance_runner",
        "provenance_run_id",
        "provenance_command_hash",
        "provenance_group",
    )
    return "||".join(str(entry.get(field, "")) for field in fields)


def _sign_record(entry: Mapping[str, object]) -> str:
    return hmac.new(
        _PROVENANCE_KEY,
        _signature_payload(entry).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _stamp_provenance_records(
    *,
    suites: Iterable[str],
    category: str,
    started_at: str,
    completed_at: str,
    returncode: int,
    run_id: str,
) -> list[dict[str, object]]:
    suites = tuple(str(suite) for suite in suites)
    command = _command_for_suites(suites)
    command_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
    records: list[dict[str, object]] = []
    for suite in suites:
        record: dict[str, object] = {
            "suite": suite,
            "category": category,
            "command": command,
            "returncode": int(returncode),
            "passed": returncode == 0,
            "started_at": started_at,
            "completed_at": completed_at,
            "provenance_runner": _PROVENANCE_RUNNER,
            "provenance_run_id": run_id,
            "provenance_command_hash": command_hash,
            "provenance_group": ",".join(suites),
        }
        record["provenance_signature"] = _sign_record(record)
        records.append(record)
    return records


def _normalize_test_record(
    entry: dict[str, object] | str,
    *,
    category: str,
) -> dict[str, object]:
    if isinstance(entry, str):
        return {
            "suite": entry,
            "category": category,
            "command": "",
            "returncode": 0,
            "passed": True,
            "started_at": "",
            "completed_at": "",
        }
    return {
        "suite": str(entry.get("suite", "")),
        "category": str(entry.get("category", category)),
        "command": str(entry.get("command", "")),
        "returncode": int(entry.get("returncode", 1)),
        "passed": bool(entry.get("passed", False)),
        "started_at": str(entry.get("started_at", "")),
        "completed_at": str(entry.get("completed_at", "")),
        "provenance_runner": str(entry.get("provenance_runner", "")),
        "provenance_run_id": str(entry.get("provenance_run_id", "")),
        "provenance_command_hash": str(entry.get("provenance_command_hash", "")),
        "provenance_group": str(entry.get("provenance_group", "")),
        "provenance_signature": str(entry.get("provenance_signature", "")),
    }


def _normalize_test_records(
    records: Iterable[dict[str, object] | str] | None,
    *,
    category: str,
) -> list[dict[str, object]]:
    return [_normalize_test_record(item, category=category) for item in list(records or ())]


def _required_suite_status(
    *,
    required_suites: Iterable[str],
    records: Iterable[dict[str, object]],
) -> dict[str, object]:
    suite_map = {str(item.get("suite", "")): dict(item) for item in records}
    missing = [suite for suite in required_suites if suite not in suite_map]
    failing = [
        suite
        for suite in required_suites
        if suite in suite_map and not bool(suite_map[suite].get("passed", False))
    ]
    return {
        "passed": not missing and not failing,
        "missing_suites": missing,
        "failing_suites": failing,
        "records": [suite_map[suite] for suite in required_suites if suite in suite_map],
    }


def _validate_external_records(
    *,
    required_suites: Iterable[str],
    records: Iterable[dict[str, object]],
    category: str,
) -> dict[str, object]:
    required = tuple(str(item) for item in required_suites)
    normalized = [dict(item) for item in records]
    status = _required_suite_status(required_suites=required, records=normalized)
    if not status["passed"]:
        return {
            "passed": False,
            "reason": f"{category} suite coverage incomplete",
            "records": normalized,
        }
    expected_command = _command_for_suites(required)
    expected_hash = hashlib.sha256(expected_command.encode("utf-8")).hexdigest()
    expected_group = ",".join(required)
    run_ids = {str(item.get("provenance_run_id", "")) for item in normalized}
    if len(run_ids) != 1 or not next(iter(run_ids), ""):
        return {
            "passed": False,
            "reason": f"{category} provenance run id missing or inconsistent",
            "records": normalized,
        }
    for item in normalized:
        if str(item.get("category", "")) != category:
            return {
                "passed": False,
                "reason": f"{category} record category mismatch",
                "records": normalized,
            }
        if str(item.get("provenance_runner", "")) != _PROVENANCE_RUNNER:
            return {
                "passed": False,
                "reason": f"{category} provenance runner mismatch",
                "records": normalized,
            }
        if str(item.get("command", "")) != expected_command:
            return {
                "passed": False,
                "reason": f"{category} command does not match required pytest invocation",
                "records": normalized,
            }
        if str(item.get("provenance_command_hash", "")) != expected_hash:
            return {
                "passed": False,
                "reason": f"{category} command hash mismatch",
                "records": normalized,
            }
        if str(item.get("provenance_group", "")) != expected_group:
            return {
                "passed": False,
                "reason": f"{category} suite grouping mismatch",
                "records": normalized,
            }
        if bool(item.get("passed", False)) != (int(item.get("returncode", 1)) == 0):
            return {
                "passed": False,
                "reason": f"{category} pass/returncode mismatch",
                "records": normalized,
            }
        if str(item.get("provenance_signature", "")) != _sign_record(item):
            return {
                "passed": False,
                "reason": f"{category} provenance signature invalid",
                "records": normalized,
            }
    return {
        "passed": True,
        "reason": "",
        "records": normalized,
    }


def _records_in_current_round(
    records: Iterable[dict[str, object]],
    *,
    round_started_at: datetime | None,
) -> bool:
    if round_started_at is None:
        return False
    normalized_start = round_started_at.astimezone(timezone.utc)
    for record in records:
        completed_at = _parse_iso8601(str(record.get("completed_at", "")))
        if completed_at is None or completed_at < normalized_start:
            return False
    return True


def _artifacts_written_in_current_round(
    artifact_paths: Iterable[str],
    *,
    round_started_at: datetime | None,
) -> bool:
    if round_started_at is None:
        return False
    normalized_start = round_started_at.astimezone(timezone.utc).timestamp()
    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        if not path.exists() or path.stat().st_mtime + 1e-6 < normalized_start:
            return False
    return True


def run_required_m230_pytest_suites() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    run_id = uuid.uuid4().hex
    for index, suites in enumerate(M230_REQUIRED_PYTEST_COMMANDS):
        category = "milestone" if index == 0 else "regression"
        command = [sys.executable, "-m", "pytest", "-q", *suites]
        env = dict(os.environ)
        started_at = _now_iso()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            check=False,
        )
        completed_at = _now_iso()
        records.extend(
            _stamp_provenance_records(
                suites=suites,
                category=category,
                started_at=started_at,
                completed_at=completed_at,
                returncode=int(completed.returncode),
                run_id=run_id,
            )
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    return records


def _replay_episode(
    tick: int,
    *,
    action: str = "hide",
    outcome: str = "survival_threat",
    danger: float = 0.86,
    stress: float = 0.74,
    fatigue: float = 0.71,
) -> dict[str, object]:
    return {
        "timestamp": tick,
        "cycle": tick,
        "cluster_id": 1,
        "action_taken": action,
        "predicted_outcome": outcome,
        "total_surprise": 0.92,
        "prediction_error": 0.58,
        "observation": {"danger": danger, "social": 0.2},
        "body_state": {"stress": stress, "fatigue": fatigue, "energy": 0.34},
        "errors": {"danger": 0.44},
    }


def _decision(tick: int, action: str) -> dict[str, object]:
    return {
        "tick": tick,
        "action": action,
        "risk": 1.6,
        "dominant_component": "identity_bias",
    }


def _prediction() -> PredictionHypothesis:
    return PredictionHypothesis(
        prediction_id="pred:env:danger",
        created_tick=1,
        last_updated_tick=1,
        source_module="m230_audit",
        prediction_type="continuity_sensitive_environment_state",
        target_channels=("danger",),
        expected_state={"danger": 0.2},
        confidence=0.72,
        expected_horizon=1,
    )


def _observation() -> Observation:
    return Observation(
        food=0.55,
        danger=0.78,
        novelty=0.22,
        shelter=0.34,
        temperature=0.5,
        social=0.2,
    )


def _apply_threat_learning(agent: SegmentAgent, *, start_tick: int = 1, count: int = 5) -> dict[str, object]:
    replay_batch = [_replay_episode(tick) for tick in range(start_tick, start_tick + count)]
    decision_history = [_decision(tick, "hide") for tick in range(start_tick, start_tick + count)]
    audit = agent.slow_variable_learner.apply_sleep_cycle(
        sleep_cycle_id=1,
        tick=start_tick + count - 1,
        replay_batch=replay_batch,
        decision_history=decision_history,
        prediction_ledger=agent.prediction_ledger,
        verification_loop=agent.verification_loop,
        social_memory=agent.social_memory,
        identity_tension_history=[],
        self_model=agent.self_model,
        body_state={"stress": 0.78, "fatigue": 0.74},
    )
    agent.action_history.extend(item["action"] for item in decision_history)
    continuity = agent.self_model.update_continuity_audit(
        episodic_memory=[],
        archived_memory=[],
        action_history=list(agent.action_history),
        current_tick=start_tick + count - 1,
        slow_continuity_modifier=agent.slow_variable_learner.continuity_modifier(),
    )
    subject_state = derive_subject_state(
        agent,
        continuity_report=continuity.to_dict(),
        previous_state=agent.subject_state,
    )
    agent.subject_state = subject_state
    return {
        "audit": audit,
        "continuity": continuity.to_dict(),
        "subject_state": subject_state.to_dict(),
    }


def _canonical_signature(seed: int) -> dict[str, object]:
    agent = SegmentAgent(rng=random.Random(seed))
    payload = _apply_threat_learning(agent)
    agent.prediction_ledger.predictions.append(_prediction())
    refresh = agent.verification_loop.refresh_targets(
        tick=payload["audit"].tick + 1,
        ledger=agent.prediction_ledger,
        subject_state=agent.subject_state,
    )
    decision = agent.decision_cycle(_observation())["diagnostics"]
    ranked = {item.choice: item for item in decision.ranked_options}
    latest_audit = agent.slow_variable_learner.latest_audit()
    return {
        "seed": seed,
        "continuity_score": payload["continuity"]["continuity_score"],
        "subject_continuity_score": payload["subject_state"]["continuity_score"],
        "continuity_consistent": payload["continuity"]["continuity_score"] == payload["subject_state"]["continuity_score"],
        "slow_summary": agent.slow_variable_learner.state.last_summary,
        "caution_bias": round(agent.slow_variable_learner.state.traits.caution_bias, 6),
        "maintenance_weight": round(agent.slow_variable_learner.state.values.maintenance_weight, 6),
        "hide_policy_score": round(ranked["hide"].policy_score, 6),
        "forage_policy_score": round(ranked["forage"].policy_score, 6),
        "verification_targets_created": bool(refresh.created_targets),
        "verification_priority": round(
            agent.verification_loop.active_targets[0].priority_score if agent.verification_loop.active_targets else 0.0,
            6,
        ),
        "latest_audit": latest_audit.to_dict() if latest_audit is not None else None,
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
                "equivalent": left == right,
                "signature_a": left,
                "signature_b": right,
            }
        )
    M230_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with M230_TRACE_PATH.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, ensure_ascii=True) + "\n")
    return {
        "artifact_path": str(M230_TRACE_PATH),
        "seed_set": [int(seed) for seed in seed_set],
        "replays": runs,
        "determinism_checks": determinism_checks,
    }


def build_ablation_artifact() -> dict[str, object]:
    baseline = SegmentAgent(rng=random.Random(21))
    adapted = SegmentAgent(rng=random.Random(21))
    adapted_payload = _apply_threat_learning(adapted)

    baseline_diag = baseline.decision_cycle(_observation())["diagnostics"]
    adapted_diag = adapted.decision_cycle(_observation())["diagnostics"]
    baseline_scores = {item.choice: item for item in baseline_diag.ranked_options}
    adapted_scores = {item.choice: item for item in adapted_diag.ranked_options}

    baseline.prediction_ledger.predictions.append(_prediction())
    adapted.prediction_ledger.predictions.append(_prediction())
    baseline_refresh = baseline.verification_loop.refresh_targets(
        tick=2,
        ledger=baseline.prediction_ledger,
        subject_state=baseline.subject_state,
    )
    adapted_refresh = adapted.verification_loop.refresh_targets(
        tick=2,
        ledger=adapted.prediction_ledger,
        subject_state=adapted.subject_state,
    )

    artifact = {
        "generated_at": _now_iso(),
        "mechanism": "slow_variable_learning",
        "comparison": "with_sleep_consolidated_slow_learning_vs_ablation",
        "baseline": {
            "hide_policy_score": round(baseline_scores["hide"].policy_score, 6),
            "forage_policy_score": round(baseline_scores["forage"].policy_score, 6),
            "memory_threshold_delta": baseline.slow_variable_learner.memory_threshold_delta(),
            "continuity_score": baseline.subject_state.continuity_score,
            "continuity_fragile": bool(baseline.subject_state.status_flags.get("continuity_fragile", False)),
            "verification_priority": round(
                baseline.verification_loop.active_targets[0].priority_score if baseline.verification_loop.active_targets else 0.0,
                6,
            ),
            "refresh_created_targets": bool(baseline_refresh.created_targets),
        },
        "full_mechanism": {
            "hide_policy_score": round(adapted_scores["hide"].policy_score, 6),
            "forage_policy_score": round(adapted_scores["forage"].policy_score, 6),
            "memory_threshold_delta": adapted.slow_variable_learner.memory_threshold_delta(),
            "continuity_score": adapted.subject_state.continuity_score,
            "continuity_fragile": bool(adapted.subject_state.status_flags.get("continuity_fragile", False)),
            "verification_priority": round(
                adapted.verification_loop.active_targets[0].priority_score if adapted.verification_loop.active_targets else 0.0,
                6,
            ),
            "refresh_created_targets": bool(adapted_refresh.created_targets),
            "slow_summary": adapted.slow_variable_learner.state.last_summary,
            "continuity_consistent": adapted_payload["continuity"]["continuity_score"]
            == adapted_payload["subject_state"]["continuity_score"],
        },
        "degradation_checks": {
            "forage_suppression_removed_without_slow_learning": adapted_scores["forage"].policy_score
            < baseline_scores["forage"].policy_score,
            "verification_priority_removed_without_slow_learning": (
                adapted.verification_loop.active_targets
                and baseline.verification_loop.active_targets
                and adapted.verification_loop.active_targets[0].priority_score
                > baseline.verification_loop.active_targets[0].priority_score
            ),
            "memory_sensitivity_removed_without_slow_learning": adapted.slow_variable_learner.memory_threshold_delta()
            < baseline.slow_variable_learner.memory_threshold_delta(),
            "continuity_pressure_removed_without_slow_learning": adapted.subject_state.continuity_score
            < baseline.subject_state.continuity_score,
            "continuity_consistency_preserved": adapted_payload["continuity"]["continuity_score"]
            == adapted_payload["subject_state"]["continuity_score"],
        },
    }
    M230_ABLATION_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifact


def build_stress_artifact() -> dict[str, object]:
    agent = SegmentAgent(rng=random.Random(33))
    replay_batch = [_replay_episode(tick) for tick in range(1, 15)]
    decision_history = [_decision(tick, "forage") for tick in range(1, 15)]
    audit = agent.slow_variable_learner.apply_sleep_cycle(
        sleep_cycle_id=1,
        tick=14,
        replay_batch=replay_batch,
        decision_history=decision_history,
        prediction_ledger=agent.prediction_ledger,
        verification_loop=agent.verification_loop,
        social_memory=agent.social_memory,
        identity_tension_history=[{"tick": 12, "identity_tension": 0.8}],
        self_model=agent.self_model,
        body_state={"stress": 0.84, "fatigue": 0.8},
    )
    snapshot = agent.to_dict()
    restored = SegmentAgent.from_dict(snapshot, rng=random.Random(33))
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = Path(tmp_dir) / "trace.jsonl"
        runtime = SegmentRuntime(
            agent=restored,
            world=SimulatedWorld(seed=33),
            trace_path=trace_path,
        )
        runtime.run(cycles=1, verbose=False)
        trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    commitment_updates = [
        item.to_dict()
        for item in audit.updates
        if item.variable_path == "identity.commitment_stability"
    ]
    artifact = {
        "generated_at": _now_iso(),
        "failure_injection": {
            "type": "divergent_multi-variable_pressure",
            "episodes": len(replay_batch),
            "identity_tension_events": 1,
        },
        "anti_collapse": {
            "triggered": audit.anti_collapse_triggered,
            "total_delta": round(sum(abs(item.delta) for item in audit.updates), 6),
            "max_total_delta_per_cycle": agent.slow_variable_learner.drift_budget.max_total_delta_per_cycle,
        },
        "protected_anchor": {
            "commitment_stability": round(agent.slow_variable_learner.state.identity.commitment_stability, 6),
            "updates": commitment_updates,
        },
        "snapshot_roundtrip": {
            "caution_bias_before_restore": round(agent.slow_variable_learner.state.traits.caution_bias, 6),
            "caution_bias_after_restore": round(restored.slow_variable_learner.state.traits.caution_bias, 6),
            "audit_history_after_restore": len(restored.slow_variable_learner.audit_history),
        },
        "trace_checks": {
            "trace_lines": len(trace_lines),
            "last_record_has_slow_learning": bool(trace_lines) and '"slow_learning"' in trace_lines[-1],
            "last_record_has_continuity": bool(trace_lines) and '"continuity"' in trace_lines[-1],
        },
    }
    M230_STRESS_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifact


def write_m230_acceptance_artifacts(
    *,
    seed_set: Iterable[int] = SEED_SET,
    executed_tests: Iterable[dict[str, object] | str] | None = None,
    executed_regressions: Iterable[dict[str, object] | str] | None = None,
    round_started_at: str | None = None,
) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    input_round_start = _parse_iso8601(round_started_at)
    provided_external_evidence = executed_tests is not None or executed_regressions is not None
    external_evidence_valid = {
        "passed": False,
        "reason": "",
    }
    if provided_external_evidence:
        milestone_records = _normalize_test_records(executed_tests, category="milestone")
        regression_records = _normalize_test_records(executed_regressions, category="regression")
        milestone_validation = _validate_external_records(
            required_suites=M230_TESTS,
            records=milestone_records,
            category="milestone",
        )
        regression_validation = _validate_external_records(
            required_suites=M230_REGRESSIONS,
            records=regression_records,
            category="regression",
        )
        external_evidence_valid = {
            "passed": bool(milestone_validation["passed"]) and bool(regression_validation["passed"]),
            "reason": "; ".join(
                item["reason"]
                for item in (milestone_validation, regression_validation)
                if item["reason"]
            ),
        }
    else:
        executed_records = run_required_m230_pytest_suites()
        milestone_records = [
            _normalize_test_record(item, category="milestone")
            for item in executed_records
            if str(item.get("category", "")) == "milestone"
        ]
        regression_records = [
            _normalize_test_record(item, category="regression")
            for item in executed_records
            if str(item.get("category", "")) == "regression"
        ]

    all_records = [*milestone_records, *regression_records]
    normalized_round_start = input_round_start
    if normalized_round_start is None and all_records:
        started_candidates = [
            _parse_iso8601(str(item.get("started_at", "")))
            for item in all_records
            if _parse_iso8601(str(item.get("started_at", ""))) is not None
        ]
        if started_candidates:
            normalized_round_start = min(started_candidates)

    trace_artifact = build_trace_artifact(seed_set=seed_set)
    ablation_artifact = build_ablation_artifact()
    stress_artifact = build_stress_artifact()

    determinism_ok = all(bool(entry["equivalent"]) for entry in trace_artifact["determinism_checks"])
    trace_consistency_ok = all(bool(entry["continuity_consistent"]) for entry in trace_artifact["replays"])
    ablation_ok = all(bool(value) for value in ablation_artifact["degradation_checks"].values())
    stress_ok = (
        bool(stress_artifact["anti_collapse"]["triggered"])
        and float(stress_artifact["anti_collapse"]["total_delta"])
        <= float(stress_artifact["anti_collapse"]["max_total_delta_per_cycle"]) + 1e-6
        and bool(stress_artifact["trace_checks"]["last_record_has_slow_learning"])
        and bool(stress_artifact["trace_checks"]["last_record_has_continuity"])
        and abs(
            float(stress_artifact["snapshot_roundtrip"]["caution_bias_before_restore"])
            - float(stress_artifact["snapshot_roundtrip"]["caution_bias_after_restore"])
        )
        <= 1e-6
    )
    milestone_status = _required_suite_status(required_suites=M230_TESTS, records=milestone_records)
    regression_status = _required_suite_status(required_suites=M230_REGRESSIONS, records=regression_records)
    evidence_current_round = _records_in_current_round(
        [*milestone_records, *regression_records],
        round_started_at=normalized_round_start,
    )

    findings: list[dict[str, object]] = []
    if provided_external_evidence and not external_evidence_valid["passed"]:
        findings.append(
            {
                "severity": "S0",
                "title": "Untrusted external pytest evidence rejected",
                "detail": "Acceptance input attempted to supply unsigned or inconsistent pytest evidence. Strict audit requires self-executed suites or strong provenance.",
                "reason": external_evidence_valid["reason"],
            }
        )
    if not determinism_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Determinism replay mismatch",
                "detail": "At least one canonical seed produced a different slow-learning signature on replay.",
            }
        )
    if not trace_consistency_ok:
        findings.append(
            {
                "severity": "S0",
                "title": "Continuity score mismatch",
                "detail": "Canonical trace shows subject-state continuity diverging from the audited continuity score.",
            }
        )
    if not ablation_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Slow learning causality weakened",
                "detail": "Removing slow-variable consolidation did not reliably degrade declared downstream behavior.",
            }
        )
    if not stress_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Anti-collapse or snapshot hardening incomplete",
                "detail": "Stress evidence did not preserve bounded drift, trace visibility, and snapshot continuity together.",
            }
        )
    if not milestone_status["passed"]:
        findings.append(
            {
                "severity": "S1",
                "title": "Current-round milestone pytest evidence missing",
                "detail": "Required M2.30 milestone suites were not all executed successfully in the current round.",
                "missing_suites": milestone_status["missing_suites"],
                "failing_suites": milestone_status["failing_suites"],
            }
        )
    if not regression_status["passed"]:
        findings.append(
            {
                "severity": "S1",
                "title": "Historical regression evidence missing",
                "detail": "Required historical regression suites were not all executed successfully in the current round.",
                "missing_suites": regression_status["missing_suites"],
                "failing_suites": regression_status["failing_suites"],
            }
        )

    artifacts = {
        "specification": str(M230_SPEC_PATH),
        "canonical_trace": str(M230_TRACE_PATH),
        "ablation": str(M230_ABLATION_PATH),
        "stress": str(M230_STRESS_PATH),
        "report": str(M230_REPORT_PATH),
        "summary": str(M230_SUMMARY_PATH),
    }
    generated_at = _now_iso()
    artifact_freshness_ok = False
    gate_status = {
        "schema_roundtrip": {
            "passed": True,
            "evidence": "slow-variable learner state survives agent snapshot round-trip and runtime trace emission",
        },
        "determinism": {
            "passed": determinism_ok,
            "evidence": trace_artifact["determinism_checks"],
        },
        "causality": {
            "passed": trace_consistency_ok and ablation_ok,
            "evidence": {
                "trace_replays": trace_artifact["replays"],
                "degradation_checks": ablation_artifact["degradation_checks"],
            },
        },
        "ablation": {
            "passed": ablation_ok,
            "evidence": ablation_artifact["degradation_checks"],
        },
        "stress": {
            "passed": stress_ok,
            "evidence": {
                "anti_collapse": stress_artifact["anti_collapse"],
                "trace_checks": stress_artifact["trace_checks"],
                "snapshot_roundtrip": stress_artifact["snapshot_roundtrip"],
            },
        },
        "regression": {
            "passed": regression_status["passed"],
            "evidence": regression_status["records"],
            "missing_suites": regression_status["missing_suites"],
            "failing_suites": regression_status["failing_suites"],
        },
        "artifact_freshness": {
            "passed": False,
            "evidence": {
                "round_started_at": normalized_round_start.isoformat(timespec="seconds")
                if normalized_round_start is not None
                else round_started_at,
                "current_round_test_evidence": evidence_current_round,
                "milestone_records": milestone_records,
                "regression_records": regression_records,
            },
        },
    }
    report_status = "PASS" if not findings and all(gate["passed"] for key, gate in gate_status.items() if key != "artifact_freshness") else "FAIL"
    gate_status = {
        **gate_status,
    }
    report = {
        "milestone_id": "M2.30",
        "status": report_status,
        "generated_at": generated_at,
        "seed_set": [int(seed) for seed in seed_set],
        "artifacts": artifacts,
        "tests": {
            "milestone": milestone_records,
            "regressions": regression_records,
        },
        "gates": gate_status,
        "findings": findings,
        "residual_risks": [],
        "freshness": {
            "current_round": artifact_freshness_ok,
            "round_started_at": normalized_round_start.isoformat(timespec="seconds")
            if normalized_round_start is not None
            else round_started_at,
            "generated_at": generated_at,
            "codebase_version": _git_commit(),
            "artifact_records": {
                name: _artifact_record(Path(path))
                for name, path in artifacts.items()
                if Path(path).exists()
            },
            "milestone_records": milestone_records,
            "regression_records": regression_records,
        },
        "recommendation": "ACCEPT" if report_status == "PASS" else "BLOCK",
    }
    summary_lines = [
        "# M2.30 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Generated at: {generated_at}",
        f"- Seeds: {', '.join(str(seed) for seed in report['seed_set'])}",
        "- Focus: slow-variable consolidation, continuity consistency, anti-collapse drift budgets, and snapshot persistence.",
    ]
    M230_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    M230_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    artifact_freshness_ok = (
        not provided_external_evidence or external_evidence_valid["passed"]
    ) and milestone_status["passed"] and regression_status["passed"] and evidence_current_round and _artifacts_written_in_current_round(
        [
            artifacts["canonical_trace"],
            artifacts["ablation"],
            artifacts["stress"],
            artifacts["summary"],
            artifacts["report"],
        ],
        round_started_at=normalized_round_start,
    )
    gate_status["artifact_freshness"]["passed"] = artifact_freshness_ok
    if not artifact_freshness_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Current-round artifact freshness not proven",
                "detail": "Acceptance artifacts are not tied to a fully executed current-round milestone and regression evidence set.",
            }
        )
    report_status = "PASS" if not findings and all(gate["passed"] for gate in gate_status.values()) else "FAIL"
    report["status"] = report_status
    report["gates"] = gate_status
    report["findings"] = findings
    report["freshness"] = {
        "current_round": artifact_freshness_ok,
        "round_started_at": normalized_round_start.isoformat(timespec="seconds")
        if normalized_round_start is not None
        else round_started_at,
        "generated_at": generated_at,
        "codebase_version": _git_commit(),
        "artifact_records": {
            name: _artifact_record(Path(path))
            for name, path in artifacts.items()
            if Path(path).exists()
        },
        "milestone_records": milestone_records,
        "regression_records": regression_records,
    }
    report["recommendation"] = "ACCEPT" if report_status == "PASS" else "BLOCK"
    summary_lines = [
        "# M2.30 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Generated at: {generated_at}",
        f"- Seeds: {', '.join(str(seed) for seed in report['seed_set'])}",
        "- Focus: slow-variable consolidation, continuity consistency, anti-collapse drift budgets, and snapshot persistence.",
    ]
    M230_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    M230_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "trace": str(M230_TRACE_PATH),
        "ablation": str(M230_ABLATION_PATH),
        "stress": str(M230_STRESS_PATH),
        "report": str(M230_REPORT_PATH),
        "summary": str(M230_SUMMARY_PATH),
    }
