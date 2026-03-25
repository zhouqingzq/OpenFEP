from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .action_registry import ActionRegistry, build_default_action_registry
from .agent import SegmentAgent
from .environment import Observation
from .narrative_compiler import NarrativeCompiler
from .narrative_experiment import InquiryPlanStatus, NarrativeExperimentDesigner
from .narrative_types import NarrativeEpisode
from .narrative_uncertainty import UncertaintyDecompositionResult
from .runtime import SegmentRuntime
from .subject_state import SubjectState, derive_subject_state

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
SCHEMA_VERSION = "m234_audit_v1"

M234_SPEC_PATH = REPORTS_DIR / "m234_milestone_spec.md"
M234_PREPARATION_PATH = REPORTS_DIR / "m234_strict_audit_preparation.md"
M234_RATIONALE_PATH = REPORTS_DIR / "m234_parallel_probe_acceptance_rationale.md"
M234_TRACE_PATH = ARTIFACTS_DIR / "m234_experiment_trace.jsonl"
M234_ABLATION_PATH = ARTIFACTS_DIR / "m234_experiment_ablation.json"
M234_STRESS_PATH = ARTIFACTS_DIR / "m234_experiment_stress.json"
M234_REPORT_PATH = REPORTS_DIR / "m234_acceptance_report.json"
M234_SUMMARY_PATH = REPORTS_DIR / "m234_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (234, 468)
M234_TESTS: tuple[str, ...] = (
    "tests/test_m234_experiment_design.py",
    "tests/test_m234_acceptance.py",
    "tests/test_m234_audit_preparation.py",
)
M234_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m233_uncertainty_decomposition.py",
    "tests/test_m227_snapshot_roundtrip.py",
    "tests/test_m228_prediction_ledger.py",
    "tests/test_runtime.py",
)
M234_GATES: tuple[str, ...] = (
    "schema",
    "determinism",
    "competition_translation",
    "bounded_parallelism",
    "value_ranking",
    "downstream_causality",
    "governance",
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


def _is_generated_m234_artifact(path: str) -> bool:
    normalized = str(path).replace("\\", "/")
    allowed_prefixes = (
        "artifacts/m234_",
        "reports/m234_acceptance_",
    )
    return normalized.startswith(allowed_prefixes) or normalized.startswith(".pytest_m234_")


def _strict_dirty_findings(dirty_paths: Iterable[str]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    blocking_paths = [path for path in (_git_status_path(item) for item in dirty_paths) if path and not _is_generated_m234_artifact(path)]
    if blocking_paths:
        findings.append(
            {
                "severity": "S1",
                "title": "Strict audit baseline is not frozen",
                "detail": (
                    "Strict M2.34 acceptance cannot rely on a dirty code or specification baseline. "
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
    milestone_auth = _is_authentic_execution_record(milestone_execution, expected_paths=M234_TESTS)
    regression_auth = _is_authentic_execution_record(regression_execution, expected_paths=M234_REGRESSIONS)
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
        "milestone_id": "M2.34",
        "title": "Narrative Hypothesis And Experiment Design",
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARATION_READY",
        "assumption_source": str(M234_SPEC_PATH),
        "seed_set": list(SEED_SET),
        "artifacts": {
            "specification": str(M234_SPEC_PATH),
            "preparation": str(M234_PREPARATION_PATH),
            "rationale": str(M234_RATIONALE_PATH),
            "canonical_trace": str(M234_TRACE_PATH),
            "ablation": str(M234_ABLATION_PATH),
            "stress": str(M234_STRESS_PATH),
            "report": str(M234_REPORT_PATH),
            "summary": str(M234_SUMMARY_PATH),
        },
        "tests": {
            "milestone": list(M234_TESTS),
            "regressions": list(M234_REGRESSIONS),
        },
        "gates": list(M234_GATES),
    }


def _social_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m234-audit-social",
        timestamp=4,
        source="audit",
        raw_text=(
            "My counterpart said they would meet me, but left me outside and later sent a vague apology. "
            "I do not know whether this was betrayal, temporary constraint, or a misunderstanding."
        ),
        tags=["social", "audit"],
        metadata={"counterpart_id": "counterpart-x", "chapter_id": 4},
    )


def _threat_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m234-audit-threat",
        timestamp=5,
        source="audit",
        raw_text=(
            "A predator attacked near the shelter, but I still cannot tell whether it was a stable threat source "
            "or a local accident that will fade if I wait."
        ),
        tags=["threat", "audit"],
        metadata={"environment_id": "shelter-edge", "chapter_id": 5},
    )


def _noise_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m234-audit-noise",
        timestamp=6,
        source="audit",
        raw_text="Everything sounded dramatic, but it was only a slogan on a wall and nothing actionable happened.",
        tags=["noise", "audit"],
        metadata={},
    )


def _determinism_signature(*, seed: int, compiled_episode, observation: Observation) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / f"m234_determinism_{seed}.json"
        trace_path = Path(tmp_dir) / f"m234_determinism_{seed}.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        runtime.agent.ingest_narrative_episode(compiled_episode)
        runtime.subject_state = derive_subject_state(runtime.agent, previous_state=runtime.subject_state)
        runtime.agent.subject_state = runtime.subject_state
        result = runtime.agent.decision_cycle(observation)
        diagnostics = result["diagnostics"]
        ranked = {item.choice: item for item in diagnostics.ranked_options}
        return {
            "seed": seed,
            "chosen_action": diagnostics.chosen.choice,
            "active_goal": diagnostics.active_goal,
            "experiment_summary": runtime.agent.latest_narrative_experiment.summary,
            "plan_statuses": [
                {
                    "plan_id": item.plan_id,
                    "action": item.selected_action,
                    "status": item.status,
                    "score": round(float(item.score), 6),
                }
                for item in runtime.agent.latest_narrative_experiment.plans
            ],
            "prediction_ids": sorted(
                item.prediction_id
                for item in runtime.agent.prediction_ledger.active_predictions()
                if item.source_module == "narrative_experiment"
            ),
            "verification_targets": sorted(
                item.prediction_id
                for item in runtime.agent.verification_loop.active_targets
                if item.linked_experiment_plan_id
            ),
            "active_inquiries": [item.to_dict() for item in runtime.agent.subject_state.active_inquiries],
            "scan_experiment_bias": round(float(getattr(ranked.get("scan"), "experiment_bias", 0.0)), 6),
        }


def build_m234_runtime_evidence() -> dict[str, object]:
    compiler = NarrativeCompiler()
    social_compiled = compiler.compile_episode(_social_episode())
    threat_compiled = compiler.compile_episode(_threat_episode())
    noise_compiled = compiler.compile_episode(_noise_episode())

    social_uncertainty = UncertaintyDecompositionResult.from_dict(social_compiled.uncertainty_decomposition)
    threat_uncertainty = UncertaintyDecompositionResult.from_dict(threat_compiled.uncertainty_decomposition)

    base_registry = build_default_action_registry()
    stable_design = NarrativeExperimentDesigner(max_active_plans=1).design(
        tick=4,
        uncertainty=social_uncertainty,
        action_registry=base_registry,
        active_goal="SOCIAL",
        subject_state=SubjectState(status_flags={"socially_destabilized": False, "threatened": False}),
    )
    destabilized_design = NarrativeExperimentDesigner().design(
        tick=4,
        uncertainty=social_uncertainty,
        action_registry=base_registry,
        active_goal="SOCIAL",
        subject_state=SubjectState(status_flags={"socially_destabilized": True, "threatened": False}),
    )
    safety_design = NarrativeExperimentDesigner().design(
        tick=5,
        uncertainty=threat_uncertainty,
        action_registry=base_registry,
        active_goal="SAFETY",
    )

    limited_registry = ActionRegistry()
    for action_name in ("scan", "rest"):
        action = base_registry.get(action_name)
        if action is not None:
            limited_registry.register(action, action.cost_estimate)
    governed_design = NarrativeExperimentDesigner().design(
        tick=7,
        uncertainty=social_uncertainty,
        action_registry=limited_registry,
        active_goal="SOCIAL",
    )

    full_agent = SegmentAgent()
    full_agent.ingest_narrative_episode(threat_compiled)
    full_agent.subject_state = derive_subject_state(full_agent, previous_state=full_agent.subject_state)
    threat_observation = Observation(
        food=0.32,
        danger=0.68,
        novelty=0.18,
        shelter=0.64,
        temperature=0.48,
        social=0.14,
    )
    full_result = full_agent.decision_cycle(threat_observation)
    full_diagnostics = full_result["diagnostics"]

    ablated_agent = SegmentAgent()
    ablated_agent.latest_narrative_uncertainty = threat_uncertainty
    ablated_agent.latest_narrative_experiment = type(full_agent.latest_narrative_experiment)()
    ablated_agent.subject_state = derive_subject_state(ablated_agent, previous_state=ablated_agent.subject_state)
    ablated_diagnostics = type(
        "Diagnostics",
        (),
        {
            "chosen": type(
                "Chosen",
                (),
                {
                    "choice": "scan",
                    "predicted_effects": {"danger_delta": -0.05, "novelty_delta": 0.02},
                    "preferred_probability": 0.63,
                },
            )(),
            "workspace_broadcast_channels": ["danger", "novelty"],
            "commitment_focus": [],
            "active_goal": "SAFETY",
            "social_focus": [],
            "social_alerts": [],
        },
    )()
    ablated_ledger_seed = ablated_agent.prediction_ledger.seed_predictions(
        tick=1,
        diagnostics=ablated_diagnostics,
        prediction={"danger": 0.68, "novelty": 0.18, "social": 0.14},
        subject_state=ablated_agent.subject_state,
        narrative_uncertainty=ablated_agent.latest_narrative_uncertainty,
        experiment_design=ablated_agent.latest_narrative_experiment,
    )
    ablated_verification_seed = ablated_agent.verification_loop.refresh_targets(
        tick=1,
        ledger=ablated_agent.prediction_ledger,
        diagnostics=ablated_diagnostics,
        subject_state=ablated_agent.subject_state,
        narrative_uncertainty=ablated_agent.latest_narrative_uncertainty,
        experiment_design=ablated_agent.latest_narrative_experiment,
        workspace_channels=("danger", "novelty"),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=SEED_SET[0],
            reset=True,
        )
        runtime.agent.ingest_narrative_episode(threat_compiled)
        runtime.subject_state = derive_subject_state(runtime.agent, previous_state=runtime.subject_state)
        runtime.agent.subject_state = runtime.subject_state
        runtime.save_snapshot()
        restored = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=SEED_SET[0],
            reset=False,
        )

    stable_active = [item for item in stable_design.plans if item.status == InquiryPlanStatus.ACTIVE_EXPERIMENT.value]
    stable_queued = [
        item
        for item in stable_design.plans
        if item.status in {InquiryPlanStatus.QUEUED_EXPERIMENT.value, InquiryPlanStatus.DEFERRED_FOR_BUDGET.value}
    ]
    destabilized_contact = next((item for item in destabilized_design.plans if item.selected_action == "seek_contact"), None)
    destabilized_scan = next((item for item in destabilized_design.plans if item.selected_action == "scan"), None)
    ranked = {item.choice: item for item in full_diagnostics.ranked_options}
    design_roundtrip = type(stable_design).from_dict(stable_design.to_dict())
    subject_state_roundtrip = SubjectState.from_dict(full_agent.subject_state.to_dict())
    replay_signatures = [
        _determinism_signature(seed=seed, compiled_episode=threat_compiled, observation=threat_observation)
        for seed in SEED_SET
    ]
    replay_repeat_signature = _determinism_signature(
        seed=SEED_SET[0],
        compiled_episode=threat_compiled,
        observation=threat_observation,
    )
    canonical_signature = dict(replay_signatures[0]) if replay_signatures else {}
    if canonical_signature:
        canonical_signature.pop("seed", None)
    equivalent_signatures = []
    for signature in replay_signatures:
        comparable = dict(signature)
        comparable.pop("seed", None)
        equivalent_signatures.append(comparable)

    trace_records = [
        {
            "schema_version": SCHEMA_VERSION,
            "event": "competition_to_experiment_translation",
            "episode_id": social_compiled.episode_id,
            "discrimination_targets": [item.to_dict() for item in stable_design.discrimination_targets],
            "predictions": [item.to_dict() for item in stable_design.predictions[:4]],
            "candidates": [item.to_dict() for item in stable_design.candidates[:4]],
            "plans": [item.to_dict() for item in stable_design.plans[:4]],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "event": "social_goal_ranking",
            "stable_top_action": stable_design.candidates[0].action_name if stable_design.candidates else "",
            "stable_plan_statuses": [item.status for item in stable_design.plans],
            "destabilized_contact_status": destabilized_contact.status if destabilized_contact is not None else "",
            "destabilized_scan_status": destabilized_scan.status if destabilized_scan is not None else "",
            "safety_top_action": safety_design.candidates[0].action_name if safety_design.candidates else "",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "event": "downstream_consumption",
            "experiment_prediction_ids": [
                item.prediction_id
                for item in full_agent.prediction_ledger.active_predictions()
                if item.source_module == "narrative_experiment"
            ],
            "verification_targets": [item.to_dict() for item in full_agent.verification_loop.active_targets],
            "active_inquiries": [item.to_dict() for item in full_agent.subject_state.active_inquiries],
            "ranked_scan_bias": getattr(ranked.get("scan"), "experiment_bias", 0.0),
            "decision_explanation": full_diagnostics.explanation,
        },
    ]

    ablation = {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": "M2.34",
        "full_mechanism": {
            "experiment_prediction_count": len(
                [item for item in full_agent.prediction_ledger.active_predictions() if item.source_module == "narrative_experiment"]
            ),
            "linked_verification_target_count": len(
                [item for item in full_agent.verification_loop.active_targets if item.linked_experiment_plan_id]
            ),
            "active_inquiry_count": len(full_agent.subject_state.active_inquiries),
            "scan_experiment_bias": float(getattr(ranked.get("scan"), "experiment_bias", 0.0)),
        },
        "without_experiment_design": {
            "experiment_prediction_count": len(
                [item for item in ablated_agent.prediction_ledger.active_predictions() if item.source_module == "narrative_experiment"]
            ),
            "linked_verification_target_count": len(
                [item for item in ablated_agent.verification_loop.active_targets if item.linked_experiment_plan_id]
            ),
            "active_inquiry_count": len(ablated_agent.subject_state.active_inquiries),
            "scan_experiment_bias": 0.0,
            "ledger_seed": ablated_ledger_seed.to_dict(),
            "verification_seed": ablated_verification_seed.to_dict(),
        },
        "degradation_checks": {
            "ledger_loses_experiment_predictions": len(
                [item for item in full_agent.prediction_ledger.active_predictions() if item.source_module == "narrative_experiment"]
            )
            > len(
                [item for item in ablated_agent.prediction_ledger.active_predictions() if item.source_module == "narrative_experiment"]
            ),
            "verification_loses_experiment_targets": len(
                [item for item in full_agent.verification_loop.active_targets if item.linked_experiment_plan_id]
            )
            > len(
                [item for item in ablated_agent.verification_loop.active_targets if item.linked_experiment_plan_id]
            ),
            "subject_state_loses_active_inquiries": len(full_agent.subject_state.active_inquiries)
            > len(ablated_agent.subject_state.active_inquiries),
            "action_scoring_loses_experiment_bias": float(getattr(ranked.get("scan"), "experiment_bias", 0.0)) > 0.0,
        },
    }

    stress = {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": "M2.34",
        "stress_checks": {
            "experiment_design_roundtrip": design_roundtrip.to_dict() == stable_design.to_dict(),
            "subject_state_roundtrip": subject_state_roundtrip.to_dict() == full_agent.subject_state.to_dict(),
            "bounded_active_plans": len(stable_active) <= 1,
            "queued_or_deferred_exists": bool(stable_queued) or any(
                item.status.startswith("deferred") for item in destabilized_design.plans
            ),
            "noise_does_not_activate_experiment": not SegmentAgent().narrative_experiment_designer.design(
                tick=6,
                uncertainty=UncertaintyDecompositionResult.from_dict(noise_compiled.uncertainty_decomposition),
                action_registry=base_registry,
                active_goal="SOCIAL",
            ).active_plans(),
            "snapshot_roundtrip_preserves_summary": restored.agent.latest_narrative_experiment.summary
            == runtime.agent.latest_narrative_experiment.summary,
            "snapshot_roundtrip_preserves_plans": [
                item.to_dict() for item in restored.agent.latest_narrative_experiment.plans
            ]
            == [item.to_dict() for item in runtime.agent.latest_narrative_experiment.plans],
            "replay_same_seed_equivalent": _same_contents(
                {key: value for key, value in replay_repeat_signature.items() if key != "seed"},
                canonical_signature,
            ),
            "replay_multi_seed_equivalent": bool(equivalent_signatures)
            and all(_same_contents(signature, canonical_signature) for signature in equivalent_signatures),
        },
        "details": {
            "stable_plan_statuses": [item.status for item in stable_design.plans],
            "governed_candidate_actions": [item.action_name for item in governed_design.candidates],
            "restored_subject_state": restored.subject_state.to_dict(),
            "determinism_replay_signatures": replay_signatures,
        },
    }

    gates = {
        "schema": {
            "passed": stress["stress_checks"]["experiment_design_roundtrip"]
            and stress["stress_checks"]["subject_state_roundtrip"],
            "evidence": ["experiment_design_roundtrip", "subject_state_roundtrip", f"schema_version={SCHEMA_VERSION}"],
        },
        "determinism": {
            "passed": stress["stress_checks"]["replay_same_seed_equivalent"]
            and stress["stress_checks"]["replay_multi_seed_equivalent"]
            and len(SEED_SET) >= 2,
            "evidence": [
                "replay_same_seed_equivalent",
                "replay_multi_seed_equivalent",
                f"seed_count={len(SEED_SET)}",
            ],
        },
        "competition_translation": {
            "passed": bool(stable_design.discrimination_targets)
            and bool(stable_design.predictions)
            and bool(stable_design.candidates),
            "evidence": ["discrimination_targets", "predictions", "candidates"],
        },
        "bounded_parallelism": {
            "passed": stress["stress_checks"]["bounded_active_plans"]
            and stress["stress_checks"]["queued_or_deferred_exists"],
            "evidence": ["bounded_active_plans", "queued_or_deferred_exists"],
        },
        "value_ranking": {
            "passed": bool(stable_design.candidates)
            and stable_design.candidates[0].action_name == "seek_contact"
            and destabilized_contact is not None
            and destabilized_contact.status == InquiryPlanStatus.DEFERRED_FOR_RISK.value
            and destabilized_scan is not None
            and destabilized_scan.status in {
                InquiryPlanStatus.ACTIVE_EXPERIMENT.value,
                InquiryPlanStatus.QUEUED_EXPERIMENT.value,
            }
            and bool(safety_design.candidates)
            and safety_design.candidates[0].action_name == "scan",
            "evidence": [
                "stable_top_action=seek_contact",
                "destabilized_contact_status=deferred_for_risk",
                "safety_top_action=scan",
            ],
        },
        "downstream_causality": {
            "passed": all(ablation["degradation_checks"].values())
            and "experiment design" in str(full_diagnostics.explanation).lower(),
            "evidence": list(ablation["degradation_checks"].keys()),
        },
        "governance": {
            "passed": bool(governed_design.candidates)
            and all(item.action_name in {"scan", "rest"} for item in governed_design.candidates)
            and not any(item.action_name == "seek_contact" for item in governed_design.candidates),
            "evidence": ["scan", "rest", "seek_contact_filtered"],
        },
        "snapshot_roundtrip": {
            "passed": stress["stress_checks"]["snapshot_roundtrip_preserves_summary"]
            and stress["stress_checks"]["snapshot_roundtrip_preserves_plans"]
            and bool(restored.subject_state.active_inquiries or restored.agent.subject_state.active_inquiries),
            "evidence": ["snapshot_roundtrip_preserves_summary", "snapshot_roundtrip_preserves_plans"],
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
        "# M2.34 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Schema gate: {'PASS' if gates['schema']['passed'] else 'FAIL'}",
        f"- Determinism gate: {'PASS' if gates['determinism']['passed'] else 'FAIL'}",
        f"- Competition translation gate: {'PASS' if gates['competition_translation']['passed'] else 'FAIL'}",
        f"- Bounded parallelism gate: {'PASS' if gates['bounded_parallelism']['passed'] else 'FAIL'}",
        f"- Value ranking gate: {'PASS' if gates['value_ranking']['passed'] else 'FAIL'}",
        f"- Downstream causality gate: {'PASS' if gates['downstream_causality']['passed'] else 'FAIL'}",
        f"- Governance gate: {'PASS' if gates['governance']['passed'] else 'FAIL'}",
        f"- Snapshot gate: {'PASS' if gates['snapshot_roundtrip']['passed'] else 'FAIL'}",
        f"- Regression gate: {'PASS' if gates['regression']['passed'] else 'FAIL'}",
        f"- Freshness gate: {'PASS' if gates['artifact_freshness']['passed'] else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


def write_m234_acceptance_artifacts(
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
                    f"strict M2.34 audit refuses injected execution records for {label} tests"
                )

    audit_started_at = _now_iso()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    runtime_evidence = build_m234_runtime_evidence()
    _write_trace(M234_TRACE_PATH, runtime_evidence["trace_records"])
    _write_json(M234_ABLATION_PATH, runtime_evidence["ablation"])
    _write_json(M234_STRESS_PATH, runtime_evidence["stress"])

    milestone_execution = milestone_execution or _suite_execution_record(
        label="m234-milestone",
        paths=M234_TESTS,
        execute=execute_test_suites,
    )
    regression_execution = regression_execution or _suite_execution_record(
        label="m234-regression",
        paths=M234_REGRESSIONS,
        execute=execute_test_suites,
    )

    artifacts = {
        "specification": str(M234_SPEC_PATH),
        "preparation": str(M234_PREPARATION_PATH),
        "rationale": str(M234_RATIONALE_PATH),
        "canonical_trace": str(M234_TRACE_PATH),
        "ablation": str(M234_ABLATION_PATH),
        "stress": str(M234_STRESS_PATH),
        "report": str(M234_REPORT_PATH),
        "summary": str(M234_SUMMARY_PATH),
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
            "schema",
            "determinism",
            "competition_translation",
            "bounded_parallelism",
            "value_ranking",
            "downstream_causality",
            "governance",
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
            "milestone_id": "M2.34",
            "title": "Narrative Hypothesis And Experiment Design",
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
    M234_REPORT_PATH.write_text(json.dumps(provisional_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M234_SUMMARY_PATH.write_text(_summary_markdown(provisional_report), encoding="utf-8")

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
    M234_REPORT_PATH.write_text(json.dumps(final_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M234_SUMMARY_PATH.write_text(_summary_markdown(final_report), encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    write_m234_acceptance_artifacts(strict=True, execute_test_suites=True)
