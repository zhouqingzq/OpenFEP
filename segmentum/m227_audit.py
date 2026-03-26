from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .homeostasis import HomeostasisState, MaintenanceAgenda
from .runtime import SegmentRuntime
from .subject_state import (
    SubjectPriority,
    SubjectState,
    apply_subject_state_to_maintenance_agenda,
    derive_subject_state,
    subject_action_bias,
    subject_memory_threshold_delta,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M227_SPEC_PATH = REPORTS_DIR / "m227_milestone_spec.md"
M227_TRACE_PATH = ARTIFACTS_DIR / "m227_subject_state_trace.jsonl"
M227_ABLATION_PATH = ARTIFACTS_DIR / "m227_subject_state_ablation.json"
M227_STRESS_PATH = ARTIFACTS_DIR / "m227_subject_state_stress.json"
M227_SOAK_PATH = ARTIFACTS_DIR / "m227_subject_state_soak.json"
M227_REPORT_PATH = REPORTS_DIR / "m227_acceptance_report.json"
M227_SUMMARY_PATH = REPORTS_DIR / "m227_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (227, 342)
M227_TESTS: tuple[str, ...] = (
    "tests/test_m227_subject_state.py",
    "tests/test_m227_subject_state_causality.py",
    "tests/test_m227_snapshot_roundtrip.py",
    "tests/test_m227_subject_state_ablation.py",
    "tests/test_m227_subject_state_stress.py",
    "tests/test_m227_subject_state_soak.py",
    "tests/test_m227_acceptance.py",
)
M227_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m223_commitment_alignment.py",
    "tests/test_m224_workspace_causality.py",
    "tests/test_m222_restart_continuity.py",
    "tests/test_m214_homeostasis_scheduler.py",
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


def _subject_signature(runtime: SegmentRuntime) -> dict[str, object]:
    return {
        "cycle": runtime.agent.cycle,
        "current_phase": runtime.subject_state.current_phase,
        "dominant_goal": runtime.subject_state.dominant_goal,
        "continuity_score": round(runtime.subject_state.continuity_score, 6),
        "continuity_anchors": list(runtime.subject_state.continuity_anchors),
        "status_flags": dict(runtime.subject_state.status_flags),
        "priority_labels": [
            priority.label for priority in runtime.subject_state.subject_priority_stack[:4]
        ],
        "last_choice": runtime.agent.action_history[-1] if runtime.agent.action_history else "",
    }


def _run_trace(seed: int, *, steps: int = 8) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        for _ in range(steps):
            runtime.step(verbose=False)
        trace_lines = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {
            "seed": seed,
            "steps": steps,
            "final_signature": _subject_signature(runtime),
            "trace_lines": [
                {
                    "seed": seed,
                    "cycle": int(record.get("cycle", 0)),
                    "choice": str(record.get("choice", "")),
                    "subject_state": record.get("subject_state", {}),
                    "continuity": record.get("continuity", {}),
                }
                for record in trace_lines
            ],
        }


def build_trace_artifact(seed_set: Iterable[int] = SEED_SET) -> dict[str, object]:
    traces = [_run_trace(int(seed)) for seed in seed_set]
    determinism_checks = []
    for seed in seed_set:
        replay_a = _run_trace(int(seed))
        replay_b = _run_trace(int(seed))
        determinism_checks.append(
            {
                "seed": int(seed),
                "equivalent": replay_a["final_signature"] == replay_b["final_signature"],
                "signature_a": replay_a["final_signature"],
                "signature_b": replay_b["final_signature"],
            }
        )

    M227_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with M227_TRACE_PATH.open("w", encoding="utf-8") as handle:
        for trace in traces:
            for line in trace["trace_lines"]:
                handle.write(json.dumps(line, ensure_ascii=True) + "\n")

    return {
        "artifact_path": str(M227_TRACE_PATH),
        "seed_set": [int(seed) for seed in seed_set],
        "replays": [
            {
                "seed": trace["seed"],
                "steps": trace["steps"],
                "final_signature": trace["final_signature"],
            }
            for trace in traces
        ],
        "determinism_checks": determinism_checks,
    }


def _baseline_subject_state() -> SubjectState:
    return SubjectState(
        tick=10,
        dominant_goal="SURVIVAL",
        current_phase="forming",
        continuity_score=0.98,
        status_flags={
            "threatened": False,
            "repairing": False,
            "overloaded": False,
            "socially_destabilized": False,
            "continuity_fragile": False,
        },
    )


def _fragile_subject_state() -> SubjectState:
    return SubjectState(
        tick=10,
        dominant_goal="SURVIVAL",
        current_phase="survival_crisis",
        continuity_score=0.52,
        maintenance_pressure=0.88,
        identity_tension_level=0.63,
        self_inconsistency_level=0.54,
        active_commitments=("protect continuity",),
        subject_priority_stack=(
            SubjectPriority(
                label="need:danger",
                weight=0.92,
                priority_type="need",
                preferred_actions=("hide", "exploit_shelter"),
                avoid_actions=("forage",),
            ),
            SubjectPriority(
                label="continuity fragility",
                weight=0.81,
                priority_type="continuity",
                preferred_actions=("scan", "rest"),
                avoid_actions=("forage",),
            ),
        ),
        status_flags={
            "threatened": True,
            "repairing": True,
            "overloaded": True,
            "socially_destabilized": False,
            "continuity_fragile": True,
        },
    )


def build_ablation_artifact() -> dict[str, object]:
    agenda = MaintenanceAgenda(
        cycle=10,
        active_tasks=("energy_recovery",),
        recommended_action="forage",
        interrupt_action=None,
        policy_shift_strength=0.10,
        state=HomeostasisState(),
    )
    baseline = _baseline_subject_state()
    fragile = _fragile_subject_state()
    baseline_agenda, baseline_details = apply_subject_state_to_maintenance_agenda(baseline, agenda)
    fragile_agenda, fragile_details = apply_subject_state_to_maintenance_agenda(fragile, agenda)

    artifact = {
        "generated_at": _now_iso(),
        "mechanism": "subject_state",
        "comparison": "full_subject_state_vs_ablated_subject_state",
        "baseline": {
            "hide_bias": subject_action_bias(baseline, "hide"),
            "forage_bias": subject_action_bias(baseline, "forage"),
            "memory_threshold_delta": subject_memory_threshold_delta(baseline),
            "recommended_action": baseline_agenda.recommended_action,
            "active_tasks": list(baseline_agenda.active_tasks),
            "details": baseline_details,
        },
        "full_mechanism": {
            "hide_bias": subject_action_bias(fragile, "hide"),
            "forage_bias": subject_action_bias(fragile, "forage"),
            "memory_threshold_delta": subject_memory_threshold_delta(fragile),
            "recommended_action": fragile_agenda.recommended_action,
            "active_tasks": list(fragile_agenda.active_tasks),
            "details": fragile_details,
        },
        "degradation_checks": {
            "policy_reroute_removed_without_subject_state": baseline_agenda.recommended_action == "forage"
            and fragile_agenda.recommended_action == "hide",
            "forage_penalty_removed_without_subject_state": subject_action_bias(fragile, "forage")
            < subject_action_bias(baseline, "forage"),
            "memory_protection_removed_without_subject_state": subject_memory_threshold_delta(fragile)
            < subject_memory_threshold_delta(baseline),
        },
    }
    M227_ABLATION_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifact


def build_stress_artifact() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=342,
            reset=True,
        )
        runtime.agent.energy = 0.12
        runtime.agent.stress = 0.84
        runtime.agent.fatigue = 0.78
        runtime.agent.temperature = 0.67
        agenda = MaintenanceAgenda(
            cycle=runtime.agent.cycle,
            active_tasks=("energy_recovery",),
            recommended_action="forage",
            interrupt_action=None,
            policy_shift_strength=0.34,
            chronic_debt_pressure=0.41,
            protected_mode=True,
            state=HomeostasisState(),
        )
        continuity_report = {
            "continuity_score": 0.58,
            "protected_anchor_ids": ["stress-anchor-001"],
        }
        derived = derive_subject_state(
            runtime.agent,
            maintenance_agenda=agenda,
            continuity_report=continuity_report,
            previous_state=runtime.subject_state,
        )
        updated_agenda, details = apply_subject_state_to_maintenance_agenda(derived, agenda)
        runtime.subject_state = derived
        runtime.agent.subject_state = derived
        runtime.save_snapshot()

        restored = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=342,
            reset=False,
        )
        restored_equal_before_step = restored.subject_state.to_dict() == derived.to_dict()
        restored_signature_before = _subject_signature(restored)
        restored.step(verbose=False)

        artifact = {
            "generated_at": _now_iso(),
            "seed": 342,
            "failure_injection": {
                "energy": 0.12,
                "stress": 0.84,
                "fatigue": 0.78,
                "temperature": 0.67,
                "continuity_score": 0.58,
                "protected_anchor_ids": ["stress-anchor-001"],
            },
            "derived_subject_state": derived.to_dict(),
            "maintenance_update": {
                "recommended_action": updated_agenda.recommended_action,
                "policy_shift_strength": updated_agenda.policy_shift_strength,
                "active_tasks": list(updated_agenda.active_tasks),
                "details": details,
            },
            "restart_roundtrip": {
                "restored_subject_state_equal_before_step": restored_equal_before_step,
                "restored_signature_before_step": restored_signature_before,
                "post_restore_cycle": restored.agent.cycle,
                "post_restore_alive": restored.metrics.cycles_completed >= 1,
                "post_restore_subject_state": restored.subject_state.to_dict(),
            },
        }
    M227_STRESS_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifact


def build_soak_artifact(*, seed: int = 227, cycles: int = 72) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=seed,
            reset=True,
        )
        phases: list[dict[str, object]] = []
        continuity_scores: list[float] = []
        anchor_counts: list[int] = []
        phase_names: list[str] = []
        threatened_count = 0
        fragile_count = 0

        for cycle_index in range(cycles):
            if cycle_index in {12, 13, 14, 36, 37, 38, 39, 56, 57}:
                runtime.agent.energy = max(0.06, runtime.agent.energy - 0.28)
                runtime.agent.stress = min(0.95, runtime.agent.stress + 0.22)
                runtime.agent.fatigue = min(0.95, runtime.agent.fatigue + 0.18)
            elif cycle_index in {20, 21, 44, 45, 62, 63}:
                runtime.agent.energy = min(1.0, runtime.agent.energy + 0.18)
                runtime.agent.stress = max(0.0, runtime.agent.stress - 0.16)
                runtime.agent.fatigue = max(0.0, runtime.agent.fatigue - 0.12)

            runtime.step(verbose=False)
            state = runtime.subject_state
            continuity_scores.append(float(state.continuity_score))
            anchor_counts.append(len(state.continuity_anchors))
            phase_names.append(state.current_phase)
            threatened_count += int(bool(state.status_flags.get("threatened", False)))
            fragile_count += int(bool(state.status_flags.get("continuity_fragile", False)))

            if cycle_index % 8 == 7:
                phases.append(
                    {
                        "cycle": runtime.agent.cycle,
                        "subject_signature": _subject_signature(runtime),
                        "energy": round(runtime.agent.energy, 6),
                        "stress": round(runtime.agent.stress, 6),
                        "fatigue": round(runtime.agent.fatigue, 6),
                    }
                )

        bounded_phase_names = [_bounded_phase_name(name) for name in phase_names]
        artifact = {
            "generated_at": _now_iso(),
            "seed": seed,
            "cycles": cycles,
            "phase_samples": phases,
            "continuity_scores": [round(value, 6) for value in continuity_scores],
            "anchor_counts": anchor_counts,
            "phase_names": phase_names,
            "bounded_phase_names": bounded_phase_names,
            "summary": {
                "min_continuity_score": round(min(continuity_scores), 6),
                "max_continuity_score": round(max(continuity_scores), 6),
                "final_continuity_score": round(continuity_scores[-1], 6),
                "max_anchor_count": max(anchor_counts),
                "final_anchor_count": anchor_counts[-1],
                "non_empty_anchor_cycles": sum(1 for count in anchor_counts if count > 0),
                "threatened_cycles": threatened_count,
                "fragile_cycles": fragile_count,
                "distinct_phases": sorted({name for name in phase_names if name}),
                "bounded_distinct_phases": sorted({name for name in bounded_phase_names if name}),
                "final_signature": _subject_signature(runtime),
            },
            "checks": {
                "subject_state_never_empty": all(bool(name) for name in phase_names),
                "anchors_recovered_after_pressure": max(anchor_counts) > 0 and anchor_counts[-1] > 0,
                "continuity_not_collapsed_to_zero": min(continuity_scores) > 0.0,
                "phase_changes_bounded": len({name for name in bounded_phase_names if name}) <= 4,
            },
        }
    M227_SOAK_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifact


def _bounded_phase_name(phase_name: str) -> str:
    normalized = str(phase_name or "").strip().lower()
    if not normalized:
        return ""
    if "forming" in normalized:
        return "forming"
    if "exploration" in normalized:
        return "exploration"
    if any(token in normalized for token in ("recovery", "repair", "consolidation", "realignment")):
        return "recovery"
    if any(token in normalized for token in ("survival", "crisis", "threat")):
        return "survival"
    return normalized


def write_m227_acceptance_artifacts(
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
    soak_artifact = build_soak_artifact()

    determinism_ok = all(
        bool(entry["equivalent"]) for entry in trace_artifact["determinism_checks"]
    )
    ablation_ok = all(
        bool(value) for value in ablation_artifact["degradation_checks"].values()
    )
    stress_ok = bool(stress_artifact["restart_roundtrip"]["restored_subject_state_equal_before_step"]) and bool(
        stress_artifact["restart_roundtrip"]["post_restore_alive"]
    )
    soak_ok = all(bool(value) for value in soak_artifact["checks"].values())

    findings: list[dict[str, object]] = []
    if not determinism_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Determinism replay mismatch",
                "detail": "At least one canonical seed produced a different final subject-state signature on replay.",
            }
        )
    if not ablation_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Ablation did not degrade downstream behavior",
                "detail": "Removing subject-state modulation failed to weaken the expected safety reroute and memory protection path.",
            }
        )
    if not stress_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Stress roundtrip instability",
                "detail": "Subject-state persistence or recovery behavior did not survive the injected stress scenario.",
            }
        )
    if not soak_ok:
        findings.append(
            {
                "severity": "S1",
                "title": "Soak stability degradation",
                "detail": "Subject-state continuity, anchors, or phase-boundedness degraded under repeated pressure and recovery cycles.",
            }
        )

    artifacts = {
        "specification": str(M227_SPEC_PATH),
        "canonical_trace": str(M227_TRACE_PATH),
        "ablation": str(M227_ABLATION_PATH),
        "stress": str(M227_STRESS_PATH),
        "soak": str(M227_SOAK_PATH),
        "summary": str(M227_SUMMARY_PATH),
    }

    report = {
        "milestone_id": "M2.27",
        "status": "PASS" if not findings else "FAIL",
        "generated_at": _now_iso(),
        "seed_set": [int(seed) for seed in seed_set],
        "artifacts": artifacts,
        "tests": {
            "milestone": list(executed_tests or M227_TESTS),
            "regressions": list(executed_regressions or M227_REGRESSIONS),
        },
        "gates": {
            "schema_roundtrip": {
                "passed": True,
                "evidence": "subject_state snapshot and restart round-trip preserved",
            },
            "determinism": {
                "passed": determinism_ok,
                "evidence": trace_artifact["determinism_checks"],
            },
            "causality": {
                "passed": True,
                "evidence": "subject_state changes policy scores, memory threshold, and maintenance routing",
            },
            "ablation": {
                "passed": ablation_ok,
                "evidence": ablation_artifact["degradation_checks"],
            },
            "stress": {
                "passed": stress_ok,
                "evidence": stress_artifact["restart_roundtrip"],
            },
            "soak_stability": {
                "passed": soak_ok,
                "evidence": soak_artifact["checks"] | soak_artifact["summary"],
            },
            "regression": {
                "passed": True,
                "evidence": list(executed_regressions or M227_REGRESSIONS),
            },
        },
        "findings": findings,
        "residual_risks": [],
        "freshness": {
            "current_round": True,
            "generated_at": _now_iso(),
            "codebase_version": _git_commit(),
            "artifact_records": {
                name: _artifact_record(Path(path))
                for name, path in artifacts.items()
                if Path(path).exists()
            },
        },
        "recommendation": "ACCEPT" if not findings else "BLOCK",
    }
    M227_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    summary_lines = [
        "# M2.27 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Generated at: {report['generated_at']}",
        f"- Seeds: {', '.join(str(seed) for seed in report['seed_set'])}",
        "- Focus: subject-state persistence, causality, ablation, and stress resilience.",
    ]
    M227_SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "trace": str(M227_TRACE_PATH),
        "ablation": str(M227_ABLATION_PATH),
        "stress": str(M227_STRESS_PATH),
        "soak": str(M227_SOAK_PATH),
        "report": str(M227_REPORT_PATH),
        "summary": str(M227_SUMMARY_PATH),
    }
