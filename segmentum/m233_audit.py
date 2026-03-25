from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .agent import SegmentAgent
from .narrative_compiler import NarrativeCompiler
from .narrative_types import NarrativeEpisode
from .prediction_ledger import PredictionLedger
from .runtime import SegmentRuntime
from .subject_state import SubjectState, derive_subject_state
from .verification import VerificationLoop

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M233_SPEC_PATH = REPORTS_DIR / "m233_milestone_spec.md"
M233_PREPARATION_PATH = REPORTS_DIR / "m233_strict_audit_preparation.md"
M233_TRACE_PATH = ARTIFACTS_DIR / "m233_uncertainty_trace.jsonl"
M233_ABLATION_PATH = ARTIFACTS_DIR / "m233_uncertainty_ablation.json"
M233_STRESS_PATH = ARTIFACTS_DIR / "m233_uncertainty_stress.json"
M233_REPORT_PATH = REPORTS_DIR / "m233_acceptance_report.json"
M233_SUMMARY_PATH = REPORTS_DIR / "m233_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (233, 466)
M233_TESTS: tuple[str, ...] = (
    "tests/test_m233_uncertainty_decomposition.py",
    "tests/test_m233_narrative_robustness.py",
    "tests/test_m233_acceptance.py",
    "tests/test_m233_audit_preparation.py",
)
M233_REGRESSIONS: tuple[str, ...] = (
    "tests/test_narrative_compiler.py",
    "tests/test_m220_narrative_initialization.py",
    "tests/test_m227_snapshot_roundtrip.py",
    "tests/test_m228_prediction_ledger.py",
    "tests/test_runtime.py",
)
M233_GATES: tuple[str, ...] = (
    "schema",
    "competition",
    "downstream_causality",
    "surface_latent_separation",
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
    milestone_auth = _is_authentic_execution_record(milestone_execution, expected_paths=M233_TESTS)
    regression_auth = _is_authentic_execution_record(regression_execution, expected_paths=M233_REGRESSIONS)
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
    freshness_ok = evidence_ok and report_ok and (not strict or (milestone_auth and regression_auth and suite_ok))
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
        "current_round": freshness_ok,
        "git": {
            "head": _git_commit(),
            "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
        },
    }


def preparation_manifest() -> dict[str, object]:
    return {
        "milestone_id": "M2.33",
        "title": "Narrative Uncertainty Decomposition",
        "status": "PREPARATION_READY",
        "assumption_source": str(M233_SPEC_PATH),
        "seed_set": list(SEED_SET),
        "artifacts": {
            "specification": str(M233_SPEC_PATH),
            "preparation": str(M233_PREPARATION_PATH),
            "canonical_trace": str(M233_TRACE_PATH),
            "ablation": str(M233_ABLATION_PATH),
            "stress": str(M233_STRESS_PATH),
            "report": str(M233_REPORT_PATH),
            "summary": str(M233_SUMMARY_PATH),
        },
        "tests": {
            "milestone": list(M233_TESTS),
            "regressions": list(M233_REGRESSIONS),
        },
        "gates": list(M233_GATES),
    }


def _episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m233-audit-social",
        timestamp=1,
        source="audit",
        raw_text=(
            "My counterpart promised to help, but left me outside and gave mixed signals. "
            "I still do not know whether it was betrayal, pressure, or a misunderstanding."
        ),
        tags=["social", "audit"],
        metadata={"counterpart_id": "counterpart-a", "chapter_id": 9},
    )


def _threat_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m233-audit-threat",
        timestamp=2,
        source="audit",
        raw_text=(
            "A predator attacked near the shelter, but I still cannot tell whether it was "
            "a persistent threat source or a local accident."
        ),
        tags=["threat", "audit"],
        metadata={"environment_id": "shelter-edge", "chapter_id": 10},
    )


def _surface_episode() -> NarrativeEpisode:
    return NarrativeEpisode(
        episode_id="m233-audit-surface",
        timestamp=3,
        source="audit",
        raw_text="Very dramatic!!! Nothing really happened; it was just a slogan on a poster.",
        tags=["surface", "audit"],
        metadata={},
    )


def _seed_diagnostics():
    class Payload:
        workspace_broadcast_channels = ["danger", "social"]
        commitment_focus = ["protect continuity"]
        active_goal = "SURVIVAL"
        social_focus = ["counterpart-a"]
        social_alerts = ["rupture"]
        identity_tension = 0.0
        self_inconsistency_error = 0.0
        violated_commitments: list[str] = []
        conflict_type = "none"
        repair_triggered = False
        chosen = type(
            "Chosen",
            (),
            {
                "choice": "scan",
                "predicted_effects": {"danger_delta": -0.04, "social_delta": 0.02},
                "preferred_probability": 0.61,
            },
        )()

    return Payload()


def build_m233_runtime_evidence() -> dict[str, object]:
    compiler = NarrativeCompiler()
    agent = SegmentAgent()
    social_compiled = compiler.compile_episode(_episode())
    threat_compiled = compiler.compile_episode(_threat_episode())
    surface_compiled = compiler.compile_episode(_surface_episode())

    agent.ingest_narrative_episode(social_compiled)
    social_state = derive_subject_state(agent, previous_state=agent.subject_state)
    agent.subject_state = social_state

    ledger = PredictionLedger()
    verification = VerificationLoop()
    diagnostics = _seed_diagnostics()
    ledger_update = ledger.seed_predictions(
        tick=1,
        diagnostics=diagnostics,
        prediction={"danger": 0.42, "social": 0.28},
        subject_state=social_state,
        narrative_uncertainty=agent.latest_narrative_uncertainty,
    )
    verification_update = verification.refresh_targets(
        tick=2,
        ledger=ledger,
        diagnostics=diagnostics,
        subject_state=social_state,
        narrative_uncertainty=agent.latest_narrative_uncertainty,
        workspace_channels=("social",),
    )

    ablated_agent = SegmentAgent()
    ablated_agent.latest_narrative_uncertainty = type(agent.latest_narrative_uncertainty)()
    ablated_state = derive_subject_state(ablated_agent, previous_state=SubjectState())
    ablated_ledger = PredictionLedger()
    ablated_verification = VerificationLoop()
    ablated_ledger.seed_predictions(
        tick=1,
        diagnostics=diagnostics,
        prediction={"danger": 0.42, "social": 0.28},
        subject_state=ablated_state,
        narrative_uncertainty=ablated_agent.latest_narrative_uncertainty,
    )
    ablated_verification.refresh_targets(
        tick=2,
        ledger=ablated_ledger,
        diagnostics=diagnostics,
        subject_state=ablated_state,
        narrative_uncertainty=ablated_agent.latest_narrative_uncertainty,
        workspace_channels=("social",),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        runtime = SegmentRuntime.load_or_create(
            state_path=Path(tmp_dir) / "state.json",
            trace_path=Path(tmp_dir) / "trace.jsonl",
            seed=SEED_SET[0],
            reset=True,
        )
        runtime.agent.ingest_narrative_episode(threat_compiled)
        runtime.subject_state = derive_subject_state(runtime.agent, previous_state=runtime.subject_state)
        runtime.agent.subject_state = runtime.subject_state
        runtime.save_snapshot()
        restored = SegmentRuntime.load_or_create(
            state_path=Path(tmp_dir) / "state.json",
            trace_path=Path(tmp_dir) / "trace.jsonl",
            seed=SEED_SET[0],
            reset=False,
        )

    trace_records = [
        {
            "event": "unknowns_extracted",
            "episode_id": social_compiled.episode_id,
            "summary": social_compiled.uncertainty_decomposition.get("summary", ""),
            "unknowns": social_compiled.uncertainty_decomposition.get("unknowns", []),
            "profile": social_compiled.uncertainty_decomposition.get("profile", {}),
        },
        {
            "event": "downstream_promotion",
            "subject_uncertainty_count": len(social_state.narrative_uncertainties),
            "ledger_created_predictions": list(ledger_update.created_predictions),
            "verification_targets": [item.to_dict() for item in verification.active_targets],
            "top_ledger_summary": ledger.explanation_payload()["summary"],
        },
        {
            "event": "surface_cue_filtering",
            "surface_profile": surface_compiled.uncertainty_decomposition.get("profile", {}),
            "surface_summary": surface_compiled.uncertainty_decomposition.get("summary", ""),
            "surface_cues": surface_compiled.uncertainty_decomposition.get("surface_cues", []),
        },
    ]

    ablation = {
        "full_mechanism": {
            "subject_uncertainty_count": len(social_state.narrative_uncertainties),
            "narrative_prediction_count": len(
                [item for item in ledger.active_predictions() if item.source_module == "narrative_uncertainty"]
            ),
            "verification_target_count": len(verification.active_targets),
        },
        "without_retained_uncertainty": {
            "subject_uncertainty_count": len(ablated_state.narrative_uncertainties),
            "narrative_prediction_count": len(
                [item for item in ablated_ledger.active_predictions() if item.source_module == "narrative_uncertainty"]
            ),
            "verification_target_count": len(ablated_verification.active_targets),
        },
        "degradation_checks": {
            "subject_state_loses_uncertainty_surface": len(social_state.narrative_uncertainties)
            > len(ablated_state.narrative_uncertainties),
            "ledger_loses_narrative_predictions": len(
                [item for item in ledger.active_predictions() if item.source_module == "narrative_uncertainty"]
            )
            > len(
                [item for item in ablated_ledger.active_predictions() if item.source_module == "narrative_uncertainty"]
            ),
            "verification_loses_uncertainty_targets": any(
                "narrative ambiguity" in item.selected_reason for item in verification.active_targets
            )
            and not any(
                "narrative ambiguity" in item.selected_reason for item in ablated_verification.active_targets
            ),
        },
    }

    stress = {
        "stress_checks": {
            "multilingual_or_noisy_input_remains_bounded": len(
                compiler.compile_episode(
                    NarrativeEpisode(
                        episode_id="m233-audit-mixed",
                        timestamp=4,
                        source="audit",
                        raw_text="He said danger was gone, 但是 later the trap snapped again and I 不知道 why.",
                        tags=["mixed"],
                        metadata={"environment_id": "ridge"},
                    )
                ).uncertainty_decomposition.get("unknowns", [])
            )
            <= 3,
            "surface_only_input_not_promoted": surface_compiled.uncertainty_decomposition["profile"][
                "decision_relevant_unknown_count"
            ]
            == 0,
            "snapshot_roundtrip_preserves_uncertainty_summary": restored.agent.latest_narrative_uncertainty.summary
            == runtime.agent.latest_narrative_uncertainty.summary,
            "snapshot_roundtrip_preserves_unknowns": bool(
                restored.agent.latest_narrative_uncertainty.unknowns
            ),
        },
        "details": {
            "restored_summary": restored.agent.latest_narrative_uncertainty.summary,
            "restored_subject_state": restored.subject_state.to_dict(),
        },
    }

    gates = {
        "schema": {
            "passed": bool(trace_records[0]["unknowns"])
            and bool(trace_records[0]["profile"])
            and "unknown_type" in trace_records[0]["unknowns"][0],
            "evidence": ["unknowns", "profile", "unknown_type"],
        },
        "competition": {
            "passed": len(social_compiled.uncertainty_decomposition.get("competing_hypotheses", [])) >= 2,
            "evidence": ["competing_hypotheses"],
        },
        "downstream_causality": {
            "passed": all(ablation["degradation_checks"].values()),
            "evidence": list(ablation["degradation_checks"].keys()),
        },
        "surface_latent_separation": {
            "passed": stress["stress_checks"]["surface_only_input_not_promoted"]
            and bool(surface_compiled.uncertainty_decomposition.get("surface_cues", [])),
            "evidence": ["surface_only_input_not_promoted", "surface_cues"],
        },
        "snapshot_roundtrip": {
            "passed": stress["stress_checks"]["snapshot_roundtrip_preserves_uncertainty_summary"]
            and stress["stress_checks"]["snapshot_roundtrip_preserves_unknowns"],
            "evidence": ["restored_summary", "restored_unknowns"],
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
        "# M2.33 Acceptance Summary",
        "",
        f"- Status: {report['status']}",
        f"- Recommendation: {report['recommendation']}",
        f"- Competition gate: {'PASS' if gates['competition']['passed'] else 'FAIL'}",
        f"- Downstream causality gate: {'PASS' if gates['downstream_causality']['passed'] else 'FAIL'}",
        f"- Surface/latent separation gate: {'PASS' if gates['surface_latent_separation']['passed'] else 'FAIL'}",
        f"- Snapshot gate: {'PASS' if gates['snapshot_roundtrip']['passed'] else 'FAIL'}",
        f"- Regression gate: {'PASS' if gates['regression']['passed'] else 'FAIL'}",
        f"- Freshness gate: {'PASS' if gates['artifact_freshness']['passed'] else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


def write_m233_acceptance_artifacts(
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
                    f"strict M2.33 audit refuses injected execution records for {label} tests"
                )

    audit_started_at = _now_iso()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    runtime_evidence = build_m233_runtime_evidence()
    _write_trace(M233_TRACE_PATH, runtime_evidence["trace_records"])
    _write_json(M233_ABLATION_PATH, runtime_evidence["ablation"])
    _write_json(M233_STRESS_PATH, runtime_evidence["stress"])

    milestone_execution = milestone_execution or _suite_execution_record(
        label="m233-milestone",
        paths=M233_TESTS,
        execute=execute_test_suites,
    )
    regression_execution = regression_execution or _suite_execution_record(
        label="m233-regression",
        paths=M233_REGRESSIONS,
        execute=execute_test_suites,
    )

    artifacts = {
        "specification": str(M233_SPEC_PATH),
        "preparation": str(M233_PREPARATION_PATH),
        "canonical_trace": str(M233_TRACE_PATH),
        "ablation": str(M233_ABLATION_PATH),
        "stress": str(M233_STRESS_PATH),
        "report": str(M233_REPORT_PATH),
        "summary": str(M233_SUMMARY_PATH),
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
            "competition",
            "downstream_causality",
            "surface_latent_separation",
            "snapshot_roundtrip",
        ):
            gates[gate_name]["passed"] = bool(gates[gate_name]["passed"]) and milestone_ok
        all_passed = all(bool(item["passed"]) for item in gates.values())
        return {
            "milestone_id": "M2.33",
            "title": "Narrative Uncertainty Decomposition",
            "strict": strict,
            "status": "PASS" if all_passed else "FAIL",
            "recommendation": "ACCEPT" if all_passed else "BLOCK",
            "generated_at": generated_at_value,
            "seed_set": list(SEED_SET),
            "artifacts": artifacts,
            "tests": {
                "milestone": milestone_execution,
                "regressions": regression_execution,
            },
            "gates": gates,
            "findings": [],
            "residual_risks": [] if all_passed else ["runtime evidence or freshness gate missing"],
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
            "current_round": False,
            "git": {
                "head": _git_commit(),
                "dirty": bool(_git_dirty_paths()),
                "dirty_paths": _git_dirty_paths(),
            },
        },
    )
    M233_REPORT_PATH.write_text(json.dumps(provisional_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M233_SUMMARY_PATH.write_text(_summary_markdown(provisional_report), encoding="utf-8")

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
    M233_REPORT_PATH.write_text(json.dumps(final_report, indent=2, ensure_ascii=True), encoding="utf-8")
    M233_SUMMARY_PATH.write_text(_summary_markdown(final_report), encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    write_m233_acceptance_artifacts(strict=True, execute_test_suites=True)
