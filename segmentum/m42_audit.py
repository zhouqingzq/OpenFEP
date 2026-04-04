from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .m4_benchmarks import (
    DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
    compute_behavioral_seed_summaries,
    default_acceptance_benchmark_root,
    run_confidence_database_benchmark,
    run_iowa_gambling_benchmark,
    run_two_armed_bandit_benchmark,
    same_seed_triple_replay,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M42_PROTOCOL_PATH = ARTIFACTS_DIR / "m42_benchmark_protocol.json"
M42_CONFIDENCE_TRACE_PATH = ARTIFACTS_DIR / "m42_confidence_trace.json"
M42_IGT_TRACE_PATH = ARTIFACTS_DIR / "m42_igt_trace.json"
M42_BANDIT_TRACE_PATH = ARTIFACTS_DIR / "m42_two_armed_bandit_trace.json"
M42_LEAKAGE_PATH = ARTIFACTS_DIR / "m42_leakage_report.json"
M42_REPRO_PATH = ARTIFACTS_DIR / "m42_reproducibility_report.json"
M42_REPORT_PATH = REPORTS_DIR / "m42_acceptance_report.json"
M42_SUMMARY_PATH = REPORTS_DIR / "m42_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _artifact_status(*, blocked: bool, acceptance_ok: bool, smoke_only: bool = False) -> str:
    if blocked:
        return "blocked"
    if smoke_only:
        return "smoke-only"
    return "pass" if acceptance_ok else "fail"


def _resolve_acceptance_root(benchmark_root: Path | str | None = None) -> Path | None:
    if benchmark_root is None:
        return default_acceptance_benchmark_root()
    candidate = Path(benchmark_root).resolve()
    required = (
        candidate / "confidence_database" / "manifest.json",
        candidate / "iowa_gambling_task" / "manifest.json",
    )
    return candidate if all(path.exists() for path in required) else None


def _choose_igt_subject_id(benchmark_root: Path) -> str:
    subject_ids: set[str] = set()
    with (benchmark_root / "iowa_gambling_task" / "iowa_gambling_task_external.jsonl").open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("split") == "heldout":
                subject_ids.add(str(row["subject_id"]))
                if len(subject_ids) >= 1:
                    break
    return sorted(subject_ids)[0] if subject_ids else "s-01"


def _external_track_payload(
    run_payload: dict[str, object],
    *,
    selected_source_dataset: str | None = None,
    selected_subject_id: str | None = None,
    protocol_mode: str | None = None,
    standard_trial_count: int | None = None,
) -> dict[str, object]:
    bundle = dict(run_payload["bundle"])
    manifest = dict(bundle["manifest"])
    track: dict[str, object] = {
        "benchmark_id": str(run_payload["benchmark_id"]),
        "manifest_path": bundle["manifest_path"],
        "data_path": bundle["data_path"],
        "source_type": bundle["source_type"],
        "smoke_test_only": bundle["smoke_test_only"],
        "is_synthetic": bundle["is_synthetic"],
        "acceptance_requires_external_bundle": bool(
            manifest.get("acceptance_requires_external_bundle", bundle.get("external_bundle_preferred", False))
        ),
        "record_count": manifest.get("record_count"),
        "subject_count": run_payload["subject_summary"]["subject_count"],
        "bundle_mode": run_payload["bundle_mode"],
        "claim_envelope": run_payload["claim_envelope"],
    }
    if selected_source_dataset is not None:
        track["selected_source_dataset"] = selected_source_dataset
    if selected_subject_id is not None:
        track["selected_subject_id"] = selected_subject_id
    if protocol_mode is not None:
        track["protocol_mode"] = protocol_mode
    if standard_trial_count is not None:
        track["standard_trial_count"] = standard_trial_count
    return track


def write_m42_acceptance_artifacts(
    *,
    round_started_at: str | None = None,
    benchmark_root: Path | str | None = None,
) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    acceptance_root = _resolve_acceptance_root(benchmark_root)
    blocked = acceptance_root is None

    protocol_payload = {
        "benchmark_root": str(acceptance_root) if acceptance_root else None,
        "tasks": {
            "confidence_database": {
                "mode": "acceptance-grade" if not blocked else "blocked",
                "selected_source_dataset": DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
                "export_type": "human_aligned_trial_trace",
            },
            "iowa_gambling_task": {
                "mode": "acceptance-grade" if not blocked else "blocked",
                "protocol_mode": "standard_100",
                "standard_trial_count": 100,
            },
            "two_armed_bandit": {
                "mode": "smoke-only",
                "trial_count": 50,
                "integration_rule": "adapter_only_no_agent_core_changes",
            },
        },
    }
    M42_PROTOCOL_PATH.write_text(json.dumps(protocol_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report: dict[str, object] = {
        "milestone_id": "M4.2",
        "generated_at": _now_iso(),
        "round_started_at": started_at,
        "status": "FAIL" if blocked else "PASS",
        "acceptance_state": "blocked_missing_external_bundle" if blocked else "acceptance_pass",
        "reference_modes": {
            "repo_smoke_fixtures": {
                "mode": "smoke-only",
                "status": "available_but_not_used_for_acceptance",
            }
        },
        "artifacts": {
            "protocol": str(M42_PROTOCOL_PATH),
            "confidence_trace": str(M42_CONFIDENCE_TRACE_PATH),
            "igt_trace": str(M42_IGT_TRACE_PATH),
            "bandit_trace": str(M42_BANDIT_TRACE_PATH),
            "leakage": str(M42_LEAKAGE_PATH),
            "reproducibility": str(M42_REPRO_PATH),
            "summary": str(M42_SUMMARY_PATH),
        },
        "tracks": {},
        "findings": [],
    }

    if blocked:
        M42_LEAKAGE_PATH.write_text(
            json.dumps(
                {
                    "benchmark_root": None,
                    "status": "blocked",
                    "summary": {
                        "all_ok": False,
                        "failing_tracks": ["confidence_database", "iowa_gambling_task"],
                    },
                    "tracks": {
                        "confidence_database": {"mode": "blocked", "status": "blocked"},
                        "iowa_gambling_task": {"mode": "blocked", "status": "blocked"},
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        M42_REPRO_PATH.write_text(
            json.dumps(
                {
                    "benchmark_root": None,
                    "status": "blocked",
                    "same_seed_triple_replay": {"exact_match": False},
                    "different_seed_behavior": {"different_seeds_differ": False},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report["tracks"] = {
            "confidence_database": {"mode": "blocked", "status": "blocked"},
            "iowa_gambling_task": {"mode": "blocked", "status": "blocked"},
            "two_armed_bandit": {"mode": "smoke-only", "status": "smoke-only"},
            "reproducibility": {"mode": "blocked", "status": "blocked"},
        }
        report["gate_summary"] = {
            "leakage_passed": False,
            "reproducibility_passed": False,
        }
        report["findings"] = [{"severity": "S1", "label": "external_bundle_missing", "detail": "Acceptance-grade bundles were not found under external_benchmark_registry."}]
    else:
        igt_subject_id = _choose_igt_subject_id(acceptance_root)
        confidence_run = run_confidence_database_benchmark(
            seed=42,
            benchmark_root=acceptance_root,
            selected_source_dataset=DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
        )
        igt_run = run_iowa_gambling_benchmark(
            seed=44,
            benchmark_root=acceptance_root,
            selected_subject_id=igt_subject_id,
        )
        bandit_run = run_two_armed_bandit_benchmark(seed=46, trial_count=50)
        igt_replay = same_seed_triple_replay(
            "iowa_gambling_task",
            seed=44,
            run_kwargs={
                "benchmark_root": acceptance_root,
                "selected_subject_id": igt_subject_id,
                "protocol_mode": "standard_100",
                "include_predictions": False,
                "include_subject_summary": False,
            },
        )
        igt_seed_summary = compute_behavioral_seed_summaries(
            "iowa_gambling_task",
            seeds=[41, 42, 43, 45],
            run_kwargs={
                "benchmark_root": acceptance_root,
                "selected_subject_id": igt_subject_id,
                "protocol_mode": "standard_100",
                "include_predictions": False,
                "include_subject_summary": False,
            },
        )

        M42_CONFIDENCE_TRACE_PATH.write_text(json.dumps(confidence_run, indent=2, ensure_ascii=False), encoding="utf-8")
        M42_IGT_TRACE_PATH.write_text(json.dumps(igt_run, indent=2, ensure_ascii=False), encoding="utf-8")
        M42_BANDIT_TRACE_PATH.write_text(json.dumps(bandit_run, indent=2, ensure_ascii=False), encoding="utf-8")
        leakage_report = {
            "benchmark_root": str(acceptance_root),
            "status": "pass"
            if bool(confidence_run.get("leakage_check", {}).get("ok")) and bool(igt_run.get("leakage_check", {}).get("ok"))
            else "fail",
            "tracks": {
                "confidence_database": {
                    "benchmark_id": confidence_run["benchmark_id"],
                    "mode": "acceptance-grade",
                    "status": confidence_run["leakage_check"]["status"],
                    "bundle_mode": confidence_run["bundle_mode"],
                    "claim_envelope": confidence_run["claim_envelope"],
                    "checks": confidence_run["leakage_check"],
                },
                "iowa_gambling_task": {
                    "benchmark_id": igt_run["benchmark_id"],
                    "mode": "acceptance-grade",
                    "status": igt_run["leakage_check"]["status"],
                    "bundle_mode": igt_run["bundle_mode"],
                    "claim_envelope": igt_run["claim_envelope"],
                    "checks": igt_run["leakage_check"],
                },
            },
            "summary": {
                "all_ok": bool(confidence_run.get("leakage_check", {}).get("ok")) and bool(igt_run.get("leakage_check", {}).get("ok")),
                "failing_tracks": [
                    track_name
                    for track_name, payload in {
                        "confidence_database": confidence_run["leakage_check"],
                        "iowa_gambling_task": igt_run["leakage_check"],
                    }.items()
                    if not bool(payload.get("ok"))
                ],
            },
        }
        M42_LEAKAGE_PATH.write_text(json.dumps(leakage_report, indent=2, ensure_ascii=False), encoding="utf-8")
        M42_REPRO_PATH.write_text(
            json.dumps(
                {
                    "benchmark_root": str(acceptance_root),
                    "replay_basis": {
                        "benchmark_id": "iowa_gambling_task",
                        "manifest_path": igt_run["bundle"]["manifest_path"],
                        "data_path": igt_run["bundle"]["data_path"],
                        "source_type": igt_run["bundle"]["source_type"],
                        "smoke_test_only": igt_run["bundle"]["smoke_test_only"],
                        "is_synthetic": igt_run["bundle"]["is_synthetic"],
                        "acceptance_requires_external_bundle": bool(
                            igt_run["bundle"]["manifest"].get("acceptance_requires_external_bundle", False)
                        ),
                        "selected_subject_id": igt_subject_id,
                        "protocol_mode": igt_run["protocol_validation"]["protocol_mode"],
                        "standard_trial_count": igt_run["protocol_validation"]["standard_trial_count"],
                    },
                    "same_seed_triple_replay": {
                        "task_id": igt_replay["task_id"],
                        "seed": igt_replay["seed"],
                        "exact_match": igt_replay["exact_match"],
                        "trial_count": igt_replay["runs"][0]["trial_count"],
                    },
                    "different_seed_behavior": {
                        "task_id": igt_seed_summary["task_id"],
                        "seed_summaries": igt_seed_summary["seed_summaries"],
                        "behavioral_summary": igt_seed_summary["behavioral_summary"],
                        "unique_behavior_sequences": igt_seed_summary["unique_behavior_sequences"],
                        "different_seeds_differ": igt_seed_summary["different_seeds_differ"],
                        "sequence_differences": igt_seed_summary["sequence_differences"],
                        "sequence_diff_summary": igt_seed_summary["sequence_diff_summary"],
                        "behavioral_evidence": igt_seed_summary["behavioral_evidence"],
                        "deck_distribution_difference": igt_seed_summary["deck_distribution_difference"],
                        "reproducibility_gate": igt_seed_summary["reproducibility_gate"],
                    },
                    "smoke_reference": {
                        "benchmark_id": bandit_run["benchmark_id"],
                        "bundle_mode": bandit_run["bundle_mode"],
                        "claim_envelope": bandit_run["claim_envelope"],
                        "trial_count": bandit_run["trial_count"],
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        confidence_leakage_ok = bool(confidence_run.get("leakage_check", {}).get("ok"))
        igt_leakage_ok = bool(igt_run.get("leakage_check", {}).get("ok"))
        confidence_ok = (
            bool(confidence_run["trial_export_validation"]["ok"])
            and confidence_run["trial_count"] == len(confidence_run["trial_trace"])
            and confidence_leakage_ok
        )
        igt_ok = (
            bool(igt_run["trial_export_validation"]["ok"])
            and str(igt_run["protocol_validation"]["protocol_mode"]) == "standard_100"
            and bool(igt_run["protocol_validation"]["standard_trial_count"] == 100)
            and int(igt_run["trial_count"]) == 100
            and bool(igt_run["trial_trace"])
            and igt_run["trial_trace"][-1]["trial_index"] == 100
            and igt_leakage_ok
        )
        bandit_ok = bool(bandit_run["trial_export_validation"]["ok"]) and bandit_run["trial_count"] == 50
        repro_gate = dict(igt_seed_summary["reproducibility_gate"])
        repro_ok = bool(igt_replay["exact_match"]) and bool(repro_gate["passed"])

        report["tracks"] = {
            "confidence_database": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=confidence_ok),
                "trial_count": confidence_run["trial_count"],
                "export_schema_ok": confidence_run["trial_export_validation"]["ok"],
                "leakage_status": _artifact_status(blocked=False, acceptance_ok=confidence_leakage_ok),
                "leakage_ok": confidence_leakage_ok,
                "leakage_check": confidence_run.get("leakage_check"),
                **_external_track_payload(
                    confidence_run,
                    selected_source_dataset=DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
                ),
            },
            "iowa_gambling_task": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=igt_ok),
                "trial_count": igt_run["trial_count"],
                "export_schema_ok": igt_run["trial_export_validation"]["ok"],
                "leakage_status": _artifact_status(blocked=False, acceptance_ok=igt_leakage_ok),
                "leakage_ok": igt_leakage_ok,
                "leakage_check": igt_run.get("leakage_check"),
                **_external_track_payload(
                    igt_run,
                    selected_subject_id=igt_subject_id,
                    protocol_mode=str(igt_run["protocol_validation"]["protocol_mode"]),
                    standard_trial_count=igt_run["protocol_validation"]["standard_trial_count"],
                ),
            },
            "two_armed_bandit": {
                "mode": "smoke-only",
                "status": _artifact_status(blocked=False, acceptance_ok=bandit_ok, smoke_only=True),
                "trial_count": bandit_run["trial_count"],
                "benchmark_state": bandit_run["benchmark_status"]["benchmark_state"],
                "bundle_mode": bandit_run["bundle_mode"],
                "claim_envelope": bandit_run["claim_envelope"],
                "extensibility_claim": "adapter_only",
            },
            "reproducibility": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=repro_ok),
                "benchmark_id": "iowa_gambling_task",
                "manifest_path": igt_run["bundle"]["manifest_path"],
                "data_path": igt_run["bundle"]["data_path"],
                "source_type": igt_run["bundle"]["source_type"],
                "smoke_test_only": igt_run["bundle"]["smoke_test_only"],
                "is_synthetic": igt_run["bundle"]["is_synthetic"],
                "acceptance_requires_external_bundle": bool(
                    igt_run["bundle"]["manifest"].get("acceptance_requires_external_bundle", False)
                ),
                "selected_subject_id": igt_subject_id,
                "protocol_mode": igt_run["protocol_validation"]["protocol_mode"],
                "standard_trial_count": igt_run["protocol_validation"]["standard_trial_count"],
                "same_seed_triple_replay_exact_match": igt_replay["exact_match"],
                "different_seeds_differ": igt_seed_summary["different_seeds_differ"],
                "unique_behavior_sequences": igt_seed_summary["unique_behavior_sequences"],
                "sequence_diff_summary": igt_seed_summary["sequence_diff_summary"],
                "varying_behavior_metric_count": igt_seed_summary["behavioral_evidence"]["varying_behavior_metric_count"],
                "varying_metric_names": igt_seed_summary["behavioral_evidence"]["varying_metric_names"],
                "reproducibility_gate": repro_gate,
            },
        }
        report["gate_summary"] = {
            "leakage_passed": leakage_report["summary"]["all_ok"],
            "reproducibility_passed": repro_ok,
            "reproducibility_requirements": repro_gate["requirements"],
        }
        acceptance_track_names = ("confidence_database", "iowa_gambling_task", "reproducibility")
        report["status"] = "PASS" if all(report["tracks"][name]["status"] == "pass" for name in acceptance_track_names) else "FAIL"
        report["acceptance_state"] = "acceptance_pass" if report["status"] == "PASS" else "acceptance_fail"

    M42_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M42_SUMMARY_PATH.write_text(
        "# M4.2 Acceptance Summary\n\n"
        + (
            "PASS: acceptance-grade Confidence Database and Iowa Gambling Task artifacts were regenerated from external bundles, leakage checks passed, IGT replay remained deterministic with distinguishable cross-seed evidence, and the two-armed bandit adapter remained smoke-only.\n"
            if report["status"] == "PASS"
            else "FAIL: at least one acceptance-grade external-bundle track remains blocked or failing, including leakage and/or reproducibility gates.\n"
        ),
        encoding="utf-8",
    )
    return {
        "protocol": str(M42_PROTOCOL_PATH),
        "confidence_trace": str(M42_CONFIDENCE_TRACE_PATH),
        "igt_trace": str(M42_IGT_TRACE_PATH),
        "bandit_trace": str(M42_BANDIT_TRACE_PATH),
        "leakage": str(M42_LEAKAGE_PATH),
        "reproducibility": str(M42_REPRO_PATH),
        "report": str(M42_REPORT_PATH),
        "summary": str(M42_SUMMARY_PATH),
    }
