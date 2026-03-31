from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .m4_benchmarks import (
    DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
    EXTERNAL_BENCHMARK_ROOT,
    bootstrap_seed_summary_ci,
    compute_behavioral_seed_summaries,
    default_acceptance_benchmark_root,
    evaluate_seed_tolerance_gate,
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


def _choose_igt_subject_id() -> str:
    subject_ids: set[str] = set()
    with (EXTERNAL_BENCHMARK_ROOT / "iowa_gambling_task" / "iowa_gambling_task_external.jsonl").open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("split") == "heldout":
                subject_ids.add(str(row["subject_id"]))
                if len(subject_ids) >= 1:
                    break
    return sorted(subject_ids)[0] if subject_ids else "s-01"


def write_m42_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    started_at = round_started_at or _now_iso()
    acceptance_root = default_acceptance_benchmark_root()
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
                "mode": "acceptance-grade",
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
            "reproducibility": str(M42_REPRO_PATH),
            "summary": str(M42_SUMMARY_PATH),
        },
        "tracks": {},
        "findings": [],
    }

    if blocked:
        report["tracks"] = {
            "confidence_database": {"mode": "blocked", "status": "blocked"},
            "iowa_gambling_task": {"mode": "blocked", "status": "blocked"},
            "two_armed_bandit": {"mode": "acceptance-grade", "status": "pass"},
            "reproducibility": {"mode": "blocked", "status": "blocked"},
        }
        report["findings"] = [{"severity": "S1", "label": "external_bundle_missing", "detail": "Acceptance-grade bundles were not found under external_benchmark_registry."}]
    else:
        igt_subject_id = _choose_igt_subject_id()
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
        triple_replay = same_seed_triple_replay(
            "two_armed_bandit",
            seed=91,
            run_kwargs={"trial_count": 50, "include_predictions": False, "include_subject_summary": False},
        )
        seed_summary = compute_behavioral_seed_summaries(
            "two_armed_bandit",
            seeds=[1, 2, 3, 4, 5, 6, 7, 8],
            run_kwargs={"trial_count": 50, "include_predictions": False, "include_subject_summary": False},
        )
        reward_ci = bootstrap_seed_summary_ci(seed_summary["seed_summaries"], metric_name="mean_reward", bootstrap_seed=99)
        tolerance_gate = evaluate_seed_tolerance_gate(
            seed_summary["seed_summaries"],
            metric_name="mean_reward",
            bootstrap_seed=99,
            lower_bound=0.55,
            upper_bound=0.8,
            min_variance=0.0001,
        )

        M42_CONFIDENCE_TRACE_PATH.write_text(json.dumps(confidence_run, indent=2, ensure_ascii=False), encoding="utf-8")
        M42_IGT_TRACE_PATH.write_text(json.dumps(igt_run, indent=2, ensure_ascii=False), encoding="utf-8")
        M42_BANDIT_TRACE_PATH.write_text(json.dumps(bandit_run, indent=2, ensure_ascii=False), encoding="utf-8")
        M42_REPRO_PATH.write_text(
            json.dumps(
                {
                    "same_seed_triple_replay": {
                        "task_id": triple_replay["task_id"],
                        "seed": triple_replay["seed"],
                        "exact_match": triple_replay["exact_match"],
                    },
                    "different_seed_behavior": seed_summary,
                    "bootstrap_ci": reward_ci,
                    "tolerance_gate": tolerance_gate,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        confidence_ok = bool(confidence_run["trial_export_validation"]["ok"]) and confidence_run["trial_count"] == len(confidence_run["trial_trace"])
        igt_ok = (
            bool(igt_run["trial_export_validation"]["ok"])
            and bool(igt_run["protocol_validation"]["standard_trial_count"] == 100)
            and bool(igt_run["trial_trace"])
            and igt_run["trial_trace"][-1]["trial_index"] == 100
        )
        bandit_ok = bool(bandit_run["trial_export_validation"]["ok"]) and bandit_run["trial_count"] == 50
        repro_ok = bool(triple_replay["exact_match"]) and bool(seed_summary["different_seeds_differ"]) and bool(tolerance_gate["passed"])

        report["tracks"] = {
            "confidence_database": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=confidence_ok),
                "trial_count": confidence_run["trial_count"],
                "subject_count": confidence_run["subject_summary"]["subject_count"],
                "export_schema_ok": confidence_run["trial_export_validation"]["ok"],
                "selected_source_dataset": DEFAULT_CONFIDENCE_ACCEPTANCE_DATASET,
            },
            "iowa_gambling_task": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=igt_ok),
                "trial_count": igt_run["trial_count"],
                "subject_count": igt_run["subject_summary"]["subject_count"],
                "protocol_mode": igt_run["protocol_validation"]["protocol_mode"],
                "standard_trial_count": igt_run["protocol_validation"]["standard_trial_count"],
                "selected_subject_id": igt_subject_id,
            },
            "two_armed_bandit": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=bandit_ok),
                "trial_count": bandit_run["trial_count"],
                "extensibility_claim": "adapter_only",
            },
            "reproducibility": {
                "mode": "acceptance-grade",
                "status": _artifact_status(blocked=False, acceptance_ok=repro_ok),
                "same_seed_triple_replay_exact_match": triple_replay["exact_match"],
                "different_seeds_differ": seed_summary["different_seeds_differ"],
                "bootstrap_metric": reward_ci,
                "tolerance_gate": tolerance_gate,
            },
        }
        report["status"] = "PASS" if all(track["status"] == "pass" for track in report["tracks"].values()) else "FAIL"
        report["acceptance_state"] = "acceptance_pass" if report["status"] == "PASS" else "acceptance_fail"

    M42_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M42_SUMMARY_PATH.write_text(
        "# M4.2 Acceptance Summary\n\n"
        + ("PASS: acceptance-grade Confidence export, standard 100-trial IGT, reproducibility gate, and two-armed bandit extensibility demo were regenerated.\n" if report["status"] == "PASS" else "FAIL: at least one M4.2 acceptance track remains blocked or failing.\n"),
        encoding="utf-8",
    )
    return {
        "protocol": str(M42_PROTOCOL_PATH),
        "confidence_trace": str(M42_CONFIDENCE_TRACE_PATH),
        "igt_trace": str(M42_IGT_TRACE_PATH),
        "bandit_trace": str(M42_BANDIT_TRACE_PATH),
        "reproducibility": str(M42_REPRO_PATH),
        "report": str(M42_REPORT_PATH),
        "summary": str(M42_SUMMARY_PATH),
    }
