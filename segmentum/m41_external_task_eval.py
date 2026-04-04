from __future__ import annotations

import ast
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .benchmark_registry import DEFAULT_BENCHMARK_ROOT, benchmark_status, load_benchmark_bundle, validate_benchmark_bundle
from .m4_benchmarks import EXTERNAL_BENCHMARK_ROOT, run_confidence_database_benchmark, run_iowa_gambling_benchmark
from .m4_cognitive_style import CognitiveStyleParameters


ROOT = Path(__file__).resolve().parents[1]

CONFIDENCE_MIN_TRIALS = 40
IGT_MIN_TRIALS = 40

FORBIDDEN_IMPORT_MODULES = {"m41_external_generator", "m41_inference"}
FORBIDDEN_CALL_NAMES = {"run_cognitive_style_trial", "compute_observable_metrics", "infer_cognitive_style"}


def _round(value: float) -> float:
    return round(float(value), 6)


def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _leakage_report(rows: list[dict[str, Any]], *, key_field: str) -> dict[str, Any]:
    observed: dict[str, set[str]] = {}
    for row in rows:
        key = str(row.get(key_field, "")).strip()
        split = str(row.get("split", "")).strip()
        if not key or not split:
            continue
        observed.setdefault(key, set()).add(split)
    leaking_keys = sorted(key for key, splits in observed.items() if len(splits) > 1)
    return {
        "key_field": key_field,
        "ok": not leaking_keys,
        "group_count": len(observed),
        "leaking_keys": leaking_keys[:25],
        "leaking_key_count": len(leaking_keys),
    }


def _bundle_evidence(benchmark_id: str) -> dict[str, Any]:
    bundle = load_benchmark_bundle(benchmark_id, root=EXTERNAL_BENCHMARK_ROOT)
    validation = validate_benchmark_bundle(benchmark_id, root=EXTERNAL_BENCHMARK_ROOT)
    status = benchmark_status(benchmark_id, root=EXTERNAL_BENCHMARK_ROOT)
    rows = _iter_jsonl(bundle.data_path)
    split_values = sorted({str(row.get("split", "")).strip() for row in rows if str(row.get("split", "")).strip()})
    evidence = {
        "bundle": bundle.to_dict(),
        "validation": validation.to_dict(),
        "status": status.to_dict(),
        "split_values": split_values,
        "record_count_observed": len(rows),
        "subject_split_integrity": _leakage_report(rows, key_field="subject_id"),
    }
    if any("session_id" in row for row in rows):
        evidence["session_split_integrity"] = _leakage_report(rows, key_field="session_id")
    return evidence


@lru_cache(maxsize=1)
def _confidence_run() -> dict[str, Any]:
    return run_confidence_database_benchmark(
        CognitiveStyleParameters(),
        seed=42,
        split="heldout",
        allow_smoke_test=False,
        prefer_external=True,
        include_subject_summary=True,
        include_predictions=True,
        include_trial_trace=False,
        max_trials=CONFIDENCE_MIN_TRIALS,
    )


@lru_cache(maxsize=1)
def _igt_run() -> dict[str, Any]:
    return run_iowa_gambling_benchmark(
        CognitiveStyleParameters(),
        seed=44,
        split="heldout",
        allow_smoke_test=False,
        prefer_external=True,
        include_subject_summary=True,
        include_predictions=True,
        include_trial_trace=False,
        max_trials=IGT_MIN_TRIALS,
        protocol_mode="nonstandard",
    )


def _run_summary(payload: dict[str, Any], *, metric_names: list[str]) -> dict[str, Any]:
    metrics = dict(payload.get("metrics", {}))
    return {
        "benchmark_id": payload["benchmark_id"],
        "claim_envelope": payload.get("claim_envelope"),
        "split": payload.get("split"),
        "trial_count": payload.get("trial_count"),
        "bundle_mode": payload.get("bundle_mode"),
        "benchmark_status": dict(payload.get("benchmark_status", {})),
        "protocol_validation": dict(payload.get("protocol_validation", {})),
        "metrics": {name: metrics.get(name) for name in metric_names},
        "subject_summary": dict(payload.get("subject_summary", {})),
        "leakage_check": dict(payload.get("leakage_check", {})),
    }


def _confidence_metric_gate(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload["metrics"])
    passed = (
        float(metrics.get("accuracy", 0.0)) >= 0.85
        and float(metrics.get("auroc2", 0.0)) >= 0.80
        and float(metrics.get("calibration_error", 1.0)) <= 0.50
        and float(metrics.get("heldout_likelihood", -99.0)) >= -1.0
        and int(payload.get("trial_count", 0)) >= CONFIDENCE_MIN_TRIALS
    )
    return {
        "passed": passed,
        "thresholds": {
            "accuracy_gte": 0.85,
            "auroc2_gte": 0.80,
            "calibration_error_lte": 0.50,
            "heldout_likelihood_gte": -1.0,
            "trial_count_gte": CONFIDENCE_MIN_TRIALS,
        },
        "observed": {
            "accuracy": _round(metrics.get("accuracy", 0.0)),
            "auroc2": _round(metrics.get("auroc2", 0.0)),
            "calibration_error": _round(metrics.get("calibration_error", 0.0)),
            "heldout_likelihood": _round(metrics.get("heldout_likelihood", 0.0)),
            "trial_count": int(payload.get("trial_count", 0)),
        },
    }


def _igt_metric_gate(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload["metrics"])
    passed = (
        float(metrics.get("advantageous_choice_rate", 0.0)) >= 0.80
        and float(metrics.get("policy_alignment_rate", 0.0)) >= 0.35
        and float(metrics.get("final_cumulative_gain", -1.0)) >= 1000.0
        and int(payload.get("trial_count", 0)) >= IGT_MIN_TRIALS
    )
    return {
        "passed": passed,
        "thresholds": {
            "advantageous_choice_rate_gte": 0.80,
            "policy_alignment_rate_gte": 0.35,
            "final_cumulative_gain_gte": 1000.0,
            "trial_count_gte": IGT_MIN_TRIALS,
        },
        "observed": {
            "advantageous_choice_rate": _round(metrics.get("advantageous_choice_rate", 0.0)),
            "policy_alignment_rate": _round(metrics.get("policy_alignment_rate", 0.0)),
            "final_cumulative_gain": _round(metrics.get("final_cumulative_gain", 0.0)),
            "trial_count": int(payload.get("trial_count", 0)),
        },
    }


def smoke_fixture_rejection_report() -> dict[str, Any]:
    checks = {}
    for benchmark_id, runner, kwargs in (
        (
            "confidence_database",
            run_confidence_database_benchmark,
            {"split": "heldout", "include_subject_summary": False, "include_predictions": False, "include_trial_trace": False, "max_trials": 5},
        ),
        (
            "iowa_gambling_task",
            run_iowa_gambling_benchmark,
            {
                "split": "heldout",
                "include_subject_summary": False,
                "include_predictions": False,
                "include_trial_trace": False,
                "max_trials": 5,
                "protocol_mode": "nonstandard",
            },
        ),
    ):
        blocked = False
        error_message = ""
        try:
            runner(
                CognitiveStyleParameters(),
                seed=41,
                prefer_external=False,
                allow_smoke_test=False,
                **kwargs,
            )
        except ValueError as exc:
            blocked = "external bundle" in str(exc).lower() or "smoke test" in str(exc).lower()
            error_message = str(exc)
        checks[benchmark_id] = {"blocked": blocked, "error": error_message}
    return {"all_blocked": all(item["blocked"] for item in checks.values()), "checks": checks}


def downgraded_claims_inventory() -> dict[str, Any]:
    return {
        "downgraded_claims": [
            {
                "claim": "external validation",
                "replacement": "same-framework synthetic holdout validation sidecar",
                "status": "downgraded",
            },
            {
                "claim": "identifiability",
                "replacement": "within synthetic family recoverability",
                "status": "downgraded",
            },
            {
                "claim": "blind classification",
                "replacement": "profile distinguishability on synthetic holdout",
                "status": "downgraded",
            },
        ],
        "module_inventory": {
            "keep": [
                "segmentum/m41_external_task_eval.py",
                "segmentum/m4_benchmarks.py",
                "segmentum/benchmark_registry.py",
            ],
            "downgrade_to_same_framework": [
                "segmentum/m41_inference.py",
                "segmentum/m41_baselines.py",
                "segmentum/m41_identifiability.py",
                "segmentum/m41_blind_classifier.py",
                "segmentum/m41_falsification.py",
                "segmentum/m41_external_generator.py",
                "segmentum/m41_external_dataset.py",
                "segmentum/m41_external_observables.py",
            ],
            "stop_relying_on_for_m41_acceptance": [
                "segmentum/m41_inference.py",
                "segmentum/m41_baselines.py",
                "segmentum/m41_identifiability.py",
                "segmentum/m41_blind_classifier.py",
                "segmentum/m41_falsification.py",
                "segmentum/m41_external_dataset.py",
                "segmentum/m41_external_observables.py",
                "scripts/generate_m41_external_data.py",
            ],
        },
        "still_not_claimed": [
            "external validation of latent cognitive parameters",
            "identifiability of latent style semantics on human data",
            "falsification of the latent parameter ontology",
            "blind classification of true human cognitive profiles",
            "completion of M4.2 benchmark/task-layer recovery",
        ],
    }


def evaluation_chain_audit() -> dict[str, Any]:
    # This audit is intentionally limited to the historical task-bundle
    # evaluation wrapper. M4.1 interface acceptance in m41_audit.py
    # legitimately uses the internal generator and is not part of the
    # external-task evaluation chain.
    paths = [
        ROOT / "segmentum" / "m41_external_task_eval.py",
        ROOT / "segmentum" / "m41_external_validation.py",
    ]
    findings = {}
    all_clear = True
    for path in paths:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        present: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_IMPORT_MODULES:
                present.append(f"import:{node.module}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                    present.append(f"call:{node.func.id}")
                if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALL_NAMES:
                    present.append(f"call:{node.func.attr}")
        findings[str(path)] = {"forbidden_references_present": present}
        all_clear = all_clear and not present
    return {"paths": findings, "all_clear": all_clear}


@lru_cache(maxsize=1)
def run_external_task_bundle_evaluation() -> dict[str, Any]:
    confidence_bundle = _bundle_evidence("confidence_database")
    igt_bundle = _bundle_evidence("iowa_gambling_task")
    confidence_run = _run_summary(
        _confidence_run(),
        metric_names=["accuracy", "auroc2", "calibration_error", "heldout_likelihood", "confidence_alignment"],
    )
    igt_run = _run_summary(
        _igt_run(),
        metric_names=["advantageous_choice_rate", "policy_alignment_rate", "final_cumulative_gain", "deck_match_rate", "late_advantageous_rate"],
    )
    smoke_rejection = smoke_fixture_rejection_report()
    scope_boundary = downgraded_claims_inventory()
    chain_audit = evaluation_chain_audit()
    return {
        "analysis_type": "m41_minimal_external_task_validation",
        "validation_type": "external_task_bundle",
        "benchmark_scope": "task-level evaluation on imported external human benchmark bundles",
        "training_design": {
            "source": "none",
            "claim": "no trainable latent recovery model is used in the M4.1 acceptance chain",
        },
        "scope_boundary": scope_boundary,
        "external_bundle_provenance": {
            "confidence_database": confidence_bundle,
            "iowa_gambling_task": igt_bundle,
        },
        "smoke_fixture_rejection": smoke_rejection,
        "confidence_benchmark": {
            **confidence_run,
            "metric_gate": _confidence_metric_gate(confidence_run),
        },
        "igt_benchmark": {
            **igt_run,
            "metric_gate": _igt_metric_gate(igt_run),
        },
        "evaluation_chain_audit": chain_audit,
        "repo_fixture_root": str(DEFAULT_BENCHMARK_ROOT),
    }


def run_minimal_external_task_validation() -> dict[str, Any]:
    return run_external_task_bundle_evaluation()


__all__ = [
    "downgraded_claims_inventory",
    "evaluation_chain_audit",
    "run_external_task_bundle_evaluation",
    "run_minimal_external_task_validation",
    "smoke_fixture_rejection_report",
]
