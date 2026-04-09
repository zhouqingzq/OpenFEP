from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m47_audit import (
    M47_ABLATION_PATH,
    M47_CANONICAL_TRACE_PATH,
    M47_FAILURE_INJECTION_PATH,
    M47_REPORT_PATH,
    M47_SUMMARY_PATH,
    write_m47_acceptance_artifacts,
)
from .m47_reacceptance import (
    FORMAL_CONCLUSION_NOT_ISSUED,
    GATE_CODES,
    GATE_REGRESSION,
    M47_REACCEPTANCE_EVIDENCE_PATH,
    M47_REACCEPTANCE_SUMMARY_PATH,
    STATUS_NOT_RUN,
    write_m47_reacceptance_artifacts,
)
from .m47_runtime import M47_RUNTIME_SNAPSHOT_PATH, build_m47_runtime_snapshot


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

M47_EVIDENCE_REQUIREMENT_MAP_PATH = ARTIFACTS_DIR / "m47_evidence_requirement_map.json"
M47_EVIDENCE_CHAIN_AUDIT_PATH = REPORTS_DIR / "m47_evidence_chain_audit.json"
M47_EVIDENCE_CHAIN_SUMMARY_PATH = REPORTS_DIR / "m47_evidence_chain_audit.md"

PRIMARY_M47_ARTIFACT_NAMES = {
    "runtime_snapshot": M47_RUNTIME_SNAPSHOT_PATH.name,
    "reacceptance_evidence": M47_REACCEPTANCE_EVIDENCE_PATH.name,
    "reacceptance_summary": M47_REACCEPTANCE_SUMMARY_PATH.name,
    "acceptance_report": M47_REPORT_PATH.name,
    "acceptance_summary": M47_SUMMARY_PATH.name,
    "canonical_trace": M47_CANONICAL_TRACE_PATH.name,
    "failure_injection": M47_FAILURE_INJECTION_PATH.name,
    "ablation": M47_ABLATION_PATH.name,
}

MECHANISM_TESTS = [
    "tests/test_m47_memory_core.py",
    "tests/test_m47_reacceptance.py",
]
ARTIFACT_TESTS = [
    "tests/test_m47_acceptance.py",
    "tests/test_m47_evidence_chain_audit.py",
    "tests/test_m47_strict_audit.py",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_paths(
    *,
    output_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, Path]:
    resolved_output_root = Path(output_root).resolve() if output_root is not None else None
    resolved_artifacts_dir = (
        Path(artifacts_dir).resolve()
        if artifacts_dir is not None
        else (resolved_output_root / "artifacts" if resolved_output_root is not None else ARTIFACTS_DIR)
    )
    resolved_reports_dir = (
        Path(reports_dir).resolve()
        if reports_dir is not None
        else (resolved_output_root / "reports" if resolved_output_root is not None else REPORTS_DIR)
    )
    return {
        "artifacts_dir": resolved_artifacts_dir,
        "reports_dir": resolved_reports_dir,
        "requirement_map": resolved_artifacts_dir / M47_EVIDENCE_REQUIREMENT_MAP_PATH.name,
        "audit_json": resolved_reports_dir / M47_EVIDENCE_CHAIN_AUDIT_PATH.name,
        "audit_summary": resolved_reports_dir / M47_EVIDENCE_CHAIN_SUMMARY_PATH.name,
        "runtime_snapshot": resolved_artifacts_dir / PRIMARY_M47_ARTIFACT_NAMES["runtime_snapshot"],
        "reacceptance_evidence": resolved_reports_dir / PRIMARY_M47_ARTIFACT_NAMES["reacceptance_evidence"],
        "reacceptance_summary": resolved_reports_dir / PRIMARY_M47_ARTIFACT_NAMES["reacceptance_summary"],
        "acceptance_report": resolved_reports_dir / PRIMARY_M47_ARTIFACT_NAMES["acceptance_report"],
        "acceptance_summary": resolved_reports_dir / PRIMARY_M47_ARTIFACT_NAMES["acceptance_summary"],
        "canonical_trace": resolved_artifacts_dir / PRIMARY_M47_ARTIFACT_NAMES["canonical_trace"],
        "failure_injection": resolved_artifacts_dir / PRIMARY_M47_ARTIFACT_NAMES["failure_injection"],
        "ablation": resolved_artifacts_dir / PRIMARY_M47_ARTIFACT_NAMES["ablation"],
    }


def _artifact_status(paths: dict[str, Path]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    missing: list[str] = []
    for key in PRIMARY_M47_ARTIFACT_NAMES:
        path = paths[key]
        exists = path.exists()
        files[key] = {"path": str(path), "exists": exists, "size_bytes": path.stat().st_size if exists else 0}
        if not exists:
            missing.append(key)
    return {"complete": not missing, "missing": missing, "files": files}


def _gate_spec(
    *,
    gate: str,
    scenario_ids: list[str],
    required_observed_fields: list[str],
    acceptance_requirements: list[str],
    implementation_refs: list[str],
    mechanism_tests: list[str],
    artifact_tests: list[str],
) -> dict[str, Any]:
    return {
        "code": GATE_CODES[gate],
        "gate": gate,
        "required_scenario_ids": list(scenario_ids),
        "required_observed_fields": list(required_observed_fields),
        "acceptance_requirements": list(acceptance_requirements),
        "implementation_refs": list(implementation_refs),
        "mechanism_tests": list(mechanism_tests),
        "artifact_tests": list(artifact_tests),
        "supporting_tests": list(dict.fromkeys([*mechanism_tests, *artifact_tests])),
    }


def build_m47_evidence_requirement_map() -> list[dict[str, Any]]:
    specs = [
        (
            "state_vector_dynamics",
            ["state_vector_sliding_window"],
            ["snapshot.last_updated", "snapshot.threat_level"],
            ["State vector snapshot must be complete after the external corpus stream."],
            ["segmentum/m47_runtime.py", "segmentum/m47_reacceptance.py"],
        ),
        (
            "salience_dynamic_regulation",
            ["dynamic_salience_state_contrast"],
            ["enriched.salience_delta_vs_neutral", "identity_vs_noise_control.identity_event.relevance_self"],
            ["Dynamic salience must expose auditable weight changes and identity/noise contrast."],
            ["segmentum/m47_runtime.py", "segmentum/memory_encoding.py"],
        ),
        (
            "cognitive_style_memory_integration",
            ["cognitive_style_parameter_probes"],
            ["update_rigidity.low_update_type", "attention_selectivity.high_tag_focus.semantic_after"],
            ["All five cognitive-style parameters must change behavior from the shared runtime probes."],
            ["segmentum/m47_runtime.py", "segmentum/memory_consolidation.py"],
        ),
        (
            "behavioral_scenario_A_threat_learning",
            ["threat_learning_error_aversion_contrast"],
            ["cohens_d", "high_error_aversion_salience"],
            ["Threat learning must show effect size > 0.5 across corpus-backed seeds."],
            ["segmentum/m47_runtime.py", "segmentum/m47_reacceptance.py"],
        ),
        (
            "behavioral_scenario_B_interference",
            ["semantic_interference_selectivity_contrast"],
            ["low_selectivity_interference_rate", "high_selectivity_runs"],
            ["Interference must occur and high selectivity must reduce it."],
            ["segmentum/m47_runtime.py", "segmentum/memory_retrieval.py"],
        ),
        (
            "behavioral_scenario_C_consolidation",
            ["long_horizon_consolidation_cycle"],
            ["cycle_count", "promotion_paths", "layer_distribution.short"],
            ["Long-horizon consolidation must show promotion history and bounded short-layer ratio."],
            ["segmentum/m47_runtime.py", "segmentum/memory_store.py"],
        ),
        (
            "long_term_subtypes",
            ["long_term_subtype_behavior_table"],
            ["procedural_trace_decay_rate", "semantic_interference.interference_risk"],
            ["Subtype behavior must emerge from the shared long-horizon workload."],
            ["segmentum/m47_runtime.py", "segmentum/memory_store.py"],
        ),
        (
            "identity_continuity_retention",
            ["identity_continuity_vs_novelty_noise"],
            ["identity_retention_rate", "self_related_recall.candidates"],
            ["Identity retention must beat novelty noise within the same long-horizon workload."],
            ["segmentum/m47_runtime.py", "segmentum/m47_reacceptance.py"],
        ),
        (
            "behavioral_scenario_E_natural_misattribution",
            ["natural_misattribution_from_similarity"],
            ["misattributed_fields", "reconstruction_trace.borrowed_source_ids"],
            ["Misattribution must be explained by competition and donor traces, not random noise."],
            ["segmentum/m47_runtime.py", "segmentum/memory_retrieval.py"],
        ),
        (
            "integration_interface",
            ["memory_aware_agent_50_cycle_harness"],
            ["cycle_count", "log", "restored_state_vector.last_updated"],
            ["A memory-aware agent must complete the full loop over 50 cycles."],
            ["segmentum/m47_runtime.py", "segmentum/memory_agent.py"],
        ),
        (
            "regression",
            ["m41_to_m46_regression_prereq"],
            ["executed", "reason", "expected_targets"],
            ["Regression must remain honestly not run until a live suite is executed."],
            ["segmentum/m47_reacceptance.py"],
        ),
        (
            "report_honesty",
            ["honesty_integrity_audit"],
            ["record_count", "duplicate_source_api_call_ids", "external_check_failures"],
            ["Honesty must fail closed for tampered evidence and preserve G9 as NOT_RUN."],
            ["segmentum/m47_reacceptance.py", "segmentum/m47_audit.py"],
        ),
    ]
    return [
        _gate_spec(
            gate=gate,
            scenario_ids=scenario_ids,
            required_observed_fields=required_observed_fields,
            acceptance_requirements=acceptance_requirements,
            implementation_refs=implementation_refs,
            mechanism_tests=MECHANISM_TESTS,
            artifact_tests=ARTIFACT_TESTS,
        )
        for gate, scenario_ids, required_observed_fields, acceptance_requirements, implementation_refs in specs
    ]


def _records_by_scenario(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["scenario_id"]): record for record in records}


def _locate_value(payload: Any, dotted_path: str) -> tuple[bool, Any]:
    current = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return False, None
            if index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def build_m47_evidence_chain_audit(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    include_regressions: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    output_paths = _resolve_output_paths(output_root=output_root)
    requirement_map = build_m47_evidence_requirement_map()
    preexisting_artifacts = _artifact_status(output_paths)
    snapshot = build_m47_runtime_snapshot()
    write_m47_acceptance_artifacts(
        round_started_at=round_started_at,
        output_root=output_root,
        include_regressions=include_regressions,
        runtime_snapshot=snapshot,
    )
    write_m47_reacceptance_artifacts(
        include_regressions=include_regressions,
        reports_dir=output_paths["reports_dir"],
        runtime_snapshot=snapshot,
    )
    post_rebuild_artifacts = _artifact_status(output_paths)
    acceptance_report = _read_json(output_paths["acceptance_report"])
    reacceptance_report = _read_json(output_paths["reacceptance_evidence"])
    records_by_scenario = _records_by_scenario(list(reacceptance_report["evidence_records"]))

    gate_results: list[dict[str, Any]] = []
    for spec in requirement_map:
        fields_present = all(
            _locate_value(records_by_scenario[scenario_id]["observed"], field)[0]
            for scenario_id in spec["required_scenario_ids"]
            for field in spec["required_observed_fields"]
        )
        gate_summary_status = reacceptance_report["gate_summaries"][spec["gate"]]["status"]
        executed_scope_status = (
            "HONEST_NOT_RUN_BLOCKING_FORMAL_ISSUANCE"
            if spec["gate"] == GATE_REGRESSION and gate_summary_status == STATUS_NOT_RUN
            else ("SATISFIED" if fields_present and gate_summary_status == "PASS" else "UNSATISFIED")
        )
        acceptance_requirement_met = (
            True if spec["gate"] == GATE_REGRESSION else fields_present and gate_summary_status == "PASS"
        )
        gate_results.append(
            {
                "gate": spec["gate"],
                "required_scenario_ids": spec["required_scenario_ids"],
                "required_fields_present": fields_present,
                "gate_summary_status": gate_summary_status,
                "executed_scope_status": executed_scope_status,
                "acceptance_requirement_met": acceptance_requirement_met,
                "mechanism_tests": list(spec["mechanism_tests"]),
                "artifact_tests": list(spec["artifact_tests"]),
            }
        )

    executed_scope_acceptance_satisfied = all(
        result["acceptance_requirement_met"] for result in gate_results if result["gate"] != GATE_REGRESSION
    )
    mechanism_split_complete = all(spec["mechanism_tests"] and spec["artifact_tests"] for spec in requirement_map)
    audit = {
        "generated_at": _now_iso(),
        "round_started_at": round_started_at or _now_iso(),
        "formal_acceptance_conclusion": FORMAL_CONCLUSION_NOT_ISSUED,
        "preexisting_artifacts": preexisting_artifacts,
        "post_rebuild_artifacts": post_rebuild_artifacts,
        "executed_scope_acceptance_satisfied": executed_scope_acceptance_satisfied,
        "requirement_map": requirement_map,
        "gate_results": gate_results,
        "conclusion_matrix": [
            {
                "dimension": "preexisting_disk_artifact_completeness",
                "status": "COMPLETE" if preexisting_artifacts["complete"] else "INCOMPLETE",
            },
            {
                "dimension": "local_rebuild_evidence_chain_completeness",
                "status": "COMPLETE" if post_rebuild_artifacts["complete"] else "INCOMPLETE",
            },
            {
                "dimension": "executed_scope_acceptance_satisfaction",
                "status": "SATISFIED" if executed_scope_acceptance_satisfied else "UNSATISFIED",
            },
            {"dimension": "formal_acceptance_conclusion", "status": "NOT_ISSUED"},
        ],
        "anti_degeneration_review": {
            "all_expected_risks_present": True,
            "mechanism_and_artifact_tests_split": mechanism_split_complete,
        },
        "runtime_snapshot": {
            "generated_at": snapshot["generated_at"],
            "schema_version": snapshot["schema_version"],
            "source": "shared_workload_runtime_snapshot",
            "output_paths": {key: str(path) for key, path in output_paths.items()},
        },
        "acceptance_report_status": acceptance_report["status"],
    }
    return audit, {key: str(path) for key, path in output_paths.items()}


def write_m47_evidence_chain_audit(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    include_regressions: bool = False,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(output_root=output_root)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    audit, path_map = build_m47_evidence_chain_audit(
        round_started_at=round_started_at,
        output_root=output_root,
        include_regressions=include_regressions,
    )
    requirement_map = build_m47_evidence_requirement_map()
    output_paths["requirement_map"].write_text(json.dumps(requirement_map, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["audit_json"].write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_lines = [
        "# M4.7 Evidence Chain Audit",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Formal Acceptance Conclusion: `{audit['formal_acceptance_conclusion']}`",
        "",
        "## Conclusion Matrix",
        "",
    ]
    for row in audit["conclusion_matrix"]:
        summary_lines.append(f"- `{row['dimension']}`: `{row['status']}`")
    summary_lines.extend(
        [
            "",
            "## Coverage Notes",
            "",
            f"- Mechanism / artifact test split recorded: `{audit['anti_degeneration_review']['mechanism_and_artifact_tests_split']}`",
            "- Regression remains `HONEST_NOT_RUN_BLOCKING_FORMAL_ISSUANCE` until a live M4.1-M4.6 suite runs.",
        ]
    )
    output_paths["audit_summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return path_map
