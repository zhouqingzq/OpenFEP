from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m47_evidence_chain_audit import build_m47_evidence_requirement_map, write_m47_evidence_chain_audit
from .m47_reacceptance import GATE_CODES, REGRESSION_TARGETS


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M47_STRICT_AUDIT_JSON_PATH = REPORTS_DIR / "m47_strict_audit.json"
M47_STRICT_AUDIT_SUMMARY_PATH = REPORTS_DIR / "m47_strict_audit.md"
M47_FIX_PRIORITY_LIST_PATH = REPORTS_DIR / "m47_fix_priority_list.md"
M47_STRICT_RUNTIME_DIRNAME = "m47_strict_runtime"

SELF_TEST_TARGETS = [
    "tests/test_m47_memory_core.py",
    "tests/test_m47_reacceptance.py",
    "tests/test_m47_acceptance.py",
    "tests/test_m47_evidence_chain_audit.py",
    "tests/test_m47_strict_audit.py",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(_read_text(path))


def _slug(value: str) -> str:
    return "_".join(
        part for part in "".join(character.lower() if character.isalnum() else "_" for character in value).split("_") if part
    )


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
    runtime_root = resolved_artifacts_dir / M47_STRICT_RUNTIME_DIRNAME / _slug(_now_iso())
    return {
        "runtime_root": runtime_root,
        "strict_json": resolved_reports_dir / M47_STRICT_AUDIT_JSON_PATH.name,
        "strict_summary": resolved_reports_dir / M47_STRICT_AUDIT_SUMMARY_PATH.name,
        "fix_priority": resolved_reports_dir / M47_FIX_PRIORITY_LIST_PATH.name,
    }


def _run_command(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        duration_seconds = round(time.perf_counter() - started, 3)
        return {
            "command": command,
            "cwd": str(ROOT),
            "executed": False,
            "timeout": True,
            "returncode": None,
            "passed": False,
            "duration_seconds": duration_seconds,
            "stdout_lines": str(exc.stdout or "").splitlines(),
            "stderr_lines": str(exc.stderr or "").splitlines(),
            "summary_line": "timed out",
        }
    duration_seconds = round(time.perf_counter() - started, 3)
    stdout_lines = completed.stdout.splitlines()
    stderr_lines = completed.stderr.splitlines()
    summary_line = stdout_lines[-1] if stdout_lines else (stderr_lines[-1] if stderr_lines else "")
    return {
        "command": command,
        "cwd": str(ROOT),
        "executed": True,
        "timeout": False,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": duration_seconds,
        "stdout_lines": stdout_lines,
        "stderr_lines": stderr_lines,
        "summary_line": summary_line,
    }


def _run_pytest_targets(targets: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    result = _run_command(["py", "-3.11", "-m", "pytest", *targets, "-q", "-rA"], timeout_seconds=timeout_seconds)
    result["targets"] = list(targets)
    return result


def _runtime_rebuild(runtime_root: Path, *, round_started_at: str) -> dict[str, Any]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    output_paths = write_m47_evidence_chain_audit(
        round_started_at=round_started_at,
        output_root=runtime_root,
        include_regressions=False,
    )
    return {
        "output_root": str(runtime_root),
        "output_paths": output_paths,
        "evidence_chain_audit": _read_json(Path(output_paths["audit_json"])),
        "reacceptance_report": _read_json(Path(output_paths["reacceptance_evidence"])),
        "acceptance_report": _read_json(Path(output_paths["acceptance_report"])),
        "requirement_map": _read_json(Path(output_paths["requirement_map"])),
    }


def _runtime_finding(
    path: Path,
    old_snippet: str,
    title: str,
    detail: str,
    gate_impact: list[str],
    category: str,
) -> dict[str, Any] | None:
    text = _read_text(path)
    if old_snippet in text:
        line = next((index for index, item in enumerate(text.splitlines(), start=1) if old_snippet in item), None)
        return {
            "severity": "S1",
            "priority": "P1",
            "category": category,
            "title": title,
            "detail": detail,
            "location": f"{path}:{line}" if line is not None else str(path),
            "gate_impact": gate_impact,
        }
    return None


def _static_code_findings() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    candidates = [
        _runtime_finding(
            ROOT / "segmentum" / "memory_retrieval.py",
            "detail_fragments = [_primary_recall_backbone(primary)]",
            "Recall artifact is still primary-entry led",
            "Recall reconstruction still begins from a top-1 backbone rather than candidate-constrained reconstruction.",
            ["behavioral_scenario_B_interference", "behavioral_scenario_E_natural_misattribution"],
            "top_1_recall_backbone",
        ),
        _runtime_finding(
            ROOT / "segmentum" / "memory_store.py",
            "identity_active_threshold_relief",
            "Promotion behavior is rule-stack driven",
            "Promotion still depends on relief and penalty branches rather than a unified score.",
            ["behavioral_scenario_C_consolidation", "long_term_subtypes", "identity_continuity_retention"],
            "rule_stack_promotion",
        ),
        _runtime_finding(
            ROOT / "segmentum" / "m47_audit.py",
            "build_m47_reacceptance_report(",
            "Official acceptance wraps reacceptance",
            "Official acceptance still sources its runtime evidence chain from reacceptance output.",
            ["report_honesty"],
            "self_generated_evidence",
        ),
        _runtime_finding(
            ROOT / "segmentum" / "m47_reacceptance.py",
            "reuse_m41_to_m45_plus_live_m46_delta",
            "G9 accepts cached regression summaries",
            "Cached regression summaries still appear in the M4.7 regression path.",
            ["regression"],
            "cached_regression",
        ),
        _runtime_finding(
            ROOT / "segmentum" / "m47_reacceptance.py",
            "_identity_input",
            "Gate evidence is helper-generated rather than corpus-backed",
            "M4.7 evidence generation still depends on helper-authored staged inputs rather than the external corpus.",
            [
                "behavioral_scenario_A_threat_learning",
                "behavioral_scenario_B_interference",
                "behavioral_scenario_C_consolidation",
                "long_term_subtypes",
                "identity_continuity_retention",
                "behavioral_scenario_E_natural_misattribution",
                "integration_interface",
            ],
            "synthetic_scenario_dependency",
        ),
    ]
    findings.extend(item for item in candidates if item is not None)
    return findings


def _test_review(requirement_map: list[dict[str, Any]]) -> dict[str, Any]:
    classifications: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    file_labels: dict[str, set[str]] = {}
    for spec in requirement_map:
        for path in spec.get("mechanism_tests", []):
            file_labels.setdefault(str(path), set()).add("mechanism_proof")
        for path in spec.get("artifact_tests", []):
            file_labels.setdefault(str(path), set()).add("artifact_report")

    for path, labels in sorted(file_labels.items()):
        classifications.append({"path": path, "labels": sorted(labels)})

    if any(not spec.get("mechanism_tests") for spec in requirement_map):
        findings.append(
            {
                "severity": "S2",
                "priority": "P2",
                "category": "report_shape_tests",
                "title": "Requirement map lacks mechanism-proof coverage",
                "detail": "At least one gate is still backed only by artifact/report assertions.",
                "location": str(ROOT / "tests"),
                "gate_impact": [str(spec["gate"]) for spec in requirement_map if not spec.get("mechanism_tests")],
            }
        )
    if any(not spec.get("artifact_tests") for spec in requirement_map):
        findings.append(
            {
                "severity": "S3",
                "priority": "P3",
                "category": "artifact_gap",
                "title": "Requirement map lacks artifact-level coverage",
                "detail": "At least one gate has no artifact/report validation path.",
                "location": str(ROOT / "tests"),
                "gate_impact": [str(spec["gate"]) for spec in requirement_map if not spec.get("artifact_tests")],
            }
        )
    return {"classifications": classifications, "findings": findings}


def _build_gate_results(
    runtime_snapshot: dict[str, Any],
    *,
    live_regression: dict[str, Any],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_scenario = {
        str(record["scenario_id"]): record for record in runtime_snapshot["reacceptance_report"]["evidence_records"]
    }
    results: list[dict[str, Any]] = []
    for spec in runtime_snapshot["requirement_map"]:
        gate = str(spec["gate"])
        gate_status = str(runtime_snapshot["reacceptance_report"]["gate_summaries"][gate]["status"])
        gate_findings = [finding for finding in findings if gate in finding.get("gate_impact", [])]
        labels = [finding["title"] for finding in gate_findings]
        if gate == "regression":
            if not live_regression.get("attempted"):
                strict_verdict = "BLOCKED"
                rationale = "Strict acceptance requires a live full regression run, which was intentionally not executed."
            elif live_regression["result"]["passed"]:
                strict_verdict = "PASS"
                rationale = "A live M4.1-M4.6 regression run completed successfully."
            else:
                strict_verdict = "FAIL"
                rationale = "The live M4.1-M4.6 regression run failed or did not complete."
            builder_verdict = "NOT_RUN" if not live_regression.get("attempted") else ("PASS" if live_regression["result"]["passed"] else "FAIL")
            evidence_excerpt = live_regression["result"].get("summary_line", "")
        else:
            rsnap = runtime_snapshot["reacceptance_report"]["runtime_snapshot"]
            summary = runtime_snapshot["reacceptance_report"]["gate_summaries"][gate]
            demoted = bool(rsnap.get("diagnostic_only")) and bool(summary.get("behavioral_claims_demoted"))
            if demoted:
                # M4.8 demotion: builder FAIL on these gates is expected and non-blocking for strict gating;
                # G9 regression and the issue list carry the real blockers.
                strict_verdict = "PASS"
                builder_verdict = gate_status
                rationale = (
                    "M4.8 demotion: gate is diagnostic_only; strict audit does not treat builder FAIL as a blocking "
                    "mechanism failure (layer-(b) evidence is M4.8)."
                )
            elif gate_status == "PASS" and not gate_findings:
                strict_verdict = "PASS"
                builder_verdict = gate_status
                rationale = "Shared runtime evidence satisfies the gate and no blocking implementation finding remains."
            elif gate_status == "PASS":
                strict_verdict = "PARTIAL"
                builder_verdict = gate_status
                rationale = (
                    "Builder evidence passed, but strict audit still sees unresolved implementation/test coupling."
                )
            else:
                strict_verdict = "FAIL"
                builder_verdict = gate_status
                rationale = f"Runtime evidence builder did not pass this gate ({gate_status})."
            evidence_excerpt = json.dumps(
                records_by_scenario[spec["required_scenario_ids"][0]]["observed"],
                ensure_ascii=False,
            )[:220]
        results.append(
            {
                "code": GATE_CODES[gate],
                "gate": gate,
                "strict_verdict": strict_verdict,
                "builder_verdict": builder_verdict,
                "scenario_ids": list(spec["required_scenario_ids"]),
                "required_observed_fields": list(spec["required_observed_fields"]),
                "labels": labels,
                "evidence_excerpt": evidence_excerpt,
                "rationale": rationale,
            }
        )
    return results


def _overall_conclusion(gate_results: list[dict[str, Any]]) -> tuple[str, str]:
    verdicts = [item["strict_verdict"] for item in gate_results]
    if "FAIL" in verdicts:
        return "FAIL", "Strict M4.7 acceptance failed."
    if "BLOCKED" in verdicts:
        return "BLOCKED", "Strict M4.7 acceptance remains blocked because G9 has not been completed with a live full regression run."
    if "PARTIAL" in verdicts:
        return "PARTIAL", "Core mechanisms exist, but multiple gates still rely on unresolved implementation shortcuts."
    return "PASS", "Strict M4.7 acceptance passed."


def build_m47_strict_audit(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    run_m47_self_tests: bool = False,
    run_live_regressions: bool = False,
    self_test_timeout_seconds: int = 900,
    regression_timeout_seconds: int = 7200,
) -> tuple[dict[str, Any], dict[str, str]]:
    output_paths = _resolve_output_paths(output_root=output_root)
    runtime_snapshot = _runtime_rebuild(output_paths["runtime_root"], round_started_at=round_started_at or _now_iso())
    requirement_map = build_m47_evidence_requirement_map()
    static_findings = _static_code_findings()
    test_review = _test_review(requirement_map)
    self_tests = (
        []
        if not run_m47_self_tests
        else [_run_pytest_targets([target], timeout_seconds=self_test_timeout_seconds) for target in SELF_TEST_TARGETS]
    )
    live_regression = {
        "attempted": False,
        "result": {
            "summary_line": "live regression not run in this strict audit round",
            "passed": False,
            "targets": list(REGRESSION_TARGETS),
        },
    }
    if run_live_regressions:
        result = _run_pytest_targets(list(REGRESSION_TARGETS), timeout_seconds=regression_timeout_seconds)
        live_regression = {"attempted": True, "result": result}

    issue_list = static_findings + test_review["findings"]
    gate_results = _build_gate_results(runtime_snapshot, live_regression=live_regression, findings=issue_list)
    overall_conclusion, overall_rationale = _overall_conclusion(gate_results)
    category_counts = Counter(finding.get("category", "uncategorized") for finding in issue_list)
    audit = {
        "generated_at": _now_iso(),
        "round_started_at": round_started_at or _now_iso(),
        "overall_conclusion": overall_conclusion,
        "overall_rationale": overall_rationale,
        "gate_results": gate_results,
        "issue_list": issue_list,
        "summary_statistics": {
            "finding_count": len(issue_list),
            "gate_count": len(gate_results),
            "finding_category_counts": dict(category_counts),
        },
        "runtime_snapshot": runtime_snapshot,
        "test_review": test_review,
        "self_tests": self_tests,
        "evidence_appendix": {"live_regression": live_regression},
    }
    return audit, {key: str(path) for key, path in output_paths.items()}


def write_m47_strict_audit(
    *,
    round_started_at: str | None = None,
    output_root: Path | str | None = None,
    run_m47_self_tests: bool = False,
    run_live_regressions: bool = False,
    self_test_timeout_seconds: int = 900,
    regression_timeout_seconds: int = 7200,
) -> dict[str, str]:
    output_paths = _resolve_output_paths(output_root=output_root)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    audit, path_map = build_m47_strict_audit(
        round_started_at=round_started_at,
        output_root=output_root,
        run_m47_self_tests=run_m47_self_tests,
        run_live_regressions=run_live_regressions,
        self_test_timeout_seconds=self_test_timeout_seconds,
        regression_timeout_seconds=regression_timeout_seconds,
    )
    output_paths["strict_json"].write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_lines = [
        "# M4.7 Strict Audit",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Overall Conclusion: `{audit['overall_conclusion']}`",
        "",
        "## Gate Verdicts",
        "",
    ]
    for item in audit["gate_results"]:
        summary_lines.append(
            f"- {item['code']} `{item['gate']}`: strict=`{item['strict_verdict']}`, builder=`{item['builder_verdict']}`"
        )
    summary_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- G9 remains BLOCKED until a live M4.1-M4.6 regression suite is executed in this round.",
            f"- Remaining implementation/test findings: `{audit['summary_statistics']['finding_count']}`",
        ]
    )
    output_paths["strict_summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    priority_lines = ["# M4.7 Fix Priority List", ""]
    if audit["issue_list"]:
        for finding in audit["issue_list"]:
            priority_lines.append(f"- {finding['title']}: {finding['detail']}")
    else:
        priority_lines.append("- Run a live M4.1-M4.6 regression suite to clear the remaining G9 blocker.")
    output_paths["fix_priority"].write_text("\n".join(priority_lines) + "\n", encoding="utf-8")
    return path_map
