from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M232_SPEC_PATH = REPORTS_DIR / "m232_milestone_spec.md"
M232_PREPARATION_PATH = REPORTS_DIR / "m232_strict_audit_preparation.md"
M232_TRACE_PATH = ARTIFACTS_DIR / "m232_claim_reconciliation_trace.jsonl"
M232_ABLATION_PATH = ARTIFACTS_DIR / "m232_claim_reconciliation_ablation.json"
M232_STRESS_PATH = ARTIFACTS_DIR / "m232_claim_reconciliation_stress.json"
M232_REPORT_PATH = REPORTS_DIR / "m232_acceptance_report.json"
M232_SUMMARY_PATH = REPORTS_DIR / "m232_acceptance_summary.md"

SEED_SET: tuple[int, ...] = (232, 464)
M232_TESTS: tuple[str, ...] = (
    "tests/test_m232_claim_reconciliation.py",
    "tests/test_m232_claim_containment.py",
    "tests/test_m232_acceptance.py",
)
M232_REGRESSIONS: tuple[str, ...] = (
    "tests/test_m229_acceptance.py",
    "tests/test_m230_acceptance.py",
    "tests/test_m231_acceptance.py",
    "tests/test_narrative_evolution.py",
)
M232_GATES: tuple[str, ...] = (
    "schema",
    "determinism",
    "causality",
    "ablation",
    "stress",
    "mixed_state_visibility",
    "regression",
    "artifact_freshness",
)


def preparation_manifest() -> dict[str, object]:
    return {
        "milestone_id": "M2.32",
        "title": "Claim-Level Narrative Reconciliation Hardening",
        "status": "PREPARATION_READY",
        "assumption_source": str(M232_SPEC_PATH),
        "seed_set": list(SEED_SET),
        "artifacts": {
            "specification": str(M232_SPEC_PATH),
            "preparation": str(M232_PREPARATION_PATH),
            "canonical_trace": str(M232_TRACE_PATH),
            "ablation": str(M232_ABLATION_PATH),
            "stress": str(M232_STRESS_PATH),
            "report": str(M232_REPORT_PATH),
            "summary": str(M232_SUMMARY_PATH),
        },
        "tests": {
            "milestone": list(M232_TESTS),
            "regressions": list(M232_REGRESSIONS),
        },
        "gates": list(M232_GATES),
    }
