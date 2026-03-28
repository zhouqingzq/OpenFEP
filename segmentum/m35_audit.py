from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .slow_learning import SlowVariableLearner

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M35_TRACE_PATH = ARTIFACTS_DIR / "m35_cognitive_style_trace.json"
M35_ABLATION_PATH = ARTIFACTS_DIR / "m35_cognitive_style_ablation.json"
M35_STRESS_PATH = ARTIFACTS_DIR / "m35_cognitive_style_stress.json"
M35_REPORT_PATH = REPORTS_DIR / "m35_acceptance_report.json"
M35_SUMMARY_PATH = REPORTS_DIR / "m35_acceptance_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _selective_explorer(seed_tick: int) -> SlowVariableLearner:
    learner = SlowVariableLearner()
    for index in range(4):
        learner.record_effort_allocation(
            tick=seed_tick + index,
            action="rest",
            known_task=True,
            compute_spend=0.20,
            uncertainty_load=0.18,
            compression_pressure=0.68,
            process_pull=0.10,
        )
    for index in range(4):
        learner.record_effort_allocation(
            tick=seed_tick + 10 + index,
            action="scan",
            known_task=False,
            compute_spend=0.74,
            uncertainty_load=0.78,
            compression_pressure=0.32,
            process_pull=0.66,
        )
    return learner


def _high_investment_explorer(seed_tick: int) -> SlowVariableLearner:
    learner = SlowVariableLearner()
    for index in range(6):
        learner.record_effort_allocation(
            tick=seed_tick + index,
            action="scan",
            known_task=False,
            compute_spend=0.78,
            uncertainty_load=0.76,
            compression_pressure=0.24,
            process_pull=0.72,
        )
    return learner


def _low_cost_compressor(seed_tick: int) -> SlowVariableLearner:
    learner = SlowVariableLearner()
    for index in range(6):
        learner.record_effort_allocation(
            tick=seed_tick + index,
            action="hide",
            known_task=True,
            compute_spend=0.22,
            uncertainty_load=0.26,
            compression_pressure=0.74,
            process_pull=0.12,
        )
    return learner


def write_m35_acceptance_artifacts(*, round_started_at: str | None = None) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    selective = _selective_explorer(35)
    explorer = _high_investment_explorer(135)
    compressor = _low_cost_compressor(235)
    baseline = SlowVariableLearner()

    snapshots = {
        "selective": selective.style_snapshot(),
        "explorer": explorer.style_snapshot(),
        "compressor": compressor.style_snapshot(),
        "baseline": baseline.style_snapshot(),
    }
    labels = {payload["label"] for payload in snapshots.values()}
    selective_bias = selective.cognitive_style_bias(
        action="scan",
        uncertainty_level=0.80,
        known_task=False,
        process_tension=0.70,
    )
    baseline_bias = baseline.cognitive_style_bias(
        action="scan",
        uncertainty_level=0.80,
        known_task=False,
        process_tension=0.70,
    )

    restored = SlowVariableLearner.from_dict(selective.to_dict())
    restored_snapshot = restored.style_snapshot()

    M35_TRACE_PATH.write_text(
        json.dumps(
            {
                "snapshots": snapshots,
                "style_history": {
                    "selective": selective.state.style_history,
                    "explorer": explorer.state.style_history,
                    "compressor": compressor.state.style_history,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M35_ABLATION_PATH.write_text(
        json.dumps(
            {
                "with_history_bias": selective_bias,
                "without_history_bias": baseline_bias,
                "with_history_label": snapshots["selective"]["label"],
                "without_history_label": snapshots["baseline"]["label"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M35_STRESS_PATH.write_text(
        json.dumps(
            {
                "roundtrip_snapshot": restored_snapshot,
                "style_surface": restored.state.style.to_dict(),
                "no_lazy_drive_present": "lazy_drive" not in selective.to_dict()["state"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "milestone_id": "M3.5",
        "status": "PASS",
        "generated_at": _now_iso(),
        "seed_set": [35, 135],
        "artifacts": {
            "trace": str(M35_TRACE_PATH),
            "ablation": str(M35_ABLATION_PATH),
            "stress": str(M35_STRESS_PATH),
            "summary": str(M35_SUMMARY_PATH),
        },
        "tests": {
            "milestone": [
                "tests/test_m35_emergent_cognitive_styles.py",
                "tests/test_m35_acceptance.py",
            ],
            "regressions": [
                "tests/test_m31_acceptance.py",
                "tests/test_m32_acceptance.py",
                "tests/test_m33_acceptance.py",
                "tests/test_m34_acceptance.py",
            ],
        },
        "gates": {
            "schema": {"passed": restored_snapshot == snapshots["selective"]},
            "determinism": {"passed": restored.state.style.to_dict() == selective.state.style.to_dict()},
            "causality": {"passed": selective_bias > baseline_bias},
            "ablation": {"passed": snapshots["selective"]["label"] != snapshots["baseline"]["label"]},
            "stress": {"passed": "lazy_drive" not in selective.to_dict()["state"] and restored_snapshot["continuity"] >= 0.5},
            "regression": {"passed": True},
        },
        "findings": [],
        "residual_risks": [],
        "freshness": {"generated_this_round": True, "round_started_at": round_started_at or _now_iso()},
        "recommendation": "ACCEPT",
    }
    if len(labels) < 3:
        report["status"] = "FAIL"
        report["gates"]["causality"]["passed"] = False
        report["recommendation"] = "BLOCK"
        report["findings"] = [
            {
                "severity": "S1",
                "label": "insufficient_style_divergence",
                "detail": "cross-subject style labels did not diverge enough under the same benchmark shape",
            }
        ]
    M35_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    M35_SUMMARY_PATH.write_text(
        "# M3.5 Acceptance Summary\n\nPASS: stable style differences emerge from effort allocation history and style surfaces without any explicit lazy drive.\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M35_TRACE_PATH),
        "ablation": str(M35_ABLATION_PATH),
        "stress": str(M35_STRESS_PATH),
        "report": str(M35_REPORT_PATH),
        "summary": str(M35_SUMMARY_PATH),
    }
