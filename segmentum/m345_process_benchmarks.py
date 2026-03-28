from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .drives import DriveSystem, ProcessValenceState
from .inquiry_scheduler import InquiryCandidate, process_valence_priority_adjustment
from .slow_learning import SlowVariableLearner

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

M345_TRACE_PATH = ARTIFACTS_DIR / "m345_process_benchmark_trace.json"
M345_ABLATION_PATH = ARTIFACTS_DIR / "m345_process_benchmark_ablation.json"
M345_REPORT_PATH = REPORTS_DIR / "m345_process_benchmark_report.json"
M345_SUMMARY_PATH = REPORTS_DIR / "m345_process_benchmark_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class ProcessBenchmarkMetrics:
    focus_bonus_gain: float
    closure_penalty_gain: float
    boredom_bonus_gain: float
    scan_reorientation_gain: float
    style_label_diversity: int
    selective_unknown_scan_gain: float
    explorer_unknown_scan_gain: float
    compressor_known_rest_gain: float
    style_continuity_mean: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = _round(value)
        return payload


def _focus_candidate() -> InquiryCandidate:
    return InquiryCandidate(
        candidate_id="benchmark:focus",
        source_subsystem="m345",
        linked_target_id="unknown:m345-focus",
        linked_unknown_id="unknown:m345-focus",
        target_channels=("danger", "social"),
        action_name="scan",
        uncertainty_level=0.64,
        decision_relevance=0.70,
        expected_information_gain=0.74,
        falsification_importance=0.60,
        practical_risk=0.18,
        cost=0.26,
        urgency=0.62,
        active=True,
        summary="persistent unresolved focus",
    )


def _novel_candidate() -> InquiryCandidate:
    return InquiryCandidate(
        candidate_id="benchmark:novel",
        source_subsystem="m345",
        linked_target_id="unknown:m345-novel",
        linked_unknown_id="unknown:m345-novel",
        target_channels=("novelty", "social"),
        action_name="seek_contact",
        uncertainty_level=0.56,
        decision_relevance=0.58,
        expected_information_gain=0.68,
        falsification_importance=0.46,
        practical_risk=0.12,
        cost=0.24,
        urgency=0.44,
        active=True,
        summary="novel inquiry after closure",
    )


def _process_states() -> tuple[DriveSystem, ProcessValenceState, ProcessValenceState, ProcessValenceState]:
    drives = DriveSystem()
    wanting = ProcessValenceState()
    for _ in range(3):
        wanting = drives.update_process_valence(
            current_focus_id="unknown:m345-focus",
            unresolved_targets={"unknown:m345-focus"},
            focus_strength=0.74,
            maintenance_pressure=0.18,
        )
    closure = drives.update_process_valence(
        current_focus_id="",
        unresolved_targets=set(),
        focus_strength=0.0,
        maintenance_pressure=0.18,
        closure_signal=1.0,
    )
    boredom = closure
    for _ in range(4):
        boredom = drives.update_process_valence(
            current_focus_id="",
            unresolved_targets=set(),
            focus_strength=0.0,
            maintenance_pressure=0.08,
        )
    return drives, wanting, closure, boredom


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
            uncertainty_load=0.80,
            compression_pressure=0.32,
            process_pull=0.68,
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


def _compressor_surface(seed_tick: int) -> SlowVariableLearner:
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


def _style_probe(learner: SlowVariableLearner, *, action: str, known_task: bool, uncertainty_level: float, process_tension: float) -> float:
    return learner.cognitive_style_bias(
        action=action,
        uncertainty_level=uncertainty_level,
        known_task=known_task,
        process_tension=process_tension,
    )


def run_m345_process_benchmark() -> dict[str, object]:
    drives, wanting, closure, boredom = _process_states()
    empty_state = ProcessValenceState()
    focus_candidate = _focus_candidate()
    novel_candidate = _novel_candidate()

    focus_with = process_valence_priority_adjustment(candidate=focus_candidate, process_valence_state=wanting)
    focus_without = process_valence_priority_adjustment(candidate=focus_candidate, process_valence_state=empty_state)
    closure_with = process_valence_priority_adjustment(candidate=focus_candidate, process_valence_state=closure)
    closure_without = process_valence_priority_adjustment(candidate=focus_candidate, process_valence_state=empty_state)
    boredom_with = process_valence_priority_adjustment(candidate=novel_candidate, process_valence_state=boredom)
    boredom_without = process_valence_priority_adjustment(candidate=novel_candidate, process_valence_state=empty_state)

    process_scan = drives.process_action_bias("scan")
    process_rest = drives.process_action_bias("rest")
    closure_drives = DriveSystem(process_valence=closure)
    closure_scan = closure_drives.process_action_bias("scan")
    boredom_drives = DriveSystem(process_valence=boredom)
    boredom_scan = boredom_drives.process_action_bias("scan")
    boredom_rest = boredom_drives.process_action_bias("rest")

    baseline = SlowVariableLearner()
    selective = _selective_explorer(345)
    explorer = _high_investment_explorer(445)
    compressor = _compressor_surface(545)

    snapshots = {
        "baseline": baseline.style_snapshot(),
        "selective": selective.style_snapshot(),
        "explorer": explorer.style_snapshot(),
        "compressor": compressor.style_snapshot(),
    }
    labels = {str(snapshot["label"]) for snapshot in snapshots.values()}

    selective_unknown_scan = _style_probe(selective, action="scan", known_task=False, uncertainty_level=0.80, process_tension=0.70)
    baseline_unknown_scan = _style_probe(baseline, action="scan", known_task=False, uncertainty_level=0.80, process_tension=0.70)
    explorer_unknown_scan = _style_probe(explorer, action="scan", known_task=False, uncertainty_level=0.82, process_tension=0.74)
    compressor_known_rest = _style_probe(compressor, action="rest", known_task=True, uncertainty_level=0.22, process_tension=0.08)
    baseline_known_rest = _style_probe(baseline, action="rest", known_task=True, uncertainty_level=0.22, process_tension=0.08)

    metrics = ProcessBenchmarkMetrics(
        focus_bonus_gain=focus_with["process_bonus"] - focus_without["process_bonus"],
        closure_penalty_gain=closure_with["closure_penalty"] - closure_without["closure_penalty"],
        boredom_bonus_gain=boredom_with["process_bonus"] - boredom_without["process_bonus"],
        scan_reorientation_gain=(boredom_scan - boredom_rest) - (closure_scan - process_rest),
        style_label_diversity=len(labels),
        selective_unknown_scan_gain=selective_unknown_scan - baseline_unknown_scan,
        explorer_unknown_scan_gain=explorer_unknown_scan - baseline_unknown_scan,
        compressor_known_rest_gain=compressor_known_rest - baseline_known_rest,
        style_continuity_mean=mean(
            float(snapshots[name]["continuity"])
            for name in ("selective", "explorer", "compressor")
        ),
    )

    gates = {
        "process_reorientation": {
            "passed": metrics.focus_bonus_gain > 0.12 and metrics.closure_penalty_gain > 0.10 and metrics.boredom_bonus_gain > 0.07,
            "focus_bonus_gain": _round(metrics.focus_bonus_gain),
            "closure_penalty_gain": _round(metrics.closure_penalty_gain),
            "boredom_bonus_gain": _round(metrics.boredom_bonus_gain),
        },
        "action_surface_shift": {
            "passed": metrics.scan_reorientation_gain > 0.05,
            "scan_reorientation_gain": _round(metrics.scan_reorientation_gain),
            "wanting_scan_bias": _round(process_scan),
            "closure_scan_bias": _round(closure_scan),
            "boredom_scan_bias": _round(boredom_scan),
            "boredom_rest_bias": _round(boredom_rest),
        },
        "style_differentiation": {
            "passed": metrics.style_label_diversity >= 4 and metrics.selective_unknown_scan_gain > 0.01 and metrics.explorer_unknown_scan_gain > 0.03,
            "style_label_diversity": metrics.style_label_diversity,
            "selective_unknown_scan_gain": _round(metrics.selective_unknown_scan_gain),
            "explorer_unknown_scan_gain": _round(metrics.explorer_unknown_scan_gain),
            "compressor_known_rest_gain": _round(metrics.compressor_known_rest_gain),
        },
        "style_continuity": {
            "passed": metrics.style_continuity_mean >= 0.50,
            "style_continuity_mean": _round(metrics.style_continuity_mean),
        },
    }

    findings: list[dict[str, object]] = []
    residual_risks: list[str] = []
    if metrics.style_label_diversity < 4:
        residual_risks.append("Style surfaces are differentiated, but not every seeded archetype crystallizes into a unique label yet.")
    if snapshots["compressor"]["label"] == "balanced_adaptive":
        findings.append(
            {
                "severity": "S2",
                "label": "compressor_not_fully_separated",
                "detail": "The compressor seed produces a distinct low-cost rest bias, but still collapses to balanced_adaptive at the label layer.",
            }
        )

    status = "PASS" if all(bool(gate["passed"]) for gate in gates.values()) else "FAIL"
    recommendation = "ACCEPT" if status == "PASS" else "BLOCK"
    return {
        "benchmark_id": "m345_process_benchmark",
        "status": status,
        "recommendation": recommendation,
        "generated_at": _now_iso(),
        "metrics": metrics.to_dict(),
        "gates": gates,
        "process_trace": {
            "wanting": wanting.to_dict(),
            "closure": closure.to_dict(),
            "boredom": boredom.to_dict(),
            "focus_with": focus_with,
            "focus_without": focus_without,
            "closure_with": closure_with,
            "closure_without": closure_without,
            "boredom_with": boredom_with,
            "boredom_without": boredom_without,
        },
        "style_trace": {
            "snapshots": snapshots,
            "bias_probe": {
                "baseline_unknown_scan": _round(baseline_unknown_scan),
                "selective_unknown_scan": _round(selective_unknown_scan),
                "explorer_unknown_scan": _round(explorer_unknown_scan),
                "baseline_known_rest": _round(baseline_known_rest),
                "compressor_known_rest": _round(compressor_known_rest),
            },
        },
        "findings": findings,
        "residual_risks": residual_risks,
    }


def write_m345_process_benchmark_artifacts() -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    payload = run_m345_process_benchmark()
    M345_TRACE_PATH.write_text(
        json.dumps(
            {
                "process_trace": payload["process_trace"],
                "style_trace": payload["style_trace"],
                "metrics": payload["metrics"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M345_ABLATION_PATH.write_text(
        json.dumps(
            {
                "focus": {
                    "with_process": payload["process_trace"]["focus_with"],
                    "without_process": payload["process_trace"]["focus_without"],
                },
                "closure": {
                    "with_process": payload["process_trace"]["closure_with"],
                    "without_process": payload["process_trace"]["closure_without"],
                },
                "boredom": {
                    "with_process": payload["process_trace"]["boredom_with"],
                    "without_process": payload["process_trace"]["boredom_without"],
                },
                "style_probe": payload["style_trace"]["bias_probe"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    M345_REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    M345_SUMMARY_PATH.write_text(
        "# M3.4-M3.5 Process Benchmark\n\n"
        f"- Status: {payload['status']}\n"
        f"- Recommendation: {payload['recommendation']}\n"
        f"- Focus bonus gain: {payload['metrics']['focus_bonus_gain']}\n"
        f"- Closure penalty gain: {payload['metrics']['closure_penalty_gain']}\n"
        f"- Boredom bonus gain: {payload['metrics']['boredom_bonus_gain']}\n"
        f"- Style label diversity: {payload['metrics']['style_label_diversity']}\n",
        encoding="utf-8",
    )
    return {
        "trace": str(M345_TRACE_PATH),
        "ablation": str(M345_ABLATION_PATH),
        "report": str(M345_REPORT_PATH),
        "summary": str(M345_SUMMARY_PATH),
    }
