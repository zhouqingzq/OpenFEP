from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from .pipeline import ValidationReport
from .statistics import ComparisonResult, paired_comparison

ACCEPTANCE_RULES_VERSION = "m54_v2"
AGENT_STATE_MIN_MEAN = 0.80


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_metric_vectors(
    reports: list[ValidationReport], metric_name: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    personality: list[float] = []
    baseline_a: list[float] = []
    baseline_b: list[float] = []
    baseline_c: list[float] = []
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if strategy_result.get("skipped", False):
                continue
            p = strategy_result.get("personality_metrics", {})
            a = strategy_result.get("baseline_a_metrics", {})
            b = strategy_result.get("baseline_b_metrics", {})
            c = strategy_result.get("baseline_c_metrics", {})
            if metric_name in p and metric_name in a and metric_name in c:
                personality.append(float(p.get(metric_name, 0.0)))
                baseline_a.append(float(a.get(metric_name, 0.0)))
                baseline_b.append(float(b.get(metric_name, 0.0)))
                baseline_c.append(float(c.get(metric_name, 0.0)))
    return personality, baseline_a, baseline_b, baseline_c


def _comparison_for_metric(reports: list[ValidationReport], metric_name: str) -> ComparisonResult:
    p, a, b, c = _collect_metric_vectors(reports, metric_name)
    p_vs_a = paired_comparison(p, a, test="wilcoxon", alpha=0.05)
    p_vs_b = paired_comparison(p, b, test="wilcoxon", alpha=0.05)
    p_vs_c = paired_comparison(p, c, test="wilcoxon", alpha=0.05)
    return ComparisonResult(
        metric_name=metric_name,
        personality_mean=round(float(mean(p)) if p else 0.0, 6),
        baseline_a_mean=round(float(mean(a)) if a else 0.0, 6),
        baseline_b_mean=round(float(mean(b)) if b else 0.0, 6),
        baseline_c_mean=round(float(mean(c)) if c else 0.0, 6),
        vs_a_pvalue=float(p_vs_a[0]),
        vs_b_pvalue=float(p_vs_b[0]),
        vs_c_pvalue=float(p_vs_c[0]),
        vs_a_mean_diff=round(float(p_vs_a[2]), 6),
        vs_b_mean_diff=round(float(p_vs_b[2]), 6),
        vs_c_mean_diff=round(float(p_vs_c[2]), 6),
        vs_a_better=bool(p_vs_a[3]),
        vs_b_better=bool(p_vs_b[3]),
        vs_c_better=bool(p_vs_c[3]),
        vs_a_significant=bool(p_vs_a[1]),
        vs_b_significant=bool(p_vs_b[1]),
        vs_c_significant=bool(p_vs_c[1]),
    )


def _compute_hard_pass(comparisons: dict[str, dict[str, object]]) -> tuple[bool, dict[str, bool]]:
    """Hard gate: semantic and behavioral(strategy) significantly *better* than baseline A; agent state mean >= threshold."""
    sem_ok = bool(comparisons["semantic_similarity"]["vs_a_significant"])
    beh_ok = bool(comparisons["behavioral_similarity_strategy"]["vs_a_significant"])
    as_row = comparisons["agent_state_similarity"]
    agent_state_ok = float(as_row["personality_mean"]) >= AGENT_STATE_MIN_MEAN
    hard_pass = bool(sem_ok and beh_ok and agent_state_ok)
    breakdown = {
        "semantic_similarity_vs_baseline_a_significant_better": sem_ok,
        "behavioral_similarity_strategy_vs_baseline_a_significant_better": beh_ok,
        f"agent_state_similarity_mean_ge_{AGENT_STATE_MIN_MEAN:.2f}": agent_state_ok,
    }
    return hard_pass, breakdown


def generate_report(
    reports: list[ValidationReport],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user_dir = output_dir / "per_user"
    per_user_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        _write_json(per_user_dir / f"{report.user_uid}_report.json", asdict(report))

    metric_names = [
        "semantic_similarity",
        "behavioral_similarity_strategy",
        "behavioral_similarity_action11",
        "stylistic_similarity",
        "personality_similarity",
        "agent_state_similarity",
    ]
    comparisons = {
        name: asdict(_comparison_for_metric(reports, name)) for name in metric_names
    }
    hard_metrics = ("semantic_similarity", "behavioral_similarity_strategy", "agent_state_similarity")
    hard_pass, hard_pass_breakdown = _compute_hard_pass(comparisons)

    acceptance_rules = {
        "version": ACCEPTANCE_RULES_VERSION,
        "hard_metrics": list(hard_metrics),
        "semantic_similarity": "paired Wilcoxon one-sided greater vs baseline A (p < 0.05) and mean paired diff > 0",
        "behavioral_similarity_strategy": "same vs baseline A",
        "agent_state_similarity": f"mean across users >= {AGENT_STATE_MIN_MEAN} (no baseline-A significance required)",
        "statistical_engine": "scipy.stats.wilcoxon when available; else conservative fallback (see statistics.py)",
    }

    aggregate_payload = {
        "user_count": int(len(reports)),
        "metric_version": "m54_v2",
        "behavioral_labeling": "dialogue_act_classifier_both",
        "comparisons": comparisons,
        "hard_metrics": list(hard_metrics),
        "soft_metrics": ["behavioral_similarity_action11", "stylistic_similarity", "personality_similarity"],
        "acceptance_rules": acceptance_rules,
        "hard_pass": bool(hard_pass),
        "hard_pass_breakdown": hard_pass_breakdown,
        "overall_conclusion": "pass" if hard_pass else "partial",
        "reports": [asdict(item) for item in reports],
    }
    _write_json(output_dir / "aggregate_report.json", aggregate_payload)

    lines = [
        "# M5.4 Validation Aggregate Report",
        "",
        f"- Users: {len(reports)}",
        f"- Metric version: {aggregate_payload['metric_version']} ({aggregate_payload['behavioral_labeling']})",
        f"- Overall conclusion: {aggregate_payload['overall_conclusion']}",
        f"- Hard pass: {hard_pass}",
        "",
        "## Acceptance (hard metrics)",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check_name, ok in hard_pass_breakdown.items():
        lines.append(f"| {check_name} | {ok} |")
    lines.extend(
        [
            "",
            "## Comparisons vs baseline A (directional)",
            "",
            "| Metric | personality_mean | baseline_a_mean | mean_diff | p (greater) | significant better |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in metric_names:
        row = comparisons[name]
        lines.append(
            f"| `{name}` | {row['personality_mean']:.4f} | {row['baseline_a_mean']:.4f} | "
            f"{row['vs_a_mean_diff']:.4f} | {row['vs_a_pvalue']:.4f} | {row['vs_a_significant']} |"
        )
    lines.extend(
        [
            "",
            "## Hard metric rows (legacy summary)",
        ]
    )
    for name in hard_metrics:
        row = comparisons[name]
        lines.append(
            f"- `{name}`: personality={row['personality_mean']:.4f}, "
            f"baseline_a={row['baseline_a_mean']:.4f}, p(vs_a)={row['vs_a_pvalue']:.4f}, "
            f"better={row['vs_a_better']}, sig_better={row['vs_a_significant']}"
        )
    lines.extend(["", "## Soft Metrics"])
    for name in aggregate_payload["soft_metrics"]:
        row = comparisons[name]
        lines.append(
            f"- `{name}`: personality={row['personality_mean']:.4f}, baseline_a={row['baseline_a_mean']:.4f}"
        )
    md_path = output_dir / "aggregate_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path
