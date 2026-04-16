from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from .pipeline import ValidationReport
from .statistics import ComparisonResult, paired_comparison

ACCEPTANCE_RULES_VERSION = "m54_v3"
AGENT_STATE_MIN_MEAN = 0.80


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strategy_has_metric(
    strategy_result: dict[str, object],
    metric_name: str,
) -> bool:
    p = strategy_result.get("personality_metrics", {})
    a = strategy_result.get("baseline_a_metrics", {})
    b = strategy_result.get("baseline_b_metrics", {})
    c = strategy_result.get("baseline_c_metrics", {})
    return (
        metric_name in p
        and metric_name in a
        and metric_name in b
        and metric_name in c
    )


def collect_per_user_metric_vectors(
    reports: list[ValidationReport], metric_name: str
) -> tuple[list[float], list[float], list[float], list[float], int, int]:
    """One paired sample per user: mean of scores across non-skipped strategies.

    Returns (personality, baseline_a, baseline_b, baseline_c, users_used, users_skipped_no_strategy).
    """
    personality: list[float] = []
    baseline_a: list[float] = []
    baseline_b: list[float] = []
    baseline_c: list[float] = []
    users_skipped_no_strategy = 0
    for report in reports:
        p_vals: list[float] = []
        a_vals: list[float] = []
        b_vals: list[float] = []
        c_vals: list[float] = []
        for strategy_result in report.per_strategy.values():
            if strategy_result.get("skipped", False):
                continue
            if not _strategy_has_metric(strategy_result, metric_name):
                continue
            p = strategy_result.get("personality_metrics", {})
            a = strategy_result.get("baseline_a_metrics", {})
            b = strategy_result.get("baseline_b_metrics", {})
            c = strategy_result.get("baseline_c_metrics", {})
            p_vals.append(float(p.get(metric_name, 0.0)))
            a_vals.append(float(a.get(metric_name, 0.0)))
            b_vals.append(float(b.get(metric_name, 0.0)))
            c_vals.append(float(c.get(metric_name, 0.0)))
        if not p_vals:
            users_skipped_no_strategy += 1
            continue
        personality.append(float(mean(p_vals)))
        baseline_a.append(float(mean(a_vals)))
        baseline_b.append(float(mean(b_vals)))
        baseline_c.append(float(mean(c_vals)))
    users_used = len(personality)
    return personality, baseline_a, baseline_b, baseline_c, users_used, users_skipped_no_strategy


def _classifier_3class_gate_passed(reports: list[ValidationReport]) -> bool:
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if strategy_result.get("skipped", False):
                continue
            cv = strategy_result.get("classifier_validation")
            if isinstance(cv, dict) and "passed_3class_gate" in cv:
                return bool(cv["passed_3class_gate"])
    return False


def _comparison_for_metric(reports: list[ValidationReport], metric_name: str) -> ComparisonResult:
    p, a, b, c, _, _ = collect_per_user_metric_vectors(reports, metric_name)
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


def _compute_hard_pass(
    comparisons: dict[str, dict[str, object]],
    *,
    classifier_3class_gate_passed: bool,
) -> tuple[bool, dict[str, bool]]:
    """Semantic vs A; behavioral(strategy) vs C when classifier gate passes; agent state threshold."""
    sem_ok = bool(comparisons["semantic_similarity"]["vs_a_significant"])
    beh_row = comparisons["behavioral_similarity_strategy"]
    beh_vs_c_sig = bool(beh_row["vs_c_significant"])
    if classifier_3class_gate_passed:
        beh_ok = beh_vs_c_sig
    else:
        beh_ok = True
    as_row = comparisons["agent_state_similarity"]
    agent_state_ok = float(as_row["personality_mean"]) >= AGENT_STATE_MIN_MEAN
    hard_pass = bool(sem_ok and beh_ok and agent_state_ok)
    breakdown = {
        "classifier_3class_gate_passed": bool(classifier_3class_gate_passed),
        "behavioral_hard_metric_required": bool(classifier_3class_gate_passed),
        "semantic_similarity_vs_baseline_a_significant_better": sem_ok,
        "behavioral_similarity_strategy_vs_baseline_c_significant_better": beh_vs_c_sig,
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

    _, _, _, _, users_used, users_skipped_no_strategy = collect_per_user_metric_vectors(
        reports, "semantic_similarity"
    )
    classifier_gate = _classifier_3class_gate_passed(reports)
    behavioral_degraded = not classifier_gate

    if classifier_gate:
        hard_metrics_list = [
            "semantic_similarity",
            "behavioral_similarity_strategy",
            "agent_state_similarity",
        ]
        soft_metrics_list = [
            "behavioral_similarity_action11",
            "stylistic_similarity",
            "personality_similarity",
        ]
    else:
        hard_metrics_list = ["semantic_similarity", "agent_state_similarity"]
        soft_metrics_list = [
            "behavioral_similarity_strategy",
            "behavioral_similarity_action11",
            "stylistic_similarity",
            "personality_similarity",
        ]

    hard_pass, hard_pass_breakdown = _compute_hard_pass(
        comparisons,
        classifier_3class_gate_passed=classifier_gate,
    )

    acceptance_rules = {
        "version": ACCEPTANCE_RULES_VERSION,
        "hard_metrics": list(hard_metrics_list),
        "soft_metrics": list(soft_metrics_list),
        "semantic_similarity": "paired Wilcoxon one-sided greater vs baseline A (p < 0.05) and mean paired diff > 0; one paired sample per user (mean across split strategies)",
        "behavioral_similarity_strategy": "paired Wilcoxon one-sided greater vs baseline C when 3-class classifier gate passes; otherwise soft-only",
        "agent_state_similarity": f"mean across users >= {AGENT_STATE_MIN_MEAN} (no baseline significance required)",
        "statistical_engine": "scipy.stats.wilcoxon when available; else conservative fallback (see statistics.py)",
        "aggregation": "per_user_mean_across_strategies",
    }

    aggregate_payload = {
        "user_count": int(len(reports)),
        "users_tested": int(users_used),
        "users_skipped_no_strategy": int(users_skipped_no_strategy),
        "metric_version": "m54_v3",
        "behavioral_labeling": "dialogue_act_classifier_both",
        "behavioral_hard_metric_degraded": bool(behavioral_degraded),
        "comparisons": comparisons,
        "hard_metrics": list(hard_metrics_list),
        "soft_metrics": list(soft_metrics_list),
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
        f"- Users: {len(reports)} (tested: {users_used}, skipped no strategy: {users_skipped_no_strategy})",
        f"- Metric version: {aggregate_payload['metric_version']} ({aggregate_payload['behavioral_labeling']})",
        f"- Classifier 3-class gate: {classifier_gate}",
        f"- Behavioral hard metric degraded (soft-only): {behavioral_degraded}",
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
            "## Comparisons vs baseline C (directional)",
            "",
            "| Metric | personality_mean | baseline_c_mean | mean_diff | p (greater) | significant better |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in metric_names:
        row = comparisons[name]
        lines.append(
            f"| `{name}` | {row['personality_mean']:.4f} | {row['baseline_c_mean']:.4f} | "
            f"{row['vs_c_mean_diff']:.4f} | {row['vs_c_pvalue']:.4f} | {row['vs_c_significant']} |"
        )
    lines.extend(
        [
            "",
            "## Hard metric rows (summary)",
        ]
    )
    for name in hard_metrics_list:
        row = comparisons[name]
        lines.append(
            f"- `{name}`: personality={row['personality_mean']:.4f}, "
            f"baseline_a={row['baseline_a_mean']:.4f}, p(vs_a)={row['vs_a_pvalue']:.4f}, "
            f"baseline_c={row['baseline_c_mean']:.4f}, p(vs_c)={row['vs_c_pvalue']:.4f}"
        )
    lines.extend(["", "## Soft Metrics"])
    for name in soft_metrics_list:
        row = comparisons[name]
        lines.append(
            f"- `{name}`: personality={row['personality_mean']:.4f}, "
            f"baseline_a={row['baseline_a_mean']:.4f}, baseline_c={row['baseline_c_mean']:.4f}"
        )
    md_path = output_dir / "aggregate_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path
