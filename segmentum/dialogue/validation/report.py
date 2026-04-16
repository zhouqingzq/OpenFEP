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


def collect_per_user_personality_only_metric(
    reports: list[ValidationReport], metric_name: str
) -> tuple[list[float], int, int]:
    """Per-user scores from personality_metrics only (no baseline pairing).

    For each user, averages ``metric_name`` across all non-skipped strategies that define it.
    Returns (per_user_values, users_used, users_skipped_no_metric).
    """
    personality: list[float] = []
    users_skipped_no_metric = 0
    for report in reports:
        p_vals: list[float] = []
        for strategy_result in report.per_strategy.values():
            if strategy_result.get("skipped", False):
                continue
            p = strategy_result.get("personality_metrics", {})
            if metric_name not in p:
                continue
            p_vals.append(float(p.get(metric_name, 0.0)))
        if not p_vals:
            users_skipped_no_metric += 1
            continue
        personality.append(float(mean(p_vals)))
    return personality, len(personality), users_skipped_no_metric


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


def _comparison_for_agent_state(reports: list[ValidationReport]) -> ComparisonResult:
    """Train-only vs full-data agent cosine similarity; no baseline A/B/C (not defined)."""
    p, users_used, users_skipped = collect_per_user_personality_only_metric(
        reports, "agent_state_similarity"
    )
    overall = round(float(mean(p)) if p else 0.0, 6)
    return ComparisonResult(
        metric_name="agent_state_similarity",
        personality_mean=overall,
        baseline_a_mean=None,
        baseline_b_mean=None,
        baseline_c_mean=None,
        vs_a_pvalue=None,
        vs_b_pvalue=None,
        vs_c_pvalue=None,
        vs_a_mean_diff=None,
        vs_b_mean_diff=None,
        vs_c_mean_diff=None,
        vs_a_better=None,
        vs_b_better=None,
        vs_c_better=None,
        vs_a_significant=None,
        vs_b_significant=None,
        vs_c_significant=None,
        users_included=users_used,
        users_skipped_no_metric=users_skipped,
        interpretation_notes="cosine(train_only_implanted_agent, full_data_implanted_agent); no baseline comparison",
    )


def _strategy_keys_union(reports: list[ValidationReport]) -> list[str]:
    keys: set[str] = set()
    for report in reports:
        keys.update(report.per_strategy.keys())
    return sorted(keys)


def collect_per_user_metric_vectors_for_strategy(
    reports: list[ValidationReport],
    metric_name: str,
    strategy_key: str,
) -> tuple[list[float], list[float], list[float], list[float], int, int]:
    """Like collect_per_user_metric_vectors but only one split strategy per user."""
    personality: list[float] = []
    baseline_a: list[float] = []
    baseline_b: list[float] = []
    baseline_c: list[float] = []
    users_skipped = 0
    for report in reports:
        strategy_result = report.per_strategy.get(strategy_key)
        if strategy_result is None or strategy_result.get("skipped", False):
            users_skipped += 1
            continue
        if not _strategy_has_metric(strategy_result, metric_name):
            users_skipped += 1
            continue
        p = strategy_result.get("personality_metrics", {})
        a = strategy_result.get("baseline_a_metrics", {})
        b = strategy_result.get("baseline_b_metrics", {})
        c = strategy_result.get("baseline_c_metrics", {})
        personality.append(float(p.get(metric_name, 0.0)))
        baseline_a.append(float(a.get(metric_name, 0.0)))
        baseline_b.append(float(b.get(metric_name, 0.0)))
        baseline_c.append(float(c.get(metric_name, 0.0)))
    users_used = len(personality)
    return personality, baseline_a, baseline_b, baseline_c, users_used, users_skipped


def collect_per_user_personality_only_metric_for_strategy(
    reports: list[ValidationReport],
    metric_name: str,
    strategy_key: str,
) -> tuple[list[float], int, int]:
    """Personality-only metric for a single split strategy."""
    personality: list[float] = []
    users_skipped = 0
    for report in reports:
        strategy_result = report.per_strategy.get(strategy_key)
        if strategy_result is None or strategy_result.get("skipped", False):
            users_skipped += 1
            continue
        p = strategy_result.get("personality_metrics", {})
        if metric_name not in p:
            users_skipped += 1
            continue
        personality.append(float(p.get(metric_name, 0.0)))
    return personality, len(personality), users_skipped


def _comparison_for_metric_for_strategy(
    reports: list[ValidationReport], metric_name: str, strategy_key: str
) -> ComparisonResult:
    p, a, b, c, _, _ = collect_per_user_metric_vectors_for_strategy(
        reports, metric_name, strategy_key
    )
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


def _comparison_for_agent_state_for_strategy(
    reports: list[ValidationReport], strategy_key: str
) -> ComparisonResult:
    p, users_used, users_skipped = collect_per_user_personality_only_metric_for_strategy(
        reports, "agent_state_similarity", strategy_key
    )
    overall = round(float(mean(p)) if p else 0.0, 6)
    return ComparisonResult(
        metric_name="agent_state_similarity",
        personality_mean=overall,
        baseline_a_mean=None,
        baseline_b_mean=None,
        baseline_c_mean=None,
        vs_a_pvalue=None,
        vs_b_pvalue=None,
        vs_c_pvalue=None,
        vs_a_mean_diff=None,
        vs_b_mean_diff=None,
        vs_c_mean_diff=None,
        vs_a_better=None,
        vs_b_better=None,
        vs_c_better=None,
        vs_a_significant=None,
        vs_b_significant=None,
        vs_c_significant=None,
        users_included=users_used,
        users_skipped_no_metric=users_skipped,
        interpretation_notes=f"strategy={strategy_key}; cosine(train_only, full_data); no baseline",
    )


def _topic_split_summary(reports: list[ValidationReport]) -> dict[str, object]:
    """Counts topic split applicability from per-user topic strategy metadata."""
    total = 0
    not_applicable = 0
    for report in reports:
        tr = report.per_strategy.get("topic")
        if tr is None:
            continue
        total += 1
        meta = tr.get("split_metadata", {})
        if isinstance(meta, dict) and bool(meta.get("topic_split_not_applicable")):
            not_applicable += 1
    return {
        "users_with_topic_strategy_row": int(total),
        "users_topic_split_not_applicable": int(not_applicable),
        "users_topic_split_applicable": int(total - not_applicable),
    }


def _per_strategy_hard_pass(
    comparisons_by_strategy: dict[str, dict[str, dict[str, object]]],
    *,
    classifier_3class_gate_passed: bool,
) -> dict[str, dict[str, bool]]:
    """Same hard rules as _compute_hard_pass but per split strategy."""
    out: dict[str, dict[str, bool]] = {}
    for sk, comp in comparisons_by_strategy.items():
        sem_ok = bool(comp["semantic_similarity"]["vs_a_significant"])
        beh_row = comp["behavioral_similarity_strategy"]
        beh_vs_c_sig = bool(beh_row["vs_c_significant"])
        if classifier_3class_gate_passed:
            beh_ok = beh_vs_c_sig
        else:
            beh_ok = True
        as_row = comp["agent_state_similarity"]
        agent_state_ok = float(as_row["personality_mean"]) >= AGENT_STATE_MIN_MEAN
        out[sk] = {
            "hard_pass": bool(sem_ok and beh_ok and agent_state_ok),
            "semantic_similarity_vs_baseline_a_significant_better": sem_ok,
            "behavioral_similarity_strategy_vs_baseline_c_significant_better": beh_vs_c_sig,
            f"agent_state_similarity_mean_ge_{AGENT_STATE_MIN_MEAN:.2f}": agent_state_ok,
        }
    return out


def _fmt_cell_num(val: object) -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        return f"{float(val):.4f}"
    return str(val)


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
    baseline_metrics = [name for name in metric_names if name != "agent_state_similarity"]
    comparisons: dict[str, dict[str, object]] = {
        name: asdict(_comparison_for_metric(reports, name)) for name in baseline_metrics
    }
    comparisons["agent_state_similarity"] = asdict(_comparison_for_agent_state(reports))

    strategy_keys = _strategy_keys_union(reports)
    per_strategy_comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for sk in strategy_keys:
        per_strategy_comparisons[sk] = {
            name: asdict(_comparison_for_metric_for_strategy(reports, name, sk))
            for name in baseline_metrics
        }
        per_strategy_comparisons[sk]["agent_state_similarity"] = asdict(
            _comparison_for_agent_state_for_strategy(reports, sk)
        )

    _, _, _, _, users_used, users_skipped_no_strategy = collect_per_user_metric_vectors(
        reports, "semantic_similarity"
    )
    _, ast_users_used, ast_users_skipped = collect_per_user_personality_only_metric(
        reports, "agent_state_similarity"
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
    per_strategy_hard_pass = _per_strategy_hard_pass(
        per_strategy_comparisons,
        classifier_3class_gate_passed=classifier_gate,
    )
    partner_hard_pass = per_strategy_hard_pass.get("partner", {}).get("hard_pass")
    topic_hard_pass = per_strategy_hard_pass.get("topic", {}).get("hard_pass")
    topic_split_summary = _topic_split_summary(reports)

    acceptance_rules = {
        "version": ACCEPTANCE_RULES_VERSION,
        "hard_metrics": list(hard_metrics_list),
        "soft_metrics": list(soft_metrics_list),
        "semantic_similarity": "paired Wilcoxon one-sided greater vs baseline A (p < 0.05) and mean paired diff > 0; one paired sample per user (mean across split strategies)",
        "behavioral_similarity_strategy": "paired Wilcoxon one-sided greater vs baseline C when 3-class classifier gate passes; otherwise soft-only",
        "agent_state_similarity": f"mean across users >= {AGENT_STATE_MIN_MEAN} (no baseline significance required)",
        "statistical_engine": "scipy.stats.wilcoxon when available; else conservative fallback (see statistics.py)",
        "aggregation": "per_user_mean_across_strategies",
        "per_strategy_comparisons": "same Wilcoxon rules computed separately per split strategy (see per_strategy_comparisons)",
        "agent_state_similarity_detail": "mean of per-user means across strategies; threshold on global mean; not a baseline contrast",
        "pilot": "run_pilot_validation estimates sd of per-user semantic (personality vs baseline A) and behavioral (personality vs baseline C) mean differences; suggested_min_users uses max of thresholds when either exceeds pilot_sd_threshold",
    }

    metric_interpretations = {
        "agent_state_similarity": comparisons["agent_state_similarity"].get("interpretation_notes"),
    }

    aggregate_payload = {
        "user_count": int(len(reports)),
        "users_tested": int(users_used),
        "users_skipped_no_strategy": int(users_skipped_no_strategy),
        "agent_state_users_tested": int(ast_users_used),
        "agent_state_users_skipped_no_metric": int(ast_users_skipped),
        "metric_version": "m54_v3",
        "behavioral_labeling": "dialogue_act_classifier_11class_and_3strategy",
        "behavioral_hard_metric_degraded": bool(behavioral_degraded),
        "comparisons": comparisons,
        "per_strategy_comparisons": per_strategy_comparisons,
        "per_strategy_hard_pass": per_strategy_hard_pass,
        "partner_hard_pass": partner_hard_pass,
        "topic_hard_pass": topic_hard_pass,
        "topic_split_summary": topic_split_summary,
        "metric_interpretations": metric_interpretations,
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
        f"- Agent state: users with metric {ast_users_used}, skipped {ast_users_skipped}",
        f"- Topic split: {topic_split_summary}",
        f"- Metric version: {aggregate_payload['metric_version']} ({aggregate_payload['behavioral_labeling']})",
        f"- Classifier 3-class gate: {classifier_gate}",
        f"- Behavioral hard metric degraded (soft-only): {behavioral_degraded}",
        f"- Overall conclusion: {aggregate_payload['overall_conclusion']}",
        f"- Hard pass: {hard_pass}",
        f"- Partner strategy hard pass: {partner_hard_pass}",
        f"- Topic strategy hard pass: {topic_hard_pass}",
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
            f"| `{name}` | {_fmt_cell_num(row['personality_mean'])} | {_fmt_cell_num(row.get('baseline_a_mean'))} | "
            f"{_fmt_cell_num(row.get('vs_a_mean_diff'))} | {_fmt_cell_num(row.get('vs_a_pvalue'))} | "
            f"{row.get('vs_a_significant') if row.get('vs_a_significant') is not None else '—'} |"
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
            f"| `{name}` | {_fmt_cell_num(row['personality_mean'])} | {_fmt_cell_num(row.get('baseline_c_mean'))} | "
            f"{_fmt_cell_num(row.get('vs_c_mean_diff'))} | {_fmt_cell_num(row.get('vs_c_pvalue'))} | "
            f"{row.get('vs_c_significant') if row.get('vs_c_significant') is not None else '—'} |"
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
            f"- `{name}`: personality={_fmt_cell_num(row['personality_mean'])}, "
            f"baseline_a={_fmt_cell_num(row.get('baseline_a_mean'))}, p(vs_a)={_fmt_cell_num(row.get('vs_a_pvalue'))}, "
            f"baseline_c={_fmt_cell_num(row.get('baseline_c_mean'))}, p(vs_c)={_fmt_cell_num(row.get('vs_c_pvalue'))}"
        )
    lines.extend(["", "## Soft Metrics"])
    for name in soft_metrics_list:
        row = comparisons[name]
        lines.append(
            f"- `{name}`: personality={_fmt_cell_num(row['personality_mean'])}, "
            f"baseline_a={_fmt_cell_num(row.get('baseline_a_mean'))}, baseline_c={_fmt_cell_num(row.get('baseline_c_mean'))}"
        )
    lines.extend(["", "## Per-strategy hard pass"])
    if per_strategy_hard_pass:
        lines.append("")
        lines.append("| Strategy | hard_pass | semantic vs A sig | behavioral vs C sig | agent_state mean ok |")
        lines.append("| --- | --- | --- | --- | --- |")
        for sk in sorted(per_strategy_hard_pass.keys()):
            br = per_strategy_hard_pass[sk]
            lines.append(
                f"| `{sk}` | {br.get('hard_pass')} | "
                f"{br.get('semantic_similarity_vs_baseline_a_significant_better')} | "
                f"{br.get('behavioral_similarity_strategy_vs_baseline_c_significant_better')} | "
                f"{br.get(f'agent_state_similarity_mean_ge_{AGENT_STATE_MIN_MEAN:.2f}')} |"
            )
    else:
        lines.append("_No per-strategy rows._")
    md_path = output_dir / "aggregate_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path
