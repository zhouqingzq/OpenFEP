from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean, median

from .pipeline import ValidationReport
from .statistics import ComparisonResult, paired_comparison, scipy_wilcoxon_available

ACCEPTANCE_RULES_VERSION = "m54_v3"
AGENT_STATE_MIN_MEAN = 0.80
REQUIRED_STRATEGIES = ("random", "temporal", "partner", "topic")


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


def _strategy_eligible(strategy_result: dict[str, object]) -> bool:
    return not strategy_result.get("skipped", False) and bool(
        strategy_result.get("eligible_for_hard_gate", True)
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
            if not _strategy_eligible(strategy_result):
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
            if not _strategy_eligible(strategy_result):
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
                return bool(cv["passed_3class_gate"]) and bool(cv.get("formal_gate_eligible", False))
    return False


def _classifier_gate_summary(reports: list[ValidationReport]) -> dict[str, object]:
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if strategy_result.get("skipped", False):
                continue
            cv = strategy_result.get("classifier_validation")
            if isinstance(cv, dict):
                return dict(cv)
    return {
        "passed_3class_gate": False,
        "formal_gate_eligible": False,
        "engine": "missing",
    }


def _formal_requested(reports: list[ValidationReport]) -> bool:
    return any(bool(report.aggregate.get("formal_requested", False)) for report in reports)


def _semantic_engine_gate(reports: list[ValidationReport]) -> dict[str, object]:
    methods: list[str] = []
    fallback_reasons: list[str] = []
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if not _strategy_eligible(strategy_result):
                continue
            details = strategy_result.get("personality_metric_details", {})
            if not isinstance(details, dict):
                continue
            sem = details.get("semantic_similarity", {})
            if not isinstance(sem, dict):
                continue
            method = str(sem.get("method", "unknown"))
            methods.append(method)
            if sem.get("fallback_reason") is not None:
                fallback_reasons.append(str(sem.get("fallback_reason")))
    unique_methods = sorted(set(methods))
    passed = bool(methods) and all(method == "sentence_embedding_cosine" for method in methods)
    return {
        "passed": bool(passed),
        "methods": unique_methods,
        "fallback_reasons": sorted(set(fallback_reasons)),
        "policy": "formal acceptance requires sentence_embedding_cosine for personality semantic metrics",
    }


def _comparison_for_metric(reports: list[ValidationReport], metric_name: str) -> ComparisonResult:
    p, a, b, c, users_used, users_skipped = collect_per_user_metric_vectors(reports, metric_name)
    p_vs_a = paired_comparison(p, a, test="wilcoxon", alpha=0.05)
    p_vs_b = paired_comparison(p, b, test="wilcoxon", alpha=0.05)
    p_vs_c = paired_comparison(p, c, test="wilcoxon", alpha=0.05)
    stat_valid = scipy_wilcoxon_available()
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
        statistical_valid=bool(stat_valid),
        statistical_error=None if stat_valid else "scipy.stats.wilcoxon unavailable",
        users_included=users_used,
        users_skipped_no_metric=users_skipped,
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
        statistical_valid=True,
        statistical_error=None,
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
        if strategy_result is None or not _strategy_eligible(strategy_result):
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
        if strategy_result is None or not _strategy_eligible(strategy_result):
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
    p, a, b, c, users_used, users_skipped = collect_per_user_metric_vectors_for_strategy(
        reports, metric_name, strategy_key
    )
    p_vs_a = paired_comparison(p, a, test="wilcoxon", alpha=0.05)
    p_vs_b = paired_comparison(p, b, test="wilcoxon", alpha=0.05)
    p_vs_c = paired_comparison(p, c, test="wilcoxon", alpha=0.05)
    stat_valid = scipy_wilcoxon_available()
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
        statistical_valid=bool(stat_valid),
        statistical_error=None if stat_valid else "scipy.stats.wilcoxon unavailable",
        users_included=users_used,
        users_skipped_no_metric=users_skipped,
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
        statistical_valid=True,
        statistical_error=None,
        users_included=users_used,
        users_skipped_no_metric=users_skipped,
        interpretation_notes=f"strategy={strategy_key}; cosine(train_only, full_data); no baseline",
    )


def _topic_split_summary(reports: list[ValidationReport]) -> dict[str, object]:
    """Counts topic split applicability from per-user topic strategy metadata."""
    total = 0
    not_applicable = 0
    valid = 0
    for report in reports:
        tr = report.per_strategy.get("topic")
        if tr is None:
            continue
        total += 1
        meta = tr.get("split_metadata", {})
        if isinstance(meta, dict) and bool(meta.get("topic_split_not_applicable")):
            not_applicable += 1
        elif _strategy_eligible(tr):
            valid += 1
    return {
        "users_with_topic_strategy_row": int(total),
        "users_topic_split_not_applicable": int(not_applicable),
        "users_topic_split_applicable": int(total - not_applicable),
        "users_topic_split_valid_for_hard_gate": int(valid),
    }


def _required_users(reports: list[ValidationReport]) -> int:
    values: list[int] = []
    for report in reports:
        raw = report.aggregate.get("required_users")
        if raw is None:
            pilot = report.aggregate.get("pilot")
            if isinstance(pilot, dict):
                raw = pilot.get("required_users", pilot.get("suggested_min_users"))
        try:
            if raw is not None:
                values.append(int(raw))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 10


def _pilot_gate(reports: list[ValidationReport], required_users: int) -> dict[str, object]:
    pilot_present = any(isinstance(report.aggregate.get("pilot"), dict) for report in reports)
    user_count_ok = len(reports) >= int(required_users)
    return {
        "passed": bool(pilot_present and user_count_ok),
        "pilot_present": bool(pilot_present),
        "user_count_ok": bool(user_count_ok),
        "user_count": int(len(reports)),
        "required_users": int(required_users),
    }


def _formal_baseline_c_gate(
    reports: list[ValidationReport],
    *,
    required_users: int,
    formal_requested: bool,
) -> dict[str, object]:
    skip_used = any(bool(report.aggregate.get("skip_population_average_implant")) for report in reports)
    leave_one_out_values: list[bool] = []
    population_counts: list[int] = []
    metrics_present_values: list[bool] = []
    excluded_uid_values: list[bool] = []
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if not _strategy_eligible(strategy_result):
                continue
            leave_one_out_values.append(bool(strategy_result.get("baseline_c_leave_one_out", False)))
            try:
                population_counts.append(int(strategy_result.get("baseline_c_population_user_count", 0) or 0))
            except (TypeError, ValueError):
                population_counts.append(0)
            c_metrics = strategy_result.get("baseline_c_metrics", {})
            metrics_present_values.append(isinstance(c_metrics, dict) and bool(c_metrics))
            try:
                excluded_raw = strategy_result.get("baseline_c_population_excluded_uid", -999999)
                excluded_uid = int(excluded_raw if excluded_raw is not None else -999999)
            except (TypeError, ValueError):
                excluded_uid = -999999
            excluded_uid_values.append(excluded_uid == int(report.user_uid))
    leave_one_out_ok = bool(leave_one_out_values and all(leave_one_out_values))
    metrics_present_ok = bool(metrics_present_values and all(metrics_present_values))
    target_excluded_ok = bool(excluded_uid_values and all(excluded_uid_values))
    min_population = int(min(population_counts)) if population_counts else 0
    population_count_ok = bool(min_population >= max(0, int(required_users) - 1))
    if formal_requested:
        passed = bool(
            (not skip_used)
            and leave_one_out_ok
            and metrics_present_ok
            and target_excluded_ok
            and population_count_ok
        )
    else:
        passed = bool(not skip_used)
    return {
        "passed": bool(passed),
        "skip_population_average_implant_used": bool(skip_used),
        "leave_one_out_population_average": bool(leave_one_out_ok),
        "target_user_excluded": bool(target_excluded_ok),
        "baseline_c_metrics_present": bool(metrics_present_ok),
        "population_count_ok": bool(population_count_ok),
        "min_population_user_count": int(min_population),
        "required_population_user_count": max(0, int(required_users) - 1),
        "formal_requested": bool(formal_requested),
    }


def _split_gate(
    per_strategy_comparisons: dict[str, dict[str, dict[str, object]]],
    required_users: int,
) -> dict[str, object]:
    per_strategy: dict[str, dict[str, object]] = {}
    for key in REQUIRED_STRATEGIES:
        row = per_strategy_comparisons.get(key, {}).get("semantic_similarity", {})
        users = int(row.get("users_included") or 0)
        per_strategy[key] = {
            "present": key in per_strategy_comparisons,
            "users_included": users,
            "required_users": int(required_users),
            "passed": bool(key in per_strategy_comparisons and users >= int(required_users)),
        }
    return {
        "passed": all(bool(item["passed"]) for item in per_strategy.values()),
        "required_strategies": list(REQUIRED_STRATEGIES),
        "per_strategy": per_strategy,
    }


def _topic_gate(
    per_strategy_hard_pass: dict[str, dict[str, bool]],
    topic_split_summary: dict[str, object],
    required_users: int,
) -> dict[str, object]:
    valid_users = int(topic_split_summary.get("users_topic_split_valid_for_hard_gate", 0))
    enough_users = valid_users >= int(required_users)
    topic_hard_pass = bool(per_strategy_hard_pass.get("topic", {}).get("hard_pass"))
    return {
        "passed": bool(enough_users and topic_hard_pass),
        "policy": "exclude_not_applicable_but_require_min_users",
        "valid_topic_users": int(valid_users),
        "required_users": int(required_users),
        "topic_hard_pass": bool(topic_hard_pass),
    }


def _partner_gate(
    per_strategy_hard_pass: dict[str, dict[str, bool]],
    per_strategy_comparisons: dict[str, dict[str, dict[str, object]]],
    required_users: int,
) -> dict[str, object]:
    row = per_strategy_comparisons.get("partner", {}).get("semantic_similarity", {})
    users = int(row.get("users_included") or 0)
    enough_users = users >= int(required_users)
    partner_hard_pass = bool(per_strategy_hard_pass.get("partner", {}).get("hard_pass"))
    return {
        "passed": bool(enough_users and partner_hard_pass),
        "users_included": int(users),
        "required_users": int(required_users),
        "partner_hard_pass": bool(partner_hard_pass),
    }


def _per_strategy_hard_pass(
    comparisons_by_strategy: dict[str, dict[str, dict[str, object]]],
    *,
    classifier_3class_gate_passed: bool,
) -> dict[str, dict[str, bool]]:
    """Same hard rules as _compute_hard_pass but per split strategy."""
    out: dict[str, dict[str, bool]] = {}
    for sk, comp in comparisons_by_strategy.items():
        sem_stat_ok = bool(comp["semantic_similarity"].get("statistical_valid", True))
        sem_ok = bool(comp["semantic_similarity"]["vs_a_significant"]) and sem_stat_ok
        beh_row = comp["behavioral_similarity_strategy"]
        beh_vs_c_sig = bool(beh_row["vs_c_significant"])
        beh_stat_ok = bool(beh_row.get("statistical_valid", True))
        if classifier_3class_gate_passed:
            beh_ok = beh_vs_c_sig and beh_stat_ok
        else:
            beh_ok = True
        as_row = comp["agent_state_similarity"]
        agent_state_ok = float(as_row["personality_mean"]) >= AGENT_STATE_MIN_MEAN
        out[sk] = {
            "hard_pass": bool(sem_ok and beh_ok and agent_state_ok),
            "semantic_similarity_vs_baseline_a_significant_better": sem_ok,
            "behavioral_similarity_strategy_vs_baseline_c_significant_better": beh_vs_c_sig,
            "semantic_wilcoxon_valid": sem_stat_ok,
            "behavioral_wilcoxon_valid": beh_stat_ok,
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
    sem_stat_ok = bool(comparisons["semantic_similarity"].get("statistical_valid", True))
    sem_ok = bool(comparisons["semantic_similarity"]["vs_a_significant"]) and sem_stat_ok
    beh_row = comparisons["behavioral_similarity_strategy"]
    beh_vs_c_sig = bool(beh_row["vs_c_significant"])
    beh_stat_ok = bool(beh_row.get("statistical_valid", True))
    if classifier_3class_gate_passed:
        beh_ok = beh_vs_c_sig and beh_stat_ok
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
        "semantic_wilcoxon_valid": sem_stat_ok,
        "behavioral_wilcoxon_valid": beh_stat_ok,
        f"agent_state_similarity_mean_ge_{AGENT_STATE_MIN_MEAN:.2f}": agent_state_ok,
    }
    return hard_pass, breakdown


def _quartiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"q1": 0.0, "median": 0.0, "q3": 0.0, "iqr": 0.0}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    q1 = ordered[max(0, int((n - 1) * 0.25))]
    q2 = float(median(ordered))
    q3 = ordered[min(n - 1, int((n - 1) * 0.75))]
    return {
        "q1": round(float(q1), 6),
        "median": round(float(q2), 6),
        "q3": round(float(q3), 6),
        "iqr": round(float(q3 - q1), 6),
    }


def _semantic_delta_summary(reports: list[ValidationReport]) -> dict[str, object]:
    user_rows: list[dict[str, object]] = []
    per_strategy: dict[str, list[float]] = {}
    pair_counts: dict[int, int] = {}
    length_buckets: dict[str, list[float]] = {}
    diagnostic_strategy: dict[str, list[float]] = {}
    diagnostic_rows_total = 0

    for report in reports:
        user_diffs: list[float] = []
        for strategy_key, strategy_result in report.per_strategy.items():
            if not _strategy_eligible(strategy_result):
                continue
            p = strategy_result.get("personality_metrics", {})
            a = strategy_result.get("baseline_a_metrics", {})
            if not isinstance(p, dict) or not isinstance(a, dict):
                continue
            if "semantic_similarity" not in p or "semantic_similarity" not in a:
                continue
            diff = float(p["semantic_similarity"]) - float(a["semantic_similarity"])
            user_diffs.append(diff)
            per_strategy.setdefault(strategy_key, []).append(diff)
            details = strategy_result.get("personality_metric_details", {})
            if isinstance(details, dict):
                sem = details.get("semantic_similarity", {})
                if isinstance(sem, dict):
                    try:
                        pc = int(sem.get("pair_count", 0))
                        pair_counts[pc] = pair_counts.get(pc, 0) + 1
                    except (TypeError, ValueError):
                        pass
            trace = strategy_result.get("diagnostic_trace", [])
            if isinstance(trace, list):
                diagnostic_rows_total += len(trace)
                for row in trace:
                    if not isinstance(row, dict):
                        continue
                    try:
                        pair_delta = float(row.get("personality_vs_a_pair_delta", 0.0))
                    except (TypeError, ValueError):
                        continue
                    bucket = str(row.get("reply_length_bucket", "unknown"))
                    length_buckets.setdefault(bucket, []).append(pair_delta)
                    diagnostic_strategy.setdefault(strategy_key, []).append(pair_delta)
        if user_diffs:
            user_rows.append(
                {
                    "user_uid": int(report.user_uid),
                    "mean_delta": round(float(mean(user_diffs)), 6),
                    "positive_strategies": int(sum(1 for item in user_diffs if item > 0.0)),
                    "negative_strategies": int(sum(1 for item in user_diffs if item < 0.0)),
                }
            )

    user_deltas = [float(row["mean_delta"]) for row in user_rows]
    user_rows_sorted = sorted(user_rows, key=lambda row: float(row["mean_delta"]))
    return {
        "users": {
            "count": int(len(user_rows)),
            "positive": int(sum(1 for item in user_deltas if item > 0.0)),
            "negative": int(sum(1 for item in user_deltas if item < 0.0)),
            "zero": int(sum(1 for item in user_deltas if abs(item) <= 1e-12)),
            **_quartiles(user_deltas),
            "worst": user_rows_sorted[:5],
            "best": list(reversed(user_rows_sorted[-5:])),
        },
        "per_strategy": {
            key: {
                "count": int(len(values)),
                "mean_delta": round(float(mean(values)) if values else 0.0, 6),
                "positive": int(sum(1 for item in values if item > 0.0)),
                "negative": int(sum(1 for item in values if item < 0.0)),
                **_quartiles(values),
            }
            for key, values in sorted(per_strategy.items())
        },
        "pair_count_distribution": {str(k): int(v) for k, v in sorted(pair_counts.items())},
        "diagnostic_pairs": {
            "rows": int(diagnostic_rows_total),
            "by_strategy": {
                key: {
                    "count": int(len(values)),
                    "mean_pair_delta": round(float(mean(values)) if values else 0.0, 6),
                    "positive": int(sum(1 for item in values if item > 0.0)),
                    "negative": int(sum(1 for item in values if item < 0.0)),
                }
                for key, values in sorted(diagnostic_strategy.items())
            },
            "by_reply_length_bucket": {
                key: {
                    "count": int(len(values)),
                    "mean_pair_delta": round(float(mean(values)) if values else 0.0, 6),
                    "positive": int(sum(1 for item in values if item > 0.0)),
                    "negative": int(sum(1 for item in values if item < 0.0)),
                }
                for key, values in sorted(length_buckets.items())
            },
        },
    }


def _counter_jsd(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left.keys()) | set(right.keys()))
    if not keys:
        return 0.0
    l_total = float(sum(left.values()))
    r_total = float(sum(right.values()))
    if l_total <= 0.0 or r_total <= 0.0:
        return 0.0
    p = [float(left.get(key, 0)) / l_total for key in keys]
    q = [float(right.get(key, 0)) / r_total for key in keys]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        out = 0.0
        for x, y in zip(a, b):
            if x > 0.0:
                out += x * math.log(x / max(y, 1e-12))
        return out

    return round(float(0.5 * _kl(p, m) + 0.5 * _kl(q, m)), 6)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean_or_zero(values: object) -> float:
    if not isinstance(values, list) or not values:
        return 0.0
    return float(mean([_safe_float(item) for item in values]))


def _baseline_audit_summary(reports: list[ValidationReport]) -> dict[str, object]:
    baselines = {
        "baseline_a": {
            "action_key": "baseline_a_action",
            "strategy_key": "baseline_a_strategy",
            "text_key": "baseline_a_text",
            "chars_key": "baseline_a_generated_chars",
            "text_similarity_key": "personality_vs_a_text_similarity",
            "semantic_score_key": "baseline_a_semantic_pair_score",
            "template_key": "baseline_a_template_id",
        },
        "baseline_c": {
            "action_key": "baseline_c_action",
            "strategy_key": "baseline_c_strategy",
            "text_key": "baseline_c_text",
            "chars_key": "baseline_c_generated_chars",
            "text_similarity_key": "personality_vs_c_text_similarity",
            "semantic_score_key": "baseline_c_semantic_pair_score",
            "template_key": "baseline_c_template_id",
        },
        "baseline_b_best": {
            "action_key": "baseline_b_best_action",
            "strategy_key": "baseline_b_best_strategy",
            "text_key": "baseline_b_best_text",
            "chars_key": "baseline_b_best_generated_chars",
            "text_similarity_key": "personality_vs_b_best_text_similarity",
            "semantic_score_key": "baseline_b_best_semantic_pair_score",
            "template_key": "",
        },
    }
    overall: dict[str, dict[str, object]] = {}
    per_user: dict[int, dict[str, list[float] | int]] = {}
    for report in reports:
        per_user.setdefault(
            int(report.user_uid),
            {
                "semantic_deltas": [],
                "a_action_agreements": [],
                "a_text_similarities": [],
                "c_action_agreements": [],
                "c_text_similarities": [],
            },
        )
        for strategy_result in report.per_strategy.values():
            trace = strategy_result.get("diagnostic_trace", [])
            if not isinstance(trace, list):
                continue
            for row in trace:
                if not isinstance(row, dict):
                    continue
                p_action = str(row.get("personality_action", ""))
                p_strategy = str(row.get("personality_strategy", ""))
                p_text = str(row.get("personality_text", ""))
                p_chars = int(row.get("personality_generated_chars", 0) or 0)
                p_sem = _safe_float(row.get("personality_semantic_pair_score"))
                per_user[int(report.user_uid)]["semantic_deltas"].append(  # type: ignore[index, union-attr]
                    _safe_float(row.get("personality_vs_a_pair_delta"))
                )
                for baseline, spec in baselines.items():
                    bucket = overall.setdefault(
                        baseline,
                        {
                            "rows": 0,
                            "action_agree": 0,
                            "strategy_agree": 0,
                            "exact_duplicate": 0,
                            "template_agree": 0,
                            "template_rows": 0,
                            "text_similarity_values": [],
                            "length_deltas": [],
                            "semantic_deltas": [],
                            "baseline_semantic_values": [],
                            "personality_actions": Counter(),
                            "baseline_actions": Counter(),
                            "personality_strategies": Counter(),
                            "baseline_strategies": Counter(),
                        },
                    )
                    b_action = str(row.get(str(spec["action_key"]), ""))
                    b_strategy = str(row.get(str(spec["strategy_key"]), ""))
                    raw_b_text = row.get(str(spec["text_key"]), "")
                    b_text = "" if raw_b_text is None else str(raw_b_text)
                    if not b_text:
                        continue
                    bucket["rows"] = int(bucket["rows"]) + 1
                    bucket["action_agree"] = int(bucket["action_agree"]) + int(p_action == b_action)
                    bucket["strategy_agree"] = int(bucket["strategy_agree"]) + int(p_strategy == b_strategy)
                    bucket["exact_duplicate"] = int(bucket["exact_duplicate"]) + int(p_text == b_text)
                    p_template = str(row.get("personality_template_id", ""))
                    template_key = str(spec.get("template_key", ""))
                    b_template = str(row.get(template_key, "")) if template_key else ""
                    if p_template and b_template:
                        bucket["template_rows"] = int(bucket["template_rows"]) + 1
                        bucket["template_agree"] = int(bucket["template_agree"]) + int(p_template == b_template)
                    bucket["text_similarity_values"].append(_safe_float(row.get(str(spec["text_similarity_key"]))))  # type: ignore[union-attr]
                    bucket["length_deltas"].append(p_chars - int(row.get(str(spec["chars_key"]), 0) or 0))  # type: ignore[union-attr]
                    b_sem = _safe_float(row.get(str(spec["semantic_score_key"])), default=p_sem)
                    bucket["semantic_deltas"].append(p_sem - b_sem)  # type: ignore[union-attr]
                    bucket["baseline_semantic_values"].append(b_sem)  # type: ignore[union-attr]
                    bucket["personality_actions"].update([p_action])  # type: ignore[union-attr]
                    bucket["baseline_actions"].update([b_action])  # type: ignore[union-attr]
                    bucket["personality_strategies"].update([p_strategy])  # type: ignore[union-attr]
                    bucket["baseline_strategies"].update([b_strategy])  # type: ignore[union-attr]
                    if baseline == "baseline_a":
                        per_user[int(report.user_uid)]["a_action_agreements"].append(float(p_action == b_action))  # type: ignore[index, union-attr]
                        per_user[int(report.user_uid)]["a_text_similarities"].append(_safe_float(row.get(str(spec["text_similarity_key"]))))  # type: ignore[index, union-attr]
                    elif baseline == "baseline_c":
                        per_user[int(report.user_uid)]["c_action_agreements"].append(float(p_action == b_action))  # type: ignore[index, union-attr]
                        per_user[int(report.user_uid)]["c_text_similarities"].append(_safe_float(row.get(str(spec["text_similarity_key"]))))  # type: ignore[index, union-attr]
    summarized: dict[str, dict[str, object]] = {}
    for baseline, bucket in overall.items():
        rows = int(bucket["rows"])
        text_vals = list(bucket["text_similarity_values"])  # type: ignore[arg-type]
        length_vals = list(bucket["length_deltas"])  # type: ignore[arg-type]
        sem_vals = list(bucket["semantic_deltas"])  # type: ignore[arg-type]
        baseline_sem_vals = list(bucket["baseline_semantic_values"])  # type: ignore[arg-type]
        p_actions = bucket["personality_actions"]  # type: ignore[assignment]
        b_actions = bucket["baseline_actions"]  # type: ignore[assignment]
        p_strats = bucket["personality_strategies"]  # type: ignore[assignment]
        b_strats = bucket["baseline_strategies"]  # type: ignore[assignment]
        summarized[baseline] = {
            "rows": rows,
            "action_agreement_rate": round(float(bucket["action_agree"]) / float(rows), 6) if rows else 0.0,
            "strategy_agreement_rate": round(float(bucket["strategy_agree"]) / float(rows), 6) if rows else 0.0,
            "exact_duplicate_rate": round(float(bucket["exact_duplicate"]) / float(rows), 6) if rows else 0.0,
            "template_agreement_rate": (
                round(float(bucket["template_agree"]) / float(bucket["template_rows"]), 6)
                if int(bucket["template_rows"])
                else 0.0
            ),
            "mean_text_similarity": round(float(mean(text_vals)) if text_vals else 0.0, 6),
            "mean_length_delta": round(float(mean(length_vals)) if length_vals else 0.0, 6),
            "mean_semantic_delta": round(float(mean(sem_vals)) if sem_vals else 0.0, 6),
            "mean_baseline_semantic": round(float(mean(baseline_sem_vals)) if baseline_sem_vals else 0.0, 6),
            "action_distribution_delta": _counter_jsd(p_actions, b_actions),  # type: ignore[arg-type]
            "strategy_jsd": _counter_jsd(p_strats, b_strats),  # type: ignore[arg-type]
            "personality_action_distribution": dict(p_actions),  # type: ignore[arg-type]
            "baseline_action_distribution": dict(b_actions),  # type: ignore[arg-type]
        }
    user_rows: list[dict[str, object]] = []
    for uid, bucket in per_user.items():
        deltas = list(bucket.get("semantic_deltas", []))  # type: ignore[arg-type]
        if not deltas:
            continue
        user_rows.append(
            {
                "user_uid": int(uid),
                "mean_semantic_delta": round(float(mean(deltas)), 6),
                "baseline_a_action_agreement_rate": round(_mean_or_zero(bucket.get("a_action_agreements")), 6),
                "baseline_a_text_similarity_mean": round(_mean_or_zero(bucket.get("a_text_similarities")), 6),
                "baseline_c_action_agreement_rate": round(_mean_or_zero(bucket.get("c_action_agreements")), 6),
                "baseline_c_text_similarity_mean": round(_mean_or_zero(bucket.get("c_text_similarities")), 6),
            }
        )
    user_rows.sort(key=lambda row: float(row["mean_semantic_delta"]))
    b_summary = summarized.get("baseline_b_best", {})
    warning = bool(
        b_summary
        and int(b_summary.get("rows", 0) or 0) > 0
        and (
            _safe_float(b_summary.get("mean_text_similarity")) >= 0.85
            or abs(_safe_float(b_summary.get("mean_semantic_delta"))) <= 0.005
        )
    )
    c_summary = summarized.get("baseline_c", {})
    c_reasons: list[str] = []
    c_weak_reasons: list[str] = []
    if c_summary:
        if _safe_float(c_summary.get("action_agreement_rate")) >= 0.90:
            c_reasons.append("action_agreement_high")
        if _safe_float(c_summary.get("mean_text_similarity")) >= 0.85:
            c_reasons.append("text_similarity_high")
        if _safe_float(c_summary.get("template_agreement_rate")) >= 0.75:
            c_reasons.append("template_overlap_high")
        if abs(_safe_float(c_summary.get("mean_semantic_delta"))) <= 0.005:
            c_reasons.append("semantic_delta_near_zero")
        if _safe_float(c_summary.get("mean_text_similarity")) <= 0.05:
            c_weak_reasons.append("text_similarity_low")
        if _safe_float(c_summary.get("mean_baseline_semantic")) <= 0.01:
            c_weak_reasons.append("semantic_mean_low")
    baseline_c_too_close = bool(c_reasons)
    baseline_c_too_weak = bool(c_weak_reasons)
    return {
        "baselines": summarized,
        "per_user_rankings": {
            "worst": user_rows[:5],
            "best": list(reversed(user_rows[-5:])),
        },
        "wrong_user_masked_by_generic_template_warning": warning,
        "baseline_c_too_close_warning": baseline_c_too_close,
        "baseline_c_too_close_reason": ",".join(c_reasons),
        "baseline_c_too_weak_warning": baseline_c_too_weak,
        "baseline_c_too_weak_reason": ",".join(c_weak_reasons),
        "baseline_c_too_close_fields": {
            "action_agreement_rate": _safe_float(c_summary.get("action_agreement_rate")),
            "mean_text_similarity": _safe_float(c_summary.get("mean_text_similarity")),
            "template_agreement_rate": _safe_float(c_summary.get("template_agreement_rate")),
            "mean_semantic_delta": _safe_float(c_summary.get("mean_semantic_delta")),
            "mean_baseline_semantic": _safe_float(c_summary.get("mean_baseline_semantic")),
        },
    }


def _ablation_summary(reports: list[ValidationReport]) -> dict[str, object]:
    by_name: dict[str, list[dict[str, object]]] = {}
    for report in reports:
        for strategy_result in report.per_strategy.values():
            rows = strategy_result.get("ablation_summary", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    by_name.setdefault(str(row.get("name", "unknown")), []).append(row)
    return {
        name: {
            "count": len(rows),
            "semantic_mean": round(float(mean([_safe_float(row.get("semantic_mean")) for row in rows])), 6),
            "semantic_vs_baseline_a_diff_mean": round(
                float(mean([_safe_float(row.get("semantic_vs_baseline_a_diff")) for row in rows])),
                6,
            ),
            "action_agreement_vs_personality_mean": round(
                float(mean([_safe_float(row.get("action_agreement_vs_personality")) for row in rows])),
                6,
            ),
            "text_similarity_vs_personality_mean": round(
                float(mean([_safe_float(row.get("text_similarity_vs_personality")) for row in rows])),
                6,
            ),
        }
        for name, rows in sorted(by_name.items())
        if rows
    }


def _profile_expression_source_summary(reports: list[ValidationReport]) -> dict[str, object]:
    buckets = {
        "personality": {
            "source_key": "personality_profile_expression_sources",
            "move_key": "personality_rhetorical_move",
        },
        "baseline_c": {
            "source_key": "baseline_c_profile_expression_sources",
            "move_key": "baseline_c_rhetorical_move",
        },
    }
    summary: dict[str, object] = {}
    for name, spec in buckets.items():
        source_counts: Counter[str] = Counter()
        move_counts: Counter[str] = Counter()
        rows = 0
        for report in reports:
            for strategy_result in report.per_strategy.values():
                trace = strategy_result.get("diagnostic_trace", [])
                if not isinstance(trace, list):
                    continue
                for row in trace:
                    if not isinstance(row, dict):
                        continue
                    rows += 1
                    raw_sources = row.get(str(spec["source_key"]), [])
                    if isinstance(raw_sources, list):
                        sources = [str(item) for item in raw_sources if str(item)]
                    else:
                        sources = [part for part in str(raw_sources).split(",") if part]
                    if not sources:
                        sources = ["generic"]
                    source_counts.update(sources)
                    move = str(row.get(str(spec["move_key"]), "") or "unknown")
                    move_counts.update([move])
        summary[name] = {
            "rows": int(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "source_rates": {
                key: round(float(value) / float(max(1, rows)), 6)
                for key, value in sorted(source_counts.items())
            },
            "rhetorical_move_counts": dict(sorted(move_counts.items())),
            "rhetorical_move_rates": {
                key: round(float(value) / float(max(1, rows)), 6)
                for key, value in sorted(move_counts.items())
            },
        }
    return summary


def _state_saturation_summary(reports: list[ValidationReport]) -> dict[str, object]:
    personality_values: list[float] = []
    distances: dict[str, list[float]] = {}
    variances: dict[str, list[float]] = {}
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if not _strategy_eligible(strategy_result):
                continue
            p = strategy_result.get("personality_metrics", {})
            if isinstance(p, dict) and "personality_similarity" in p:
                personality_values.append(_safe_float(p.get("personality_similarity")))
            state = strategy_result.get("state_distance_diagnostics", {})
            if isinstance(state, dict):
                for key in ("train_full", "train_default", "train_wrong_user"):
                    row = state.get(key)
                    if isinstance(row, dict):
                        distances.setdefault(f"{key}_cosine", []).append(_safe_float(row.get("cosine")))
                        distances.setdefault(f"{key}_l2", []).append(_safe_float(row.get("l2")))
                var_row = state.get("per_dimension_variance")
                if isinstance(var_row, dict):
                    for key, value in var_row.items():
                        variances.setdefault(str(key), []).append(_safe_float(value))
    value_sd = 0.0
    if len(personality_values) > 1:
        value_sd = math.sqrt(sum((x - mean(personality_values)) ** 2 for x in personality_values) / len(personality_values))
    saturated = bool(personality_values and (value_sd <= 1e-6 or min(personality_values) >= 0.999))
    return {
        "personality_similarity": {
            "count": len(personality_values),
            "mean": round(float(mean(personality_values)) if personality_values else 0.0, 6),
            "std": round(float(value_sd), 6),
            "saturation_warning": saturated,
            "diagnostic_only": True,
        },
        "state_distances": {
            key: round(float(mean(values)) if values else 0.0, 6)
            for key, values in sorted(distances.items())
        },
        "per_dimension_variance_mean": {
            key: round(float(mean(values)) if values else 0.0, 6)
            for key, values in sorted(variances.items())
        },
    }


def _debug_readiness_gate(
    *,
    baseline_audit_summary: dict[str, object],
    ablation_summary: dict[str, object],
    state_saturation_summary: dict[str, object],
    comparisons: dict[str, dict[str, object]],
) -> dict[str, object]:
    distances = state_saturation_summary.get("state_distances", {})
    distances = distances if isinstance(distances, dict) else {}
    train_default_l2 = _safe_float(distances.get("train_default_l2"))
    train_wrong_user_l2 = _safe_float(distances.get("train_wrong_user_l2"))
    wrong_user_warning = bool(
        baseline_audit_summary.get("wrong_user_masked_by_generic_template_warning", False)
    )
    baseline_c_warning = bool(baseline_audit_summary.get("baseline_c_too_close_warning", False))
    no_surface = ablation_summary.get("no_surface_profile", {})
    no_surface_diff = (
        _safe_float(no_surface.get("semantic_vs_baseline_a_diff_mean"))
        if isinstance(no_surface, dict)
        else 0.0
    )
    full_diff = _safe_float(
        comparisons.get("semantic_similarity", {}).get("vs_a_mean_diff")
        if isinstance(comparisons.get("semantic_similarity"), dict)
        else None
    )
    no_surface_present = isinstance(no_surface, dict) and bool(no_surface)
    checks = {
        "train_default_l2_positive": train_default_l2 > 0.0,
        "train_wrong_user_l2_positive": train_wrong_user_l2 > 0.0,
        "wrong_user_masked_warning_false": not wrong_user_warning,
        "no_surface_not_better_than_full": bool((not no_surface_present) or no_surface_diff <= full_diff),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "ablation_evaluated": bool(no_surface_present),
        "train_default_l2_mean": round(float(train_default_l2), 6),
        "train_wrong_user_l2_mean": round(float(train_wrong_user_l2), 6),
        "full_personality_semantic_vs_baseline_a_diff_mean": round(float(full_diff), 6),
        "no_surface_profile_semantic_vs_baseline_a_diff_mean": round(float(no_surface_diff), 6),
        "wrong_user_masked_by_generic_template_warning": wrong_user_warning,
        "baseline_c_too_close_warning": baseline_c_warning,
        "baseline_c_too_close_reason": str(
            baseline_audit_summary.get("baseline_c_too_close_reason", "")
        ),
        "policy": "debug-only gate for state/profile differentiation before mini-formal",
    }


def _diagnostic_trace_gate(
    *,
    formal_requested: bool,
    diagnostic_trace_rows: int,
) -> dict[str, object]:
    passed = (not formal_requested) or int(diagnostic_trace_rows) > 0
    return {
        "passed": bool(passed),
        "formal_requested": bool(formal_requested),
        "diagnostic_trace_rows": int(diagnostic_trace_rows),
        "policy": "formal acceptance requires non-empty diagnostic_trace.jsonl",
    }


def _agent_state_differentiation_gate(
    reports: list[ValidationReport],
    *,
    formal_requested: bool,
) -> dict[str, object]:
    full_l2: list[float] = []
    default_l2: list[float] = []
    wrong_l2: list[float] = []
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if not _strategy_eligible(strategy_result):
                continue
            diag = strategy_result.get("state_distance_diagnostics", {})
            if not isinstance(diag, dict):
                continue
            train_full = diag.get("train_full", {})
            train_default = diag.get("train_default", {})
            train_wrong = diag.get("train_wrong_user", {})
            if not isinstance(train_full, dict) or not isinstance(train_default, dict):
                continue
            full_l2.append(_safe_float(train_full.get("l2")))
            default_l2.append(_safe_float(train_default.get("l2")))
            if isinstance(train_wrong, dict):
                wrong_l2.append(_safe_float(train_wrong.get("l2")))
    full_mean = float(mean(full_l2)) if full_l2 else 0.0
    default_mean = float(mean(default_l2)) if default_l2 else 0.0
    wrong_mean = float(mean(wrong_l2)) if wrong_l2 else 0.0
    default_ok = bool(full_l2 and default_l2 and full_mean < default_mean)
    wrong_ok = bool(full_l2 and wrong_l2 and full_mean < wrong_mean)
    diagnostic_passed = bool(default_ok and wrong_ok)
    passed = (not formal_requested) or diagnostic_passed
    return {
        "passed": bool(passed),
        "formal_requested": bool(formal_requested),
        "diagnostic_passed": bool(diagnostic_passed),
        "comparisons_evaluated": int(len(full_l2)),
        "train_full_l2_mean": round(float(full_mean), 6),
        "train_default_l2_mean": round(float(default_mean), 6),
        "train_wrong_user_l2_mean": round(float(wrong_mean), 6),
        "train_full_closer_than_default": bool(default_ok),
        "train_full_closer_than_wrong_user": bool(wrong_ok),
        "policy": "train-only state must be closer to full-data state than default or wrong-user state",
    }


def _behavioral_majority_baseline_gate(
    reports: list[ValidationReport],
    *,
    formal_requested: bool,
) -> dict[str, object]:
    personality_values: list[float] = []
    majority_values: list[float] = []
    for report in reports:
        for strategy_result in report.per_strategy.values():
            if not _strategy_eligible(strategy_result):
                continue
            p = strategy_result.get("personality_metrics", {})
            majority = strategy_result.get("majority_baseline_metrics", {})
            if not isinstance(p, dict) or not isinstance(majority, dict):
                continue
            if "behavioral_similarity_strategy" not in p or "behavioral_similarity_strategy" not in majority:
                continue
            personality_values.append(_safe_float(p.get("behavioral_similarity_strategy")))
            majority_values.append(_safe_float(majority.get("behavioral_similarity_strategy")))
    personality_mean = float(mean(personality_values)) if personality_values else 0.0
    majority_mean = float(mean(majority_values)) if majority_values else 0.0
    warning = bool(majority_values and majority_mean >= personality_mean - 1e-12)
    passed = (not formal_requested) or bool(majority_values and not warning)
    return {
        "passed": bool(passed),
        "formal_requested": bool(formal_requested),
        "behavioral_metric_majority_warning": bool(warning),
        "personality_behavioral_strategy_mean": round(float(personality_mean), 6),
        "majority_behavioral_strategy_mean": round(float(majority_mean), 6),
        "comparisons_evaluated": int(len(majority_values)),
        "policy": "formal acceptance blocks when train-majority behavioral baseline matches or beats personality",
    }


def _baseline_c_behavioral_failure_audit(reports: list[ValidationReport]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for report in reports:
        for strategy_key, strategy_result in report.per_strategy.items():
            if not _strategy_eligible(strategy_result):
                continue
            trace = strategy_result.get("diagnostic_trace", [])
            if not isinstance(trace, list):
                trace = []
            real_strategy: Counter[str] = Counter()
            personality_strategy: Counter[str] = Counter()
            baseline_c_strategy: Counter[str] = Counter()
            real_action: Counter[str] = Counter()
            personality_action: Counter[str] = Counter()
            baseline_c_action: Counter[str] = Counter()
            for item in trace:
                if not isinstance(item, dict):
                    continue
                if item.get("real_text") is not None:
                    # Real action labels are not persisted in diagnostic rows; the real
                    # distribution is available from metric details only for majority rows.
                    pass
                personality_strategy.update([str(item.get("personality_strategy", "unknown"))])
                baseline_c_strategy.update([str(item.get("baseline_c_strategy", "unknown"))])
                personality_action.update([str(item.get("personality_action", "unknown"))])
                baseline_c_action.update([str(item.get("baseline_c_action", "unknown"))])
            majority = strategy_result.get("majority_baseline_metrics", {})
            if isinstance(majority, dict):
                real_strategy_raw = majority.get("real_strategy_distribution", {})
                real_action_raw = majority.get("real_action_distribution", {})
                real_strategy_map = real_strategy_raw if isinstance(real_strategy_raw, dict) else {}
                real_action_map = real_action_raw if isinstance(real_action_raw, dict) else {}
                real_strategy.update(
                    {
                        str(k): int(v)
                        for k, v in real_strategy_map.items()
                    }
                )
                real_action.update(
                    {
                        str(k): int(v)
                        for k, v in real_action_map.items()
                    }
                )
            p = strategy_result.get("personality_metrics", {})
            c = strategy_result.get("baseline_c_metrics", {})
            rows.append(
                {
                    "user_uid": int(report.user_uid),
                    "strategy": str(strategy_key),
                    "pair_count": int(len(trace)),
                    "real_strategy_distribution": dict(real_strategy),
                    "personality_strategy_distribution": dict(personality_strategy),
                    "baseline_c_strategy_distribution": dict(baseline_c_strategy),
                    "majority_strategy": majority.get("majority_strategy") if isinstance(majority, dict) else None,
                    "real_action_distribution": dict(real_action),
                    "personality_action_distribution": dict(personality_action),
                    "baseline_c_action_distribution": dict(baseline_c_action),
                    "majority_action": majority.get("majority_action") if isinstance(majority, dict) else None,
                    "personality_behavioral_strategy": (
                        _safe_float(p.get("behavioral_similarity_strategy")) if isinstance(p, dict) else 0.0
                    ),
                    "baseline_c_behavioral_strategy": (
                        _safe_float(c.get("behavioral_similarity_strategy")) if isinstance(c, dict) else 0.0
                    ),
                    "majority_behavioral_strategy": (
                        _safe_float(majority.get("behavioral_similarity_strategy"))
                        if isinstance(majority, dict)
                        else 0.0
                    ),
                    "personality_balanced_behavioral_strategy": (
                        _safe_float(p.get("balanced_behavioral_similarity_strategy")) if isinstance(p, dict) else 0.0
                    ),
                    "majority_balanced_behavioral_strategy": (
                        _safe_float(majority.get("balanced_behavioral_similarity_strategy"))
                        if isinstance(majority, dict)
                        else 0.0
                    ),
                }
            )
    return {
        "rows": rows,
        "row_count": int(len(rows)),
        "policy": "diagnostic distribution audit for Baseline C and train-majority behavioral baselines",
    }


def _write_diagnostic_trace(reports: list[ValidationReport], output_dir: Path) -> int:
    rows: list[dict[str, object]] = []
    for report in reports:
        for strategy_key, strategy_result in report.per_strategy.items():
            trace = strategy_result.get("diagnostic_trace", [])
            if not isinstance(trace, list):
                continue
            for row in trace:
                if not isinstance(row, dict):
                    continue
                payload = dict(row)
                payload["user_uid"] = int(report.user_uid)
                payload.setdefault("strategy", strategy_key)
                rows.append(payload)
    if not rows:
        return 0
    path = output_dir / "diagnostic_trace.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


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

    metric_hard_pass, metric_hard_breakdown = _compute_hard_pass(
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
    required_users = _required_users(reports)
    formal_requested = _formal_requested(reports)
    diagnostic_trace_rows = _write_diagnostic_trace(reports, output_dir)
    pilot_gate = _pilot_gate(reports, required_users)
    split_gate = _split_gate(per_strategy_comparisons, required_users)
    partner_gate = _partner_gate(per_strategy_hard_pass, per_strategy_comparisons, required_users)
    topic_gate = _topic_gate(per_strategy_hard_pass, topic_split_summary, required_users)
    baseline_c_gate = _formal_baseline_c_gate(
        reports,
        required_users=required_users,
        formal_requested=formal_requested,
    )
    classifier_gate_summary = _classifier_gate_summary(reports)
    semantic_engine_gate = _semantic_engine_gate(reports)
    diagnostic_trace_gate = _diagnostic_trace_gate(
        formal_requested=formal_requested,
        diagnostic_trace_rows=diagnostic_trace_rows,
    )
    agent_state_differentiation_gate = _agent_state_differentiation_gate(
        reports,
        formal_requested=formal_requested,
    )
    behavioral_majority_baseline_gate = _behavioral_majority_baseline_gate(
        reports,
        formal_requested=formal_requested,
    )
    statistical_gate = {
        "passed": bool(scipy_wilcoxon_available()),
        "engine": "scipy.stats.wilcoxon",
        "policy": "formal acceptance requires scipy.stats.wilcoxon(alternative='greater')",
    }
    reproducibility_gate = {
        "passed": bool(baseline_c_gate["passed"]),
        "baseline_c_leave_one_out_population_average": bool(baseline_c_gate["passed"]),
    }
    formal_acceptance_eligible = bool(
        classifier_gate
        and semantic_engine_gate["passed"]
        and statistical_gate["passed"]
        and baseline_c_gate["passed"]
        and diagnostic_trace_gate["passed"]
        and agent_state_differentiation_gate["passed"]
        and behavioral_majority_baseline_gate["passed"]
    )
    hard_pass_breakdown = {
        **metric_hard_breakdown,
        "metric_hard_pass": bool(metric_hard_pass),
        "formal_acceptance_eligible": bool(formal_acceptance_eligible),
        "semantic_embedding_gate": bool(semantic_engine_gate["passed"]),
        "statistical_gate": bool(statistical_gate["passed"]),
        "pilot_gate": bool(pilot_gate["passed"]),
        "split_gate_all_required_strategies": bool(split_gate["passed"]),
        "partner_gate": bool(partner_gate["passed"]),
        "topic_gate": bool(topic_gate["passed"]),
        "reproducibility_gate": bool(reproducibility_gate["passed"]),
        "baseline_c_leave_one_out_population_average": bool(baseline_c_gate["passed"]),
        "diagnostic_trace_gate": bool(diagnostic_trace_gate["passed"]),
        "agent_state_differentiation_gate": bool(agent_state_differentiation_gate["passed"]),
        "behavioral_majority_baseline_gate": bool(behavioral_majority_baseline_gate["passed"]),
    }
    hard_pass = bool(
        metric_hard_pass
        and formal_acceptance_eligible
        and pilot_gate["passed"]
        and split_gate["passed"]
        and partner_gate["passed"]
        and topic_gate["passed"]
        and reproducibility_gate["passed"]
        and baseline_c_gate["passed"]
        and diagnostic_trace_gate["passed"]
        and agent_state_differentiation_gate["passed"]
        and behavioral_majority_baseline_gate["passed"]
    )
    if hard_pass:
        overall_conclusion = "pass"
    elif bool(metric_hard_pass):
        overall_conclusion = "partial"
    else:
        overall_conclusion = "fail"
    semantic_delta_summary = _semantic_delta_summary(reports)
    baseline_audit_summary = _baseline_audit_summary(reports)
    ablation_summary = _ablation_summary(reports)
    profile_expression_summary = _profile_expression_source_summary(reports)
    state_saturation_summary = _state_saturation_summary(reports)
    debug_readiness_gate = _debug_readiness_gate(
        baseline_audit_summary=baseline_audit_summary,
        ablation_summary=ablation_summary,
        state_saturation_summary=state_saturation_summary,
        comparisons=comparisons,
    )
    baseline_c_behavioral_failure_audit = _baseline_c_behavioral_failure_audit(reports)

    acceptance_rules = {
        "version": ACCEPTANCE_RULES_VERSION,
        "hard_metrics": list(hard_metrics_list),
        "soft_metrics": list(soft_metrics_list),
        "semantic_similarity": "paired Wilcoxon one-sided greater vs baseline A (p < 0.05) and mean paired diff > 0; one paired sample per user (mean across split strategies)",
        "behavioral_similarity_strategy": "paired Wilcoxon one-sided greater vs baseline C when 3-class classifier gate passes; otherwise soft-only",
        "agent_state_similarity": f"mean across users >= {AGENT_STATE_MIN_MEAN} (no baseline significance required)",
        "statistical_engine": "scipy.stats.wilcoxon(alternative='greater'); no fallback p-values are valid for formal acceptance",
        "classifier_gate": "formal acceptance requires independent non-fixture train/gate labels, class minima, separation, non-TFIDF embedding engine, cue override <= threshold, without-cue 3-class macro-F1 gate pass, and overall 3-class macro-F1 gate pass",
        "semantic_embedding_gate": "formal acceptance requires sentence_embedding_cosine; TF-IDF fallback is development-only",
        "aggregation": "per_user_mean_across_strategies",
        "per_strategy_comparisons": "same Wilcoxon rules computed separately per split strategy (see per_strategy_comparisons)",
        "agent_state_similarity_detail": "mean of per-user means across strategies; threshold on global mean; not a baseline contrast",
        "pilot": "run_pilot_validation estimates sd of per-user semantic (personality vs baseline A) and behavioral (personality vs baseline C) mean differences; suggested_min_users uses max of thresholds when either exceeds pilot_sd_threshold",
        "topic_gate": "topic_split_not_applicable users are excluded from topic statistics, but valid topic users must still meet required_users",
        "formal_baseline_c": "Baseline C must use leave-one-out population averaging and skip_population_average_implant is test-only",
        "diagnostic_trace_gate": "formal acceptance requires non-empty diagnostic trace and audit summaries",
        "behavioral_majority_baseline_gate": "formal acceptance blocks when train-majority behavioral baseline matches or beats personality",
        "agent_state_differentiation_gate": "train-only state must be closer to full-data state than default or wrong-user state",
    }

    metric_interpretations = {
        "agent_state_similarity": comparisons["agent_state_similarity"].get("interpretation_notes"),
    }

    aggregate_payload = {
        "user_count": int(len(reports)),
        "users_tested": int(users_used),
        "users_skipped_no_strategy": int(users_skipped_no_strategy),
        "required_users": int(required_users),
        "agent_state_users_tested": int(ast_users_used),
        "agent_state_users_skipped_no_metric": int(ast_users_skipped),
        "metric_version": "m54_v3",
        "behavioral_labeling": "generated_action_direct_real_reply_classifier",
        "classifier_gate": classifier_gate_summary,
        "semantic_engine_gate": semantic_engine_gate,
        "statistical_gate": statistical_gate,
        "formal_acceptance_eligible": bool(formal_acceptance_eligible),
        "behavioral_hard_metric_degraded": bool(behavioral_degraded),
        "formal_requested": bool(formal_requested),
        "comparisons": comparisons,
        "per_strategy_comparisons": per_strategy_comparisons,
        "per_strategy_hard_pass": per_strategy_hard_pass,
        "partner_hard_pass": partner_hard_pass,
        "topic_hard_pass": topic_hard_pass,
        "topic_split_summary": topic_split_summary,
        "pilot_gate": pilot_gate,
        "split_gate": split_gate,
        "partner_gate": partner_gate,
        "topic_gate": topic_gate,
        "reproducibility_gate": reproducibility_gate,
        "baseline_c_gate": baseline_c_gate,
        "diagnostic_trace_gate": diagnostic_trace_gate,
        "agent_state_differentiation_gate": agent_state_differentiation_gate,
        "behavioral_majority_baseline_gate": behavioral_majority_baseline_gate,
        "metric_interpretations": metric_interpretations,
        "semantic_delta_summary": semantic_delta_summary,
        "baseline_audit_summary": baseline_audit_summary,
        "ablation_summary": ablation_summary,
        "profile_expression_source_summary": profile_expression_summary,
        "state_saturation_summary": state_saturation_summary,
        "debug_readiness_gate": debug_readiness_gate,
        "baseline_c_behavioral_failure_audit": baseline_c_behavioral_failure_audit,
        "baseline_c_behavioral_failure_audit_json": str(
            (output_dir / "baseline_c_behavioral_failure_audit.json").as_posix()
        ),
        "baseline_audit_summary_json": str((output_dir / "baseline_audit_summary.json").as_posix()),
        "ablation_summary_json": str((output_dir / "ablation_summary.json").as_posix()),
        "profile_expression_source_summary_json": str(
            (output_dir / "profile_expression_source_summary.json").as_posix()
        ),
        "state_saturation_summary_json": str((output_dir / "state_saturation_summary.json").as_posix()),
        "diagnostic_trace_jsonl": (
            str((output_dir / "diagnostic_trace.jsonl").as_posix())
            if diagnostic_trace_rows
            else None
        ),
        "diagnostic_trace_rows": int(diagnostic_trace_rows),
        "hard_metrics": list(hard_metrics_list),
        "soft_metrics": list(soft_metrics_list),
        "acceptance_rules": acceptance_rules,
        "hard_pass": bool(hard_pass),
        "hard_pass_breakdown": hard_pass_breakdown,
        "overall_conclusion": overall_conclusion,
        "reports": [asdict(item) for item in reports],
    }
    _write_json(output_dir / "baseline_audit_summary.json", baseline_audit_summary)
    _write_json(output_dir / "ablation_summary.json", ablation_summary)
    _write_json(output_dir / "profile_expression_source_summary.json", profile_expression_summary)
    _write_json(output_dir / "state_saturation_summary.json", state_saturation_summary)
    _write_json(output_dir / "baseline_c_behavioral_failure_audit.json", baseline_c_behavioral_failure_audit)
    _write_json(output_dir / "aggregate_report.json", aggregate_payload)

    lines = [
        "# M5.4 Validation Aggregate Report",
        "",
        f"- Users: {len(reports)} (tested: {users_used}, skipped no strategy: {users_skipped_no_strategy})",
        f"- Required users: {required_users}",
        f"- Agent state: users with metric {ast_users_used}, skipped {ast_users_skipped}",
        f"- Topic split: {topic_split_summary}",
        f"- Metric version: {aggregate_payload['metric_version']} ({aggregate_payload['behavioral_labeling']})",
        f"- Classifier 3-class gate: {classifier_gate}",
        f"- Semantic embedding gate: {semantic_engine_gate['passed']}",
        f"- Statistical gate: {statistical_gate['passed']}",
        f"- Formal acceptance eligible: {formal_acceptance_eligible}",
        f"- Behavioral hard metric degraded (soft-only): {behavioral_degraded}",
        f"- Overall conclusion: {aggregate_payload['overall_conclusion']}",
        f"- Hard pass: {hard_pass}",
        f"- Pilot gate: {pilot_gate['passed']}",
        f"- Split gate: {split_gate['passed']}",
        f"- Partner strategy hard pass: {partner_hard_pass}",
        f"- Topic strategy hard pass: {topic_hard_pass}",
        f"- Formal Baseline C gate: {baseline_c_gate['passed']}",
        f"- Diagnostic trace gate: {diagnostic_trace_gate['passed']}",
        f"- Agent-state differentiation gate: {agent_state_differentiation_gate['passed']}",
        f"- Behavioral majority baseline gate: {behavioral_majority_baseline_gate['passed']}",
        f"- Diagnostic trace rows: {diagnostic_trace_rows}",
        "",
        "## Acceptance (hard metrics)",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check_name, ok in hard_pass_breakdown.items():
        lines.append(f"| {check_name} | {ok} |")
    user_delta = semantic_delta_summary["users"]
    lines.extend(
        [
            "",
            "## Semantic Delta Diagnostics",
            "",
            f"- Users positive/negative/zero: {user_delta['positive']} / {user_delta['negative']} / {user_delta['zero']}",
            f"- User delta median: {_fmt_cell_num(user_delta['median'])}; IQR: {_fmt_cell_num(user_delta['iqr'])}",
            f"- Pair-count distribution: {semantic_delta_summary['pair_count_distribution']}",
            "",
            "| Strategy | mean P-A delta | positive | negative | median | IQR |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sk, row in semantic_delta_summary["per_strategy"].items():
        lines.append(
            f"| `{sk}` | {_fmt_cell_num(row['mean_delta'])} | {row['positive']} | {row['negative']} | "
            f"{_fmt_cell_num(row['median'])} | {_fmt_cell_num(row['iqr'])} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Audit Diagnostics",
            "",
            f"- Wrong-user masked warning: {baseline_audit_summary['wrong_user_masked_by_generic_template_warning']}",
            f"- Baseline C too-close warning: {baseline_audit_summary['baseline_c_too_close_warning']} "
            f"({baseline_audit_summary['baseline_c_too_close_reason']})",
            f"- Baseline C too-weak warning (diagnostic-only): "
            f"{baseline_audit_summary['baseline_c_too_weak_warning']} "
            f"({baseline_audit_summary['baseline_c_too_weak_reason']})",
            "",
            "| Baseline | rows | action agree | strategy agree | template agree | text sim | duplicate | semantic delta | action JSD | strategy JSD |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for baseline, row in baseline_audit_summary["baselines"].items():
        lines.append(
            f"| `{baseline}` | {row['rows']} | {_fmt_cell_num(row['action_agreement_rate'])} | "
            f"{_fmt_cell_num(row['strategy_agreement_rate'])} | {_fmt_cell_num(row['template_agreement_rate'])} | "
            f"{_fmt_cell_num(row['mean_text_similarity'])} | "
            f"{_fmt_cell_num(row['exact_duplicate_rate'])} | {_fmt_cell_num(row['mean_semantic_delta'])} | "
            f"{_fmt_cell_num(row['action_distribution_delta'])} | {_fmt_cell_num(row['strategy_jsd'])} |"
            )
    if profile_expression_summary:
        lines.extend(
            [
                "",
                "## Profile Expression Diagnostics",
                "",
                "| Surface | rows | expression source rates | rhetorical move rates |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name, row in profile_expression_summary.items():
            lines.append(
                f"| `{name}` | {row['rows']} | {row['source_rates']} | "
                f"{row['rhetorical_move_rates']} |"
            )
    if ablation_summary:
        lines.extend(
            [
                "",
                "## Ablation Diagnostics",
                "",
                "| Ablation | count | semantic | semantic vs A | action agree vs P | text sim vs P |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, row in ablation_summary.items():
            lines.append(
                f"| `{name}` | {row['count']} | {_fmt_cell_num(row['semantic_mean'])} | "
                f"{_fmt_cell_num(row['semantic_vs_baseline_a_diff_mean'])} | "
                f"{_fmt_cell_num(row['action_agreement_vs_personality_mean'])} | "
                f"{_fmt_cell_num(row['text_similarity_vs_personality_mean'])} |"
            )
    lines.extend(
        [
            "",
            "## State Saturation Diagnostics",
            "",
            f"- Personality similarity diagnostic-only saturation warning: "
            f"{state_saturation_summary['personality_similarity']['saturation_warning']}",
            f"- State distance means: {state_saturation_summary['state_distances']}",
            "",
            "## Debug Readiness Gate",
            "",
            f"- Passed: {debug_readiness_gate['passed']}",
            f"- Checks: {debug_readiness_gate['checks']}",
        ]
    )
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
