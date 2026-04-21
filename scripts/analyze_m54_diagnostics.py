from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping


DEFAULT_ARTIFACT_DIR = Path("artifacts/m54_validation_formal_15x32_llm_partial")
ABLATION_NAMES = (
    "no_surface_profile",
    "no_policy_trait_bias",
    "surface_only_default_agent",
)
STRATEGY_LABELS = ("escape", "exploit", "explore")


def _load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = max(0.0, min(1.0, q)) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - float(lo)
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _counter_payload(counter: Counter[str], limit: int = 12) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit)}


def _distribution_entropy(counts: Mapping[str, object], labels: Iterable[str] | None = None) -> float:
    keys = list(labels) if labels is not None else list(counts.keys())
    total = sum(_safe_float(counts.get(key, 0.0)) for key in keys)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    active = 0
    for key in keys:
        p = _safe_float(counts.get(key, 0.0)) / total
        if p <= 0.0:
            continue
        active += 1
        entropy -= p * math.log(p)
    if active <= 1:
        return 0.0
    return float(entropy / math.log(float(len(keys))))


def _majority_coverage(counts: Mapping[str, object]) -> float:
    total = sum(_safe_float(value) for value in counts.values())
    if total <= 0.0:
        return 0.0
    return max(_safe_float(value) for value in counts.values()) / total


def _artifact_sanity(
    *,
    aggregate: Mapping[str, object],
    ablation_summary: Mapping[str, object],
    diagnostic_rows: list[dict[str, object]],
    ablation_trace_rows: list[dict[str, object]],
    artifact_dir: Path,
) -> dict[str, object]:
    comparisons = aggregate.get("comparisons", {})
    semantic = comparisons.get("semantic_similarity", {}) if isinstance(comparisons, Mapping) else {}
    baseline_c_mean = _safe_float(semantic.get("baseline_c_mean")) if isinstance(semantic, Mapping) else 0.0
    checks: list[dict[str, object]] = []
    stale = False
    for name in ABLATION_NAMES:
        row = ablation_summary.get(name, {}) if isinstance(ablation_summary, Mapping) else {}
        if not isinstance(row, Mapping):
            continue
        semantic_mean = _safe_float(row.get("semantic_mean"))
        observed = _safe_float(row.get("semantic_vs_baseline_c_diff_mean"))
        expected = semantic_mean - baseline_c_mean
        diff = abs(observed - expected)
        ok = diff <= 1e-5
        stale = stale or not ok
        checks.append(
            {
                "name": name,
                "semantic_mean": _rounded(semantic_mean),
                "baseline_c_mean": _rounded(baseline_c_mean),
                "observed_semantic_vs_baseline_c_diff_mean": _rounded(observed),
                "expected_semantic_vs_baseline_c_diff_mean": _rounded(expected),
                "absolute_error": _rounded(diff),
                "passed": bool(ok),
            }
        )
    missing_files = [
        filename
        for filename in (
            "aggregate_report.json",
            "diagnostic_trace.jsonl",
            "ablation_summary.json",
            "baseline_c_behavioral_failure_audit.json",
        )
        if not (artifact_dir / filename).exists()
    ]
    return {
        "passed": bool(not stale and not missing_files and diagnostic_rows),
        "missing_files": missing_files,
        "diagnostic_trace_rows": int(len(diagnostic_rows)),
        "ablation_trace_rows": int(len(ablation_trace_rows)),
        "ablation_baseline_c_consistency": checks,
        "stale_ablation_baseline_c_warning": bool(stale),
    }


def _baseline_c_diagnosis(diagnostic_rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in diagnostic_rows:
        key = (str(row.get("user_uid", "unknown")), str(row.get("strategy", "unknown")))
        grouped[key].append(_safe_float(row.get("personality_vs_c_pair_delta")))

    per_user_strategy = []
    for (uid, strategy), deltas in sorted(grouped.items()):
        per_user_strategy.append(
            {
                "user_uid": uid,
                "strategy": strategy,
                "turns": int(len(deltas)),
                "mean_delta": _rounded(mean(deltas) if deltas else 0.0),
                "median_delta": _rounded(median(deltas) if deltas else 0.0),
                "p10_delta": _rounded(_pct(deltas, 0.10)),
                "p90_delta": _rounded(_pct(deltas, 0.90)),
                "personality_win_rate": _rounded(sum(1 for value in deltas if value > 0.0) / max(1, len(deltas))),
            }
        )
    worst = sorted(per_user_strategy, key=lambda item: (_safe_float(item["mean_delta"]), str(item["user_uid"])))[:10]

    c_wins = [
        row
        for row in diagnostic_rows
        if _safe_float(row.get("baseline_c_semantic_pair_score")) > _safe_float(row.get("personality_semantic_pair_score"))
    ]
    action = Counter(str(row.get("baseline_c_action", "unknown")) for row in c_wins)
    strategy = Counter(str(row.get("baseline_c_strategy", "unknown")) for row in c_wins)
    template = Counter(str(row.get("baseline_c_template_id", "unknown")) for row in c_wins)
    move = Counter(str(row.get("baseline_c_rhetorical_move", "unknown")) for row in c_wins)
    source = Counter(str(row.get("baseline_c_surface_source", "unknown")) for row in c_wins)
    degraded = Counter(str(row.get("baseline_c_profile_degraded_reason", "none")) for row in c_wins)
    length_bucket = Counter(_length_bucket(_safe_int(row.get("baseline_c_generated_chars"))) for row in c_wins)
    real_length_bucket = Counter(_length_bucket(_safe_int(row.get("real_chars"))) for row in c_wins)

    population_surface_rate = sum(
        1 for row in c_wins if str(row.get("baseline_c_surface_source", "")).startswith("population")
    ) / max(1, len(c_wins))
    short_template_rate = sum(
        1 for row in c_wins if _safe_int(row.get("baseline_c_generated_chars")) <= 12
    ) / max(1, len(c_wins))
    state_like_rate = sum(
        1
        for row in c_wins
        if not str(row.get("baseline_c_surface_source", "")).startswith("population")
        and _safe_int(row.get("baseline_c_generated_chars")) > 12
    ) / max(1, len(c_wins))

    filtered = _baseline_c_filtered_deltas(diagnostic_rows)
    best_filtered_delta = max(
        [
            _safe_float(row.get("mean_personality_vs_c_delta"))
            for row in filtered.values()
            if isinstance(row, Mapping) and _safe_int(row.get("rows")) > 0
        ]
        or [0.0]
    )
    raw_delta = _safe_float(filtered.get("all_rows", {}).get("mean_personality_vs_c_delta")) if isinstance(filtered.get("all_rows"), Mapping) else 0.0
    narrowed = raw_delta < 0.0 and (best_filtered_delta - raw_delta) >= max(0.05, abs(raw_delta) * 0.25)
    no_non_short_rows = (
        isinstance(filtered.get("exclude_baseline_c_short_replies"), Mapping)
        and _safe_int(filtered.get("exclude_baseline_c_short_replies", {}).get("rows")) == 0
    )
    no_non_population_rows = (
        isinstance(filtered.get("exclude_population_surface"), Mapping)
        and _safe_int(filtered.get("exclude_population_surface", {}).get("rows")) == 0
    )
    all_wins_short_population = bool(
        c_wins
        and population_surface_rate >= 0.95
        and short_template_rate >= 0.95
        and no_non_short_rows
        and no_non_population_rows
    )
    suspect = bool(best_filtered_delta > 0.0 or narrowed or all_wins_short_population)

    return {
        "per_user_strategy": per_user_strategy,
        "worst_user_strategies": worst,
        "baseline_c_win_turns": int(len(c_wins)),
        "baseline_c_win_rate": _rounded(len(c_wins) / max(1, len(diagnostic_rows))),
        "win_slice": {
            "baseline_c_action": _counter_payload(action),
            "baseline_c_strategy": _counter_payload(strategy),
            "baseline_c_template_id": _counter_payload(template),
            "baseline_c_rhetorical_move": _counter_payload(move),
            "baseline_c_surface_source": _counter_payload(source),
            "baseline_c_profile_degraded_reason": _counter_payload(degraded),
            "baseline_c_length_bucket": _counter_payload(length_bucket),
            "real_length_bucket": _counter_payload(real_length_bucket),
        },
        "filtered_semantic_deltas": filtered,
        "baseline_c_surface_metric_suspect": suspect,
        "baseline_c_surface_metric_suspect_reason": (
            "filtered_personality_delta_positive"
            if best_filtered_delta > 0.0
            else (
                "filtered_delta_substantially_narrows_raw_gap"
                if narrowed
                else ("all_baseline_c_wins_are_short_population_surface" if all_wins_short_population else "")
            )
        ),
        "interpretation": {
            "population_surface_win_rate": _rounded(population_surface_rate),
            "short_reply_template_win_rate": _rounded(short_template_rate),
            "population_averaged_state_like_win_rate": _rounded(state_like_rate),
        },
    }


def _mean_personality_vs_c_delta(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [_safe_float(row.get("personality_vs_c_pair_delta")) for row in rows]
    wins = sum(1 for value in deltas if value > 0.0)
    return {
        "rows": int(len(rows)),
        "mean_personality_vs_c_delta": _rounded(mean(deltas) if deltas else 0.0),
        "median_personality_vs_c_delta": _rounded(median(deltas) if deltas else 0.0),
        "personality_win_rate": _rounded(wins / max(1, len(deltas))),
    }


def _baseline_c_filtered_deltas(diagnostic_rows: list[dict[str, object]]) -> dict[str, object]:
    all_rows = list(diagnostic_rows)
    non_short_c = [
        row for row in diagnostic_rows if _safe_int(row.get("baseline_c_generated_chars")) > 12
    ]
    non_population = [
        row
        for row in diagnostic_rows
        if not str(row.get("baseline_c_surface_source", "")).startswith("population")
    ]
    length_matched = [
        row
        for row in diagnostic_rows
        if _length_bucket(_safe_int(row.get("baseline_c_generated_chars")))
        == _length_bucket(_safe_int(row.get("real_chars")))
    ]
    non_short_or_population = [
        row
        for row in diagnostic_rows
        if _safe_int(row.get("baseline_c_generated_chars")) > 12
        and not str(row.get("baseline_c_surface_source", "")).startswith("population")
    ]
    return {
        "all_rows": _mean_personality_vs_c_delta(all_rows),
        "exclude_baseline_c_short_replies": _mean_personality_vs_c_delta(non_short_c),
        "exclude_population_surface": _mean_personality_vs_c_delta(non_population),
        "length_bucket_matched_to_real": _mean_personality_vs_c_delta(length_matched),
        "exclude_short_or_population_surface": _mean_personality_vs_c_delta(non_short_or_population),
    }


def _length_bucket(length: int) -> str:
    if length <= 3:
        return "ultra_short"
    if length <= 12:
        return "short"
    if length <= 40:
        return "medium"
    return "long"


def _ablation_diagnosis(
    *,
    ablation_summary: Mapping[str, object],
    ablation_trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    if not ablation_trace_rows:
        return {
            "trace_available": False,
            "summary_only": ablation_summary,
            "policy": "Rerun validation with the new diagnostic trace schema to populate ablation_trace.jsonl.",
        }
    lifts = {
        "trait_policy_lift_full_minus_no_policy": [],
        "surface_lift_full_minus_no_surface": [],
        "surface_anchor_standalone_lift_vs_baseline_c": [],
    }
    close_cases: dict[str, list[dict[str, object]]] = {
        "no_policy_trait_bias": [],
        "surface_only_default_agent": [],
    }
    beating_clusters: dict[str, Counter[str]] = {
        "no_policy_trait_bias": Counter(),
        "surface_only_default_agent": Counter(),
    }
    for row in ablation_trace_rows:
        full_score = _safe_float(row.get("full_personality_semantic_pair_score"))
        no_policy = _safe_float(row.get("no_policy_trait_bias_semantic_pair_score"))
        no_surface = _safe_float(row.get("no_surface_profile_semantic_pair_score"))
        surface_only = _safe_float(row.get("surface_only_default_agent_semantic_pair_score"))
        baseline_c = _safe_float(row.get("baseline_c_semantic_pair_score"))
        lifts["trait_policy_lift_full_minus_no_policy"].append(full_score - no_policy)
        lifts["surface_lift_full_minus_no_surface"].append(full_score - no_surface)
        lifts["surface_anchor_standalone_lift_vs_baseline_c"].append(surface_only - baseline_c)
        for name, score in (
            ("no_policy_trait_bias", no_policy),
            ("surface_only_default_agent", surface_only),
        ):
            if score >= full_score:
                beating_clusters[name].update([_ablation_cluster_key(row, name)])
            close_cases[name].append(
                {
                    "user_uid": row.get("user_uid"),
                    "strategy": row.get("strategy"),
                    "pair_index": row.get("pair_index"),
                    "full_score": _rounded(full_score),
                    "ablation_score": _rounded(score),
                    "ablation_minus_full": _rounded(score - full_score),
                    "full_action": row.get("full_personality_action"),
                    "ablation_action": row.get(f"{name}_action"),
                    "full_template": row.get("full_personality_template_id"),
                    "ablation_template": row.get(f"{name}_template_id"),
                    "full_surface_source": row.get("full_personality_surface_source"),
                    "ablation_surface_source": row.get(f"{name}_surface_source"),
                }
            )
    return {
        "trace_available": True,
        "lift_means": {
            key: _rounded(mean(values) if values else 0.0)
            for key, values in lifts.items()
        },
        "lift_medians": {
            key: _rounded(median(values) if values else 0.0)
            for key, values in lifts.items()
        },
        "close_or_beating_full_cases": {
            key: sorted(rows, key=lambda item: _safe_float(item["ablation_minus_full"]), reverse=True)[:10]
            for key, rows in close_cases.items()
        },
        "beating_full_clusters": {
            key: _counter_payload(counter, limit=16)
            for key, counter in beating_clusters.items()
        },
    }


def _ablation_cluster_key(row: Mapping[str, object], name: str) -> str:
    real_strategy = str(row.get("real_strategy", "unknown"))
    template = str(row.get(f"{name}_template_id", "unknown"))
    surface = str(row.get(f"{name}_surface_source", "unknown"))
    length = _length_bucket(_safe_int(row.get(f"{name}_generated_chars")))
    return f"real={real_strategy}|template={template}|surface={surface}|len={length}"


def _confusion(rows: list[dict[str, object]], pred_key: str, real_key: str = "real_strategy") -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = {label: Counter() for label in STRATEGY_LABELS}
    for row in rows:
        real = str(row.get(real_key, ""))
        pred = str(row.get(pred_key, ""))
        if real not in STRATEGY_LABELS or pred not in STRATEGY_LABELS:
            continue
        matrix[real][pred] += 1
    return {real: dict(counter) for real, counter in matrix.items()}


def _balanced_recall_from_confusion(matrix: Mapping[str, Mapping[str, object]]) -> float:
    recalls: list[float] = []
    for label in STRATEGY_LABELS:
        row = matrix.get(label, {})
        total = sum(_safe_float(value) for value in row.values())
        if total <= 0.0:
            continue
        recalls.append(_safe_float(row.get(label, 0.0)) / total)
    return float(mean(recalls)) if recalls else 0.0


def _majority_rows_as_trace(failure_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in failure_rows:
        real_dist = row.get("real_strategy_distribution", {})
        if not isinstance(real_dist, Mapping):
            continue
        majority = str(row.get("majority_strategy", ""))
        for real, count in real_dist.items():
            for _ in range(_safe_int(count)):
                rows.append({"real_strategy": str(real), "majority_strategy": majority})
    return rows


def _state_collapse_summary(aggregate: Mapping[str, object]) -> dict[str, object]:
    coverages: list[float] = []
    dominant_matches = 0
    evaluated = 0
    reports = aggregate.get("reports", [])
    if not isinstance(reports, list):
        return {"evaluated": 0, "mean_strategy_majority_coverage": 0.0, "policy_dominant_match_rate": 0.0}
    for report in reports:
        if not isinstance(report, Mapping):
            continue
        per_strategy = report.get("per_strategy", {})
        if not isinstance(per_strategy, Mapping):
            continue
        for strategy_result in per_strategy.values():
            if not isinstance(strategy_result, Mapping) or strategy_result.get("skipped"):
                continue
            summary = strategy_result.get("state_calibration_summary", {})
            if not isinstance(summary, Mapping):
                continue
            counts = summary.get("strategy_counts", {})
            if not isinstance(counts, Mapping) or not counts:
                continue
            evaluated += 1
            coverages.append(_majority_coverage(counts))
            dominant = str(summary.get("policy_dominant_strategy", ""))
            majority = max(counts.items(), key=lambda item: (_safe_int(item[1]), str(item[0])))[0]
            if dominant == str(majority):
                dominant_matches += 1
    return {
        "evaluated": int(evaluated),
        "mean_strategy_majority_coverage": _rounded(mean(coverages) if coverages else 0.0),
        "policy_dominant_match_rate": _rounded(dominant_matches / max(1, evaluated)),
    }


def _behavior_diagnosis(
    *,
    aggregate: Mapping[str, object],
    diagnostic_rows: list[dict[str, object]],
    failure_audit: Mapping[str, object],
) -> dict[str, object]:
    rows_with_real = [row for row in diagnostic_rows if row.get("real_strategy") in STRATEGY_LABELS]
    failure_rows = failure_audit.get("rows", []) if isinstance(failure_audit, Mapping) else []
    failure_rows = [row for row in failure_rows if isinstance(row, dict)]
    majority_trace = _majority_rows_as_trace(failure_rows)

    real_counts = Counter(str(row.get("real_strategy")) for row in rows_with_real)
    if not real_counts:
        for row in failure_rows:
            real_dist = row.get("real_strategy_distribution", {})
            if isinstance(real_dist, Mapping):
                real_counts.update({str(k): _safe_int(v) for k, v in real_dist.items()})

    personality_matrix = _confusion(rows_with_real, "personality_strategy")
    baseline_c_matrix = _confusion(rows_with_real, "baseline_c_strategy")
    majority_matrix = _confusion(majority_trace, "majority_strategy")
    classifier_gate = aggregate.get("classifier_gate", {})
    cue_override_rate = _safe_float(classifier_gate.get("cue_override_rate")) if isinstance(classifier_gate, Mapping) else 0.0
    cue_feature_assist_rate = (
        _safe_float(classifier_gate.get("cue_feature_assist_rate"))
        if isinstance(classifier_gate, Mapping)
        else 0.0
    )
    without_cue_f1 = (
        _safe_float(classifier_gate.get("macro_f1_3class_without_cue"))
        if isinstance(classifier_gate, Mapping)
        else 0.0
    )
    state_collapse = _state_collapse_summary(aggregate)
    majority_gate = aggregate.get("behavioral_majority_baseline_gate", {})
    majority_mean = _safe_float(majority_gate.get("majority_behavioral_strategy_mean")) if isinstance(majority_gate, Mapping) else 0.0
    personality_mean = _safe_float(majority_gate.get("personality_behavioral_strategy_mean")) if isinstance(majority_gate, Mapping) else 0.0
    real_entropy = _distribution_entropy(real_counts, STRATEGY_LABELS)
    majority_cov = _majority_coverage(real_counts)

    tags: list[str] = []
    if real_entropy < 0.35 or majority_cov >= 0.75 or cue_override_rate > 0.35:
        tags.append("classifier_definition_issue")
    if (
        _safe_float(state_collapse.get("mean_strategy_majority_coverage")) >= 0.75
        and _safe_float(state_collapse.get("policy_dominant_match_rate")) >= 0.80
    ):
        tags.append("state_modeling_issue")
    if majority_mean > personality_mean and "classifier_definition_issue" not in tags:
        tags.append("generator_policy_issue")
    elif majority_mean > personality_mean and _safe_float(state_collapse.get("mean_strategy_majority_coverage")) < 0.75:
        tags.append("generator_policy_issue")

    return {
        "real_strategy_distribution": dict(real_counts),
        "real_strategy_entropy_normalized": _rounded(real_entropy),
        "real_majority_class_coverage": _rounded(majority_cov),
        "cue_override_rate": _rounded(cue_override_rate),
        "cue_feature_assist_rate": _rounded(cue_feature_assist_rate),
        "macro_f1_3class_without_cue": _rounded(without_cue_f1),
        "classifier_prediction_source_counts": (
            dict(classifier_gate.get("prediction_source_counts", {}))
            if isinstance(classifier_gate, Mapping)
            and isinstance(classifier_gate.get("prediction_source_counts"), Mapping)
            else {}
        ),
        "personality_confusion": personality_matrix,
        "baseline_c_confusion": baseline_c_matrix,
        "train_majority_confusion": majority_matrix,
        "balanced_recall": {
            "personality": _rounded(_balanced_recall_from_confusion(personality_matrix)),
            "baseline_c": _rounded(_balanced_recall_from_confusion(baseline_c_matrix)),
            "train_majority": _rounded(_balanced_recall_from_confusion(majority_matrix)),
        },
        "state_collapse": state_collapse,
        "majority_vs_personality": {
            "majority_behavioral_strategy_mean": _rounded(majority_mean),
            "personality_behavioral_strategy_mean": _rounded(personality_mean),
            "majority_minus_personality": _rounded(majority_mean - personality_mean),
        },
        "label_examples": _label_examples(rows_with_real),
        "diagnosis_tags": tags,
    }


def _label_examples(rows: list[dict[str, object]], limit_per_class: int = 4) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {label: [] for label in STRATEGY_LABELS}
    for row in rows:
        label = str(row.get("real_strategy", ""))
        if label not in out or len(out[label]) >= limit_per_class:
            continue
        out[label].append(
            {
                "user_uid": row.get("user_uid"),
                "strategy": row.get("strategy"),
                "real_action": row.get("real_action"),
                "real_text": str(row.get("real_text", ""))[:120],
                "personality_strategy": row.get("personality_strategy"),
                "baseline_c_strategy": row.get("baseline_c_strategy"),
            }
        )
    return out


def build_diagnosis(artifact_dir: Path) -> dict[str, object]:
    aggregate = _load_json(artifact_dir / "aggregate_report.json", {})
    ablation_summary = _load_json(artifact_dir / "ablation_summary.json", {})
    failure_audit = _load_json(artifact_dir / "baseline_c_behavioral_failure_audit.json", {})
    diagnostic_rows = _load_jsonl(artifact_dir / "diagnostic_trace.jsonl")
    ablation_trace_rows = _load_jsonl(artifact_dir / "ablation_trace.jsonl")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    ablation_summary = ablation_summary if isinstance(ablation_summary, Mapping) else {}
    failure_audit = failure_audit if isinstance(failure_audit, Mapping) else {}
    return {
        "artifact_dir": str(artifact_dir.as_posix()),
        "artifact_sanity": _artifact_sanity(
            aggregate=aggregate,
            ablation_summary=ablation_summary,
            diagnostic_rows=diagnostic_rows,
            ablation_trace_rows=ablation_trace_rows,
            artifact_dir=artifact_dir,
        ),
        "baseline_c_diagnosis": _baseline_c_diagnosis(diagnostic_rows),
        "ablation_diagnosis": _ablation_diagnosis(
            ablation_summary=ablation_summary,
            ablation_trace_rows=ablation_trace_rows,
        ),
        "behavior_diagnosis": _behavior_diagnosis(
            aggregate=aggregate,
            diagnostic_rows=diagnostic_rows,
            failure_audit=failure_audit,
        ),
    }


def _md_table(rows: list[Mapping[str, object]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def render_markdown(diagnosis: Mapping[str, object]) -> str:
    sanity = diagnosis.get("artifact_sanity", {})
    baseline = diagnosis.get("baseline_c_diagnosis", {})
    ablation = diagnosis.get("ablation_diagnosis", {})
    behavior = diagnosis.get("behavior_diagnosis", {})
    lines = [
        "# M5.4 Failure Diagnosis",
        "",
        "## Artifact Sanity",
        "",
        f"- Passed: {sanity.get('passed') if isinstance(sanity, Mapping) else False}",
        f"- Missing files: {sanity.get('missing_files') if isinstance(sanity, Mapping) else []}",
        f"- Diagnostic trace rows: {sanity.get('diagnostic_trace_rows') if isinstance(sanity, Mapping) else 0}",
        f"- Ablation trace rows: {sanity.get('ablation_trace_rows') if isinstance(sanity, Mapping) else 0}",
        f"- Stale ablation-vs-Baseline-C warning: {sanity.get('stale_ablation_baseline_c_warning') if isinstance(sanity, Mapping) else False}",
        "",
        "## Baseline C",
        "",
        f"- Baseline C win rate: {baseline.get('baseline_c_win_rate') if isinstance(baseline, Mapping) else 0}",
        f"- Baseline C win turns: {baseline.get('baseline_c_win_turns') if isinstance(baseline, Mapping) else 0}",
        f"- Interpretation: {baseline.get('interpretation') if isinstance(baseline, Mapping) else {}}",
        f"- Surface metric suspect: {baseline.get('baseline_c_surface_metric_suspect') if isinstance(baseline, Mapping) else False} "
        f"({baseline.get('baseline_c_surface_metric_suspect_reason') if isinstance(baseline, Mapping) else ''})",
        f"- Filtered semantic deltas: {baseline.get('filtered_semantic_deltas') if isinstance(baseline, Mapping) else {}}",
        "",
        "Worst user-strategy deltas:",
    ]
    worst = baseline.get("worst_user_strategies", []) if isinstance(baseline, Mapping) else []
    lines.extend(_md_table(
        worst if isinstance(worst, list) else [],
        ["user_uid", "strategy", "turns", "mean_delta", "median_delta", "personality_win_rate"],
    ))
    lines.extend(["", "## Ablations", ""])
    if isinstance(ablation, Mapping) and ablation.get("trace_available"):
        lines.append(f"- Lift means: {ablation.get('lift_means')}")
        lines.append(f"- Lift medians: {ablation.get('lift_medians')}")
        lines.append(f"- Beating-full clusters: {ablation.get('beating_full_clusters')}")
    else:
        lines.append("- Ablation turn trace unavailable; rerun validation to populate `ablation_trace.jsonl`.")
    lines.extend(["", "## Behavior", ""])
    if isinstance(behavior, Mapping):
        lines.extend(
            [
                f"- Diagnosis tags: {behavior.get('diagnosis_tags')}",
                f"- Real strategy distribution: {behavior.get('real_strategy_distribution')}",
                f"- Real majority coverage: {behavior.get('real_majority_class_coverage')}",
                f"- Classifier cue override / cue feature / without-cue F1: "
                f"{behavior.get('cue_override_rate')} / {behavior.get('cue_feature_assist_rate')} / "
                f"{behavior.get('macro_f1_3class_without_cue')}",
                f"- Balanced recall: {behavior.get('balanced_recall')}",
                f"- Majority vs personality: {behavior.get('majority_vs_personality')}",
                f"- State collapse: {behavior.get('state_collapse')}",
                f"- Label examples: {behavior.get('label_examples')}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze M5.4 failure diagnostics from validation artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    output_dir = args.output_dir or artifact_dir
    diagnosis = build_diagnosis(artifact_dir)
    _write_json(output_dir / "m54_failure_diagnosis.json", diagnosis)
    (output_dir / "m54_failure_diagnosis.md").write_text(render_markdown(diagnosis), encoding="utf-8")
    print(f"wrote {output_dir / 'm54_failure_diagnosis.json'}")
    print(f"wrote {output_dir / 'm54_failure_diagnosis.md'}")


if __name__ == "__main__":
    main()
