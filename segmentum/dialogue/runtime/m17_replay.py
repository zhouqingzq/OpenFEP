"""Offline M17 replay, calibration, and ablation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from segmentum.dialogue.runtime.m17_bundle_features import (
    BUNDLE_TRIGGER_THRESHOLD,
    MIN_SYNERGY_MARGIN,
    REDUNDANCY_PENALTY_FLOOR,
    SINGLE_TRIGGER_THRESHOLD,
)
from segmentum.dialogue.runtime.m17_bundle_policy import (
    assemble_memory_evidence_bundles,
    evaluate_bundle_decision,
)
from segmentum.user_model.m17_calibration import BIN_ORDER
from segmentum.user_model.prediction_ledger import PredictionEntry


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _read_any_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return list(_iter_jsonl(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    return []


def _session_files(session: Path) -> list[Path]:
    if session.is_file():
        return [session]
    event_files = [
        session / "conversation_log.jsonl",
        session / "memory_dynamics_episodes.jsonl",
    ]
    state_candidates = [
        session / "m11_user_models.json",
        session / "m11_state.json",
        session / "session_state.json",
        session / "state.json",
    ]
    selected_state = next((path for path in state_candidates if path.exists()), None)
    files = [path for path in event_files if path.exists()]
    if selected_state is not None:
        files.append(selected_state)
    return files


def _prediction_entries_from_ledger_payload(payload: Mapping[str, Any]) -> list[PredictionEntry]:
    rows: list[PredictionEntry] = []
    ledger = _mapping(payload.get("prediction_ledger"))
    for row in ledger.get("entries", []):
        if isinstance(row, Mapping):
            rows.append(PredictionEntry.from_dict(row))
    return rows


def _prediction_entries_from_user_model_map(payload: Mapping[str, Any]) -> list[PredictionEntry]:
    rows: list[PredictionEntry] = []
    for value in payload.values():
        if not isinstance(value, Mapping):
            continue
        rows.extend(_prediction_entries_from_ledger_payload(dict(value)))
    return rows


def _load_prediction_entries(payload: Mapping[str, Any]) -> list[PredictionEntry]:
    rows: list[PredictionEntry] = []
    rows.extend(_prediction_entries_from_ledger_payload(payload))
    m11_state = _mapping(payload.get("m11_state"))
    rows.extend(_prediction_entries_from_ledger_payload(m11_state))
    rows.extend(_prediction_entries_from_user_model_map(_mapping(payload.get("m11_user_models"))))
    if "m11_user_models" not in payload:
        rows.extend(_prediction_entries_from_user_model_map(payload))
    return rows


def load_m17_replay_data(session_paths: Sequence[Path]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    prediction_entries: list[PredictionEntry] = []
    warnings: list[str] = []
    session_summaries: list[dict[str, Any]] = []
    for session_path in session_paths:
        files = _session_files(session_path)
        if not files:
            warnings.append(f"missing_files:{session_path}")
            continue
        state_source = ""
        event_file_count = 0
        session_entry_count = 0
        for file_path in files:
            if file_path.name.endswith(".jsonl"):
                event_file_count += 1
            elif not state_source:
                state_source = str(file_path)
            for row in _read_any_json(file_path):
                events.append(row)
                loaded_entries = _load_prediction_entries(row)
                prediction_entries.extend(loaded_entries)
                session_entry_count += len(loaded_entries)
        if session_path.is_dir() and not state_source:
            warnings.append(f"missing_prediction_state:{session_path}")
        elif state_source and session_entry_count == 0:
            warnings.append(f"no_prediction_entries:{session_path}")
        session_summaries.append(
            {
                "session_path": str(session_path),
                "state_source": state_source,
                "event_file_count": event_file_count,
                "prediction_entry_count": session_entry_count,
            }
        )
    return {
        "events": events,
        "prediction_entries": prediction_entries,
        "warnings": warnings,
        "session_summaries": session_summaries,
    }


def replay_coverage_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    events = [row for row in payload.get("events", []) if isinstance(row, Mapping)]
    entries = [row for row in payload.get("prediction_entries", []) if isinstance(row, PredictionEntry)]
    session_summaries = [row for row in payload.get("session_summaries", []) if isinstance(row, Mapping)]
    turns_seen = max(
        [int(row.get("turn_index", 0) or 0) for row in events] + [int(entry.turn_id) for entry in entries] + [0]
    )
    latest_by_prediction: dict[str, PredictionEntry] = {}
    for entry in entries:
        latest_by_prediction[entry.prediction_id] = entry
    settled = [entry for entry in latest_by_prediction.values() if entry.settlement_outcome in {"confirmed", "violated", "unclear", "expired"}]
    skip_reason_counts = Counter(
        str(row.get("reason_code", row.get("reason", "")) or "")
        for row in events
        if str(row.get("type", "")) == "PredictionLockSkippedEvent"
    )
    lock_count = sum(1 for entry in latest_by_prediction.values() if entry.created_before_response)
    novelty_seen = 0
    structured_linkable = 0
    bundle_linkable_count = 0
    retrieval_eligible_count = 0
    for row in events:
        if "novelty_signal" in row:
            novelty_seen += 1
        if str(row.get("type", "")) == "MemoryEfeEvaluationEvent":
            diagnostics = _mapping(row.get("bundle_linkage_diagnostics"))
            bundle_linkable_count += int(diagnostics.get("bundle_linkable_count", 0) or 0)
            retrieval_eligible_count += int(diagnostics.get("retrieval_eligible_count", 0) or 0)
            for candidate in row.get("bundle_candidate_rows", []):
                if not isinstance(candidate, Mapping):
                    continue
                if candidate.get("prediction_ids") or candidate.get("expectation_ids") or candidate.get("episode_ids"):
                    structured_linkable += 1
    settled_count = len([entry for entry in settled if entry.settlement_outcome in {"confirmed", "violated"}])
    unclear_count = len([entry for entry in settled if entry.settlement_outcome == "unclear"])
    expired_count = len([entry for entry in settled if entry.settlement_outcome == "expired"])
    return {
        "turns_seen": turns_seen,
        "prediction_lock_coverage_rate": round(lock_count / float(max(1, turns_seen)), 6),
        "prediction_lock_skip_reason_counts": dict(skip_reason_counts),
        "pending_prediction_count": len([entry for entry in latest_by_prediction.values() if entry.validation_status == "pending"]),
        "settled_prediction_count": settled_count,
        "unclear_rate": round(unclear_count / float(max(1, len(settled))), 6),
        "expired_rate": round(expired_count / float(max(1, len(settled))), 6),
        "episode_linkage_coverage": round(
            len([entry for entry in settled if entry.source_episode_id]) / float(max(1, len(settled))),
            6,
        ),
        "novelty_signal_coverage": round(novelty_seen / float(max(1, len(events))), 6),
        "structured_linkage_coverage": round(structured_linkable / float(max(1, retrieval_eligible_count)), 6),
        "bundle_linkable_prediction_rate": round(bundle_linkable_count / float(max(1, retrieval_eligible_count)), 6),
        "prediction_state_loaded_session_count": len(
            [row for row in session_summaries if str(row.get("state_source", "") or "").strip()]
        ),
        "prediction_state_missing_session_count": len(
            [row for row in session_summaries if not str(row.get("state_source", "") or "").strip()]
        ),
        "prediction_sample_empty_session_count": len(
            [
                row
                for row in session_summaries
                if str(row.get("state_source", "") or "").strip()
                and int(row.get("prediction_entry_count", 0) or 0) == 0
            ]
        ),
    }


def calibration_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    entries = [
        entry
        for entry in payload.get("prediction_entries", [])
        if isinstance(entry, PredictionEntry) and entry.settlement_outcome in {"confirmed", "violated"}
    ]
    if not entries:
        return {
            "overall_brier_score": None,
            "brier_by_prediction_type": {},
            "hit_rate_by_confidence_bin": {},
            "expected_calibration_error": None,
            "sample_count_by_type": {},
            "unclear_rate_by_type": {},
            "overconfidence_flags": [],
            "confidence_discrimination_by_type": {},
            "non_trivial_prediction_rate": 0.0,
            "base_rate_adjusted_lift": {},
            "mean_information_value": None,
            "low_sample_warning": True,
        }
    by_type: dict[str, list[PredictionEntry]] = defaultdict(list)
    by_bin: dict[str, list[PredictionEntry]] = defaultdict(list)
    by_type_unclear: Counter[str] = Counter()
    all_entries = [entry for entry in payload.get("prediction_entries", []) if isinstance(entry, PredictionEntry)]
    for entry in all_entries:
        by_type[entry.prediction_type].append(entry)
        if entry.settlement_outcome == "unclear":
            by_type_unclear[entry.prediction_type] += 1
    settled = [entry for entry in all_entries if entry.settlement_outcome in {"confirmed", "violated"}]
    for entry in settled:
        for label, low, high in BIN_ORDER:
            if entry.committed_confidence >= low and (entry.committed_confidence < high or label == BIN_ORDER[-1][0]):
                by_bin[label].append(entry)
                break
    brier_by_type: dict[str, float] = {}
    sample_count_by_type: dict[str, int] = {}
    discrimination: dict[str, float] = {}
    base_rate_lift: dict[str, float] = {}
    flags: list[str] = []
    non_trivial_types = 0
    info_values: list[float] = []
    for prediction_type, rows in by_type.items():
        settled_rows = [row for row in rows if row.settlement_outcome in {"confirmed", "violated"}]
        if not settled_rows:
            continue
        sample_count_by_type[prediction_type] = len(settled_rows)
        brier_scores = [float(row.m17_brier_score or 0.0) for row in settled_rows if row.m17_brier_score is not None]
        brier_by_type[prediction_type] = round(sum(brier_scores) / float(max(1, len(brier_scores))), 6)
        confirmed_conf = [row.committed_confidence for row in settled_rows if row.settlement_outcome == "confirmed"]
        violated_conf = [row.committed_confidence for row in settled_rows if row.settlement_outcome == "violated"]
        confirmed_mean = sum(confirmed_conf) / float(max(1, len(confirmed_conf)))
        violated_mean = sum(violated_conf) / float(max(1, len(violated_conf)))
        discrimination[prediction_type] = round(confirmed_mean - violated_mean, 6)
        base_rate = len(confirmed_conf) / float(max(1, len(settled_rows)))
        confidence_mean = sum(row.committed_confidence for row in settled_rows) / float(len(settled_rows))
        base_rate_lift[prediction_type] = round(confidence_mean - base_rate, 6)
        if discrimination[prediction_type] >= 0.05:
            non_trivial_types += 1
        else:
            flags.append(f"weak_discrimination:{prediction_type}")
        if confidence_mean - base_rate > 0.10:
            flags.append(f"overconfident:{prediction_type}")
        info_values.extend(
            [float(row.m17_prediction_error) for row in settled_rows if row.m17_prediction_error is not None]
        )
    hit_rate_by_bin = {
        label: round(
            len([row for row in rows if row.settlement_outcome == "confirmed"]) / float(max(1, len(rows))),
            6,
        )
        for label, rows in by_bin.items()
    }
    expected_calibration_error = 0.0
    for label, rows in by_bin.items():
        if not rows:
            continue
        empirical = hit_rate_by_bin[label]
        predicted = sum(row.committed_confidence for row in rows) / float(len(rows))
        expected_calibration_error += abs(empirical - predicted) * (len(rows) / float(len(settled)))
    overall_brier = sum(float(row.m17_brier_score or 0.0) for row in settled) / float(len(settled))
    return {
        "overall_brier_score": round(overall_brier, 6),
        "brier_by_prediction_type": brier_by_type,
        "hit_rate_by_confidence_bin": hit_rate_by_bin,
        "expected_calibration_error": round(expected_calibration_error, 6),
        "sample_count_by_type": sample_count_by_type,
        "unclear_rate_by_type": {
            prediction_type: round(by_type_unclear[prediction_type] / float(max(1, len(rows))), 6)
            for prediction_type, rows in by_type.items()
        },
        "overconfidence_flags": sorted(set(flags)),
        "confidence_discrimination_by_type": discrimination,
        "non_trivial_prediction_rate": round(non_trivial_types / float(max(1, len(sample_count_by_type))), 6),
        "base_rate_adjusted_lift": base_rate_lift,
        "mean_information_value": round(sum(info_values) / float(max(1, len(info_values))), 6),
        "low_sample_warning": len(settled) < 30 or any(count < 10 for count in sample_count_by_type.values()),
    }


def _policy_case_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in events:
        if str(row.get("type", "")) != "MemoryEfeEvaluationEvent":
            continue
        candidate_rows = [dict(item) for item in row.get("bundle_candidate_rows", []) if isinstance(item, Mapping)]
        if not candidate_rows:
            continue
        cases.append(
            {
                "turn_index": int(row.get("turn_index", 0) or 0),
                "traceable_expectation_id": str(row.get("traceable_expectation_id", "") or ""),
                "candidate_rows": candidate_rows,
            }
        )
    return cases


def _settled_prediction_error_map(entries: Sequence[PredictionEntry]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for entry in entries:
        if entry.settlement_outcome in {"confirmed", "violated"} and entry.m17_prediction_error is not None:
            mapping[entry.prediction_id] = float(entry.m17_prediction_error)
    return mapping


def _retained_prediction_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    prediction_ids: set[str] = set()
    for row in rows:
        for value in row.get("prediction_ids", []):
            prediction_ids.add(str(value))
    return prediction_ids


def _simulate_policy_case(policy: str, case: Mapping[str, Any], *, rng: random.Random) -> dict[str, Any]:
    rows = [dict(row) for row in case.get("candidate_rows", []) if isinstance(row, Mapping)]
    if not rows:
        return {"retained_rows": [], "bundle_required": False, "synergy_margin": 0.0}
    if policy == "prediction_error_bundle_policy":
        bundles, _ = assemble_memory_evidence_bundles(
            rows,
            allowed_expectation_ids=[str(case.get("traceable_expectation_id", "") or "")],
        )
        if bundles:
            decision = evaluate_bundle_decision(bundles[0], consumer_kind="reply_policy_bias")
            return {
                "retained_rows": [row for row in rows if str(row.get("id", row.get("memory_id", ""))) in bundles[0].member_memory_ids] if decision.commit else [],
                "bundle_required": decision.commit,
                "synergy_margin": bundles[0].synergy_margin,
                "best_single_counterfactual": decision.best_single_counterfactual_would_trigger,
            }
        return {"retained_rows": [], "bundle_required": False, "synergy_margin": 0.0, "best_single_counterfactual": False}
    if policy == "best_single_memory_baseline":
        best = max(rows, key=lambda row: float(row.get("item_support", 0.0) or 0.0))
        return {"retained_rows": [best] if float(best.get("item_support", 0.0) or 0.0) >= SINGLE_TRIGGER_THRESHOLD else [], "bundle_required": False, "synergy_margin": 0.0}
    if policy == "naive_additive_topn_baseline":
        top_rows = sorted(rows, key=lambda row: float(row.get("item_support", 0.0) or 0.0), reverse=True)[:4]
        score = 1.0
        for row in top_rows:
            score *= 1.0 - float(row.get("item_support", 0.0) or 0.0)
        aggregate = 1.0 - score
        return {"retained_rows": top_rows if aggregate >= BUNDLE_TRIGGER_THRESHOLD else [], "bundle_required": False, "synergy_margin": 0.0}
    if policy == "legacy_salience_proxy":
        ranked = sorted(
            rows,
            key=lambda row: float(_mapping(row.get("factor_breakdown")).get("salience_factor", row.get("item_support", 0.0)) or 0.0),
            reverse=True,
        )
        return {"retained_rows": ranked[:1], "bundle_required": False, "synergy_margin": 0.0}
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return {"retained_rows": shuffled[:1], "bundle_required": False, "synergy_margin": 0.0}


def ablation_metrics(payload: Mapping[str, Any], *, seed: int = 17) -> dict[str, Any]:
    events = [row for row in payload.get("events", []) if isinstance(row, Mapping)]
    entries = [row for row in payload.get("prediction_entries", []) if isinstance(row, PredictionEntry)]
    cases = _policy_case_rows(events)
    error_by_prediction = _settled_prediction_error_map(entries)
    policies = [
        "prediction_error_bundle_policy",
        "best_single_memory_baseline",
        "naive_additive_topn_baseline",
        "legacy_salience_proxy",
        "random_or_recency_baseline",
    ]
    metrics: dict[str, dict[str, Any]] = {}
    bundle_required_count = 0
    best_single_failure_count = 0
    top_k_equivalent_count = 0
    synergy_values: list[float] = []
    decision_sets: dict[str, set[str]] = defaultdict(set)
    linkage_eligible = 0
    for policy in policies:
        rng = random.Random(seed)
        retained_prediction_ids: set[str] = set()
        retained_count = 0
        future_errors: list[float] = []
        covered_count = 0
        for case in cases:
            result = _simulate_policy_case(policy, case, rng=rng)
            retained_rows = result.get("retained_rows", [])
            retained_count += len(retained_rows)
            prediction_ids = _retained_prediction_ids(retained_rows)
            if prediction_ids:
                linkage_eligible += 1
            retained_prediction_ids.update(prediction_ids)
            decision_sets[policy].update(str(row.get("id", row.get("memory_id", ""))) for row in retained_rows)
            if policy == "prediction_error_bundle_policy":
                if result.get("bundle_required"):
                    bundle_required_count += 1
                    synergy_values.append(float(result.get("synergy_margin", 0.0) or 0.0))
                    if not result.get("best_single_counterfactual"):
                        best_single_failure_count += 1
                else:
                    top_k_equivalent_count += 1
        for prediction_id, error in error_by_prediction.items():
            if prediction_id in retained_prediction_ids:
                covered_count += 1
                future_errors.append(error)
        metrics[policy] = {
            "future_prediction_error_mean": round(sum(future_errors) / float(max(1, len(future_errors))), 6) if future_errors else None,
            "future_brier_mean": round(
                sum(entry.m17_brier_score or 0.0 for entry in entries if entry.prediction_id in retained_prediction_ids and entry.m17_brier_score is not None)
                / float(
                    max(
                        1,
                        len([entry for entry in entries if entry.prediction_id in retained_prediction_ids and entry.m17_brier_score is not None]),
                    )
                ),
                6,
            )
            if retained_prediction_ids
            else None,
            "retained_memory_count": retained_count,
            "retained_memory_cost": retained_count,
            "coverage_of_future_predictions": round(covered_count / float(max(1, len(error_by_prediction))), 6),
        }
    bundle_error = metrics["prediction_error_bundle_policy"]["future_prediction_error_mean"]
    best_single_error = metrics["best_single_memory_baseline"]["future_prediction_error_mean"]
    naive_error = metrics["naive_additive_topn_baseline"]["future_prediction_error_mean"]
    salience_error = metrics["legacy_salience_proxy"]["future_prediction_error_mean"]
    random_error = metrics["random_or_recency_baseline"]["future_prediction_error_mean"]
    overlap = {}
    for policy in policies[1:]:
        denom = len(decision_sets["prediction_error_bundle_policy"] | decision_sets[policy]) or 1
        overlap[policy] = round(
            len(decision_sets["prediction_error_bundle_policy"] & decision_sets[policy]) / float(denom),
            6,
        )
    parameter_sensitivity = []
    for delta in (-0.05, 0.0, 0.05):
        parameter_sensitivity.append(
            {
                "single_trigger_threshold": round(SINGLE_TRIGGER_THRESHOLD + delta, 6),
                "bundle_trigger_threshold": round(BUNDLE_TRIGGER_THRESHOLD + delta, 6),
                "min_synergy_margin": MIN_SYNERGY_MARGIN,
                "redundancy_penalty_floor": REDUNDANCY_PENALTY_FLOOR,
            }
        )
    return {
        "policies": metrics,
        "future_prediction_error_mean": bundle_error,
        "future_brier_mean": metrics["prediction_error_bundle_policy"]["future_brier_mean"],
        "retained_memory_count": metrics["prediction_error_bundle_policy"]["retained_memory_count"],
        "retained_memory_cost": metrics["prediction_error_bundle_policy"]["retained_memory_cost"],
        "coverage_of_future_predictions": metrics["prediction_error_bundle_policy"]["coverage_of_future_predictions"],
        "policy_delta_vs_best_single": None if bundle_error is None or best_single_error is None else round(best_single_error - bundle_error, 6),
        "policy_delta_vs_naive_additive": None if bundle_error is None or naive_error is None else round(naive_error - bundle_error, 6),
        "policy_delta_vs_random": None if bundle_error is None or random_error is None else round(random_error - bundle_error, 6),
        "policy_delta_vs_salience": None if bundle_error is None or salience_error is None else round(salience_error - bundle_error, 6),
        "evidence_linkage_coverage": replay_coverage_metrics(payload)["structured_linkage_coverage"],
        "policy_decision_overlap": overlap,
        "bundle_required_decision_rate": round(bundle_required_count / float(max(1, len(cases))), 6),
        "best_single_failure_rate": round(best_single_failure_count / float(max(1, len(cases))), 6),
        "mean_synergy_margin": round(sum(synergy_values) / float(max(1, len(synergy_values))), 6),
        "single_memory_counterfactual_pass_rate": round(best_single_failure_count / float(max(1, bundle_required_count)), 6),
        "structured_linkage_advantage_rate": round(bundle_required_count / float(max(1, linkage_eligible)), 6),
        "top_k_equivalent_behavior_rate": round(top_k_equivalent_count / float(max(1, len(cases))), 6),
        "insufficient_evidence_linkage": linkage_eligible == 0,
        "parameter_sensitivity_summary": parameter_sensitivity,
        "low_sample_warning": len(cases) < 3,
    }


def write_replay_artifacts(
    *,
    out_dir: Path,
    coverage: Mapping[str, Any],
    calibration: Mapping[str, Any],
    ablation: Mapping[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage_metrics.json").write_text(json.dumps(dict(coverage), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "calibration_metrics.json").write_text(json.dumps(dict(calibration), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ablation_metrics.json").write_text(json.dumps(dict(ablation), ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "calibration_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["confidence_bin", "hit_rate"])
        writer.writeheader()
        for label, hit_rate in dict(calibration.get("hit_rate_by_confidence_bin", {})).items():
            writer.writerow({"confidence_bin": label, "hit_rate": hit_rate})
    with (out_dir / "policy_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "future_prediction_error_mean",
                "future_brier_mean",
                "retained_memory_count",
                "coverage_of_future_predictions",
            ],
        )
        writer.writeheader()
        for policy, row in dict(ablation.get("policies", {})).items():
            writer.writerow(
                {
                    "policy": policy,
                    "future_prediction_error_mean": row.get("future_prediction_error_mean"),
                    "future_brier_mean": row.get("future_brier_mean"),
                    "retained_memory_count": row.get("retained_memory_count"),
                    "coverage_of_future_predictions": row.get("coverage_of_future_predictions"),
                }
            )


def run_m17_replay(
    *,
    session_paths: Sequence[Path],
    out_dir: Path,
    seed: int = 17,
) -> dict[str, Any]:
    payload = load_m17_replay_data(session_paths)
    coverage = replay_coverage_metrics(payload)
    calibration = calibration_metrics(payload)
    ablation = ablation_metrics(payload, seed=seed)
    write_replay_artifacts(out_dir=out_dir, coverage=coverage, calibration=calibration, ablation=ablation)
    return {
        "coverage": coverage,
        "calibration": calibration,
        "ablation": ablation,
        "warnings": list(payload.get("warnings", [])),
    }
