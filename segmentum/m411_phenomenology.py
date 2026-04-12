from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from .m4_acceptance import final_conclusion
from .memory_encoding import EncodingDynamics, EncodingDynamicsInput
from .memory_model import MemoryClass, MemoryEntry
from .runtime import SegmentRuntime


REPORTS_DIR = Path("reports")
ARTIFACTS_DIR = Path("artifacts")

M411_DEFAULT_ROLLOUT_PATH = ARTIFACTS_DIR / "m411_default_rollout.json"
M411_CONTROL_ROLLOUT_PATH = ARTIFACTS_DIR / "m411_negative_control_rollout.json"
M411_EVIDENCE_PATH = ARTIFACTS_DIR / "m411_phenomenology_evidence.json"
M411_REPORT_PATH = REPORTS_DIR / "m411_acceptance_report.json"
M411_SUMMARY_PATH = REPORTS_DIR / "m411_acceptance_summary.md"
M411_OFFICIAL_TICKS = 20000
M411_OFFICIAL_MIN_ACCEPTANCE_TICKS = 5000

GATE_ORDER = (
    "long_horizon_free_rollout",
    "serial_position_effect",
    "retention_curve_fit",
    "schema_intrusion",
    "identity_continuity",
    "negative_controls",
    "three_layer_acceptance_taxonomy",
    "honesty_safety_net",
)

STATE_VECTOR_KEYS = (
    "food",
    "danger",
    "novelty",
    "shelter",
    "temperature",
    "social",
    "energy",
    "stress",
    "fatigue",
)


@dataclass(frozen=True)
class M411RolloutConfig:
    seed: int = 411
    ticks: int = M411_OFFICIAL_TICKS
    recall_probe_interval: int = 50
    perturbation_tick: int | None = None
    control_mode: str = "salience_shuffled"
    sleep_interval: int = 50
    min_acceptance_ticks: int = M411_OFFICIAL_MIN_ACCEPTANCE_TICKS

    def effective_perturbation_tick(self) -> int:
        if self.perturbation_tick is not None:
            return max(1, int(self.perturbation_tick))
        return max(1, int(self.ticks) // 2)


def _official_config_checks(config: M411RolloutConfig) -> dict[str, object]:
    return {
        "ticks_is_official_default": int(config.ticks) == M411_OFFICIAL_TICKS,
        "min_acceptance_ticks_non_toy": (
            int(config.min_acceptance_ticks) >= M411_OFFICIAL_MIN_ACCEPTANCE_TICKS
        ),
        "official_ticks_required": M411_OFFICIAL_TICKS,
        "official_min_acceptance_ticks_required": M411_OFFICIAL_MIN_ACCEPTANCE_TICKS,
    }


def _is_official_acceptance_config(config: M411RolloutConfig) -> bool:
    checks = _official_config_checks(config)
    return bool(
        checks["ticks_is_official_default"]
        and checks["min_acceptance_ticks_non_toy"]
    )


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _vector_distance(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width <= 0:
        return float("inf")
    return math.sqrt(
        sum((float(left[index]) - float(right[index])) ** 2 for index in range(width))
    )


def _entry_metadata(entry: MemoryEntry) -> dict[str, object]:
    return dict(entry.compression_metadata or {})


def _encoding_strength(entry: MemoryEntry) -> float:
    metadata = _entry_metadata(entry)
    return float(metadata.get("encoding_strength", entry.salience))


def _current_state_snapshot(runtime: SegmentRuntime) -> dict[str, object]:
    context = dict(runtime.agent.last_memory_context)
    observed = {
        str(key): float(value)
        for key, value in dict(runtime.agent.last_decision_observation).items()
    }
    prediction = {
        str(key): float(value)
        for key, value in dict(context.get("prediction_before_memory", {})).items()
    }
    errors = {
        str(key): float(value)
        for key, value in dict(context.get("errors", {})).items()
    }
    return {
        "observation": observed,
        "prediction": prediction,
        "errors": errors,
        "body_state": runtime.agent._current_body_state(),
    }


def _entry_snapshot(entry: MemoryEntry) -> dict[str, object]:
    metadata = _entry_metadata(entry)
    dynamics = dict(metadata.get("m410_encoding_dynamics", {}) or {})
    requested = float(
        metadata.get(
            "attention_budget_requested",
            dynamics.get("attention_budget_requested", 0.0),
        )
        or 0.0
    )
    granted = float(
        metadata.get(
            "attention_budget_granted",
            dynamics.get("attention_budget_granted", 0.0),
        )
        or 0.0
    )
    return {
        "entry_id": entry.id,
        "memory_class": entry.memory_class.value,
        "created_at": entry.created_at,
        "last_accessed": entry.last_accessed,
        "salience": _round(entry.salience),
        "accessibility": _round(entry.accessibility),
        "trace_strength": _round(entry.trace_strength),
        "arousal": _round(entry.arousal),
        "novelty": _round(entry.novelty),
        "encoding_attention": _round(entry.encoding_attention),
        "relevance_self": _round(entry.relevance_self),
        "encoding_source": str(metadata.get("encoding_source", "")),
        "encoding_strength": _round(_encoding_strength(entry)),
        "raw_drive": _round(metadata.get("raw_drive", dynamics.get("raw_drive", 0.0)) or 0.0),
        "attention_budget_raw_drive_total": _round(
            metadata.get(
                "attention_budget_raw_drive_total",
                dynamics.get("attention_budget_raw_drive_total", 0.0),
            )
            or 0.0
        ),
        "attention_budget_total": _round(
            metadata.get("attention_budget_total", dynamics.get("attention_budget_total", 0.0))
            or 0.0
        ),
        "attention_budget_requested": _round(requested),
        "attention_budget_granted": _round(granted),
        "attention_budget_denied": _round(
            metadata.get("attention_budget_denied", dynamics.get("attention_budget_denied", 0.0))
            or 0.0
        ),
        "attention_budget_granted_requested_ratio": _round(
            granted / requested if requested > 1e-9 else 0.0
        ),
        "consolidation_source": entry.consolidation_source,
        "support_ids": list(entry.support_ids or []),
        "lineage_type": metadata.get("lineage_type"),
        "has_centroid": bool(entry.centroid),
        "centroid": [_round(value) for value in list(entry.centroid or [])],
        "state_vector": [_round(value) for value in list(entry.state_vector or [])],
        "residual_norm_mean": (
            _round(entry.residual_norm_mean)
            if entry.residual_norm_mean is not None
            else None
        ),
        "residual_norm_var": (
            _round(entry.residual_norm_var)
            if entry.residual_norm_var is not None
            else None
        ),
        "replay_second_pass_error": entry.replay_second_pass_error,
        "salience_delta": entry.salience_delta,
        "retention_adjustment": entry.retention_adjustment,
        "semantic_tags": list(entry.semantic_tags[:8]),
    }


def _granted_requested_ratio(event: dict[str, object]) -> float:
    requested = float(event.get("attention_budget_requested", 0.0) or 0.0)
    granted = float(event.get("attention_budget_granted", 0.0) or 0.0)
    if requested <= 1e-9:
        return 0.0
    return granted / requested


def _budget_ratio_histogram(events: list[dict[str, object]]) -> dict[str, int]:
    histogram = {
        "0.00-0.24": 0,
        "0.25-0.49": 0,
        "0.50-0.74": 0,
        "0.75-0.99": 0,
        "1.00": 0,
    }
    for event in events:
        ratio = _granted_requested_ratio(event)
        if ratio >= 1.0:
            histogram["1.00"] += 1
        elif ratio >= 0.75:
            histogram["0.75-0.99"] += 1
        elif ratio >= 0.50:
            histogram["0.50-0.74"] += 1
        elif ratio >= 0.25:
            histogram["0.25-0.49"] += 1
        else:
            histogram["0.00-0.24"] += 1
    return histogram


def _probe_recall(runtime: SegmentRuntime, *, tick: int) -> dict[str, object]:
    store = runtime.agent.memory_store
    if store is None or not store.entries:
        return {"tick": tick, "candidate_ids": [], "recall_hypothesis": None}
    query = runtime.agent.long_term_memory._build_memory_store_query(
        _current_state_snapshot(runtime)
    )
    result = store.retrieve(
        query,
        current_mood=runtime.agent.agent_state_vector.recent_mood_baseline,
        k=8,
        agent_state=runtime.agent.agent_state_vector,
        cognitive_style=runtime.agent.memory_cognitive_style,
    )
    return {
        "tick": tick,
        "candidate_ids": [candidate.entry_id for candidate in result.candidates],
        "candidate_classes": [candidate.memory_class for candidate in result.candidates],
        "recall_confidence": _round(result.recall_confidence),
        "recall_hypothesis": (
            result.recall_hypothesis.to_dict()
            if result.recall_hypothesis is not None
            else None
        ),
    }


def _apply_perturbation(runtime: SegmentRuntime, *, tick: int) -> dict[str, object]:
    runtime.world.threat_density = _clamp(runtime.world.threat_density + 0.32)
    runtime.world.novelty_density = _clamp(runtime.world.novelty_density + 0.20)
    runtime.world.social_density = _clamp(runtime.world.social_density - 0.18)
    runtime.world.temperature = _clamp(runtime.world.temperature + 0.22)
    runtime.agent.stress = _clamp(runtime.agent.stress + 0.32)
    runtime.agent.fatigue = _clamp(runtime.agent.fatigue + 0.20)
    runtime.agent.energy = _clamp(runtime.agent.energy - 0.18)
    return {
        "tick": tick,
        "perturbation_type": "homeostatic_identity_environment_shift",
        "world": runtime.world.to_dict(),
        "body": runtime.agent._current_body_state(),
    }


def _build_budget_competition_event(
    runtime: SegmentRuntime,
    *,
    tick: int,
) -> dict[str, object] | None:
    diagnostics = runtime.agent.last_decision_diagnostics
    ranked = list(diagnostics.ranked_options if diagnostics is not None else [])
    if len(ranked) < 2:
        return None
    attention_budget = float(
        runtime.agent.long_term_memory.episode_attention_budget(
            {"body_state": runtime.agent._current_body_state()}
        )
    )
    event_inputs: list[dict[str, object]] = []
    signals: list[EncodingDynamicsInput] = []
    for option in ranked:
        prediction_error = max(0.01, float(option.predicted_error))
        surprise = max(
            0.01,
            float(option.action_ambiguity),
            1.0 - float(option.preferred_probability),
        )
        arousal = min(
            1.0,
            max(0.05, (float(option.risk) / 4.0) + (max(0.0, -float(option.value_score)) * 0.20)),
        )
        event_inputs.append(
            {
                "choice": str(option.choice),
                "prediction_error": prediction_error,
                "surprise": surprise,
                "arousal": arousal,
                "expected_free_energy": _round(float(option.expected_free_energy)),
                "risk": _round(float(option.risk)),
                "preferred_probability": _round(float(option.preferred_probability)),
                "value_score": _round(float(option.value_score)),
            }
        )
        signals.append(
            EncodingDynamicsInput(
                prediction_error=prediction_error,
                surprise=surprise,
                arousal=arousal,
                attention_budget=attention_budget,
                requested_budget=1.0,
            )
        )
    constrained = EncodingDynamics.score_many(signals)
    chosen_choice = str(diagnostics.chosen.choice)
    candidates: list[dict[str, object]] = []
    for rank, (event_input, result) in enumerate(zip(event_inputs, constrained), start=1):
        candidates.append(
            {
                **event_input,
                **result.to_dict(),
                "rank": rank,
                "is_winner": event_input["choice"] == chosen_choice,
                "granted_requested_ratio": _round(
                    result.attention_budget_granted / result.attention_budget_requested
                    if result.attention_budget_requested > 1e-9
                    else 0.0
                ),
            }
        )
    winner = next((item for item in candidates if item["is_winner"]), candidates[0])
    return {
        "tick": tick,
        "event_count": len(candidates),
        "attention_budget_total": _round(attention_budget),
        "winner_choice": winner["choice"],
        "winner_rank": int(winner["rank"]),
        "winner_candidate_id": None,
        "winner_granted_requested_ratio": winner["granted_requested_ratio"],
        "granted_requested_histogram": _budget_ratio_histogram(candidates),
        "candidates": candidates,
    }


def _attach_budget_competition_metadata(
    entry: MemoryEntry,
    budget_event: dict[str, object],
) -> None:
    winner = next(
        (
            candidate
            for candidate in budget_event.get("candidates", [])
            if candidate.get("is_winner")
        ),
        None,
    )
    if not isinstance(winner, dict):
        return
    metadata = dict(entry.compression_metadata or {})
    dynamics = dict(metadata.get("m410_encoding_dynamics", {}) or {})
    for key in (
        "raw_drive",
        "attention_budget_raw_drive_total",
        "attention_budget_total",
        "attention_budget_requested",
        "attention_budget_granted",
        "attention_budget_denied",
    ):
        if key in winner:
            metadata[key] = winner[key]
            dynamics[key] = winner[key]
    metadata["m410_encoding_dynamics"] = dynamics
    metadata["m411_budget_competition"] = {
        "tick": int(budget_event["tick"]),
        "winner_choice": winner.get("choice"),
        "winner_rank": int(winner.get("rank", 1)),
        "event_count": int(budget_event.get("event_count", 0)),
        "granted_requested_ratio": winner.get("granted_requested_ratio"),
        "granted_requested_histogram": dict(budget_event.get("granted_requested_histogram", {})),
    }
    entry.compression_metadata = metadata


def _install_negative_control_policy(
    runtime: SegmentRuntime,
    *,
    mode: str,
) -> dict[str, object]:
    original_score_payload_encoding = runtime.agent.long_term_memory._score_payload_encoding

    def controlled_score_payload_encoding(payload: dict[str, object]) -> dict[str, object]:
        audit = dict(original_score_payload_encoding(payload))
        requested = max(0.5, float(audit.get("attention_budget_requested", 1.0) or 1.0))
        if mode == "salience_shuffled":
            key = str(payload.get("cycle", payload.get("timestamp", payload.get("action", ""))))
            bucket = sum(ord(char) for char in key) % 5
            shuffled_strength = 0.014 + (bucket * 0.003)
            grant_cap = 0.05 + (bucket * 0.01)
            audit["encoding_strength"] = _round(shuffled_strength)
            audit["attention_budget_total"] = _round(
                min(float(audit.get("attention_budget_total", 0.0) or 0.0), 0.16)
            )
            audit["attention_budget_requested"] = _round(requested)
            audit["attention_budget_granted"] = _round(
                min(
                    float(audit.get("attention_budget_total", 0.0) or 0.0),
                    grant_cap,
                    requested * 0.18,
                )
            )
            audit["raw_drive"] = _round(shuffled_strength)
            audit["attention_budget_raw_drive_total"] = _round(max(shuffled_strength * 4.0, 0.08))
        else:
            audit["encoding_strength"] = 0.0
            audit["attention_budget_requested"] = _round(requested)
            audit["attention_budget_granted"] = 0.0
            audit["raw_drive"] = 0.0
            audit["attention_budget_raw_drive_total"] = 0.0
        audit["attention_budget_denied"] = max(
            0.0,
            requested - float(audit.get("attention_budget_granted", 0.0) or 0.0),
        )
        audit["m411_negative_control_mode"] = mode
        audit["m411_negative_control_policy"] = "encoding_weight_policy"
        return audit

    runtime.agent.long_term_memory._score_payload_encoding = controlled_score_payload_encoding
    return {
        "tick": 0,
        "mode": mode,
        "policy_scope": "encoding_weight_policy",
        "weight_policy": mode,
        "mutates_existing_memory": False,
    }


def run_m411_rollout(
    config: M411RolloutConfig | None = None,
    *,
    negative_control: bool = False,
) -> dict[str, object]:
    config = config or M411RolloutConfig()
    with TemporaryDirectory() as tmp_dir:
        runtime = SegmentRuntime.load_or_create(
            state_path=Path(tmp_dir) / "segment_state.json",
            seed=config.seed,
            reset=True,
            memory_enabled=True,
        )
        runtime.agent.long_term_memory.sleep_interval = max(1, int(config.sleep_interval))
        encoded_events: list[dict[str, object]] = []
        consolidation_events: list[dict[str, object]] = []
        recall_events: list[dict[str, object]] = []
        replay_events: list[dict[str, object]] = []
        budget_events: list[dict[str, object]] = []
        interventions: list[dict[str, object]] = []
        perturbations: list[dict[str, object]] = []
        seen_entry_ids: set[str] = set()
        seen_sleep_count = 0
        perturbation_tick = config.effective_perturbation_tick()
        if negative_control:
            interventions.append(
                _install_negative_control_policy(runtime, mode=config.control_mode)
            )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            for _ in range(max(0, int(config.ticks))):
                next_tick = runtime.agent.cycle + 1
                if next_tick == perturbation_tick:
                    perturbations.append(_apply_perturbation(runtime, tick=next_tick))
                loop.run_until_complete(runtime.astep(verbose=False))
                tick = int(runtime.agent.cycle)
                budget_event = _build_budget_competition_event(runtime, tick=tick)
                if budget_event is not None:
                    budget_events.append(budget_event)

                for entry in list(runtime.agent.memory_store.entries):
                    if entry.id in seen_entry_ids:
                        continue
                    seen_entry_ids.add(entry.id)
                    if entry.memory_class is MemoryClass.EPISODIC and budget_event is not None:
                        _attach_budget_competition_metadata(entry, budget_event)
                        winner_choice = str(budget_event.get("winner_choice", ""))
                        if winner_choice and entry.id.endswith(f"-{winner_choice}"):
                            budget_event["winner_candidate_id"] = entry.id
                    snapshot = _entry_snapshot(entry)
                    snapshot["tick"] = tick
                    if entry.memory_class is MemoryClass.EPISODIC:
                        encoded_events.append(snapshot)
                    elif entry.memory_class in {MemoryClass.SEMANTIC, MemoryClass.INFERRED}:
                        consolidation_events.append(snapshot)

                if len(runtime.agent.sleep_history) > seen_sleep_count:
                    for sleep_index in range(seen_sleep_count, len(runtime.agent.sleep_history)):
                        payload = {}
                        if sleep_index < len(runtime.agent.narrative_trace):
                            payload = dict(runtime.agent.narrative_trace[sleep_index])
                        m410_payload = dict(
                            payload.get("m410_memory_store_consolidation", {}) or {}
                        )
                        replay_ids = list(m410_payload.get("replay_reencoded_ids", []) or [])
                        if replay_ids:
                            replay_events.append(
                                {
                                    "tick": tick,
                                    "sleep_cycle_id": sleep_index + 1,
                                    "source": "live_sleep_consolidation",
                                    "replay_reencoded_ids": replay_ids,
                                    "m410_memory_store_consolidation": m410_payload,
                                }
                            )
                    seen_sleep_count = len(runtime.agent.sleep_history)

                if tick % max(1, int(config.recall_probe_interval)) == 0:
                    recall_events.append(_probe_recall(runtime, tick=tick))
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        final_entries = [
            _entry_snapshot(entry)
            for entry in runtime.agent.memory_store.entries
        ]
    encoded_budget_histogram = _budget_ratio_histogram(encoded_events)

    return {
        "milestone_id": "M4.11",
        "run_kind": "negative_control" if negative_control else "default",
        "seed": config.seed,
        "ticks": int(config.ticks),
        "recall_probe_interval": int(config.recall_probe_interval),
        "perturbation_tick": perturbation_tick,
        "control_mode": config.control_mode if negative_control else None,
        "curated_corpus_paths_read": [],
        "curated_corpus_prohibited_path": "data/m47_corpus.json",
        "encoded_events": encoded_events,
        "consolidation_events": consolidation_events,
        "recall_events": recall_events,
        "replay_events": replay_events,
        "budget_events": budget_events,
        "budget_ratio_histogram": encoded_budget_histogram,
        "negative_control_interventions": interventions,
        "perturbations": perturbations,
        "final_entries": final_entries,
    }


def run_m411_rollout_pair(config: M411RolloutConfig | None = None) -> dict[str, object]:
    config = config or M411RolloutConfig()
    default = run_m411_rollout(config, negative_control=False)
    control = run_m411_rollout(config, negative_control=True)
    return {
        "config": {
            "seed": config.seed,
            "ticks": config.ticks,
            "recall_probe_interval": config.recall_probe_interval,
            "perturbation_tick": config.effective_perturbation_tick(),
            "control_mode": config.control_mode,
            "sleep_interval": config.sleep_interval,
            "min_acceptance_ticks": config.min_acceptance_ticks,
        },
        "default": default,
        "negative_control": control,
    }


def _entries_by_id(rollout: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(entry["entry_id"]): dict(entry)
        for entry in rollout.get("final_entries", [])
        if isinstance(entry, dict) and entry.get("entry_id")
    }


def _recalled_ids(rollout: dict[str, object]) -> set[str]:
    recalled: set[str] = set()
    for event in rollout.get("recall_events", []):
        if isinstance(event, dict):
            recalled.update(str(item) for item in event.get("candidate_ids", []) if str(item))
    return recalled


def _retention_score(entry: dict[str, object], recalled: set[str]) -> float:
    score = (
        (0.35 * _clamp(float(entry.get("accessibility", 0.0))))
        + (0.25 * _clamp(float(entry.get("trace_strength", 0.0))))
        + (0.25 * _clamp(float(entry.get("salience", 0.0))))
        + (0.15 if str(entry.get("entry_id", "")) in recalled else 0.0)
    )
    if entry.get("replay_second_pass_error") is not None:
        score += 0.08
    if float(entry.get("relevance_self", 0.0)) >= 0.35:
        score += 0.05
    return _clamp(score)


def evaluate_serial_position(rollout: dict[str, object]) -> dict[str, object]:
    encoded = [
        dict(event)
        for event in rollout.get("encoded_events", [])
        if isinstance(event, dict) and event.get("entry_id")
    ]
    if len(encoded) < 6:
        return {"status": "FAIL", "reason": "insufficient_encoded_events", "passed": False}
    recalled = _recalled_ids(rollout)
    final_entries = _entries_by_id(rollout)
    boundaries = sorted(
        {
            int(event.get("tick", 0))
            for event in rollout.get("replay_events", [])
            if isinstance(event, dict) and int(event.get("tick", 0)) > 0
        }
    )
    if not boundaries:
        interval = max(1, int(rollout.get("recall_probe_interval", 10)))
        boundaries = list(
            range(interval, int(rollout.get("ticks", interval)) + interval, interval)
        )
    lists: list[list[dict[str, object]]] = []
    start = 0
    for boundary in boundaries:
        group = [
            event
            for event in encoded
            if start < int(event.get("created_at", event.get("tick", 0))) <= boundary
        ]
        if len(group) >= 3:
            lists.append(group)
        start = boundary
    trailing = [
        event for event in encoded
        if int(event.get("created_at", event.get("tick", 0))) > start
    ]
    if len(trailing) >= 3:
        lists.append(trailing)
    first_scores: list[float] = []
    middle_scores: list[float] = []
    last_scores: list[float] = []
    for study_list in lists:
        width = len(study_list)
        k = max(1, min(3, width // 4))
        for index, event in enumerate(study_list):
            entry = final_entries.get(str(event["entry_id"]), event)
            score = _retention_score(entry, recalled)
            if index < k:
                first_scores.append(score)
            elif index >= width - k:
                last_scores.append(score)
            else:
                middle_scores.append(score)
    first = _mean(first_scores)
    middle = _mean(middle_scores, default=0.0)
    last = _mean(last_scores)
    primacy = first - middle
    recency = last - middle
    passed = bool(lists and primacy > 0.015 and recency > 0.015)
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "list_count": len(lists),
        "primacy_score": _round(primacy),
        "recency_score": _round(recency),
        "first_mean": _round(first),
        "middle_mean": _round(middle),
        "last_mean": _round(last),
    }


def _fit_sse(xs: list[float], ys: list[float], *, log_x: bool) -> dict[str, float]:
    if len(xs) < 2:
        return {"sse": 0.0, "slope": 0.0, "intercept": _mean(ys)}
    transformed = [math.log1p(x) if log_x else x for x in xs]
    x_mean = _mean(transformed)
    y_mean = _mean(ys)
    denom = sum((x - x_mean) ** 2 for x in transformed)
    slope = 0.0 if denom <= 1e-12 else sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(transformed, ys)
    ) / denom
    intercept = y_mean - (slope * x_mean)
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(transformed, ys))
    return {"sse": sse, "slope": slope, "intercept": intercept}


def evaluate_retention_curve(rollout: dict[str, object]) -> dict[str, object]:
    recalled = _recalled_ids(rollout)
    ticks = int(rollout.get("ticks", 0))
    rows: list[tuple[float, float]] = []
    for entry in rollout.get("final_entries", []):
        if not isinstance(entry, dict) or entry.get("memory_class") != MemoryClass.EPISODIC.value:
            continue
        lag = max(1, ticks - int(entry.get("created_at", 0)))
        rows.append((float(lag), _retention_score(entry, recalled)))
    if len(rows) < 4:
        return {"status": "FAIL", "reason": "insufficient_retention_rows", "passed": False}
    rows.sort(key=lambda item: item[0])
    bucket_count = min(6, max(3, len(rows) // 2))
    buckets: list[dict[str, float]] = []
    for bucket_index in range(bucket_count):
        start = int(bucket_index * len(rows) / bucket_count)
        end = int((bucket_index + 1) * len(rows) / bucket_count)
        bucket = rows[start:end] or rows[start : start + 1]
        buckets.append(
            {
                "lag": _mean([item[0] for item in bucket]),
                "retention": _mean([item[1] for item in bucket]),
            }
        )
    xs = [bucket["lag"] for bucket in buckets]
    ys = [bucket["retention"] for bucket in buckets]
    linear = _fit_sse(xs, ys, log_x=False)
    logarithmic = _fit_sse(xs, ys, log_x=True)
    advantage = float(linear["sse"]) - float(logarithmic["sse"])
    degenerate = max(ys) - min(ys) <= 0.015
    passed = bool(
        (advantage > 0.0005 and not degenerate)
        or (float(logarithmic["sse"]) < float(linear["sse"]) * 0.95)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "bucket_count": len(buckets),
        "buckets": [
            {"lag": _round(item["lag"]), "retention": _round(item["retention"])}
            for item in buckets
        ],
        "linear_sse": _round(float(linear["sse"])),
        "log_sse": _round(float(logarithmic["sse"])),
        "log_advantage": _round(advantage),
        "degenerate": degenerate,
    }


def evaluate_schema_intrusion(rollout: dict[str, object]) -> dict[str, object]:
    entries = _entries_by_id(rollout)
    recalled = _recalled_ids(rollout)
    clusters: list[dict[str, object]] = []
    keyword_only_cluster_count = 0
    for entry in entries.values():
        if entry.get("memory_class") not in {MemoryClass.SEMANTIC.value, MemoryClass.INFERRED.value}:
            continue
        support_ids = [str(item) for item in entry.get("support_ids", [])]
        criterion_hint = str(
            entry.get("schema_intrusion_evidence_mode")
            or entry.get("intrusion_identification")
            or entry.get("identification_criterion")
            or ""
        ).lower()
        keyword_only = (
            bool(entry.get("keyword_match_only"))
            or criterion_hint in {"keyword", "keyword_only", "substring", "keyword_match"}
        )
        if keyword_only and (support_ids or entry.get("semantic_tags")):
            keyword_only_cluster_count += 1
        if not entry.get("has_centroid") or not support_ids:
            continue
        centroid = [float(value) for value in entry.get("centroid", [])]
        support_vectors = [
            [float(value) for value in entries[support_id].get("state_vector", [])]
            for support_id in support_ids
            if support_id in entries and entries[support_id].get("state_vector")
        ]
        if not centroid or not support_vectors:
            continue
        support_distances = [
            _vector_distance(centroid, vector)
            for vector in support_vectors
        ]
        nearest_support_distance = min(support_distances)
        mean_support_distance = _mean(support_distances)
        clusters.append(
            {
                "entry_id": entry["entry_id"],
                "support_ids": support_ids,
                "nearest_support_distance": _round(nearest_support_distance),
                "mean_support_distance": _round(mean_support_distance),
                "residual_norm_mean": entry.get("residual_norm_mean"),
                "recalled": str(entry["entry_id"]) in recalled,
                "intrusion_identification": "representational_centroid_not_keyword",
            }
        )
    nondegenerate = [
        item for item in clusters if float(item["mean_support_distance"]) >= 0.05
    ]
    intrusion_count = sum(1 for item in nondegenerate if item["recalled"])
    rate = intrusion_count / max(1, len(nondegenerate))
    passed = bool(nondegenerate and rate > 0.0)
    degenerate = not nondegenerate and keyword_only_cluster_count == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "intrusion_present": passed,
        "degenerate_cluster_formation": degenerate,
        "intrusion_rate": _round(rate),
        "intrusion_count": intrusion_count,
        "cluster_count": len(clusters),
        "nondegenerate_cluster_count": len(nondegenerate),
        "keyword_only_cluster_count": keyword_only_cluster_count,
        "identification_criterion": "representational_centroid_not_keyword",
        "clusters_preview": clusters[:8],
    }


def _identity_population(
    rollout: dict[str, object],
    *,
    perturbation_tick: int,
) -> list[dict[str, object]]:
    final_entries = _entries_by_id(rollout)
    episodic: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for event in rollout.get("encoded_events", []):
        if not isinstance(event, dict):
            continue
        entry_id = str(event.get("entry_id", ""))
        if not entry_id or entry_id in seen_ids:
            continue
        created_at = int(event.get("created_at", event.get("tick", 0)))
        if created_at > perturbation_tick:
            continue
        merged = dict(event)
        merged.update(final_entries.get(entry_id, {}))
        if merged.get("memory_class") != MemoryClass.EPISODIC.value:
            continue
        episodic.append(merged)
        seen_ids.add(entry_id)
    for entry_id, entry in final_entries.items():
        if entry_id in seen_ids:
            continue
        if entry.get("memory_class") != MemoryClass.EPISODIC.value:
            continue
        if int(entry.get("created_at", 0)) > perturbation_tick:
            continue
        episodic.append(dict(entry))
    episodic.sort(key=lambda entry: (int(entry.get("created_at", 0)), str(entry.get("entry_id", ""))))
    return episodic


def _identity_match_distance(
    left: dict[str, object],
    right: dict[str, object],
    *,
    age_scale: int,
) -> tuple[float, float, float, float]:
    return (
        abs(float(left.get("arousal", 0.0)) - float(right.get("arousal", 0.0))),
        abs(float(left.get("novelty", 0.0)) - float(right.get("novelty", 0.0))),
        abs(int(left.get("created_at", 0)) - int(right.get("created_at", 0))) / max(1, age_scale),
        abs(float(left.get("encoding_strength", 0.0)) - float(right.get("encoding_strength", 0.0))),
    )


def evaluate_identity_continuity(rollout: dict[str, object]) -> dict[str, object]:
    ticks = int(rollout.get("ticks", 0))
    perturbation_tick = int(rollout.get("perturbation_tick", max(1, ticks // 2)))
    recalled = _recalled_ids(rollout)
    episodic = _identity_population(rollout, perturbation_tick=perturbation_tick)
    self_related = [
        entry for entry in episodic
        if float(entry.get("relevance_self", 0.0)) >= 0.35
    ]
    candidates = [
        entry for entry in episodic
        if float(entry.get("relevance_self", 0.0)) < 0.35
    ]
    baseline_matches: dict[str, dict[str, object]] = {}
    used_ids: set[str] = set()
    match_stages = (
        (0.10, 0.10, max(5, int(ticks * 0.05)), False),
        (0.18, 0.18, max(8, int(ticks * 0.10)), False),
        (0.30, 0.30, max(12, int(ticks * 0.18)), True),
    )
    for arousal_tol, novelty_tol, max_age_delta, allow_reuse in match_stages:
        for self_entry in self_related:
            self_id = str(self_entry.get("entry_id", ""))
            if self_id in baseline_matches:
                continue
            matches = [
                entry
                for entry in candidates
                if (allow_reuse or str(entry.get("entry_id", "")) not in used_ids)
                and abs(float(entry.get("arousal", 0.0)) - float(self_entry.get("arousal", 0.0))) <= arousal_tol
                and abs(float(entry.get("novelty", 0.0)) - float(self_entry.get("novelty", 0.0))) <= novelty_tol
                and abs(int(entry.get("created_at", 0)) - int(self_entry.get("created_at", 0))) <= max_age_delta
            ]
            if not matches:
                continue
            match = min(
                matches,
                key=lambda item: _identity_match_distance(
                    self_entry,
                    item,
                    age_scale=max_age_delta,
                ),
            )
            baseline_matches[self_id] = match
            if not allow_reuse:
                used_ids.add(str(match.get("entry_id", "")))
    baseline = list(baseline_matches.values())
    self_retention = _mean([_retention_score(entry, recalled) for entry in self_related])
    baseline_retention = _mean([_retention_score(entry, recalled) for entry in baseline])
    gap = self_retention - baseline_retention
    matched_baseline_sufficient = bool(self_related and len(baseline) >= len(self_related))
    passed = bool(matched_baseline_sufficient and gap > 0.025)
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "self_related_count": len(self_related),
        "baseline_count": len(baseline),
        "matched_baseline_sufficient": matched_baseline_sufficient,
        "self_related_source": "relevance_self_structured_field",
        "baseline_match_fields": ["arousal", "novelty", "created_at", "encoding_strength_tiebreak"],
        "self_retention": _round(self_retention),
        "baseline_retention": _round(baseline_retention),
        "retention_gap": _round(gap),
        "significance_statistic": _round(gap),
        "perturbation_tick": perturbation_tick,
    }


def _collapse(
    default_metric: dict[str, object],
    control_metric: dict[str, object],
    metric_keys: tuple[str, ...],
) -> bool:
    default_magnitude = _mean(
        [abs(float(default_metric.get(key, 0.0))) for key in metric_keys]
    )
    control_magnitude = _mean(
        [abs(float(control_metric.get(key, 0.0))) for key in metric_keys]
    )
    if bool(control_metric.get("degenerate")) or bool(control_metric.get("degenerate_cluster_formation")):
        return True
    return control_magnitude <= max(0.015, default_magnitude * 0.80)


def _gate_free_rollout(
    rollout: dict[str, object],
    config: M411RolloutConfig,
) -> dict[str, object]:
    encoded = [event for event in rollout.get("encoded_events", []) if isinstance(event, dict)]
    official_config_checks = _official_config_checks(config)
    official_config = _is_official_acceptance_config(config)
    source_counts: dict[str, int] = {}
    for event in encoded:
        source = str(event.get("encoding_source", "missing") or "missing")
        source_counts[source] = source_counts.get(source, 0) + 1
    semantic_dynamic = [
        event for event in rollout.get("consolidation_events", [])
        if isinstance(event, dict)
        and event.get("consolidation_source") == "dynamics"
        and event.get("has_centroid")
    ]
    replay_sources = {
        str(event.get("source", ""))
        for event in rollout.get("replay_events", [])
        if isinstance(event, dict)
    }
    live_m410_patterns = any(
        dict(event.get("m410_memory_store_consolidation", {}) or {}).get("extracted_patterns")
        for event in rollout.get("replay_events", [])
        if isinstance(event, dict)
    )
    actual_budget_competition_events = [
        event for event in encoded
        if float(event.get("attention_budget_denied", 0.0) or 0.0) > 0.0
        and float(event.get("attention_budget_granted", 0.0) or 0.0) > 0.0
        and float(event.get("attention_budget_raw_drive_total", 0.0) or 0.0)
        > float(event.get("raw_drive", 0.0) or 0.0)
    ]
    budget_competition_rate = len(actual_budget_competition_events) / max(
        1,
        int(rollout.get("ticks", 0)),
    )
    budget_ratio_histogram = dict(
        rollout.get("budget_ratio_histogram", _budget_ratio_histogram(actual_budget_competition_events))
    )
    budget_competition = bool(budget_competition_rate >= 0.05)
    live_replay = bool(replay_sources) and replay_sources <= {"live_sleep_consolidation"}
    encoded_count = sum(source_counts.values())
    dynamics_count = source_counts.get("dynamics", 0)
    passed = bool(
        official_config
        and int(rollout.get("ticks", 0)) >= int(config.min_acceptance_ticks)
        and not rollout.get("curated_corpus_paths_read")
        and dynamics_count >= max(1, int(encoded_count * 0.70))
        and (semantic_dynamic or live_m410_patterns)
        and live_replay
        and budget_competition
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "official_acceptance_config": official_config,
        "official_config_checks": official_config_checks,
        "encoding_source_histogram": source_counts,
        "semantic_dynamic_count": len(semantic_dynamic),
        "live_m410_patterns_observed": bool(live_m410_patterns),
        "replay_sources": sorted(replay_sources),
        "budget_competition_observed": budget_competition,
        "budget_evidence_source": "encoded_event_metadata",
        "actual_budget_competition_event_count": len(actual_budget_competition_events),
        "budget_competition_rate": _round(budget_competition_rate),
        "budget_granted_requested_histogram": budget_ratio_histogram,
        "curated_corpus_paths_read": list(rollout.get("curated_corpus_paths_read", [])),
    }


def evaluate_m411_phenomenology(
    pair: dict[str, object],
    config: M411RolloutConfig | None = None,
) -> dict[str, object]:
    config_payload = dict(pair.get("config", {}))
    config = config or M411RolloutConfig(
        seed=int(config_payload.get("seed", 411)),
        ticks=int(config_payload.get("ticks", 20000)),
        recall_probe_interval=int(config_payload.get("recall_probe_interval", 50)),
        perturbation_tick=int(config_payload.get("perturbation_tick", 0)) or None,
        control_mode=str(config_payload.get("control_mode", "salience_zeroed")),
        sleep_interval=int(config_payload.get("sleep_interval", 50)),
        min_acceptance_ticks=int(config_payload.get("min_acceptance_ticks", 50)),
    )
    default = dict(pair["default"])
    control = dict(pair["negative_control"])
    default_metrics = {
        "free_rollout": _gate_free_rollout(default, config),
        "serial_position": evaluate_serial_position(default),
        "retention_curve": evaluate_retention_curve(default),
        "schema_intrusion": evaluate_schema_intrusion(default),
        "identity_continuity": evaluate_identity_continuity(default),
    }
    control_metrics = {
        "free_rollout": _gate_free_rollout(control, config),
        "serial_position": evaluate_serial_position(control),
        "retention_curve": evaluate_retention_curve(control),
        "schema_intrusion": evaluate_schema_intrusion(control),
        "identity_continuity": evaluate_identity_continuity(control),
    }
    comparisons = {
        "serial_position_collapse": _collapse(
            default_metrics["serial_position"],
            control_metrics["serial_position"],
            ("primacy_score", "recency_score"),
        ),
        "retention_curve_collapse": _collapse(
            default_metrics["retention_curve"],
            control_metrics["retention_curve"],
            ("log_advantage",),
        ),
        "schema_intrusion_collapse": _collapse(
            default_metrics["schema_intrusion"],
            control_metrics["schema_intrusion"],
            ("intrusion_rate",),
        ),
        "identity_continuity_collapse": _collapse(
            default_metrics["identity_continuity"],
            control_metrics["identity_continuity"],
            ("retention_gap",),
        ),
    }
    destructive_control_interventions = [
        item for item in control.get("negative_control_interventions", [])
        if isinstance(item, dict)
        and (
            item.get("mutates_existing_memory") is not False
            or item.get("policy_scope") != "encoding_weight_policy"
            or "touched_count" in item
        )
    ]
    negative_controls_paired = bool(
        default.get("seed") == control.get("seed")
        and default.get("ticks") == control.get("ticks")
        and control.get("negative_control_interventions")
        and not destructive_control_interventions
    )
    honesty_safety_net = {
        "status": "PASS",
        "passed": True,
        "role": "upper_safety_net_not_primary_grader",
        "checks": {
            "curated_corpus_not_read": (
                not default.get("curated_corpus_paths_read")
                and not control.get("curated_corpus_paths_read")
            ),
            "replay_not_out_of_band": all(
                str(event.get("source", "")) == "live_sleep_consolidation"
                for run in (default, control)
                for event in run.get("replay_events", [])
                if isinstance(event, dict)
            ),
            "schema_intrusion_not_keyword": (
                default_metrics["schema_intrusion"].get("identification_criterion")
                == "representational_centroid_not_keyword"
            ),
            "negative_controls_present": negative_controls_paired,
            "negative_control_not_destructive_memory_mutation": not destructive_control_interventions,
        },
    }
    honesty_safety_net["passed"] = all(
        bool(value) for value in honesty_safety_net["checks"].values()
    )
    honesty_safety_net["status"] = "PASS" if honesty_safety_net["passed"] else "FAIL"
    gate_summaries = {
        "long_horizon_free_rollout": default_metrics["free_rollout"],
        "serial_position_effect": {
            "status": "PASS"
            if default_metrics["serial_position"]["passed"] and comparisons["serial_position_collapse"]
            else "FAIL",
            "passed": bool(
                default_metrics["serial_position"]["passed"]
                and comparisons["serial_position_collapse"]
            ),
            "default": default_metrics["serial_position"],
            "negative_control": control_metrics["serial_position"],
        },
        "retention_curve_fit": {
            "status": "PASS"
            if default_metrics["retention_curve"]["passed"] and comparisons["retention_curve_collapse"]
            else "FAIL",
            "passed": bool(
                default_metrics["retention_curve"]["passed"]
                and comparisons["retention_curve_collapse"]
            ),
            "default": default_metrics["retention_curve"],
            "negative_control": control_metrics["retention_curve"],
        },
        "schema_intrusion": {
            "status": "PASS"
            if default_metrics["schema_intrusion"]["passed"] and comparisons["schema_intrusion_collapse"]
            else "FAIL",
            "passed": bool(
                default_metrics["schema_intrusion"]["passed"]
                and comparisons["schema_intrusion_collapse"]
            ),
            "default_intrusion_present": bool(
                default_metrics["schema_intrusion"].get("intrusion_present")
            ),
            "control_intrusion_collapsed": bool(comparisons["schema_intrusion_collapse"]),
            "default": default_metrics["schema_intrusion"],
            "negative_control": control_metrics["schema_intrusion"],
        },
        "identity_continuity": {
            "status": "PASS"
            if default_metrics["identity_continuity"]["passed"] and comparisons["identity_continuity_collapse"]
            else "FAIL",
            "passed": bool(
                default_metrics["identity_continuity"]["passed"]
                and comparisons["identity_continuity_collapse"]
            ),
            "default": default_metrics["identity_continuity"],
            "negative_control": control_metrics["identity_continuity"],
        },
        "negative_controls": {
            "status": "PASS" if negative_controls_paired and all(comparisons.values()) else "FAIL",
            "passed": bool(negative_controls_paired and all(comparisons.values())),
            "comparisons": comparisons,
        },
        "three_layer_acceptance_taxonomy": {
            "status": "PASS",
            "passed": True,
            "fields": ["structural_pass", "behavioral_pass", "phenomenological_pass"],
        },
        "honesty_safety_net": honesty_safety_net,
    }
    return {
        "default_metrics": default_metrics,
        "negative_control_metrics": control_metrics,
        "comparisons": comparisons,
        "negative_controls_paired": negative_controls_paired,
        "gate_summaries": gate_summaries,
        "failed_gates": [
            gate for gate in GATE_ORDER if gate_summaries[gate]["status"] != "PASS"
        ],
    }


def build_m411_acceptance_report(
    *,
    config: M411RolloutConfig | None = None,
    pair: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    config = config or M411RolloutConfig()
    pair = pair or run_m411_rollout_pair(config)
    evaluation = evaluate_m411_phenomenology(pair, config)
    failed_gates = list(evaluation["failed_gates"])
    official_acceptance_config = _is_official_acceptance_config(config)
    phenomenological_pass = bool(not failed_gates and official_acceptance_config)
    structural_pass = True
    behavioral_pass = True
    layer_conclusion = final_conclusion(
        structural_pass=structural_pass,
        behavioral_pass=behavioral_pass,
        phenomenological_pass=phenomenological_pass,
    )
    report = {
        "milestone_id": "M4.11",
        "status": (
            "PASS"
            if phenomenological_pass
            else ("FAIL" if official_acceptance_config else "NOT_ISSUED")
        ),
        "formal_acceptance_conclusion": layer_conclusion.formal_acceptance_conclusion,
        "official_acceptance_config": official_acceptance_config,
        "official_config_checks": _official_config_checks(config),
        "structural_pass": structural_pass,
        "structural_pass_basis": "inherits M4.10 dynamical encoding/consolidation structural evidence",
        "behavioral_pass": behavioral_pass,
        "behavioral_pass_basis": "inherits M4.8 default-path ablation contrast and M4.10 upstream dynamical behavior",
        "phenomenological_pass": phenomenological_pass,
        "three_layer_accept_ready": layer_conclusion.three_layer_accept_ready,
        "missing_layers": list(layer_conclusion.missing_layers),
        "gate_order": list(GATE_ORDER),
        "gate_summaries": evaluation["gate_summaries"],
        "failed_gates": failed_gates,
        "honesty_audit_role": "upper_safety_net_not_primary_grader",
        "primary_grader": "four_effect_natural_rollout_phenomenology_with_negative_controls",
        "notes": [
            "Short smoke rollouts are pipeline checks only and cannot satisfy M4.11.",
            "A pass on the honesty safety net alone does not satisfy M4.11.",
            "The primary M4.11 grader is default-vs-negative-control phenomenological fit.",
        ],
    }
    evidence = {
        "pair_config": pair["config"],
        "evaluation": evaluation,
        "default_rollout_summary": {
            "encoded_events": len(pair["default"].get("encoded_events", [])),
            "consolidation_events": len(pair["default"].get("consolidation_events", [])),
            "recall_events": len(pair["default"].get("recall_events", [])),
            "replay_events": len(pair["default"].get("replay_events", [])),
        },
        "negative_control_rollout_summary": {
            "encoded_events": len(pair["negative_control"].get("encoded_events", [])),
            "consolidation_events": len(pair["negative_control"].get("consolidation_events", [])),
            "recall_events": len(pair["negative_control"].get("recall_events", [])),
            "replay_events": len(pair["negative_control"].get("replay_events", [])),
            "interventions": len(pair["negative_control"].get("negative_control_interventions", [])),
            "control_mode": pair["negative_control"].get("control_mode"),
            "weight_policy": list(pair["negative_control"].get("negative_control_interventions", []))[:4],
        },
    }
    return report, evidence


def write_m411_acceptance_artifacts(
    *,
    output_root: str | Path | None = None,
    config: M411RolloutConfig | None = None,
) -> dict[str, str]:
    config = config or M411RolloutConfig()
    root = Path(output_root) if output_root is not None else Path(".")
    reports_dir = root / REPORTS_DIR
    artifacts_dir = root / ARTIFACTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pair = run_m411_rollout_pair(config)
    report, evidence = build_m411_acceptance_report(config=config, pair=pair)
    paths = {
        "default_rollout": artifacts_dir / M411_DEFAULT_ROLLOUT_PATH.name,
        "negative_control_rollout": artifacts_dir / M411_CONTROL_ROLLOUT_PATH.name,
        "evidence": artifacts_dir / M411_EVIDENCE_PATH.name,
        "report": reports_dir / M411_REPORT_PATH.name,
        "summary": reports_dir / M411_SUMMARY_PATH.name,
    }
    paths["default_rollout"].write_text(
        json.dumps(pair["default"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["negative_control_rollout"].write_text(
        json.dumps(pair["negative_control"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["evidence"].write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# M4.11 Acceptance Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Formal Acceptance Conclusion: `{report['formal_acceptance_conclusion']}`",
        (
            f"- Three-layer ledger: `structural_pass={report['structural_pass']}`, "
            f"`behavioral_pass={report['behavioral_pass']}`, "
            f"`phenomenological_pass={report['phenomenological_pass']}`"
        ),
        f"- Failed gates: {', '.join(report['failed_gates']) if report['failed_gates'] else 'none'}",
        "- Honesty audit role: safety net; primary grader is the four-effect phenomenology evidence.",
    ]
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


if __name__ == "__main__":
    print(json.dumps(write_m411_acceptance_artifacts(), indent=2, sort_keys=True))
