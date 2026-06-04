from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Mapping

from .adaptive_compute import fixed_budget_decision
from .agent import SegmentAgent
from .goal_priors import build_goal_prior_adjustment
from .memory_credit import MemoryCreditSignal
from .memory_model import MemoryClass, MemoryEntry, SourceType, StoreLevel
from .preferences import Goal


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_float_map(payload: Mapping[str, object] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _state_vector(payload: Mapping[str, object] | None) -> list[float]:
    if not isinstance(payload, Mapping):
        return [0.2, 0.8, 0.1, 0.7]
    vector = payload.get("state_vector")
    if isinstance(vector, list) and all(isinstance(item, (int, float)) for item in vector):
        return [float(item) for item in vector]
    return [0.2, 0.8, 0.1, 0.7]


def _make_entry(payload: Mapping[str, object]) -> MemoryEntry:
    cycle = int(payload.get("cycle", 1))
    action = str(payload.get("action", "rest"))
    outcome = str(payload.get("outcome", "neutral"))
    predicted_effects = (
        dict(payload.get("predicted_effects"))
        if isinstance(payload.get("predicted_effects"), Mapping)
        else {
            "energy_delta": 0.0,
            "stress_delta": 0.0,
            "free_energy_delta": 0.0,
        }
    )
    observation = (
        dict(payload.get("observation"))
        if isinstance(payload.get("observation"), Mapping)
        else {
            "food": 0.20,
            "danger": 0.50,
            "novelty": 0.20,
            "shelter": 0.40,
            "temperature": 0.50,
            "social": 0.20,
        }
    )
    errors = (
        dict(payload.get("errors"))
        if isinstance(payload.get("errors"), Mapping)
        else {key: 0.0 for key in observation}
    )
    free_energy_delta = float(predicted_effects.get("free_energy_delta", 0.0))
    risk = float(payload.get("risk", 0.18))
    predicted_outcome_label = str(payload.get("predicted_outcome_label", ""))
    if not predicted_outcome_label:
        if free_energy_delta >= 0.12 and risk <= 0.24:
            predicted_outcome_label = "resource_gain"
        elif free_energy_delta <= -0.16 or risk >= 0.70:
            predicted_outcome_label = "integrity_loss"
        elif free_energy_delta < 0.0:
            predicted_outcome_label = "resource_loss"
        else:
            predicted_outcome_label = "neutral"
    return MemoryEntry(
        id=str(payload.get("entry_id", "")),
        content=str(payload.get("content", f"{action} tends toward {outcome}")),
        memory_class=MemoryClass.EPISODIC,
        store_level=StoreLevel.MID,
        source_type=SourceType.EXPERIENCE,
        created_at=cycle,
        last_accessed=cycle,
        valence=float(payload.get("valence", 0.20)),
        arousal=float(payload.get("arousal", 0.28)),
        encoding_attention=float(payload.get("encoding_attention", 0.45)),
        novelty=float(payload.get("novelty", 0.22)),
        relevance_goal=float(payload.get("relevance_goal", 0.52)),
        relevance_threat=float(payload.get("relevance_threat", 0.48)),
        relevance_self=float(payload.get("relevance_self", 0.18)),
        relevance_social=float(payload.get("relevance_social", 0.10)),
        relevance_reward=float(payload.get("relevance_reward", 0.34)),
        relevance=float(payload.get("relevance", 0.54)),
        salience=float(payload.get("salience", 0.55)),
        trace_strength=float(payload.get("trace_strength", 0.52)),
        accessibility=float(payload.get("accessibility", 0.51)),
        abstractness=float(payload.get("abstractness", 0.12)),
        source_confidence=float(payload.get("source_confidence", 0.88)),
        reality_confidence=float(payload.get("reality_confidence", 0.82)),
        semantic_tags=[str(item) for item in payload.get("semantic_tags", [])],
        context_tags=[str(item) for item in payload.get("context_tags", [])],
        anchor_slots={
            "time": str(cycle),
            "place": str(payload.get("place", "cave")),
            "agents": "self",
            "action": action,
            "outcome": outcome,
        },
        mood_context=str(payload.get("mood_context", "alert")),
        state_vector=_state_vector(payload),
        compression_metadata={
            "legacy_template": {
                "action": action,
                "predicted_outcome": predicted_outcome_label,
                "preferred_probability": float(payload.get("preferred_probability", 0.72)),
                "risk": risk,
                "observation": observation,
                "errors": errors,
                "outcome": predicted_effects,
            }
        },
    )


def _goal(goal_name: str) -> Goal:
    return Goal[str(goal_name).upper()]


def _canonical_outcome_label(
    label: str,
    *,
    risk: float = 0.0,
    free_energy_delta: float = 0.0,
) -> str:
    token = str(label or "").strip()
    if token in {
        "survival_threat",
        "integrity_loss",
        "resource_loss",
        "neutral",
        "resource_gain",
    }:
        return token
    if free_energy_delta >= 0.12 and risk <= 0.24:
        return "resource_gain"
    if free_energy_delta <= -0.16 or risk >= 0.70:
        return "integrity_loss"
    if free_energy_delta < 0.0:
        return "resource_loss"
    return "neutral"


def _fixture_goal_context(fixture: Mapping[str, object]) -> dict[str, object]:
    return {
        "active_goal": str(fixture.get("active_goal", "CONTROL")),
        "urgency_scores": {
            str(key): float(value)
            for key, value in dict(fixture.get("urgency_scores", {})).items()
            if isinstance(value, (int, float))
        },
    }


def _build_agent_from_fixture(
    fixture: Mapping[str, object],
    *,
    seed: int = 0,
) -> SegmentAgent:
    agent = SegmentAgent(rng=random.Random(seed))
    body_state = _as_float_map(fixture.get("body_state"))
    if body_state:
        agent.energy = float(body_state.get("energy", agent.energy))
        agent.stress = float(body_state.get("stress", agent.stress))
        agent.fatigue = float(body_state.get("fatigue", agent.fatigue))
        agent.temperature = float(body_state.get("temperature", agent.temperature))
    agent.sync_memory_awareness_to_long_term_memory()
    assert agent.memory_store is not None
    for item in fixture.get("episodes", []):
        if isinstance(item, Mapping):
            agent.memory_store.add(_make_entry(item))
    for index, item in enumerate(fixture.get("credit_events", []), start=1):
        if not isinstance(item, Mapping):
            continue
        signal = MemoryCreditSignal(
            linked_prediction_id=str(item.get("prediction_id", f"pred:init:{index}")),
            linked_memory_ids=(str(item.get("memory_id", "")),),
            linked_path_ids=tuple(str(value) for value in item.get("linked_path_ids", [])),
            outcome=str(item.get("outcome", "confirmed")),
            support_score=float(item.get("support_score", 0.92)),
            contradiction_score=float(item.get("contradiction_score", 0.0)),
            prediction_error_delta=float(item.get("prediction_error_delta", item.get("free_energy_delta", 0.0))),
            free_energy_delta=float(item.get("free_energy_delta", 0.0)),
            confidence_weight=float(item.get("confidence_weight", 0.78)),
            source_module="m17_11_fixture",
        )
        agent.memory_store.apply_memory_credit(signal, tick=int(item.get("tick", index + 4)))
    agent.sync_memory_awareness_to_long_term_memory()
    return agent


def _apply_ablation(agent: SegmentAgent, ablation: str | None) -> None:
    if not ablation:
        return
    if ablation == "m17_6_credit":
        agent.memory_credit_enabled = False
    elif ablation == "m17_7_reconsolidation":
        agent.surprise_reconsolidation_enabled = False
    elif ablation == "m17_8_paths":
        agent.path_substrate_enabled = False
    elif ablation == "m17_9_field":
        agent.local_field_enabled = False
    elif ablation == "m17_10_goal_priors":
        agent.goal_prior_enabled = False
    elif ablation == "m17_10_adaptive_compute":
        agent.adaptive_compute_enabled = False
    else:
        raise ValueError(f"unknown ablation: {ablation}")


def _apply_credit_signal_to_agent(
    agent: SegmentAgent,
    signal_payload: Mapping[str, object],
    *,
    tick: int,
) -> None:
    if agent.memory_store is None:
        return
    signal = MemoryCreditSignal(
        linked_prediction_id=str(signal_payload.get("linked_prediction_id", "")),
        linked_memory_ids=tuple(str(item) for item in signal_payload.get("linked_memory_ids", [])),
        linked_path_ids=tuple(str(item) for item in signal_payload.get("linked_path_ids", [])),
        outcome=str(signal_payload.get("outcome", "confirmed")),
        support_score=float(signal_payload.get("support_score", 0.92)),
        contradiction_score=float(signal_payload.get("contradiction_score", 0.0)),
        prediction_error_delta=float(signal_payload.get("prediction_error_delta", signal_payload.get("free_energy_delta", 0.0))),
        free_energy_delta=float(signal_payload.get("free_energy_delta", 0.0)),
        confidence_weight=float(signal_payload.get("confidence_weight", 0.78)),
        source_module=str(signal_payload.get("source_module", "m17_11_trajectory")),
    )
    agent.memory_store.apply_memory_credit(signal, tick=tick)
    agent.long_term_memory.episodes = agent.memory_store.to_legacy_episodes()
    agent.sync_memory_awareness_to_long_term_memory()


def _evaluate_with_context(
    agent: SegmentAgent,
    fixture: Mapping[str, object],
    *,
    memory_context: dict[str, object],
    chosen_action: str | None = None,
) -> dict[str, object]:
    observation = _as_float_map(fixture.get("observation"))
    prediction = _as_float_map(fixture.get("prediction"))
    baseline_errors = {
        key: observation.get(key, 0.0) - prediction.get(key, 0.0)
        for key in sorted(set(observation) | set(prediction))
    }
    active_goal = _goal(str(fixture.get("active_goal", "CONTROL")))
    options = agent.evaluate_action_options(
        observed=observation,
        prediction=prediction,
        priors=dict(prediction),
        free_energy_before=float(fixture.get("free_energy_before", 0.40)),
        current_cluster_id=None,
        active_goal=active_goal,
        memory_context=memory_context,
    )
    if not options:
        raise RuntimeError("no action options available for fixture evaluation")
    if not chosen_action or chosen_action not in options:
        chosen_action = min(
            options.items(),
            key=lambda item: (float(item[1]["expected_free_energy"]), item[0]),
        )[0]
    chosen = dict(options[chosen_action])
    return {
        "action": chosen_action,
        "expected_free_energy": float(chosen["expected_free_energy"]),
        "predicted_outcome": str(chosen["predicted_outcome"]),
        "projected_observation": dict(chosen.get("projected_observation", {})),
    }


def _canonicalize_memory_context(memory_context: dict[str, object]) -> dict[str, object]:
    sanitized = copy.deepcopy(memory_context)
    actions = sanitized.get("actions", {})
    if not isinstance(actions, dict):
        return sanitized
    for action_payload in actions.values():
        if not isinstance(action_payload, dict):
            continue
        predicted_effects = (
            dict(action_payload.get("predicted_effects", {}))
            if isinstance(action_payload.get("predicted_effects"), dict)
            else {}
        )
        risk = _coerce_float(action_payload.get("risk", 0.0), 0.0)
        free_energy_delta = _coerce_float(predicted_effects.get("free_energy_delta", 0.0), 0.0)
        distribution = action_payload.get("outcome_distribution", {})
        if isinstance(distribution, dict) and distribution:
            canonical: dict[str, float] = {}
            for key, value in distribution.items():
                canonical_key = _canonical_outcome_label(
                    str(key),
                    risk=risk,
                    free_energy_delta=free_energy_delta,
                )
                canonical[canonical_key] = canonical.get(canonical_key, 0.0) + float(value)
            action_payload["outcome_distribution"] = canonical
    return sanitized


def _evaluate_fixture_on_agent(
    agent: SegmentAgent,
    fixture: Mapping[str, object],
    *,
    seed: int = 0,
) -> dict[str, object]:
    observation = _as_float_map(fixture.get("observation"))
    prediction = _as_float_map(fixture.get("prediction"))
    baseline_errors = {
        key: observation.get(key, 0.0) - prediction.get(key, 0.0)
        for key in sorted(set(observation) | set(prediction))
    }
    current_state_snapshot = {
        "observation": dict(observation),
        "prediction": dict(prediction),
        "errors": dict(baseline_errors),
        "body_state": agent._current_body_state(),
    }
    goal_context = _fixture_goal_context(fixture)
    similar_memories = agent._retrieve_decision_memories(
        observed=observation,
        baseline_prediction=prediction,
        baseline_errors=baseline_errors,
        current_state_snapshot=current_state_snapshot,
        k=int(fixture.get("retrieval_k", 3)),
        goal_context=goal_context,
    )
    full_context = agent._build_memory_context(
        observed=observation,
        baseline_prediction=prediction,
        errors=baseline_errors,
        similar_memories=similar_memories,
    )
    full_context = _canonicalize_memory_context(full_context)
    full_field = dict(full_context.get("local_field", {}) or {})
    active_paths = list(full_context.get("active_paths", []) or [])
    counterfactual_audit = dict(full_field.get("counterfactual_audit", {}) or {})
    best_single_action = str(counterfactual_audit.get("best_single_action", ""))
    best_single_path_id = str(counterfactual_audit.get("best_single_path_id", ""))
    naive_topk_action = str(counterfactual_audit.get("naive_topk_action", ""))
    field_action = str(counterfactual_audit.get("field_selected_action", ""))
    field_enabled_context = agent._zero_memory_context(
        observed=observation,
        baseline_prediction=prediction,
        errors=baseline_errors,
        summary="field-enabled baseline",
        active_paths=active_paths,
        local_field=full_field,
    )
    field_enabled_context["goal_prior"] = dict(full_context.get("goal_prior", {}) or {})
    field_enabled_context["adaptive_compute"] = fixed_budget_decision().to_dict()
    field_enabled_context = _canonicalize_memory_context(field_enabled_context)
    field_enabled = _evaluate_with_context(
        agent,
        fixture,
        memory_context=field_enabled_context,
        chosen_action=field_action,
    )
    best_single_context = agent._zero_memory_context(
        observed=observation,
        baseline_prediction=prediction,
        errors=baseline_errors,
        summary="best single baseline",
        active_paths=[
            payload
            for payload in active_paths
            if str(payload.get("path_id", "")) == best_single_path_id
        ] or active_paths[:1],
        local_field={},
    )
    best_single_context["goal_prior"] = dict(full_context.get("goal_prior", {}) or {})
    best_single_context["adaptive_compute"] = fixed_budget_decision().to_dict()
    best_single_context = _canonicalize_memory_context(best_single_context)
    best_single = _evaluate_with_context(
        agent,
        fixture,
        memory_context=best_single_context,
        chosen_action=best_single_action,
    )
    naive_topk_context = agent._zero_memory_context(
        observed=observation,
        baseline_prediction=prediction,
        errors=baseline_errors,
        summary="naive top-k baseline",
        active_paths=active_paths,
        local_field={},
    )
    naive_topk_context["goal_prior"] = dict(full_context.get("goal_prior", {}) or {})
    naive_topk_context["adaptive_compute"] = fixed_budget_decision().to_dict()
    naive_topk_context = _canonicalize_memory_context(naive_topk_context)
    naive_topk = _evaluate_with_context(
        agent,
        fixture,
        memory_context=naive_topk_context,
        chosen_action=naive_topk_action,
    )
    field_off_agent = _build_agent_from_fixture(fixture, seed=seed)
    field_off_agent.local_field_enabled = False
    field_off_agent.path_substrate_enabled = False
    field_off_agent.adaptive_compute_enabled = False
    field_off_similar = field_off_agent._retrieve_decision_memories(
        observed=observation,
        baseline_prediction=prediction,
        baseline_errors=baseline_errors,
        current_state_snapshot=current_state_snapshot,
        k=int(fixture.get("retrieval_k", 3)),
        goal_context=goal_context,
    )
    field_off_context = field_off_agent._build_memory_context(
        observed=observation,
        baseline_prediction=prediction,
        errors=baseline_errors,
        similar_memories=field_off_similar,
    )
    field_off_context = _canonicalize_memory_context(field_off_context)
    field_off = _evaluate_with_context(
        field_off_agent,
        fixture,
        memory_context=field_off_context,
        chosen_action=None,
    )
    return {
        "fixture_id": str(fixture.get("fixture_id", "")),
        "field_enabled": field_enabled,
        "best_single": best_single,
        "naive_topk": naive_topk,
        "field_off": field_off,
        "counterfactual_audit": counterfactual_audit,
        "goal_prior": dict(full_context.get("goal_prior", {}) or {}),
        "adaptive_compute": dict(full_context.get("adaptive_compute", {}) or {}),
    }


def evaluate_fixture_variant(
    fixture: Mapping[str, object],
    *,
    ablation: str | None = None,
    seed: int = 0,
) -> dict[str, object]:
    agent = _build_agent_from_fixture(fixture, seed=seed)
    _apply_ablation(agent, ablation)
    return _evaluate_fixture_on_agent(agent, fixture, seed=seed)


def _trajectory_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return (values[-1] - values[0]) / float(len(values) - 1)


def run_adaptation_trajectory(
    fixture: Mapping[str, object],
    *,
    frozen_memory: bool = False,
    ablation: str | None = None,
    seed: int = 0,
) -> dict[str, object]:
    agent = _build_agent_from_fixture(fixture, seed=seed)
    _apply_ablation(agent, ablation)
    scores: list[float] = []
    statuses: list[str] = []
    for index, item in enumerate(fixture.get("trajectory", []), start=1):
        evaluation = _evaluate_fixture_on_agent(agent, fixture, seed=seed)
        scores.append(float(evaluation["field_enabled"]["expected_free_energy"]))
        statuses.append(str(dict(evaluation.get("counterfactual_audit", {})).get("status", "")))
        if frozen_memory or not isinstance(item, Mapping):
            continue
        signal = {
            "linked_prediction_id": str(item.get("prediction_id", f"traj:{index}")),
            "linked_memory_ids": [str(item.get("memory_id", ""))],
            "linked_path_ids": [str(item.get("path_id", ""))] if item.get("path_id") else [],
            "outcome": str(item.get("outcome", "confirmed")),
            "support_score": float(item.get("support_score", 0.92)),
            "contradiction_score": float(item.get("contradiction_score", 0.0)),
            "prediction_error_delta": float(item.get("prediction_error_delta", item.get("free_energy_delta", 0.0))),
            "free_energy_delta": float(item.get("free_energy_delta", 0.0)),
            "confidence_weight": float(item.get("confidence_weight", 0.78)),
            "source_module": "m17_11_trajectory",
        }
        if agent.memory_credit_enabled:
            _apply_credit_signal_to_agent(agent, signal, tick=int(item.get("tick", index + 12)))
        if (
            agent.surprise_reconsolidation_enabled
            and agent.memory_store is not None
            and agent.memory_store.get(str(item.get("memory_id", ""))) is not None
        ):
            agent.apply_reuse_reconsolidation(
                [signal],
                observation=_as_float_map(fixture.get("observation")),
                tick=int(item.get("tick", index + 12)),
            )
    return {
        "fixture_id": str(fixture.get("fixture_id", "")),
        "trajectory": scores,
        "status_trajectory": statuses,
        "trajectory_slope": round(_trajectory_slope(scores), 6),
    }


def load_field_validation_corpus(
    *,
    train_path: str | Path,
    held_out_path: str | Path,
) -> dict[str, list[dict[str, object]]]:
    train = json.loads(Path(train_path).read_text(encoding="utf-8"))
    held_out = json.loads(Path(held_out_path).read_text(encoding="utf-8"))
    return {
        "train": list(train),
        "held_out": list(held_out),
    }


def _regression_p90(losses: list[float]) -> float:
    if not losses:
        return 0.0
    ordered = sorted(losses)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.90))))
    return float(ordered[index])


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    adv_best = [
        float(row["best_single"]["expected_free_energy"]) - float(row["field_enabled"]["expected_free_energy"])
        for row in rows
    ]
    adv_naive = [
        float(row["naive_topk"]["expected_free_energy"]) - float(row["field_enabled"]["expected_free_energy"])
        for row in rows
    ]
    adv_field_off = [
        float(row["field_off"]["expected_free_energy"]) - float(row["field_enabled"]["expected_free_energy"])
        for row in rows
    ]
    regressions = [
        max(
            0.0,
            float(row["field_enabled"]["expected_free_energy"])
            - min(
                float(row["best_single"]["expected_free_energy"]),
                float(row["naive_topk"]["expected_free_energy"]),
                float(row["field_off"]["expected_free_energy"]),
            ),
        )
        for row in rows
    ]
    no_gain = sum(
        1
        for row in rows
        if str(dict(row.get("counterfactual_audit", {})).get("status", "")) == "field_divergent_no_gain"
    )
    wins = sum(1 for best, naive in zip(adv_best, adv_naive) if best > 0.0 and naive > 0.0)
    regressions_only = [value for value in regressions if value > 0.0]
    return {
        "mean_fe_advantage_vs_best_single": round(mean(adv_best) if adv_best else 0.0, 6),
        "mean_fe_advantage_vs_naive_topk": round(mean(adv_naive) if adv_naive else 0.0, 6),
        "mean_fe_advantage_vs_field_off": round(mean(adv_field_off) if adv_field_off else 0.0, 6),
        "median_fe_advantage_vs_best_single": round(median(adv_best) if adv_best else 0.0, 6),
        "median_fe_advantage_vs_naive_topk": round(median(adv_naive) if adv_naive else 0.0, 6),
        "win_rate": round(float(wins) / max(1, len(rows)), 6),
        "no_gain_rate": round(float(no_gain) / max(1, len(rows)), 6),
        "regression_rate": round(float(len(regressions_only)) / max(1, len(rows)), 6),
        "p90_regression_magnitude": round(_regression_p90(regressions_only), 6),
        "paired_rows": [
            {
                "fixture_id": str(row.get("fixture_id", "")),
                "field_enabled_fe": round(float(row["field_enabled"]["expected_free_energy"]), 6),
                "best_single_fe": round(float(row["best_single"]["expected_free_energy"]), 6),
                "naive_topk_fe": round(float(row["naive_topk"]["expected_free_energy"]), 6),
                "field_off_fe": round(float(row["field_off"]["expected_free_energy"]), 6),
                "field_status": str(dict(row.get("counterfactual_audit", {})).get("status", "")),
            }
            for row in rows
        ],
        "outcome_quantity": "m17_5_expected_free_energy_surrogate",
    }


def _ablation_report(
    held_out_rows: list[dict[str, object]],
    ablated_rows: list[dict[str, object]],
    full_trajectory: list[dict[str, object]],
    ablated_trajectory: list[dict[str, object]],
    *,
    ablation: str,
) -> dict[str, object]:
    full_metrics = _aggregate_metrics(held_out_rows)
    ablated_metrics = _aggregate_metrics(ablated_rows)
    full_slope = mean(float(item["trajectory_slope"]) for item in full_trajectory) if full_trajectory else 0.0
    ablated_slope = mean(float(item["trajectory_slope"]) for item in ablated_trajectory) if ablated_trajectory else 0.0
    contribution = (
        abs(float(full_metrics["mean_fe_advantage_vs_best_single"]) - float(ablated_metrics["mean_fe_advantage_vs_best_single"]))
        > 1e-9
        or abs(full_slope - ablated_slope) > 1e-9
    )
    return {
        "ablation": ablation,
        "held_out_metrics": ablated_metrics,
        "trajectory_slope_delta": round(full_slope - ablated_slope, 6),
        "component_no_measurable_contribution": not contribution,
    }


def run_field_validation(
    *,
    train_path: str | Path,
    held_out_path: str | Path,
    seed: int = 0,
) -> dict[str, object]:
    corpus = load_field_validation_corpus(train_path=train_path, held_out_path=held_out_path)
    train_ids = {str(item.get("fixture_id", "")) for item in corpus["train"]}
    held_out_ids = {str(item.get("fixture_id", "")) for item in corpus["held_out"]}
    overlap = bool(train_ids & held_out_ids)
    held_out_rows = [evaluate_fixture_variant(item, seed=seed) for item in corpus["held_out"]]
    train_rows = [evaluate_fixture_variant(item, seed=seed) for item in corpus["train"]]
    full_trajectory = [run_adaptation_trajectory(item, seed=seed) for item in corpus["held_out"]]
    frozen_trajectory = [
        run_adaptation_trajectory(item, seed=seed, frozen_memory=True)
        for item in corpus["held_out"]
    ]
    ablations: list[dict[str, object]] = []
    for name in (
        "m17_6_credit",
        "m17_7_reconsolidation",
        "m17_8_paths",
        "m17_9_field",
        "m17_10_goal_priors",
        "m17_10_adaptive_compute",
    ):
        ablated_rows = [evaluate_fixture_variant(item, seed=seed, ablation=name) for item in corpus["held_out"]]
        ablated_trajectory = [
            run_adaptation_trajectory(item, seed=seed, ablation=name)
            for item in corpus["held_out"]
        ]
        ablations.append(
            _ablation_report(
                held_out_rows,
                ablated_rows,
                full_trajectory,
                ablated_trajectory,
                ablation=name,
            )
        )
    result = {
        "params_fit_on": "train",
        "metrics_reported_on": "held_out",
        "fixtures_overlap": overlap,
        "leakage_detected": overlap,
        "split_summary": {
            "train_fixture_ids": sorted(train_ids),
            "held_out_fixture_ids": sorted(held_out_ids),
        },
        "train_metrics_preview": _aggregate_metrics(train_rows),
        "held_out_metrics": _aggregate_metrics(held_out_rows),
        "trajectory": {
            "full_loop": full_trajectory,
            "frozen_memory_control": frozen_trajectory,
            "full_loop_mean_slope": round(
                mean(float(item["trajectory_slope"]) for item in full_trajectory) if full_trajectory else 0.0,
                6,
            ),
            "frozen_memory_mean_slope": round(
                mean(float(item["trajectory_slope"]) for item in frozen_trajectory) if frozen_trajectory else 0.0,
                6,
            ),
        },
        "ablation_matrix": ablations,
        "honesty_statement": (
            "Outcome quantity is the M17.5 expected-free-energy surrogate, "
            "not a learned generative-model variational free energy."
        ),
    }
    return result


def render_field_validation_report(result: Mapping[str, object]) -> str:
    held_out = dict(result.get("held_out_metrics", {}) or {})
    trajectory = dict(result.get("trajectory", {}) or {})
    lines = [
        "# M17.11 Field Validation",
        "",
        "## Overfitting Guard",
        f"- params_fit_on: {result.get('params_fit_on', '')}",
        f"- metrics_reported_on: {result.get('metrics_reported_on', '')}",
        f"- fixtures_overlap: {bool(result.get('fixtures_overlap', False))}",
        f"- leakage_detected: {bool(result.get('leakage_detected', False))}",
        "",
        "## Held-out Metrics",
        f"- mean_fe_advantage_vs_best_single: {held_out.get('mean_fe_advantage_vs_best_single', 0.0)}",
        f"- mean_fe_advantage_vs_naive_topk: {held_out.get('mean_fe_advantage_vs_naive_topk', 0.0)}",
        f"- mean_fe_advantage_vs_field_off: {held_out.get('mean_fe_advantage_vs_field_off', 0.0)}",
        f"- median_fe_advantage_vs_best_single: {held_out.get('median_fe_advantage_vs_best_single', 0.0)}",
        f"- median_fe_advantage_vs_naive_topk: {held_out.get('median_fe_advantage_vs_naive_topk', 0.0)}",
        f"- win_rate: {held_out.get('win_rate', 0.0)}",
        f"- no_gain_rate: {held_out.get('no_gain_rate', 0.0)}",
        f"- regression_rate: {held_out.get('regression_rate', 0.0)}",
        f"- p90_regression_magnitude: {held_out.get('p90_regression_magnitude', 0.0)}",
        "",
        "## Trajectory",
        f"- full_loop_mean_slope: {trajectory.get('full_loop_mean_slope', 0.0)}",
        f"- frozen_memory_mean_slope: {trajectory.get('frozen_memory_mean_slope', 0.0)}",
        "",
        "## Held-out Rows",
    ]
    for row in held_out.get("paired_rows", []):
        lines.append(
            f"- {row.get('fixture_id', '')}: status={row.get('field_status', '')}, "
            f"field={row.get('field_enabled_fe', 0.0)}, best_single={row.get('best_single_fe', 0.0)}, "
            f"naive_topk={row.get('naive_topk_fe', 0.0)}, field_off={row.get('field_off_fe', 0.0)}"
        )
    lines.extend(
        [
            "",
        "## Honesty Statement",
        f"- {result.get('honesty_statement', '')}",
        ]
    )
    return "\n".join(lines)
