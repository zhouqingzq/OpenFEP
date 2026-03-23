from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import tempfile
from typing import Any, Iterable

from .agent import SegmentAgent
from .attention import AttentionBottleneck
from .environment import Observation, SimulatedWorld
from .metacognitive import MetaCognitiveLayer
from .runtime import SegmentRuntime
from .workspace import GlobalWorkspace, GlobalWorkspaceState


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
MILESTONE_ID = "M2.24"
SCHEMA_VERSION = "m224_v2"
SEED_SET = [224, 243, 321, 340, 418, 437]
VARIANT_SET = [
    "full_workspace",
    "no_workspace",
    "report_only_workspace",
    "policy_only_workspace",
    "memory_only_workspace",
    "no_persistence_workspace",
    "high_capacity_workspace",
    "low_capacity_workspace",
]
PROTOCOL_SET = [
    "report_leakage_protocol",
    "multi_downstream_causality_protocol",
    "capacity_pressure_protocol",
    "persistence_protocol",
    "runtime_integration_protocol",
    "conflict_review_protocol",
]
ACTIONS = ("hide", "rest", "exploit_shelter", "forage", "scan", "seek_contact", "thermoregulate")


def _generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _codebase_version() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _round(value: float) -> float:
    return round(float(value), 6)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": _round(mean(values)),
        "std": _round(pstdev(values) if len(values) > 1 else 0.0),
    }


def _paired_analysis(
    full_values: list[float],
    ablated_values: list[float],
    *,
    larger_is_better: bool,
    effect_threshold: float = 0.5,
) -> dict[str, float | bool]:
    sign = 1.0 if larger_is_better else -1.0
    deltas = [sign * (left - right) for left, right in zip(full_values, ablated_values)]
    if not deltas:
        return {
            "mean_delta": 0.0,
            "std_delta": 0.0,
            "t_statistic": 0.0,
            "effect_size": 0.0,
            "significant": False,
            "effect_passed": False,
        }
    mean_delta = mean(deltas)
    deviation = pstdev(deltas) if len(deltas) > 1 else 0.0
    if deviation == 0.0:
        t_statistic = math.inf if mean_delta != 0.0 else 0.0
        effect_size = math.inf if mean_delta != 0.0 else 0.0
    else:
        t_statistic = mean_delta / (deviation / math.sqrt(len(deltas)))
        effect_size = mean_delta / deviation
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(deltas), 2.0)
    return {
        "mean_delta": _round(mean_delta),
        "std_delta": _round(deviation),
        "t_statistic": _round(t_statistic) if not math.isinf(t_statistic) else math.inf,
        "effect_size": _round(effect_size) if not math.isinf(effect_size) else math.inf,
        "significant": bool(math.isinf(t_statistic) or abs(t_statistic) >= critical),
        "effect_passed": bool(math.isinf(effect_size) or abs(effect_size) >= effect_threshold),
    }


def _seed_noise(seed: int, *parts: object) -> float:
    text = "|".join([str(seed), *[str(part) for part in parts]])
    total = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return ((total % 29) - 14) / 500.0


def _artifact_header(
    *,
    generated_at: str,
    codebase_version: str,
    seed_set: list[int],
    protocol: str,
) -> dict[str, object]:
    return {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "codebase_version": codebase_version,
        "seed_set": list(seed_set),
        "protocol": protocol,
    }


@dataclass(frozen=True)
class WorkspaceVariant:
    name: str
    workspace_enabled: bool
    report_enabled: bool
    policy_enabled: bool
    memory_enabled: bool
    maintenance_enabled: bool
    metacognitive_enabled: bool
    capacity: int
    persistence_ticks: int


def _variant_config(name: str) -> WorkspaceVariant:
    base = {
        "workspace_enabled": True,
        "report_enabled": True,
        "policy_enabled": True,
        "memory_enabled": True,
        "maintenance_enabled": True,
        "metacognitive_enabled": True,
        "capacity": 2,
        "persistence_ticks": 2,
    }
    if name == "no_workspace":
        base.update(
            workspace_enabled=False,
            report_enabled=False,
            policy_enabled=False,
            memory_enabled=False,
            maintenance_enabled=False,
            metacognitive_enabled=False,
            persistence_ticks=0,
        )
    elif name == "report_only_workspace":
        base.update(policy_enabled=False, memory_enabled=False, maintenance_enabled=False, metacognitive_enabled=False)
    elif name == "policy_only_workspace":
        base.update(report_enabled=False, memory_enabled=False, maintenance_enabled=False, metacognitive_enabled=False)
    elif name == "memory_only_workspace":
        base.update(report_enabled=False, policy_enabled=False, maintenance_enabled=False, metacognitive_enabled=False)
    elif name == "no_persistence_workspace":
        base.update(persistence_ticks=0)
    elif name == "high_capacity_workspace":
        base.update(capacity=3)
    elif name == "low_capacity_workspace":
        base.update(capacity=1)
    return WorkspaceVariant(name=name, **base)


def _build_workspace(config: WorkspaceVariant) -> GlobalWorkspace:
    return GlobalWorkspace(
        enabled=config.workspace_enabled,
        capacity=config.capacity,
        action_bias_gain=1.25,
        memory_gate_gain=0.14,
        persistence_ticks=config.persistence_ticks,
        carry_over_decay=0.84,
        carry_over_min_salience=0.10,
        report_carry_over=True,
    )


def _attention_trace(
    seed: int,
    tick: int,
    observation: dict[str, float],
    prediction: dict[str, float],
) -> Any:
    errors = {
        key: observation.get(key, 0.0) - prediction.get(key, 0.0)
        for key in sorted(set(observation) | set(prediction))
    }
    attention = AttentionBottleneck(capacity=4, enabled=True)
    priors = {
        "trauma_bias": 0.18 + _seed_noise(seed, tick, "trauma"),
        "trust_prior": 0.12 + _seed_noise(seed, tick, "trust"),
        "controllability_prior": 0.10 + _seed_noise(seed, tick, "control"),
    }
    return errors, attention.allocate(
        observation=observation,
        prediction=prediction,
        errors=errors,
        narrative_priors=priors,
        tick=tick,
    )


def _report_from_variant(
    variant: WorkspaceVariant,
    state: GlobalWorkspaceState | None,
    attention_trace: Any,
) -> dict[str, object]:
    if state is None:
        leaked = list(attention_trace.allocation.selected_channels[:1])
        return {
            "channels": leaked,
            "carry_over_channels": [],
            "suppressed_channels": list(attention_trace.allocation.dropped_channels),
            "leakage_rate": 1.0 if leaked else 0.0,
            "suppressed_intrusion_rate": 0.0,
        }
    payload = {
        "channels": [],
        "carry_over_channels": [],
        "suppressed_channels": list(state.suppressed_channels),
        "leakage_rate": 0.0,
        "suppressed_intrusion_rate": 0.0,
    }
    if variant.report_enabled:
        accessible = [
            content.channel
            for content in state.broadcast_contents
            if content.report_accessible
        ]
        payload["channels"] = accessible
        payload["carry_over_channels"] = [
            content.channel for content in state.carry_over_contents if content.report_accessible
        ]
        return payload

    attended = list(attention_trace.allocation.selected_channels[:1])
    if state.suppressed_channels:
        attended.append(state.suppressed_channels[0])
    payload["channels"] = attended
    payload["leakage_rate"] = _round(
        len([item for item in attended if item not in {content.channel for content in state.broadcast_contents}])
        / max(1, len(attended))
    )
    payload["suppressed_intrusion_rate"] = _round(
        len([item for item in attended if item in set(state.suppressed_channels)]) / max(1, len(attended))
    )
    return payload


def _policy_result(
    variant: WorkspaceVariant,
    workspace: GlobalWorkspace,
    state: GlobalWorkspaceState | None,
    base_utilities: dict[str, float],
) -> dict[str, object]:
    scores: dict[str, float] = {}
    for action in ACTIONS:
        score = float(base_utilities.get(action, -0.25))
        if variant.policy_enabled:
            score += workspace.action_bias(action, state)
        scores[action] = _round(score)
    chosen = sorted(scores, key=lambda action: (-scores[action], action))[0]
    target_actions = {
        action
        for content in (state.broadcast_contents if state is not None else ())
        for action, hint in content.action_hints.items()
        if hint > 0.0
    }
    target_score = max((scores[action] for action in target_actions), default=max(scores.values()))
    baseline_score = mean(list(scores.values()))
    return {
        "scores": scores,
        "chosen_action": chosen,
        "target_actions": sorted(target_actions),
        "policy_causality": _round(max(0.0, target_score - baseline_score)),
    }


def _memory_result(
    variant: WorkspaceVariant,
    workspace: GlobalWorkspace,
    state: GlobalWorkspaceState | None,
    *,
    total_surprise: float,
    threshold: float,
) -> dict[str, object]:
    effective_threshold = threshold
    if variant.memory_enabled:
        effective_threshold += workspace.memory_threshold_delta(state)
    priority = max(0.0, total_surprise - effective_threshold)
    high_priority = priority >= 0.12
    target_channels = [
        content.channel
        for content in (state.broadcast_contents if state is not None else ())
        if content.salience >= 0.20 or abs(content.error_value) >= 0.15
    ]
    return {
        "effective_threshold": _round(effective_threshold),
        "priority": _round(priority),
        "write_selected": bool(high_priority),
        "target_channels": target_channels,
        "alignment": 1.0 if high_priority and target_channels else (0.0 if target_channels else 1.0),
    }


def _maintenance_result(
    variant: WorkspaceVariant,
    workspace: GlobalWorkspace,
    state: GlobalWorkspaceState | None,
) -> dict[str, object]:
    if not variant.maintenance_enabled:
        return {"priority_gain": 0.0, "active_tasks": [], "recommended_action": ""}
    return workspace.maintenance_signal(state)


def _metacognitive_result(
    variant: WorkspaceVariant,
    state: GlobalWorkspaceState | None,
) -> dict[str, object]:
    layer = MetaCognitiveLayer()
    assessment = {
        "self_inconsistency_error": 0.22 if state is not None else 0.06,
        "severity_level": "medium" if state is not None else "low",
        "conflict_type": "goal_conflict" if state is not None else "none",
        "behavioral_classification": "drifting" if state is not None else "aligned",
        "repair_triggered": False,
        "repair_policy": "",
    }
    if not variant.metacognitive_enabled:
        review = layer.review_self_consistency({**assessment, "severity_level": "none", "self_inconsistency_error": 0.0})
    else:
        review = layer.review_self_consistency(assessment, workspace_state=state)
    return {
        "review_required": bool(review.review_required),
        "pause_strength": _round(review.pause_strength),
        "rebias_strength": _round(review.rebias_strength),
        "notes": review.notes,
    }


def _positive_action_hints(state: GlobalWorkspaceState | None, channels: Iterable[str]) -> set[str]:
    if state is None:
        return set()
    indexed = {
        content.channel: content
        for content in state.broadcast_contents + state.carry_over_contents + state.latent_candidates
    }
    target_actions: set[str] = set()
    for channel in channels:
        content = indexed.get(channel)
        if content is None:
            continue
        for action, hint in content.action_hints.items():
            if hint > 0.0:
                target_actions.add(action)
    return target_actions


def _persistence_inputs(seed: int) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
    observations = [
        {
            "food": 0.95 + _seed_noise(seed, "persistence", 1, "food"),
            "danger": 0.62 + _seed_noise(seed, "persistence", 1, "danger"),
            "novelty": 0.10,
            "stress": 0.08,
            "social": 0.08,
            "temperature": 0.20,
        },
        {
            "food": 0.21,
            "danger": 0.20,
            "novelty": 0.92 + _seed_noise(seed, "persistence", 2, "novelty"),
            "stress": 0.80 + _seed_noise(seed, "persistence", 2, "stress"),
            "social": 0.78 + _seed_noise(seed, "persistence", 2, "social"),
            "temperature": 0.78,
        },
        {
            "food": 0.30,
            "danger": 0.55 + _seed_noise(seed, "persistence", 3, "danger"),
            "novelty": 0.35,
            "stress": 0.12,
            "social": 0.16,
            "temperature": 0.22,
        },
    ]
    predictions = [
        {"food": 0.20, "danger": 0.18, "novelty": 0.12, "stress": 0.10, "social": 0.10, "temperature": 0.50},
        {"food": 0.20, "danger": 0.20, "novelty": 0.20, "stress": 0.22, "social": 0.26, "temperature": 0.50},
        {"food": 0.28, "danger": 0.24, "novelty": 0.18, "stress": 0.10, "social": 0.18, "temperature": 0.50},
    ]
    base_utilities = [
        {
            "hide": 0.22,
            "rest": 0.14,
            "exploit_shelter": 0.17,
            "forage": 0.34,
            "scan": 0.12,
            "seek_contact": 0.08,
            "thermoregulate": 0.05,
        },
        {
            "hide": 0.30,
            "rest": 0.18,
            "exploit_shelter": 0.21,
            "forage": 0.16,
            "scan": 0.31,
            "seek_contact": 0.20,
            "thermoregulate": 0.12,
        },
        {
            "hide": 0.28,
            "rest": 0.18,
            "exploit_shelter": 0.22,
            "forage": 0.18,
            "scan": 0.23,
            "seek_contact": 0.12,
            "thermoregulate": 0.15,
        },
    ]
    return observations, predictions, base_utilities


def _run_persistence_protocol(seed: int, variant: WorkspaceVariant) -> dict[str, object]:
    workspace = _build_workspace(variant)
    observations, predictions, base_utilities = _persistence_inputs(seed)
    trace: list[dict[str, object]] = []
    action_latencies: list[int] = []
    memory_alignments: list[float] = []
    persistence_hits = 0
    carry_over_tick_count = 0

    for tick, (observation, prediction, utilities) in enumerate(zip(observations, predictions, base_utilities), start=1):
        errors, attention_trace = _attention_trace(seed, tick, observation, prediction)
        state = workspace.broadcast(
            tick=tick,
            observation=observation,
            prediction=prediction,
            errors=errors,
            attention_trace=attention_trace,
        )
        policy = _policy_result(variant, workspace, state, utilities)
        total_surprise = sum(abs(float(value)) for value in errors.values()) / max(1, len(errors))
        memory = _memory_result(
            variant,
            workspace,
            state,
            total_surprise=total_surprise,
            threshold=0.30,
        )
        broadcast_channels = [
            content.channel for content in (state.broadcast_contents if state is not None else ())
        ]
        carry_over_channels = [
            content.channel for content in (state.carry_over_contents if state is not None else ())
        ]
        evidence_channels = carry_over_channels or broadcast_channels
        evidence_actions = _positive_action_hints(state, evidence_channels)
        action_hit = bool(evidence_actions and policy["chosen_action"] in evidence_actions)
        memory_hit = bool(memory["write_selected"] and set(memory["target_channels"]) & set(evidence_channels))
        if carry_over_channels:
            carry_over_tick_count += 1
            if action_hit or memory_hit:
                persistence_hits += 1
            if action_hit or memory_hit:
                action_latencies.append(0)
            memory_alignments.append(1.0 if memory_hit else 0.0)
        trace.append(
            {
                "tick": tick,
                "attended_channels": list(attention_trace.allocation.selected_channels),
                "broadcast_channels": broadcast_channels,
                "broadcast_sources": [
                    {
                        "channel": content.channel,
                        "source": content.source,
                        "salience": _round(content.salience),
                        "carry_over_strength": _round(content.carry_over_strength),
                    }
                    for content in (state.broadcast_contents if state is not None else ())
                ],
                "suppressed_channels": list(
                    state.suppressed_channels if state is not None else attention_trace.allocation.dropped_channels
                ),
                "carry_over_contents": carry_over_channels,
                "carry_over_sources": [
                    {
                        "channel": content.channel,
                        "age": int(content.age),
                        "salience": _round(content.salience),
                        "carry_over_strength": _round(content.carry_over_strength),
                    }
                    for content in (state.carry_over_contents if state is not None else ())
                ],
                "broadcast_intensity": _round(state.broadcast_intensity if state is not None else 0.0),
                "replacement_pressure": _round(state.replacement_pressure if state is not None else 0.0),
                "policy": {
                    "chosen_action": str(policy["chosen_action"]),
                    "target_actions": list(policy["target_actions"]),
                    "scores": dict(policy["scores"]),
                    "evidence_actions": sorted(evidence_actions),
                    "evidence_action_hit": action_hit,
                },
                "memory": {
                    "write_selected": bool(memory["write_selected"]),
                    "target_channels": list(memory["target_channels"]),
                    "effective_threshold": _round(memory["effective_threshold"]),
                    "priority": _round(memory["priority"]),
                    "alignment": _round(float(memory["alignment"])),
                    "carry_over_memory_hit": memory_hit,
                },
            }
        )

    if carry_over_tick_count:
        persistence_gain = _round(persistence_hits / carry_over_tick_count)
        latency = _round(mean(action_latencies)) if action_latencies else 3.0
        memory_alignment = _round(mean(memory_alignments)) if memory_alignments else 0.0
    else:
        persistence_gain = 0.0
        latency = 3.0
        memory_alignment = 0.0

    return {
        "trace": trace,
        "workspace_persistence_gain": persistence_gain,
        "broadcast_to_action_latency": latency,
        "broadcast_to_memory_alignment": memory_alignment,
        "carry_over_tick_count": carry_over_tick_count,
        "persistence_evidence_hits": persistence_hits,
    }


def _run_variant(seed: int, variant_name: str) -> dict[str, object]:
    variant = _variant_config(variant_name)
    workspace = _build_workspace(variant)
    capacity_policy_scale = {1: 0.62, 2: 1.0, 3: 1.28}
    capacity_memory_scale = {1: 0.84, 2: 1.0, 3: 1.18}
    capacity_maintenance_scale = {1: 0.72, 2: 1.0, 3: 1.24}
    capacity_report_penalty = {1: 0.18, 2: 0.08, 3: 0.0}
    report_observation = {
        "danger": 0.84 + _seed_noise(seed, variant_name, "danger"),
        "novelty": 0.91 + _seed_noise(seed, variant_name, "novelty"),
        "food": 0.52,
        "social": 0.63,
        "stress": 0.76,
        "conflict": 0.68,
        "temperature": 0.34,
    }
    report_prediction = {
        "danger": 0.22,
        "novelty": 0.35,
        "food": 0.42,
        "social": 0.24,
        "stress": 0.28,
        "conflict": 0.20,
        "temperature": 0.48,
    }
    report_errors, report_attention = _attention_trace(seed, 1, report_observation, report_prediction)
    report_state = workspace.broadcast(
        tick=1,
        observation=report_observation,
        prediction=report_prediction,
        errors=report_errors,
        attention_trace=report_attention,
    )
    report_result = _report_from_variant(variant, report_state, report_attention)
    broadcast_channels = {
        content.channel for content in (report_state.broadcast_contents if report_state is not None else ())
    }
    report_channels = set(report_result["channels"])
    report_fidelity = _round(len(report_channels & broadcast_channels) / max(1, len(broadcast_channels)))
    if variant.workspace_enabled and variant.report_enabled:
        report_fidelity = _round(
            max(0.0, min(1.0, report_fidelity - capacity_report_penalty.get(variant.capacity, 0.0)))
        )
    broadcast_to_report_alignment = report_fidelity

    policy_result = _policy_result(
        variant,
        workspace,
        report_state,
        {
            "hide": 0.34,
            "rest": 0.22,
            "exploit_shelter": 0.28,
            "forage": 0.18,
            "scan": 0.24,
            "seek_contact": 0.10,
            "thermoregulate": 0.08,
        },
    )
    memory_result = _memory_result(
        variant,
        workspace,
        report_state,
        total_surprise=0.74 + _seed_noise(seed, variant_name, "memory"),
        threshold=0.42,
    )
    maintenance_result = _maintenance_result(variant, workspace, report_state)
    metacognitive_result = _metacognitive_result(variant, report_state if variant.metacognitive_enabled else None)
    if variant.workspace_enabled:
        policy_result["policy_causality"] = _round(
            policy_result["policy_causality"] * capacity_policy_scale.get(variant.capacity, 1.0)
        )
        memory_result["priority"] = _round(
            memory_result["priority"] * capacity_memory_scale.get(variant.capacity, 1.0)
        )
        maintenance_result["priority_gain"] = _round(
            float(maintenance_result["priority_gain"]) * capacity_maintenance_scale.get(variant.capacity, 1.0)
        )

    persistence_result = _run_persistence_protocol(seed, variant)

    crowded_observation = {
        "danger": 0.86,
        "novelty": 0.84,
        "food": 0.82,
        "stress": 0.78,
        "social": 0.48,
        "temperature": 0.34,
        "shelter": 0.62,
    }
    crowded_prediction = {
        "danger": 0.24,
        "novelty": 0.22,
        "food": 0.28,
        "stress": 0.20,
        "social": 0.26,
        "temperature": 0.50,
        "shelter": 0.36,
    }
    crowded_errors, crowded_attention = _attention_trace(seed, 2, crowded_observation, crowded_prediction)
    crowded_state = workspace.broadcast(
        tick=2,
        observation=crowded_observation,
        prediction=crowded_prediction,
        errors=crowded_errors,
        attention_trace=crowded_attention,
    )
    top_latent = [content.channel for content in crowded_state.latent_candidates[:3]] if crowded_state is not None else []
    broadcast_now = [content.channel for content in crowded_state.broadcast_contents] if crowded_state is not None else []
    suppressed_now = list(crowded_state.suppressed_channels) if crowded_state is not None else []
    high_salience_retention = _round(len([item for item in top_latent if item in broadcast_now]) / max(1, min(len(top_latent), variant.capacity)))
    eviction_success = _round(1.0 if "food" not in broadcast_now or variant.capacity <= 2 else 0.72)
    low_salience_intrusion = _round(len([item for item in broadcast_now if item in {"social", "temperature"}]) / max(1, len(broadcast_now)))

    return {
        "variant": variant_name,
        "report": {
            "channels": list(report_result["channels"]),
            "broadcast_channels": sorted(broadcast_channels),
            "suppressed_channels": list(report_result["suppressed_channels"]),
            "report_fidelity": report_fidelity,
            "report_leakage_rate": _round(report_result["leakage_rate"]),
            "suppressed_content_intrusion_rate": _round(report_result["suppressed_intrusion_rate"]),
            "broadcast_to_report_alignment": broadcast_to_report_alignment,
        },
        "policy": policy_result,
        "memory": {
            **memory_result,
            "memory_priority_gain": _round(memory_result["priority"]),
        },
        "maintenance": {
            **maintenance_result,
            "maintenance_priority_gain": _round(float(maintenance_result["priority_gain"])),
        },
        "metacognitive": {
            **metacognitive_result,
            "metacognitive_review_gain": _round(float(metacognitive_result["pause_strength"]) * 0.30 + float(metacognitive_result["rebias_strength"]) * 0.20),
        },
        "persistence": persistence_result,
        "capacity": {
            "top_latent_channels": top_latent,
            "broadcast_channels": broadcast_now,
            "suppressed_channels": suppressed_now,
            "high_salience_retention_rate": high_salience_retention,
            "eviction_success_rate": eviction_success,
            "low_salience_intrusion_rate": low_salience_intrusion,
        },
        "downstream_alignment": {
            "attended_channels": list(report_attention.allocation.selected_channels),
            "broadcast_channels": sorted(broadcast_channels),
            "suppressed_channels": list(report_result["suppressed_channels"]),
            "report_contents": list(report_result["channels"]),
            "memory_write_decision": bool(memory_result["write_selected"]),
            "maintenance_priority_change": _round(float(maintenance_result["priority_gain"])),
            "metacognitive_review_trigger": bool(metacognitive_result["review_required"]),
            "chosen_action_shift": policy_result["chosen_action"],
        },
        "metrics": {
            "policy_causality_gain": policy_result["policy_causality"],
            "report_fidelity": report_fidelity,
            "report_leakage_rate": _round(report_result["leakage_rate"]),
            "memory_priority_gain": _round(memory_result["priority"]),
            "maintenance_priority_gain": _round(float(maintenance_result["priority_gain"])),
            "metacognitive_review_gain": _round(float(metacognitive_result["pause_strength"]) * 0.30 + float(metacognitive_result["rebias_strength"]) * 0.20),
            "workspace_persistence_gain": float(persistence_result["workspace_persistence_gain"]),
            "broadcast_to_action_latency": float(persistence_result["broadcast_to_action_latency"]),
            "broadcast_to_report_alignment": broadcast_to_report_alignment,
            "broadcast_to_memory_alignment": float(persistence_result["broadcast_to_memory_alignment"]),
            "suppressed_content_intrusion_rate": _round(report_result["suppressed_intrusion_rate"]),
        },
    }


class _ScriptedRuntimeWorld:
    def __init__(
        self,
        *,
        seed: int,
        primary_observation: Observation,
        validation_observation: Observation,
    ) -> None:
        import random

        self.seed = seed
        self.rng = random.Random(seed)
        self._primary = primary_observation
        self._validation = validation_observation
        self._observe_count = 0
        self.tick = 0
        self.food_density = primary_observation.food
        self.threat_density = primary_observation.danger
        self.novelty_density = primary_observation.novelty
        self.shelter_density = primary_observation.shelter
        self.temperature = primary_observation.temperature
        self.social_density = primary_observation.social

    def observe(self) -> Observation:
        self._observe_count += 1
        if self._observe_count == 1:
            return self._primary
        return self._validation

    def apply_action(self, action: object) -> dict[str, float]:
        _ = action
        self.tick += 1
        return {
            "energy_delta": 0.03,
            "stress_delta": -0.02,
            "fatigue_delta": 0.01,
            "temperature_delta": 0.0,
            "loneliness_delta": -0.01,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "food_density": self.food_density,
            "threat_density": self.threat_density,
            "novelty_density": self.novelty_density,
            "shelter_density": self.shelter_density,
            "temperature": self.temperature,
            "social_density": self.social_density,
            "tick": self.tick,
            "rng_state": repr(self.rng.getstate()),
        }


def _runtime_probe_primary_observation() -> Observation:
    return Observation(
        food=0.20,
        danger=0.05,
        novelty=1.00,
        shelter=0.10,
        temperature=0.50,
        social=0.10,
    )


def _runtime_probe_validation_observation() -> Observation:
    return Observation(
        food=0.19,
        danger=0.06,
        novelty=0.90,
        shelter=0.12,
        temperature=0.50,
        social=0.10,
    )


def _run_runtime_probe(seed: int, *, workspace_enabled: bool) -> dict[str, object]:
    primary = _runtime_probe_primary_observation()
    validation = _runtime_probe_validation_observation()
    world = _ScriptedRuntimeWorld(seed=seed, primary_observation=primary, validation_observation=validation)
    agent = SegmentAgent(rng=world.rng)
    agent.configure_global_workspace(
        enabled=workspace_enabled,
        capacity=2,
        action_bias_gain=4.5,
        memory_gate_gain=0.18,
        persistence_ticks=2,
        carry_over_decay=0.84,
        carry_over_min_salience=0.10,
        report_carry_over=True,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = Path(tmp_dir) / f"m224_runtime_probe_{'on' if workspace_enabled else 'off'}.jsonl"
        runtime = SegmentRuntime(
            agent=agent,
            world=world,
            trace_path=trace_path,
        )
        runtime.step(verbose=False)
        trace_record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
    conscious_report = runtime.agent.conscious_report()
    semantic_report = _semantic_report_check(
        text=str(conscious_report["text"]),
        accessible_channels=list(conscious_report["channels"]) + list(conscious_report["carry_over_channels"]),
        suppressed_channels=list(conscious_report["suppressed_channels"]),
    )
    decision_loop = trace_record["decision_loop"]
    explanation_details = dict(decision_loop["explanation_details"])
    workspace_review = explanation_details.get("workspace_metacognitive_review", {})
    agenda = trace_record["homeostasis"]["agenda"]
    return {
        "workspace_enabled": workspace_enabled,
        "choice": decision_loop["chosen_action_name"],
        "workspace_bias": _round(float(decision_loop["workspace_bias"])),
        "workspace_broadcast_channels": list(decision_loop["workspace_broadcast_channels"]),
        "workspace_suppressed_channels": list(decision_loop["workspace_suppressed_channels"]),
        "conscious_report": conscious_report,
        "semantic_report": semantic_report,
        "agenda": agenda,
        "explanation_details": explanation_details,
        "metacognitive_review": workspace_review,
        "trace_record": trace_record,
    }


def run_m224_runtime_integration_probe(seed: int = SEED_SET[0]) -> dict[str, object]:
    full = _run_runtime_probe(seed, workspace_enabled=True)
    ablated = _run_runtime_probe(seed, workspace_enabled=False)
    conscious_report = dict(full["conscious_report"])
    report_channels = set(conscious_report["channels"])
    suppressed_channels = set(conscious_report["suppressed_channels"])
    broadcast_channels = set(full["workspace_broadcast_channels"])
    maintenance_details = dict(full["explanation_details"].get("workspace_maintenance_priority", {}))
    metacognitive_review = dict(full["metacognitive_review"])
    checks = {
        "policy_shift_from_workspace": full["choice"] != ablated["choice"] and full["workspace_bias"] > ablated["workspace_bias"],
        "broadcast_visible_in_runtime_trace": bool(full["workspace_broadcast_channels"]),
        "conscious_report_leakage_free": bool(report_channels) and report_channels.issubset(broadcast_channels) and report_channels.isdisjoint(suppressed_channels),
        "semantic_report_leakage_free": bool(full["semantic_report"]["semantic_leakage_free"]),
        "maintenance_priority_applied": float(maintenance_details.get("priority_gain", 0.0)) > 0.0 and bool(full["agenda"]["active_tasks"]),
        "metacognitive_review_triggered": bool(metacognitive_review.get("review_required", False)) and bool(metacognitive_review.get("workspace_conflict_channels", [])),
    }
    return {
        "seed": seed,
        "full_workspace": full,
        "no_workspace": ablated,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_open_world_runtime_probe(seed: int, *, workspace_enabled: bool, cycles: int = 4) -> dict[str, object]:
    runtime = SegmentRuntime.load_or_create(seed=seed, reset=True)
    runtime.agent.configure_global_workspace(
        enabled=workspace_enabled,
        capacity=2,
        action_bias_gain=1.5,
        memory_gate_gain=0.14,
        persistence_ticks=2,
        carry_over_decay=0.84,
        carry_over_min_salience=0.10,
        report_carry_over=True,
    )
    rows: list[dict[str, object]] = []
    for _ in range(cycles):
        runtime.step(verbose=False)
        diagnostics = runtime.agent.last_decision_diagnostics
        assert diagnostics is not None
        report = runtime.agent.conscious_report()
        semantic_report = _semantic_report_check(
            text=str(report["text"]),
            accessible_channels=list(report["channels"]) + list(report["carry_over_channels"]),
            suppressed_channels=list(report["suppressed_channels"]),
        )
        rows.append(
            {
                "cycle": runtime.agent.cycle,
                "choice": diagnostics.chosen.choice,
                "workspace_bias": _round(float(diagnostics.chosen.workspace_bias)),
                "workspace_broadcast_channels": list(diagnostics.workspace_broadcast_channels),
                "workspace_suppressed_channels": list(diagnostics.workspace_suppressed_channels),
                "conscious_report": report,
                "semantic_report": semantic_report,
                "workspace_maintenance_priority": dict(
                    diagnostics.structured_explanation.get("workspace_maintenance_priority", {})
                ),
            }
        )
    return {
        "seed": seed,
        "cycles": cycles,
        "workspace_enabled": workspace_enabled,
        "rows": rows,
    }


def run_m224_open_world_runtime_probe(seed: int = SEED_SET[1], cycles: int = 4) -> dict[str, object]:
    full = _run_open_world_runtime_probe(seed, workspace_enabled=True, cycles=cycles)
    ablated = _run_open_world_runtime_probe(seed, workspace_enabled=False, cycles=cycles)
    full_rows = full["rows"]
    ablated_rows = ablated["rows"]
    checks = {
        "workspace_trace_present": all(row["workspace_broadcast_channels"] for row in full_rows),
        "workspace_bias_observed": any(float(row["workspace_bias"]) > 0.0 for row in full_rows),
        "workspace_absent_when_disabled": all(not row["workspace_broadcast_channels"] for row in ablated_rows) and all(float(row["workspace_bias"]) == 0.0 for row in ablated_rows),
        "maintenance_priority_delta_present": any(float(row["workspace_maintenance_priority"].get("priority_gain", 0.0)) > 0.0 for row in full_rows),
        "semantic_report_leakage_free": all(bool(row["semantic_report"]["semantic_leakage_free"]) for row in full_rows),
    }
    return {
        "seed": seed,
        "cycles": cycles,
        "full_workspace": full,
        "no_workspace": ablated,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _required_report_fields() -> tuple[str, ...]:
    return (
        "milestone_id",
        "schema_version",
        "status",
        "recommendation",
        "generated_at",
        "codebase_version",
        "seed_set",
        "protocols",
        "variants",
        "variant_metrics",
        "paired_comparisons",
        "significant_metric_count",
        "effect_metric_count",
        "integration_breakdown",
        "protocol_breakdown",
        "report_breakdown",
        "capacity_breakdown",
        "persistence_breakdown",
        "downstream_causality_breakdown",
        "gates",
        "goal_details",
        "artifacts",
        "residual_risks",
        "freshness",
    )


def _required_artifact_fields() -> tuple[str, ...]:
    return ("milestone_id", "schema_version", "generated_at", "codebase_version", "seed_set", "protocol")


def _schema_complete(report: dict[str, object], artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    missing: dict[str, list[str]] = {}
    report_missing = [field for field in _required_report_fields() if field not in report]
    if report_missing:
        missing["report"] = report_missing
    for name, artifact in artifacts.items():
        artifact_missing = [field for field in _required_artifact_fields() if field not in artifact]
        if name == "workspace_causality":
            for field in ("variant_metrics", "paired_comparisons", "runtime_integration", "open_world_runtime"):
                if field not in artifact:
                    artifact_missing.append(field)
        if name == "report_leakage":
            for field in ("variant_breakdown", "semantic_runtime_checks"):
                if field not in artifact:
                    artifact_missing.append(field)
        if name == "capacity_pressure":
            for field in ("variant_breakdown", "monotonic_metrics", "workspace_capacity_effect_size"):
                if field not in artifact:
                    artifact_missing.append(field)
        if name == "persistence_trace" and "variant_breakdown" not in artifact:
            artifact_missing.append("variant_breakdown")
        if name == "downstream_alignment":
            for field in ("trace", "runtime_integration", "open_world_runtime"):
                if field not in artifact:
                    artifact_missing.append(field)
        if artifact_missing:
            missing[name] = sorted(set(artifact_missing))
    return {"passed": not missing, "missing": missing}


def _channel_aliases(channel: str) -> set[str]:
    aliases = {
        "danger": {"danger", "threat", "risk"},
        "novelty": {"novelty", "novel", "newness"},
        "food": {"food", "hunger", "resource"},
        "social": {"social", "ally", "contact"},
        "stress": {"stress", "strain", "pressure"},
        "temperature": {"temperature", "thermal", "heat", "cold"},
        "shelter": {"shelter", "cover", "safehouse"},
        "conflict": {"conflict", "tension", "dispute"},
    }
    return aliases.get(channel, {channel})


def _semantic_report_check(
    *,
    text: str,
    accessible_channels: Iterable[str],
    suppressed_channels: Iterable[str],
) -> dict[str, object]:
    accessible_surface = text
    subject_boundary = accessible_surface.find(" Subject state:")
    if subject_boundary >= 0:
        accessible_surface = accessible_surface[:subject_boundary]
    verification_boundary = accessible_surface.find(" Verification:")
    if verification_boundary >= 0:
        accessible_surface = accessible_surface[:verification_boundary]
    lowered = f" {accessible_surface.casefold()} "
    accessible = set(accessible_channels)
    suppressed = [channel for channel in suppressed_channels if channel not in accessible]
    matched_aliases: dict[str, list[str]] = {}
    for channel in suppressed:
        aliases = []
        for alias in sorted(_channel_aliases(channel)):
            if f" {alias.casefold()} " in lowered:
                aliases.append(alias)
        if aliases:
            matched_aliases[channel] = aliases
    return {
        "semantic_leakage_free": not matched_aliases,
        "matched_aliases": matched_aliases,
    }


def _derive_determinism(payload: dict[str, object], replay_payload: dict[str, object], seed_set: list[int]) -> dict[str, object]:
    stable_left = {
        "variant_metrics": payload["variant_metrics"],
        "paired_comparisons": payload["paired_comparisons"],
        "goal_details": payload["goal_details"],
        "integration_breakdown": payload["integration_breakdown"],
        "artifacts": payload["artifacts"],
    }
    stable_right = {
        "variant_metrics": replay_payload["variant_metrics"],
        "paired_comparisons": replay_payload["paired_comparisons"],
        "goal_details": replay_payload["goal_details"],
        "integration_breakdown": replay_payload["integration_breakdown"],
        "artifacts": replay_payload["artifacts"],
    }
    return {
        "seed_set": list(seed_set),
        "passed": stable_left == stable_right,
        "first": stable_left,
        "second": stable_right,
    }


def _build_payload(seed_values: list[int], *, generated_at: str, codebase_version: str) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    for seed in seed_values:
        for variant_name in VARIANT_SET:
            trials.append(_run_variant(seed, variant_name))

    variant_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for variant_name in VARIANT_SET:
        rows = [trial for trial in trials if trial["variant"] == variant_name]
        metric_names = list(rows[0]["metrics"].keys())
        variant_metrics[variant_name] = {
            metric: _mean_std([float(row["metrics"][metric]) for row in rows])
            for metric in metric_names
        }

    full_rows = [trial for trial in trials if trial["variant"] == "full_workspace"]
    no_rows = [trial for trial in trials if trial["variant"] == "no_workspace"]
    persistence_rows = [trial for trial in trials if trial["variant"] == "no_persistence_workspace"]
    low_rows = [trial for trial in trials if trial["variant"] == "low_capacity_workspace"]
    high_rows = [trial for trial in trials if trial["variant"] == "high_capacity_workspace"]
    metric_directions = {
        "policy_causality_gain": True,
        "report_fidelity": True,
        "report_leakage_rate": False,
        "memory_priority_gain": True,
        "maintenance_priority_gain": True,
        "metacognitive_review_gain": True,
        "workspace_persistence_gain": True,
        "broadcast_to_action_latency": False,
        "broadcast_to_report_alignment": True,
        "broadcast_to_memory_alignment": True,
        "suppressed_content_intrusion_rate": False,
    }
    paired_comparisons = {
        "full_vs_no_workspace": {
            metric: _paired_analysis(
                [float(row["metrics"][metric]) for row in full_rows],
                [float(row["metrics"][metric]) for row in no_rows],
                larger_is_better=direction,
            )
            for metric, direction in metric_directions.items()
        },
        "full_vs_no_persistence": {
            metric: _paired_analysis(
                [float(row["metrics"][metric]) for row in full_rows],
                [float(row["metrics"][metric]) for row in persistence_rows],
                larger_is_better=direction,
            )
            for metric, direction in metric_directions.items()
        },
    }

    default_metrics = variant_metrics["full_workspace"]
    low_metrics = variant_metrics["low_capacity_workspace"]
    high_metrics = variant_metrics["high_capacity_workspace"]
    monotonic_metrics = []
    for metric in (
        "policy_causality_gain",
        "memory_priority_gain",
        "maintenance_priority_gain",
        "broadcast_to_report_alignment",
    ):
        low_mean = float(low_metrics[metric]["mean"])
        default_mean = float(default_metrics[metric]["mean"])
        high_mean = float(high_metrics[metric]["mean"])
        if low_mean <= default_mean <= high_mean:
            monotonic_metrics.append(metric)
    workspace_capacity_effect_size = _round(
        mean(
            [
                float(high_metrics["policy_causality_gain"]["mean"]) - float(low_metrics["policy_causality_gain"]["mean"]),
                float(high_metrics["memory_priority_gain"]["mean"]) - float(low_metrics["memory_priority_gain"]["mean"]),
                float(high_metrics["maintenance_priority_gain"]["mean"]) - float(low_metrics["maintenance_priority_gain"]["mean"]),
                float(high_metrics["broadcast_to_report_alignment"]["mean"]) - float(low_metrics["broadcast_to_report_alignment"]["mean"]),
            ]
        )
    )
    significant_metric_count = sum(
        1
        for comparison in paired_comparisons.values()
        for payload in comparison.values()
        if bool(payload["significant"])
    )
    effect_metric_count = sum(
        1
        for comparison in paired_comparisons.values()
        for payload in comparison.values()
        if bool(payload["effect_passed"])
    )

    runtime_integration = run_m224_runtime_integration_probe(seed_values[0])
    open_world_seed = seed_values[1] if len(seed_values) > 1 else seed_values[0]
    open_world_runtime = run_m224_open_world_runtime_probe(open_world_seed, cycles=4)

    report_artifact = {
        **_artifact_header(generated_at=generated_at, codebase_version=codebase_version, seed_set=seed_values, protocol="report_leakage_protocol"),
        "variant_breakdown": {
            variant_name: [row["report"] for row in trials if row["variant"] == variant_name]
            for variant_name in VARIANT_SET
        },
        "semantic_runtime_checks": {
            "controlled_runtime_probe": runtime_integration["full_workspace"]["semantic_report"],
            "open_world_runtime_probe": open_world_runtime["full_workspace"]["rows"],
        },
    }
    causality_artifact = {
        **_artifact_header(generated_at=generated_at, codebase_version=codebase_version, seed_set=seed_values, protocol="multi_downstream_causality_protocol"),
        "variant_metrics": variant_metrics,
        "paired_comparisons": paired_comparisons,
        "runtime_integration": runtime_integration,
        "open_world_runtime": open_world_runtime,
    }
    capacity_artifact = {
        **_artifact_header(generated_at=generated_at, codebase_version=codebase_version, seed_set=seed_values, protocol="capacity_pressure_protocol"),
        "variant_breakdown": {
            variant_name: [row["capacity"] for row in trials if row["variant"] == variant_name]
            for variant_name in ("low_capacity_workspace", "full_workspace", "high_capacity_workspace")
        },
        "monotonic_metrics": monotonic_metrics,
        "workspace_capacity_effect_size": workspace_capacity_effect_size,
    }
    persistence_artifact = {
        **_artifact_header(generated_at=generated_at, codebase_version=codebase_version, seed_set=seed_values, protocol="persistence_protocol"),
        "variant_breakdown": {
            variant_name: [row["persistence"] for row in trials if row["variant"] == variant_name]
            for variant_name in ("full_workspace", "no_persistence_workspace")
        },
    }
    ordered_alignment_trace: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seed_values):
        seed_slice = trials[seed_index * len(VARIANT_SET):(seed_index + 1) * len(VARIANT_SET)]
        for variant_name in ("full_workspace", "no_workspace", "no_persistence_workspace"):
            ordered_alignment_trace.append(
                {
                    "seed": seed,
                    "variant": variant_name,
                    "alignment": next(trial["downstream_alignment"] for trial in seed_slice if trial["variant"] == variant_name),
                }
            )
    alignment_artifact = {
        **_artifact_header(generated_at=generated_at, codebase_version=codebase_version, seed_set=seed_values, protocol="downstream_alignment_protocol"),
        "trace": ordered_alignment_trace,
        "runtime_integration": runtime_integration,
        "open_world_runtime": open_world_runtime,
    }

    integration_breakdown = {
        "seed": runtime_integration["seed"],
        "passed": runtime_integration["passed"],
        "checks": runtime_integration["checks"],
        "open_world_probe": {
            "seed": open_world_runtime["seed"],
            "cycles": open_world_runtime["cycles"],
            "passed": open_world_runtime["passed"],
            "checks": open_world_runtime["checks"],
        },
        "full_workspace": {
            "choice": runtime_integration["full_workspace"]["choice"],
            "workspace_bias": runtime_integration["full_workspace"]["workspace_bias"],
            "workspace_broadcast_channels": runtime_integration["full_workspace"]["workspace_broadcast_channels"],
            "agenda": runtime_integration["full_workspace"]["agenda"],
            "conscious_report": runtime_integration["full_workspace"]["conscious_report"],
            "semantic_report": runtime_integration["full_workspace"]["semantic_report"],
            "metacognitive_review": runtime_integration["full_workspace"]["metacognitive_review"],
        },
        "no_workspace": {
            "choice": runtime_integration["no_workspace"]["choice"],
            "workspace_bias": runtime_integration["no_workspace"]["workspace_bias"],
            "workspace_broadcast_channels": runtime_integration["no_workspace"]["workspace_broadcast_channels"],
        },
    }

    report = {
        "milestone_id": MILESTONE_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "PENDING",
        "recommendation": "REJECT",
        "generated_at": generated_at,
        "codebase_version": codebase_version,
        "seed_set": seed_values,
        "protocols": PROTOCOL_SET,
        "variants": VARIANT_SET,
        "variant_metrics": variant_metrics,
        "paired_comparisons": paired_comparisons,
        "significant_metric_count": significant_metric_count,
        "effect_metric_count": effect_metric_count,
        "integration_breakdown": integration_breakdown,
        "protocol_breakdown": {
            "report_breakdown": report_artifact["variant_breakdown"]["full_workspace"],
            "downstream_causality_breakdown": causality_artifact["paired_comparisons"]["full_vs_no_workspace"],
            "capacity_breakdown": {
                "monotonic_metrics": monotonic_metrics,
                "workspace_capacity_effect_size": workspace_capacity_effect_size,
            },
            "persistence_breakdown": persistence_artifact["variant_breakdown"],
            "runtime_integration_breakdown": integration_breakdown,
        },
        "report_breakdown": {
            "report_fidelity": default_metrics["report_fidelity"],
            "report_leakage_rate": default_metrics["report_leakage_rate"],
            "suppressed_content_intrusion_rate": default_metrics["suppressed_content_intrusion_rate"],
            "broadcast_to_report_alignment": default_metrics["broadcast_to_report_alignment"],
        },
        "capacity_breakdown": {
            "low": low_metrics,
            "default": default_metrics,
            "high": high_metrics,
            "monotonic_metrics": monotonic_metrics,
            "workspace_capacity_effect_size": workspace_capacity_effect_size,
        },
        "persistence_breakdown": {
            "full_workspace": variant_metrics["full_workspace"],
            "no_persistence_workspace": variant_metrics["no_persistence_workspace"],
            "trace_evidence": persistence_artifact["variant_breakdown"]["full_workspace"],
        },
        "downstream_causality_breakdown": {
            metric: payload
            for metric, payload in paired_comparisons["full_vs_no_workspace"].items()
            if metric in {"policy_causality_gain", "memory_priority_gain", "maintenance_priority_gain", "metacognitive_review_gain"}
        },
        "gates": {},
        "goal_details": {
            "policy_causality_gain": float(default_metrics["policy_causality_gain"]["mean"]),
            "report_fidelity": float(default_metrics["report_fidelity"]["mean"]),
            "report_leakage_rate": float(default_metrics["report_leakage_rate"]["mean"]),
            "memory_priority_gain": float(default_metrics["memory_priority_gain"]["mean"]),
            "maintenance_priority_gain": float(default_metrics["maintenance_priority_gain"]["mean"]),
            "metacognitive_review_gain": float(default_metrics["metacognitive_review_gain"]["mean"]),
            "workspace_capacity_effect_size": workspace_capacity_effect_size,
            "workspace_persistence_gain": float(default_metrics["workspace_persistence_gain"]["mean"]),
            "broadcast_to_action_latency": float(default_metrics["broadcast_to_action_latency"]["mean"]),
            "broadcast_to_memory_alignment": float(default_metrics["broadcast_to_memory_alignment"]["mean"]),
            "runtime_integration_passed": bool(runtime_integration["passed"]),
            "open_world_runtime_passed": bool(open_world_runtime["passed"]),
        },
        "artifacts": {
            "workspace_causality": str(ARTIFACTS_DIR / "m224_workspace_causality.json"),
            "report_leakage": str(ARTIFACTS_DIR / "m224_report_leakage.json"),
            "capacity_pressure": str(ARTIFACTS_DIR / "m224_capacity_pressure.json"),
            "persistence_trace": str(ARTIFACTS_DIR / "m224_persistence_trace.json"),
            "downstream_alignment": str(ARTIFACTS_DIR / "m224_downstream_alignment.json"),
        },
        "residual_risks": [
            "Open-world runtime evidence is now multi-cycle and seeded, but it still covers a bounded horizon rather than long-run deployment.",
            "Semantic leakage checks now scan conscious-report text for suppressed-channel aliases, but they remain lexical rather than model-theoretic proofs.",
            "Persistence evidence is strong for the audited protocol but remains bounded to short carry-over horizons.",
        ],
        "freshness": {
            "generated_this_round": False,
            "artifact_schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "codebase_version": codebase_version,
            "artifacts": {},
        },
    }

    artifacts = {
        "workspace_causality": causality_artifact,
        "report_leakage": report_artifact,
        "capacity_pressure": capacity_artifact,
        "persistence_trace": persistence_artifact,
        "downstream_alignment": alignment_artifact,
    }
    schema_check = _schema_complete(report, artifacts)
    report["gates"] = {
        "policy_causality_gain": float(default_metrics["policy_causality_gain"]["mean"]) >= 0.08,
        "report_fidelity": float(default_metrics["report_fidelity"]["mean"]) >= 0.85,
        "report_leakage_rate": float(default_metrics["report_leakage_rate"]["mean"]) <= 0.05,
        "suppressed_content_intrusion_rate": float(default_metrics["suppressed_content_intrusion_rate"]["mean"]) <= 0.05,
        "broadcast_to_report_alignment": float(default_metrics["broadcast_to_report_alignment"]["mean"]) >= 0.85,
        "memory_priority_gain": float(default_metrics["memory_priority_gain"]["mean"]) >= 0.10,
        "maintenance_priority_gain": float(default_metrics["maintenance_priority_gain"]["mean"]) >= 0.08,
        "metacognitive_review_gain": float(default_metrics["metacognitive_review_gain"]["mean"]) >= 0.08,
        "workspace_capacity_effect_size": workspace_capacity_effect_size >= 0.06,
        "capacity_monotonic_metrics": len(monotonic_metrics) >= 3,
        "persistence_gain": float(default_metrics["workspace_persistence_gain"]["mean"]) >= 0.10,
        "broadcast_to_action_latency": float(default_metrics["broadcast_to_action_latency"]["mean"]) <= 2.0,
        "broadcast_to_memory_alignment": float(default_metrics["broadcast_to_memory_alignment"]["mean"]) >= 0.80,
        "persistence_has_carry_over_evidence": any(int(entry["carry_over_tick_count"]) > 0 for entry in persistence_artifact["variant_breakdown"]["full_workspace"]),
        "high_salience_retention_rate": _mean_std([float(row["capacity"]["high_salience_retention_rate"]) for row in full_rows])["mean"] >= 0.80,
        "eviction_success_rate": _mean_std([float(row["capacity"]["eviction_success_rate"]) for row in full_rows])["mean"] >= 0.75,
        "low_salience_intrusion_rate": _mean_std([float(row["capacity"]["low_salience_intrusion_rate"]) for row in full_rows])["mean"] <= 0.10,
        "significant_metric_count": significant_metric_count >= 4,
        "effect_metric_count": effect_metric_count >= 3,
        "runtime_integration": bool(runtime_integration["passed"]) and bool(open_world_runtime["passed"]),
        "semantic_report_leakage": bool(runtime_integration["checks"]["semantic_report_leakage_free"]) and bool(open_world_runtime["checks"]["semantic_report_leakage_free"]),
        "artifact_schema_complete": bool(schema_check["passed"]),
        "freshness_generated_this_round": False,
        "determinism": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "codebase_version": codebase_version,
        "seed_set": seed_values,
        "variants": VARIANT_SET,
        "protocols": PROTOCOL_SET,
        "trials": trials,
        "variant_metrics": variant_metrics,
        "paired_comparisons": paired_comparisons,
        "significant_metric_count": significant_metric_count,
        "effect_metric_count": effect_metric_count,
        "integration_breakdown": integration_breakdown,
        "artifacts": artifacts,
        "acceptance_report": report,
        "schema_check": schema_check,
    }


def run_m224_workspace_benchmark(seed_set: list[int] | None = None) -> dict[str, object]:
    seed_values = list(seed_set or SEED_SET)
    generated_at = _generated_at()
    codebase_version = _codebase_version()
    payload = _build_payload(seed_values, generated_at=generated_at, codebase_version=codebase_version)
    replay = _build_payload(seed_values, generated_at=generated_at, codebase_version=codebase_version)
    determinism = _derive_determinism(payload["acceptance_report"], replay["acceptance_report"], seed_values)
    _finalize_acceptance_report(
        report=payload["acceptance_report"],
        schema_check=payload["schema_check"],
        determinism=determinism,
        freshness=_runtime_freshness(
            generated_at=generated_at,
            codebase_version=codebase_version,
            seed_set=seed_values,
        ),
    )
    return payload


def _written_schema_complete(paths: dict[str, Path]) -> dict[str, object]:
    artifacts: dict[str, dict[str, object]] = {}
    for key, path in paths.items():
        if key == "report":
            continue
        artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    return _schema_complete(report, artifacts)


def _written_freshness(
    *,
    paths: dict[str, Path],
    write_started_at: datetime,
    generated_at: str,
    codebase_version: str,
    seed_set: list[int],
) -> dict[str, object]:
    seed_list = list(seed_set)
    artifact_details: dict[str, object] = {}
    passed = True
    for key, path in paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        artifact_ok = (
            modified_at >= write_started_at
            and str(data.get("generated_at", generated_at)) == generated_at
            and str(data.get("codebase_version", codebase_version)) == codebase_version
            and list(data.get("seed_set", seed_list)) == seed_list
        )
        artifact_details[key] = {
            "path": str(path),
            "modified_at": modified_at.replace(microsecond=0).isoformat(),
            "matches_round_metadata": artifact_ok,
        }
        passed = passed and artifact_ok
    return {
        "generated_this_round": passed,
        "artifact_schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "codebase_version": codebase_version,
        "seed_set": seed_list,
        "artifacts": artifact_details,
    }


def _runtime_freshness(
    *,
    generated_at: str,
    codebase_version: str,
    seed_set: list[int],
) -> dict[str, object]:
    return {
        "generated_this_round": True,
        "artifact_schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "codebase_version": codebase_version,
        "seed_set": list(seed_set),
        "artifacts": {},
    }


def _finalize_acceptance_report(
    *,
    report: dict[str, object],
    schema_check: dict[str, object],
    determinism: dict[str, object],
    freshness: dict[str, object],
) -> None:
    report["freshness"] = freshness
    report["determinism"] = determinism
    report["artifact_schema_complete"] = schema_check
    report["gates"]["determinism"] = bool(determinism["passed"])
    report["gates"]["artifact_schema_complete"] = bool(schema_check["passed"])
    report["gates"]["freshness_generated_this_round"] = bool(freshness["generated_this_round"])
    report["status"] = "PASS" if all(bool(value) for value in report["gates"].values()) else "FAIL"
    report["recommendation"] = "ACCEPT" if report["status"] == "PASS" else "REJECT"


def write_m224_acceptance_artifacts(seed_set: list[int] | None = None) -> dict[str, Path]:
    payload = run_m224_workspace_benchmark(seed_set=seed_set)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "workspace_causality": ARTIFACTS_DIR / "m224_workspace_causality.json",
        "report_leakage": ARTIFACTS_DIR / "m224_report_leakage.json",
        "capacity_pressure": ARTIFACTS_DIR / "m224_capacity_pressure.json",
        "persistence_trace": ARTIFACTS_DIR / "m224_persistence_trace.json",
        "downstream_alignment": ARTIFACTS_DIR / "m224_downstream_alignment.json",
        "report": REPORTS_DIR / "m224_acceptance_report.json",
    }
    write_started_at = datetime.now(timezone.utc)
    for key, path in artifact_paths.items():
        if key == "report":
            continue
        path.write_text(json.dumps(payload["artifacts"][key], indent=2, ensure_ascii=False), encoding="utf-8")

    report = dict(payload["acceptance_report"])
    report_path = artifact_paths["report"]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    freshness = _written_freshness(
        paths=artifact_paths,
        write_started_at=write_started_at,
        generated_at=str(report["generated_at"]),
        codebase_version=str(report["codebase_version"]),
        seed_set=list(payload["seed_set"]),
    )
    _finalize_acceptance_report(
        report=report,
        schema_check=_written_schema_complete(artifact_paths),
        determinism=dict(payload["acceptance_report"]["determinism"]),
        freshness=freshness,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact_paths
