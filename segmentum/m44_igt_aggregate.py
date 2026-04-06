from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

from .m4_benchmarks import _safe_round


PHASE_WINDOWS: tuple[tuple[int, int], ...] = (
    (1, 20),
    (21, 40),
    (41, 60),
    (61, 80),
    (81, 100),
)
DECKS: tuple[str, ...] = ("A", "B", "C", "D")
ADVANTAGEOUS_DECKS = {"C", "D"}
MAX_DECK_L1 = 2.0
MAX_PHASE_L1 = float(len(PHASE_WINDOWS))


def _phase_name(start: int, end: int) -> str:
    return f"{start:03d}_{end:03d}"


def _is_advantageous(row: dict[str, Any], *, deck_field: str, flag_field: str) -> bool:
    if flag_field in row:
        return bool(row[flag_field])
    return str(row.get(deck_field, "")).upper() in ADVANTAGEOUS_DECKS


def _rows_by_subject(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subject_id", "unknown"))].append(dict(row))
    for subject_id, subject_rows in grouped.items():
        grouped[subject_id] = sorted(subject_rows, key=lambda item: int(item.get("trial_index", 0)))
    return grouped


def _phase_rows(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [row for row in rows if start <= int(row.get("trial_index", 0)) <= end]


def _mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _deck_distribution(rows: list[dict[str, Any]], *, deck_field: str) -> dict[str, float]:
    total = len(rows)
    if total <= 0:
        return {deck: 0.0 for deck in DECKS}
    return {
        deck: _safe_round(sum(1.0 for row in rows if str(row.get(deck_field, "")).upper() == deck) / total)
        for deck in DECKS
    }


def _distribution_l1(left: dict[str, float], right: dict[str, float]) -> float:
    return _safe_round(sum(abs(float(left[deck]) - float(right[deck])) for deck in DECKS))


def _normalized_entropy(distribution: dict[str, float]) -> float:
    entropy = 0.0
    for value in distribution.values():
        probability = float(value)
        if probability <= 0.0:
            continue
        entropy -= probability * math.log(probability, 2)
    return _safe_round(entropy / math.log(len(DECKS), 2))


def advantageous_learning_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    l1_distance = 0.0
    for start, end in PHASE_WINDOWS:
        phase_rows = _phase_rows(rows, start, end)
        agent_rate = _mean_or_zero(
            [1.0 if _is_advantageous(row, deck_field="chosen_deck", flag_field="advantageous_choice") else 0.0 for row in phase_rows]
        )
        human_rate = _mean_or_zero(
            [1.0 if _is_advantageous(row, deck_field="human_deck", flag_field="actual_advantageous") else 0.0 for row in phase_rows]
        )
        gap = abs(agent_rate - human_rate)
        l1_distance += gap
        phases.append(
            {
                "phase": _phase_name(start, end),
                "trial_count": len(phase_rows),
                "agent_advantageous_rate": _safe_round(agent_rate),
                "human_advantageous_rate": _safe_round(human_rate),
                "gap": _safe_round(gap),
            }
        )
    l1_distance = _safe_round(l1_distance)
    return {
        "phases": phases,
        "learning_curve_distance": l1_distance,
        "advantageous_learning_curve": _safe_round(max(0.0, 1.0 - (l1_distance / max(MAX_PHASE_L1, 1e-6)))),
    }


def post_loss_switching_pattern(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_switches: list[float] = []
    human_switches: list[float] = []
    by_subject = _rows_by_subject(rows)
    for subject_rows in by_subject.values():
        for previous, current in zip(subject_rows, subject_rows[1:]):
            if float(previous.get("net_outcome", 0.0)) < 0.0:
                agent_switches.append(
                    1.0 if str(previous.get("chosen_deck", "")) != str(current.get("chosen_deck", "")) else 0.0
                )
            if float(previous.get("human_net_outcome", 0.0)) < 0.0:
                human_switches.append(
                    1.0 if str(previous.get("human_deck", "")) != str(current.get("human_deck", "")) else 0.0
                )
    agent_rate = _mean_or_zero(agent_switches)
    human_rate = _mean_or_zero(human_switches)
    gap = abs(agent_rate - human_rate)
    return {
        "agent_post_loss_switch_rate": _safe_round(agent_rate),
        "human_post_loss_switch_rate": _safe_round(human_rate),
        "post_loss_switch_gap": _safe_round(gap),
        "post_loss_switching_pattern": _safe_round(max(0.0, 1.0 - gap)),
        "agent_post_loss_event_count": len(agent_switches),
        "human_post_loss_event_count": len(human_switches),
    }


def deck_distribution_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_distribution = _deck_distribution(rows, deck_field="chosen_deck")
    human_distribution = _deck_distribution(rows, deck_field="human_deck")
    l1_distance = _distribution_l1(agent_distribution, human_distribution)
    return {
        "agent_deck_distribution": agent_distribution,
        "human_deck_distribution": human_distribution,
        "deck_distribution_l1": l1_distance,
        "deck_distribution_alignment": _safe_round(max(0.0, 1.0 - (l1_distance / MAX_DECK_L1))),
    }


def exploration_exploitation_transition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    entropy_gap = 0.0
    for start, end in PHASE_WINDOWS:
        phase_rows = _phase_rows(rows, start, end)
        agent_entropy = _normalized_entropy(_deck_distribution(phase_rows, deck_field="chosen_deck"))
        human_entropy = _normalized_entropy(_deck_distribution(phase_rows, deck_field="human_deck"))
        gap = abs(agent_entropy - human_entropy)
        entropy_gap += gap
        phases.append(
            {
                "phase": _phase_name(start, end),
                "trial_count": len(phase_rows),
                "agent_entropy": agent_entropy,
                "human_entropy": human_entropy,
                "gap": _safe_round(gap),
            }
        )
    mean_gap = _safe_round(entropy_gap / max(len(PHASE_WINDOWS), 1))
    return {
        "phases": phases,
        "exploration_exploitation_entropy_gap": mean_gap,
        "exploration_exploitation_transition": _safe_round(max(0.0, 1.0 - mean_gap)),
    }


def compute_igt_aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_rows = [dict(row) for row in rows]
    learning = advantageous_learning_curve(normalized_rows)
    switching = post_loss_switching_pattern(normalized_rows)
    distribution = deck_distribution_alignment(normalized_rows)
    entropy = exploration_exploitation_transition(normalized_rows)
    behavioral_similarity = _safe_round(
        mean(
            [
                float(learning["advantageous_learning_curve"]),
                float(switching["post_loss_switching_pattern"]),
                float(distribution["deck_distribution_alignment"]),
                float(entropy["exploration_exploitation_transition"]),
            ]
        )
    )
    return {
        **learning,
        **switching,
        **distribution,
        **entropy,
        "igt_behavioral_similarity": behavioral_similarity,
    }


__all__ = [
    "ADVANTAGEOUS_DECKS",
    "DECKS",
    "PHASE_WINDOWS",
    "advantageous_learning_curve",
    "compute_igt_aggregate_metrics",
    "deck_distribution_alignment",
    "exploration_exploitation_transition",
    "post_loss_switching_pattern",
]
