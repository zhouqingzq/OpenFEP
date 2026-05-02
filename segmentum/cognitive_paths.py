"""Read-only cognitive path view over existing decision diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from .types import DecisionDiagnostics, InterventionScore


PROXY_COST_FIELDS = [
    "current_free_energy",
    "energy_cost",
    "attention_cost",
    "memory_cost",
    "control_cost",
    "social_risk",
    "long_term_value",
    "total_cost",
]


@dataclass(frozen=True)
class CognitivePath:
    path_id: str
    interpretation: str
    proposed_action: str
    expected_outcome: str
    current_free_energy: float
    expected_free_energy: float
    energy_cost: float
    attention_cost: float
    memory_cost: float
    control_cost: float
    social_risk: float
    long_term_value: float
    total_cost: float
    posterior_weight: float
    source_action: str
    source_policy_score: float
    proxy_fields: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _option_float(option: InterventionScore, name: str, default: float = 0.0) -> float:
    return _finite_float(getattr(option, name, default), default)


def _safe_action_id(value: object, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _softmax_policy_weights(options: Sequence[InterventionScore]) -> list[float]:
    if not options:
        return []
    scores = [_option_float(option, "policy_score", 0.0) for option in options]
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    denominator = sum(exps)
    if denominator <= 0.0 or not math.isfinite(denominator):
        uniform = 1.0 / float(len(options))
        return [uniform for _ in options]
    return [value / denominator for value in exps]


def _interpretation(option: InterventionScore) -> str:
    action = str(getattr(option, "choice", "") or "unknown_action")
    dominant = str(getattr(option, "dominant_component", "") or "policy_score")
    outcome = str(getattr(option, "predicted_outcome", "") or "unspecified_outcome")
    return (
        f"{action} is an existing ranked option, primarily explained by "
        f"{dominant}, with expected outcome {outcome}."
    )


def _path_from_option(
    option: InterventionScore,
    *,
    index: int,
    current_free_energy: float,
    posterior_weight: float,
) -> CognitivePath:
    expected_free_energy = _option_float(option, "expected_free_energy", 0.0)
    energy_cost = max(0.0, _option_float(option, "cost", 0.0))
    attention_cost = max(0.0, -_option_float(option, "workspace_bias", 0.0))
    memory_cost = abs(_option_float(option, "memory_bias", 0.0)) + abs(
        _option_float(option, "pattern_bias", 0.0)
    )
    control_cost = max(0.0, _option_float(option, "action_ambiguity", 0.0))
    social_risk = max(
        0.0,
        _option_float(option, "risk", 0.0)
        - max(0.0, _option_float(option, "social_bias", 0.0))
        - max(0.0, _option_float(option, "commitment_bias", 0.0)),
    )
    long_term_value = _option_float(option, "goal_alignment", 0.0) + _option_float(
        option, "value_score", 0.0
    )
    total_cost = (
        expected_free_energy
        + energy_cost
        + attention_cost
        + memory_cost
        + control_cost
        + social_risk
        - long_term_value
    )
    action = str(getattr(option, "choice", "") or "unknown_action")
    return CognitivePath(
        path_id=f"path_{index}_{_safe_action_id(action, fallback='option')}",
        interpretation=_interpretation(option),
        proposed_action=action,
        expected_outcome=str(
            getattr(option, "predicted_outcome", "") or "unspecified_outcome"
        ),
        current_free_energy=current_free_energy,
        expected_free_energy=expected_free_energy,
        energy_cost=energy_cost,
        attention_cost=attention_cost,
        memory_cost=memory_cost,
        control_cost=control_cost,
        social_risk=social_risk,
        long_term_value=long_term_value,
        total_cost=total_cost,
        posterior_weight=posterior_weight,
        source_action=action,
        source_policy_score=_option_float(option, "policy_score", 0.0),
        proxy_fields=list(PROXY_COST_FIELDS),
    )


def cognitive_paths_from_diagnostics(
    diagnostics: DecisionDiagnostics,
    *,
    max_paths: int = 5,
) -> list[CognitivePath]:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if max_paths <= 0 or not ranked:
        return []
    selected = ranked[:max_paths]
    weights = _softmax_policy_weights(selected)
    current_free_energy = _finite_float(getattr(diagnostics, "prediction_error", 0.0))
    return [
        _path_from_option(
            option,
            index=index,
            current_free_energy=current_free_energy,
            posterior_weight=weights[index],
        )
        for index, option in enumerate(selected)
    ]


def _path_ref(path: CognitivePath) -> dict[str, object]:
    return {
        "path_id": path.path_id,
        "action": path.proposed_action,
        "policy_score": path.source_policy_score,
        "expected_free_energy": path.expected_free_energy,
        "total_cost": path.total_cost,
        "posterior_weight": path.posterior_weight,
    }


def path_competition_summary(paths: Sequence[CognitivePath]) -> dict[str, object]:
    if not paths:
        return {
            "path_count": 0,
            "chosen_path": None,
            "runner_up_path": None,
            "policy_margin": 0.0,
            "efe_margin": 0.0,
            "total_cost_margin": 0.0,
            "posterior_margin": 0.0,
            "proxy_notice": "Cost fields are derived approximations, not validated measurements.",
            "proxy_fields": list(PROXY_COST_FIELDS),
        }

    chosen = paths[0]
    runner_up = paths[1] if len(paths) > 1 else None
    return {
        "path_count": len(paths),
        "chosen_path": _path_ref(chosen),
        "runner_up_path": _path_ref(runner_up) if runner_up is not None else None,
        "policy_margin": (
            chosen.source_policy_score - runner_up.source_policy_score
            if runner_up is not None
            else 0.0
        ),
        "efe_margin": (
            abs(chosen.expected_free_energy - runner_up.expected_free_energy)
            if runner_up is not None
            else 0.0
        ),
        "total_cost_margin": (
            runner_up.total_cost - chosen.total_cost if runner_up is not None else 0.0
        ),
        "posterior_margin": (
            chosen.posterior_weight - runner_up.posterior_weight
            if runner_up is not None
            else 0.0
        ),
        "proxy_notice": "Cost fields are derived approximations, not validated measurements.",
        "proxy_fields": sorted(
            {
                field_name
                for path in paths
                for field_name in getattr(path, "proxy_fields", [])
            }
        ),
    }
