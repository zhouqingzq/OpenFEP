"""Read-only cognitive path view over existing decision diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class PathScoringLambdas:
    lambda_energy: float = 0.25
    lambda_attention: float = 0.35
    lambda_memory: float = 0.25
    lambda_control: float = 0.35
    beta_efe: float = 0.5
    social_risk: float = 1.0
    long_term_value: float = 1.0
    current_free_energy: float = 0.25

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CognitivePathCandidate:
    path_id: str
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
    effective_temperature: float
    cost_components: dict[str, float]
    scoring_lambdas: dict[str, float]
    source_action: str
    source_policy_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PathSelectionDiagnostics:
    selected_path: CognitivePathCandidate | None
    runner_up_path: CognitivePathCandidate | None
    selection_margin: float
    uncertainty: float
    low_confidence_reason: str
    effective_temperature: float
    candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_path": (
                self.selected_path.to_dict() if self.selected_path is not None else None
            ),
            "runner_up_path": (
                self.runner_up_path.to_dict()
                if self.runner_up_path is not None
                else None
            ),
            "selection_margin": float(self.selection_margin),
            "uncertainty": float(self.uncertainty),
            "low_confidence_reason": self.low_confidence_reason,
            "effective_temperature": float(self.effective_temperature),
            "candidate_count": int(self.candidate_count),
        }


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


def _softmax_negative_costs(costs: Sequence[float], temperature: float) -> list[float]:
    if not costs:
        return []
    safe_temperature = max(0.01, _finite_float(temperature, 0.35))
    scores = [-(cost / safe_temperature) for cost in costs]
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    denominator = sum(exps)
    if denominator <= 0.0 or not math.isfinite(denominator):
        uniform = 1.0 / float(len(costs))
        return [uniform for _ in costs]
    return [value / denominator for value in exps]


def _meta_value(meta_control: object | None, name: str, default: float) -> float:
    if meta_control is None:
        return default
    if isinstance(meta_control, Mapping):
        return _finite_float(meta_control.get(name), default)
    return _finite_float(getattr(meta_control, name, default), default)


def path_scoring_lambdas_from_meta_control(
    meta_control: object | None,
) -> PathScoringLambdas:
    return PathScoringLambdas(
        lambda_energy=_meta_value(meta_control, "lambda_energy", 0.25),
        lambda_attention=_meta_value(meta_control, "lambda_attention", 0.35),
        lambda_memory=_meta_value(meta_control, "lambda_memory", 0.25),
        lambda_control=_meta_value(meta_control, "lambda_control", 0.35),
        beta_efe=_meta_value(meta_control, "beta_efe", 0.5),
    )


def effective_temperature_from_meta_control(meta_control: object | None) -> float:
    return max(
        0.01,
        _meta_value(
            meta_control,
            "exploration_temperature",
            0.35,
        ),
    )


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


def _candidate_cost_components(
    option: InterventionScore,
    *,
    current_free_energy: float,
) -> dict[str, float]:
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
    return {
        "current_free_energy": current_free_energy,
        "expected_free_energy": expected_free_energy,
        "energy_cost": energy_cost,
        "attention_cost": attention_cost,
        "memory_cost": memory_cost,
        "control_cost": control_cost,
        "social_risk": social_risk,
        "long_term_value": long_term_value,
    }


def score_path_candidate(
    option: InterventionScore,
    *,
    index: int,
    current_free_energy: float,
    lambdas: PathScoringLambdas,
    effective_temperature: float,
    posterior_weight: float = 0.0,
) -> CognitivePathCandidate:
    components = _candidate_cost_components(
        option,
        current_free_energy=current_free_energy,
    )
    total_cost = (
        components["current_free_energy"] * lambdas.current_free_energy
        + components["expected_free_energy"] * lambdas.beta_efe
        + components["energy_cost"] * lambdas.lambda_energy
        + components["attention_cost"] * lambdas.lambda_attention
        + components["memory_cost"] * lambdas.lambda_memory
        + components["control_cost"] * lambdas.lambda_control
        + components["social_risk"] * lambdas.social_risk
        - components["long_term_value"] * lambdas.long_term_value
    )
    action = str(getattr(option, "choice", "") or "unknown_action")
    return CognitivePathCandidate(
        path_id=f"candidate_{index}_{_safe_action_id(action, fallback='option')}",
        proposed_action=action,
        expected_outcome=str(
            getattr(option, "predicted_outcome", "") or "unspecified_outcome"
        ),
        current_free_energy=components["current_free_energy"],
        expected_free_energy=components["expected_free_energy"],
        energy_cost=components["energy_cost"],
        attention_cost=components["attention_cost"],
        memory_cost=components["memory_cost"],
        control_cost=components["control_cost"],
        social_risk=components["social_risk"],
        long_term_value=components["long_term_value"],
        total_cost=total_cost,
        posterior_weight=posterior_weight,
        effective_temperature=effective_temperature,
        cost_components=components,
        scoring_lambdas=lambdas.to_dict(),
        source_action=action,
        source_policy_score=_option_float(option, "policy_score", 0.0),
    )


def cognitive_path_candidates_from_diagnostics(
    diagnostics: DecisionDiagnostics,
    *,
    meta_control: object | None = None,
    effective_temperature: float | None = None,
    max_paths: int = 5,
) -> list[CognitivePathCandidate]:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if max_paths <= 0 or not ranked:
        return []
    selected = ranked[:max_paths]
    lambdas = path_scoring_lambdas_from_meta_control(meta_control)
    temperature = (
        max(0.01, _finite_float(effective_temperature, 0.35))
        if effective_temperature is not None
        else effective_temperature_from_meta_control(meta_control)
    )
    current_free_energy = _finite_float(getattr(diagnostics, "prediction_error", 0.0))
    provisional = [
        score_path_candidate(
            option,
            index=index,
            current_free_energy=current_free_energy,
            lambdas=lambdas,
            effective_temperature=temperature,
        )
        for index, option in enumerate(selected)
    ]
    weights = _softmax_negative_costs(
        [candidate.total_cost for candidate in provisional],
        temperature,
    )
    return [
        CognitivePathCandidate(
            **{
                **candidate.to_dict(),
                "posterior_weight": weights[index],
            }
        )
        for index, candidate in enumerate(provisional)
    ]


def select_cognitive_path_candidate(
    candidates: Sequence[CognitivePathCandidate],
    *,
    low_margin_threshold: float = 0.08,
) -> PathSelectionDiagnostics:
    if not candidates:
        return PathSelectionDiagnostics(
            selected_path=None,
            runner_up_path=None,
            selection_margin=0.0,
            uncertainty=1.0,
            low_confidence_reason="no_candidates",
            effective_temperature=0.35,
            candidate_count=0,
        )
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.posterior_weight),
            float(candidate.total_cost),
            str(candidate.path_id),
        ),
    )
    selected = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = (
        selected.posterior_weight - runner_up.posterior_weight
        if runner_up is not None
        else 1.0
    )
    uncertainty = max(0.0, min(1.0, 1.0 - margin))
    reason = ""
    if runner_up is not None and margin < low_margin_threshold:
        reason = "low_selection_margin"
    return PathSelectionDiagnostics(
        selected_path=selected,
        runner_up_path=runner_up,
        selection_margin=margin,
        uncertainty=uncertainty,
        low_confidence_reason=reason,
        effective_temperature=selected.effective_temperature,
        candidate_count=len(candidates),
    )


def path_candidate_competition_summary(
    candidates: Sequence[CognitivePathCandidate],
) -> dict[str, object]:
    selection = select_cognitive_path_candidate(candidates)
    return selection.to_dict()


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
