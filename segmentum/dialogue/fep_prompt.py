"""Compact FEP decision capsule for LLM prompt conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from ..cognitive_paths import (
    cognitive_paths_from_diagnostics,
    path_competition_summary,
)


_OUTCOME_ALIASES = {
    "social_reward": "social_reward",
    "social_threat": "social_threat",
    "epistemic_gain": "epistemic_gain",
    "epistemic_loss": "epistemic_loss",
    "identity_affirm": "identity_affirm",
    "identity_threat": "identity_threat",
    "neutral": "neutral",
}


@dataclass
class FEPPromptCapsule:
    chosen_action: str
    chosen_predicted_outcome: str
    chosen_risk: float
    chosen_risk_label: str
    chosen_expected_free_energy: float
    chosen_policy_score: float
    chosen_dominant_component: str
    top_alternatives: list[dict[str, object]]
    policy_margin: float
    efe_margin: float
    decision_uncertainty: str
    prediction_error: float
    prediction_error_label: str
    workspace_focus: list[str]
    workspace_suppressed: list[str]
    previous_outcome: str
    hidden_intent_score: float
    hidden_intent_label: str
    observation_channels: dict[str, float]
    cognitive_paths: list[dict[str, object]]
    path_competition: dict[str, object]
    meta_control_guidance: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_dialogue_outcome(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    text = str(value or "").strip()
    if not text:
        return "neutral"
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = text.lower()
    return _OUTCOME_ALIASES.get(normalized, normalized or "neutral")


def _float_attr(obj: object, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name, default))
    except (TypeError, ValueError):
        return default


def _str_attr(obj: object, name: str, default: str = "") -> str:
    value = getattr(obj, name, default)
    return str(value if value is not None else default)


def _risk_label(value: float) -> str:
    if value <= 1.0:
        return "low"
    if value <= 3.0:
        return "medium"
    return "high"


def _prediction_error_label(value: float) -> str:
    if value >= 0.35:
        return "volatile"
    if value >= 0.18:
        return "uncertain"
    return "stable"


def _hidden_intent_label(value: float) -> str:
    if value >= 0.70:
        return "clear_subtext"
    if value >= 0.55:
        return "possible_subtext"
    return "surface_level"


def _option_summary(option: object) -> dict[str, object]:
    return {
        "action": _str_attr(option, "choice", "ask_question"),
        "predicted_outcome": _str_attr(option, "predicted_outcome", "neutral"),
        "risk": _float_attr(option, "risk", 0.0),
        "risk_label": _risk_label(_float_attr(option, "risk", 0.0)),
        "expected_free_energy": _float_attr(option, "expected_free_energy", 0.0),
        "policy_score": _float_attr(option, "policy_score", 0.0),
        "dominant_component": _str_attr(option, "dominant_component", ""),
    }


def build_fep_prompt_capsule(
    diagnostics: Any,
    observation: Mapping[str, float],
    *,
    previous_outcome: str = "",
    meta_control_guidance: Mapping[str, object] | None = None,
) -> FEPPromptCapsule:
    obs = {str(k): float(v) for k, v in dict(observation or {}).items()}
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    chosen = getattr(diagnostics, "chosen", ranked[0] if ranked else None)
    paths = cognitive_paths_from_diagnostics(diagnostics) if diagnostics is not None else []
    cognitive_paths = [path.to_dict() for path in paths]
    path_competition = path_competition_summary(paths)
    if chosen is None:
        chosen_summary = {
            "action": "ask_question",
            "predicted_outcome": "neutral",
            "risk": 0.0,
            "risk_label": "low",
            "expected_free_energy": 0.0,
            "policy_score": 0.0,
            "dominant_component": "",
        }
        top_alternatives: list[dict[str, object]] = []
        prediction_error = 0.0
        workspace_focus: list[str] = []
        workspace_suppressed: list[str] = []
    else:
        chosen_summary = _option_summary(chosen)
        top_alternatives = [_option_summary(option) for option in ranked[:3]]
        prediction_error = _float_attr(diagnostics, "prediction_error", 0.0)
        workspace_focus = [
            str(ch) for ch in getattr(diagnostics, "workspace_broadcast_channels", []) or []
        ]
        workspace_suppressed = [
            str(ch) for ch in getattr(diagnostics, "workspace_suppressed_channels", []) or []
        ]

    if len(ranked) >= 2:
        first = ranked[0]
        second = ranked[1]
        policy_margin = _float_attr(first, "policy_score", 0.0) - _float_attr(
            second, "policy_score", 0.0
        )
        efe_margin = abs(
            _float_attr(first, "expected_free_energy", 0.0)
            - _float_attr(second, "expected_free_energy", 0.0)
        )
    else:
        policy_margin = 1.0
        efe_margin = 1.0

    if policy_margin < 0.05 or efe_margin < 0.02:
        decision_uncertainty = "high"
    elif policy_margin < 0.12 or efe_margin < 0.05:
        decision_uncertainty = "medium"
    else:
        decision_uncertainty = "low"

    hidden_intent_score = float(obs.get("hidden_intent", 0.5))
    return FEPPromptCapsule(
        chosen_action=str(chosen_summary["action"]),
        chosen_predicted_outcome=str(chosen_summary["predicted_outcome"]),
        chosen_risk=float(chosen_summary["risk"]),
        chosen_risk_label=str(chosen_summary["risk_label"]),
        chosen_expected_free_energy=float(chosen_summary["expected_free_energy"]),
        chosen_policy_score=float(chosen_summary["policy_score"]),
        chosen_dominant_component=str(chosen_summary["dominant_component"]),
        top_alternatives=top_alternatives,
        policy_margin=policy_margin,
        efe_margin=efe_margin,
        decision_uncertainty=decision_uncertainty,
        prediction_error=prediction_error,
        prediction_error_label=_prediction_error_label(prediction_error),
        workspace_focus=workspace_focus,
        workspace_suppressed=workspace_suppressed,
        previous_outcome=normalize_dialogue_outcome(previous_outcome),
        hidden_intent_score=hidden_intent_score,
        hidden_intent_label=_hidden_intent_label(hidden_intent_score),
        observation_channels=obs,
        cognitive_paths=cognitive_paths,
        path_competition=path_competition,
        meta_control_guidance=dict(meta_control_guidance) if meta_control_guidance else None,
    )
