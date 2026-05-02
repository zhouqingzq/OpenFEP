"""Compact FEP decision capsule for LLM prompt conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

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

_SENSITIVE_KEY_FRAGMENTS = (
    "api",
    "authorization",
    "body",
    "content",
    "conversation_history",
    "diagnostic",
    "event",
    "history",
    "key",
    "markdown",
    "message",
    "payload",
    "prompt",
    "raw",
    "secret",
    "self-consciousness",
    "system",
    "token",
    "user",
)

_SENSITIVE_TEXT_MARKERS = (
    "Self-consciousness.md",
    "Conscious.md",
    "```",
)

_AFFECTIVE_RAW_KEYS = {"affective_notes", "raw_affective_notes", "notes_raw"}


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
    persona_id: str | None = None
    session_id: str | None = None
    self_prior_summary: dict[str, object] | None = None
    selected_path_summary: dict[str, object] | None = None
    path_competition_summary: dict[str, object] | None = None
    active_gaps: dict[str, list[str]] | None = None
    affective_state_summary: dict[str, object] | None = None
    meta_control_guidance: dict[str, object] | None = None
    cognitive_control_guidance: dict[str, object] | None = None
    affective_guidance: dict[str, object] | None = None
    memory_use_guidance: dict[str, object] | None = None
    omitted_signals: list[str] | None = None
    prompt_budget_summary: dict[str, object] | None = None

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


def _to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {}
    return {}


def _compact_text(value: object, *, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if any(marker.lower() in text.lower() for marker in _SENSITIVE_TEXT_MARKERS):
        return "[redacted]"
    return text[:limit]


def _safe_key(key: object) -> bool:
    lower = str(key).lower()
    return not any(fragment in lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _compact_object(
    value: object,
    *,
    depth: int = 2,
    list_limit: int = 6,
    text_limit: int = 160,
) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _compact_text(value, limit=text_limit)
    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        if depth <= 0:
            return {"summary": "[compressed]"}
        result: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in _AFFECTIVE_RAW_KEYS or not _safe_key(key_text):
                continue
            compacted = _compact_object(
                item,
                depth=depth - 1,
                list_limit=list_limit,
                text_limit=text_limit,
            )
            if compacted not in ({}, [], "", None):
                result[key_text] = compacted
            if len(result) >= list_limit:
                break
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            item
            for item in (
                _compact_object(
                    entry,
                    depth=depth - 1,
                    list_limit=list_limit,
                    text_limit=text_limit,
                )
                for entry in list(value)[:list_limit]
            )
            if item not in ({}, [], "", None)
        ]
    return _compact_text(value, limit=text_limit)


def _compact_mapping(value: object, *, depth: int = 2) -> dict[str, object] | None:
    compacted = _compact_object(value, depth=depth)
    return dict(compacted) if isinstance(compacted, Mapping) and compacted else None


def _compact_signal_names(values: object, *, limit: int = 12) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        return None
    result: list[str] = []
    for item in values:
        text = _compact_text(item, limit=64)
        if text and text != "[redacted]" and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result or None


def _active_gaps_from_state(cognitive_state: object) -> dict[str, list[str]] | None:
    state = _to_mapping(cognitive_state)
    gaps = _to_mapping(state.get("gaps")) if state else _to_mapping(getattr(cognitive_state, "gaps", None))
    if not gaps:
        return None
    result: dict[str, list[str]] = {}
    for key in (
        "epistemic_gaps",
        "contextual_gaps",
        "instrumental_gaps",
        "resource_gaps",
        "social_gaps",
        "blocking_gaps",
    ):
        values = _compact_signal_names(gaps.get(key), limit=5)
        if values:
            result[key] = values
    return result or None


def _affective_summary_from_state(
    cognitive_state: object,
    affective_state: object | None,
) -> dict[str, object] | None:
    source = affective_state
    if source is None:
        state = _to_mapping(cognitive_state)
        source = state.get("affect") if state else getattr(cognitive_state, "affect", None)
    summary = _compact_mapping(source, depth=1)
    if not summary:
        return None
    summary.pop("affective_notes", None)
    return summary or None


def _memory_guidance_from_state(
    cognitive_state: object,
    meta_control_guidance: Mapping[str, object] | None,
) -> dict[str, object] | None:
    state = _to_mapping(cognitive_state)
    memory = _to_mapping(state.get("memory")) if state else _to_mapping(getattr(cognitive_state, "memory", None))
    guidance = dict(meta_control_guidance or {})
    result: dict[str, object] = {}
    if memory:
        activated = memory.get("activated_memories", [])
        conflicts = memory.get("memory_conflicts", [])
        reusable = _compact_signal_names(memory.get("reusable_patterns"), limit=4)
        result["activated_memory_count"] = len(activated) if isinstance(activated, list) else 0
        result["memory_conflict_count"] = len(conflicts) if isinstance(conflicts, list) else 0
        if reusable:
            result["reusable_patterns"] = reusable
        if "memory_helpfulness" in memory:
            result["memory_helpfulness"] = memory["memory_helpfulness"]
    if guidance:
        result["reduce_memory_reliance"] = bool(guidance.get("reduce_memory_reliance", False))
    return result or None


def _selected_path_summary(paths: list[dict[str, object]]) -> dict[str, object] | None:
    if not paths:
        return None
    first = paths[0]
    return {
        key: first[key]
        for key in (
            "path_id",
            "proposed_action",
            "expected_outcome",
            "expected_free_energy",
            "total_cost",
            "posterior_weight",
        )
        if key in first
    } or None


def _prompt_budget_summary(prompt_budget: object) -> dict[str, object] | None:
    budget = _compact_mapping(prompt_budget, depth=1)
    if not budget:
        return None
    allowed = {
        "max_tokens",
        "used_tokens",
        "remaining_tokens",
        "tokens_remaining",
        "used_ratio",
        "usage_ratio",
        "budget_used_ratio",
        "prompt_usage_ratio",
        "omitted_signals",
        "included_signals",
    }
    return {key: value for key, value in budget.items() if key in allowed} or budget


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
    cognitive_state: object | None = None,
    self_prior_summary: Mapping[str, object] | str | None = None,
    cognitive_paths: Sequence[object] | None = None,
    path_summary: Mapping[str, object] | None = None,
    meta_control_guidance: Mapping[str, object] | None = None,
    cognitive_control_guidance: Mapping[str, object] | None = None,
    affective_state: object | None = None,
    affective_guidance: Mapping[str, object] | None = None,
    prompt_budget: Mapping[str, object] | None = None,
    included_signals: Sequence[str] | None = None,
    omitted_signals: Sequence[str] | None = None,
    persona_id: str | None = None,
    session_id: str | None = None,
) -> FEPPromptCapsule:
    obs = {str(k): float(v) for k, v in dict(observation or {}).items()}
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    chosen = getattr(diagnostics, "chosen", ranked[0] if ranked else None)
    if cognitive_paths is None:
        paths = cognitive_paths_from_diagnostics(diagnostics) if diagnostics is not None else []
        cognitive_path_dicts = [path.to_dict() for path in paths]
    else:
        paths = []
        cognitive_path_dicts = [
            dict(item.to_dict() if hasattr(item, "to_dict") else item)
            for item in cognitive_paths
            if isinstance(item, Mapping) or hasattr(item, "to_dict")
        ][:5]
    path_competition = (
        dict(path_summary)
        if isinstance(path_summary, Mapping)
        else path_competition_summary(paths)
    )
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
        cognitive_paths=cognitive_path_dicts,
        path_competition=path_competition,
        persona_id=_compact_text(persona_id, limit=80) if persona_id else None,
        session_id=_compact_text(session_id, limit=80) if session_id else None,
        self_prior_summary=_compact_mapping(self_prior_summary, depth=2),
        selected_path_summary=_selected_path_summary(cognitive_path_dicts),
        path_competition_summary=_compact_mapping(path_competition, depth=2),
        active_gaps=_active_gaps_from_state(cognitive_state),
        affective_state_summary=_affective_summary_from_state(cognitive_state, affective_state),
        meta_control_guidance=_compact_mapping(meta_control_guidance, depth=2),
        cognitive_control_guidance=_compact_mapping(cognitive_control_guidance, depth=2),
        affective_guidance=_compact_mapping(affective_guidance, depth=2),
        memory_use_guidance=_memory_guidance_from_state(cognitive_state, meta_control_guidance),
        omitted_signals=_compact_signal_names(omitted_signals),
        prompt_budget_summary=_prompt_budget_summary(prompt_budget),
    )
