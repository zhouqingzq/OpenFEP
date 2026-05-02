"""Deterministic meta-control guidance derived from compact cognitive state.

The guidance produced here is a prompt-conditioning and audit signal only. It
must not mutate ranked options, policy scores, selected actions, memory stores,
or conscious artifacts.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from typing import Mapping, Sequence, TypeVar

from .cognitive_state import CognitiveStateMVP
from .types import DecisionDiagnostics


T = TypeVar("T")

_NEGATIVE_OUTCOME_TOKENS = (
    "fail",
    "failed",
    "failure",
    "negative",
    "rupture",
    "worse",
    "miss",
    "regress",
    "error",
)
_REPEATED_OUTCOME_TOKENS = ("repeat", "repeated", "again", "streak")
_ACCUSATORY_TERMS = (
    "accuse",
    "blame",
    "deceive",
    "deception",
    "manipulate",
    "manipulation",
    "paranoia",
    "paranoid",
    "suspicion",
    "suspicious",
)


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(lo, min(hi, numeric)), 6)


def _float_attr(obj: object, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name, default))
    except (TypeError, ValueError):
        return default


def _strings(values: object, *, limit: int = 12) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        return []
    result: list[str] = []
    for item in values:
        text = " ".join(str(item or "").split())[:160]
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dataclass_from_dict(cls: type[T], payload: Mapping[str, object] | None) -> T:
    source = dict(payload or {})
    kwargs: dict[str, object] = {}
    for item in fields(cls):  # type: ignore[arg-type]
        value = source.get(item.name, item.default)
        if value is MISSING:
            factory = getattr(item, "default_factory", MISSING)
            value = factory() if factory is not MISSING else None
        if item.type in {bool, "bool"}:
            kwargs[item.name] = _as_bool(value)
        elif item.type in {float, "float"}:
            kwargs[item.name] = _clamp(value)
        elif str(item.type).startswith("list[str]"):
            kwargs[item.name] = _strings(value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _policy_margin(diagnostics: DecisionDiagnostics | None) -> float:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if len(ranked) < 2:
        return 1.0
    return abs(
        _float_attr(ranked[0], "policy_score", 0.0)
        - _float_attr(ranked[1], "policy_score", 0.0)
    )


def _efe_margin(diagnostics: DecisionDiagnostics | None) -> float:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if len(ranked) < 2:
        return 1.0
    return abs(
        _float_attr(ranked[0], "expected_free_energy", 0.0)
        - _float_attr(ranked[1], "expected_free_energy", 0.0)
    )


def _path_margin(path_summary: Mapping[str, object], key: str, default: float) -> float:
    try:
        return float(path_summary.get(key, default))
    except (TypeError, ValueError):
        return default


def _outcome_negative(previous_outcome: str) -> bool:
    lowered = str(previous_outcome or "").lower()
    return any(token in lowered for token in _NEGATIVE_OUTCOME_TOKENS)


def _outcome_repeated(previous_outcome: str) -> bool:
    lowered = str(previous_outcome or "").lower()
    return any(token in lowered for token in _REPEATED_OUTCOME_TOKENS)


def _prompt_overloaded(prompt_budget: Mapping[str, object] | None) -> bool:
    budget = _mapping(prompt_budget)
    if not budget:
        return False
    used_ratio = _clamp(
        budget.get("used_ratio")
        or budget.get("usage_ratio")
        or budget.get("budget_used_ratio")
        or budget.get("prompt_usage_ratio")
    )
    omitted_count = 0
    omitted = budget.get("omitted_signals") or budget.get("omitted")
    if isinstance(omitted, Sequence) and not isinstance(omitted, (str, bytes, bytearray)):
        omitted_count = len(omitted)
    try:
        remaining_tokens = float(
            budget.get("remaining_tokens") or budget.get("tokens_remaining") or 999999
        )
    except (TypeError, ValueError):
        remaining_tokens = 999999
    return used_ratio >= 0.85 or omitted_count >= 3 or remaining_tokens <= 256


def _safe_notes(notes: list[str]) -> list[str]:
    safe: list[str] = []
    for note in notes:
        lowered = note.lower()
        if any(term in lowered for term in _ACCUSATORY_TERMS):
            continue
        if note not in safe:
            safe.append(note)
    return safe[:12]


@dataclass(frozen=True)
class MetaControlGuidance:
    increase_caution: bool
    ask_clarifying_question: bool
    lower_assertiveness: bool
    compress_context: bool
    reduce_memory_reliance: bool
    increase_control_gain: bool
    increase_exploration_temperature: bool
    prefer_repair_strategy: bool
    avoid_overinterpreting_hidden_intent: bool
    deescalate_affect: bool
    preserve_warmth: bool
    reduce_intensity: bool
    guidance_notes: list[str]
    trigger_reasons: list[str]
    intensity: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MetaControlGuidance":
        return _dataclass_from_dict(cls, payload)


def generate_meta_control_guidance(
    state: CognitiveStateMVP,
    *,
    diagnostics: DecisionDiagnostics | None = None,
    path_summary: Mapping[str, object] | None = None,
    previous_outcome: str = "",
    prompt_budget: Mapping[str, object] | None = None,
) -> MetaControlGuidance:
    """Produce deterministic guidance without changing decision diagnostics."""
    summary = _mapping(path_summary)
    policy_margin = min(
        _policy_margin(diagnostics),
        abs(_path_margin(summary, "policy_margin", 1.0)),
    )
    efe_margin = min(
        _efe_margin(diagnostics),
        abs(_path_margin(summary, "efe_margin", 1.0)),
    )
    total_cost_margin = _path_margin(summary, "total_cost_margin", 1.0)
    posterior_margin = abs(_path_margin(summary, "posterior_margin", 1.0))
    low_margin = (
        policy_margin < 0.12
        or efe_margin < 0.05
        or total_cost_margin < 0.08
        or posterior_margin < 0.05
    )

    affect = state.affect
    gaps = state.gaps
    memory = state.memory
    meta_control = state.meta_control

    high_conflict = any("conflict" in item.lower() for item in gaps.social_gaps)
    low_social_safety = affect.social_safety <= 0.58
    high_repair_need = affect.repair_need >= 0.22 or bool(gaps.blocking_gaps)
    elevated_irritation = affect.irritation >= 0.14
    elevated_arousal = affect.arousal >= 0.25
    memory_conflict = bool(memory.memory_conflicts)
    prior_failure = _outcome_negative(previous_outcome)
    repeated_failure = prior_failure and _outcome_repeated(previous_outcome)
    prompt_overload = _prompt_overloaded(prompt_budget)
    hidden_low_observable = (
        any("hidden-intent" in item.lower() or "hidden intent" in item.lower() for item in gaps.social_gaps)
        or any("intent signal is ambiguous" in item.lower() for item in gaps.contextual_gaps)
    )
    identity_tension = (
        bool(getattr(diagnostics, "repair_triggered", False))
        or _float_attr(diagnostics, "identity_tension", 0.0) >= 0.35
        or _float_attr(diagnostics, "self_inconsistency_error", 0.0) >= 0.35
        or bool(getattr(diagnostics, "violated_commitments", []) or [])
    )

    notes: list[str] = []
    reasons: list[str] = []

    increase_caution = False
    ask_clarifying_question = False
    lower_assertiveness = False
    compress_context = False
    reduce_memory_reliance = False
    increase_control_gain = False
    increase_exploration_temperature = False
    prefer_repair_strategy = False
    avoid_overinterpreting_hidden_intent = False
    deescalate_affect = False
    preserve_warmth = False
    reduce_intensity = False
    pressure = 0.0

    if low_margin:
        increase_caution = True
        ask_clarifying_question = True
        lower_assertiveness = True
        increase_exploration_temperature = True
        pressure = max(pressure, 0.35)
        reasons.append("low decision margin")
        notes.append("Use provisional wording and ask for the missing constraint before committing.")

    if high_conflict:
        increase_control_gain = True
        prefer_repair_strategy = True
        deescalate_affect = True
        preserve_warmth = True
        lower_assertiveness = True
        pressure = max(pressure, 0.55)
        reasons.append("high conflict tension")
        notes.append("Prioritize repair, slow pacing, and concrete next steps.")

    if low_social_safety or high_repair_need:
        deescalate_affect = True
        preserve_warmth = True
        prefer_repair_strategy = prefer_repair_strategy or high_repair_need
        pressure = max(pressure, 0.45)
        reasons.append("affective maintenance pressure")
        notes.append("Maintain warmth without making claims about the user's inner state.")

    if elevated_irritation or elevated_arousal:
        reduce_intensity = True
        lower_assertiveness = True
        deescalate_affect = True
        pressure = max(pressure, 0.42)
        reasons.append("elevated arousal or irritation")
        notes.append("Keep the response calm, brief, and non-accusatory.")

    if prior_failure:
        increase_caution = True
        prefer_repair_strategy = True
        lower_assertiveness = True
        pressure = max(pressure, 0.50 if repeated_failure else 0.38)
        reasons.append("previous negative outcome")
        notes.append("Repair the prior miss before escalating the strategy.")

    if memory_conflict:
        reduce_memory_reliance = True
        increase_caution = True
        pressure = max(pressure, 0.40)
        reasons.append("memory conflict")
        notes.append("Treat memory as tentative when it conflicts with current evidence.")

    if prompt_overload:
        compress_context = True
        increase_caution = True
        pressure = max(pressure, 0.32)
        reasons.append("prompt overload")
        notes.append("Compress context to the few signals needed for the next response.")

    if hidden_low_observable:
        avoid_overinterpreting_hidden_intent = True
        lower_assertiveness = True
        pressure = max(pressure, 0.34)
        reasons.append("hidden intent has low observability")
        notes.append("Name observable ambiguity only; do not infer hidden motives.")

    if identity_tension:
        increase_control_gain = True
        prefer_repair_strategy = True
        increase_caution = True
        pressure = max(pressure, 0.50)
        reasons.append("identity or commitment tension")
        notes.append("Favor commitment repair over broad reinterpretation.")

    if meta_control.control_gain >= 0.55:
        increase_control_gain = True
    if meta_control.exploration_temperature >= 0.45 and low_margin:
        increase_exploration_temperature = True

    active_count = sum(
        1
        for value in (
            increase_caution,
            ask_clarifying_question,
            lower_assertiveness,
            compress_context,
            reduce_memory_reliance,
            increase_control_gain,
            increase_exploration_temperature,
            prefer_repair_strategy,
            avoid_overinterpreting_hidden_intent,
            deescalate_affect,
            preserve_warmth,
            reduce_intensity,
        )
        if value
    )
    intensity = _clamp(max(pressure, min(0.75, active_count / 14.0)))
    return MetaControlGuidance(
        increase_caution=increase_caution,
        ask_clarifying_question=ask_clarifying_question,
        lower_assertiveness=lower_assertiveness,
        compress_context=compress_context,
        reduce_memory_reliance=reduce_memory_reliance,
        increase_control_gain=increase_control_gain,
        increase_exploration_temperature=increase_exploration_temperature,
        prefer_repair_strategy=prefer_repair_strategy,
        avoid_overinterpreting_hidden_intent=avoid_overinterpreting_hidden_intent,
        deescalate_affect=deescalate_affect,
        preserve_warmth=preserve_warmth,
        reduce_intensity=reduce_intensity,
        guidance_notes=_safe_notes(notes),
        trigger_reasons=_safe_notes(reasons),
        intensity=intensity,
    )


def summarize_affective_maintenance(
    guidance: MetaControlGuidance | Mapping[str, object] | None,
) -> dict[str, object]:
    """Return dashboard/trace-safe affective maintenance hints in Chinese."""
    if guidance is None:
        return {}
    if isinstance(guidance, Mapping):
        guidance = MetaControlGuidance.from_dict(guidance)
    actions: list[str] = []
    if guidance.deescalate_affect:
        actions.append("降温")
    if guidance.preserve_warmth:
        actions.append("保留温度")
    if guidance.reduce_intensity:
        actions.append("降低表达强度")
    if guidance.lower_assertiveness:
        actions.append("降低断言")
    return {
        "actions": actions,
        "intensity": guidance.intensity,
        "summary": "；".join(actions) if actions else "维持默认表达强度",
        "source": "meta_control_guidance",
    }
