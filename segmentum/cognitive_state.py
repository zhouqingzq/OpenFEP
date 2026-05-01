"""M6.3 derived cognitive state MVP.

`CognitiveStateMVP` is a compact read model derived from turn events,
`DecisionDiagnostics`, observation channels, previous outcome labels, and an
optional compressed `self_prior_summary`.

Conscious artifact boundary:
- `Self-consciousness.md` is a persona-scoped long-term self prior. It is
  updated only by slow consolidation gates: repeated evidence across turns or
  sessions, expected future usefulness, low conflict with the existing
  self-prior, bounded maintenance cost, and an explicit consolidation step.
- `Conscious.md` is a session-scoped current-context projection.

This module must not read or write either file inside the per-turn policy cycle.
Callers may pass a compressed, redacted `self_prior_summary`; raw artifact text,
raw event payloads, full diagnostics, ranked-option mutation, and policy score
updates are intentionally out of scope for M6.3.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from typing import Mapping, Sequence, TypeVar

from .cognitive_events import CognitiveEvent
from .types import DecisionDiagnostics


T = TypeVar("T")


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(lo, min(hi, numeric)), 6)


def _text(value: object, *, limit: int = 96) -> str:
    cleaned = " ".join(str(value or "").split())
    return cleaned[:limit]


def _strings(values: object, *, limit: int = 6, item_limit: int = 96) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        return []
    result: list[str] = []
    for item in values:
        text = _text(item, limit=item_limit)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _mapping(payload: object) -> dict[str, object]:
    return dict(payload) if isinstance(payload, Mapping) else {}


def _dataclass_from_dict(cls: type[T], payload: Mapping[str, object] | None) -> T:
    source = dict(payload or {})
    kwargs: dict[str, object] = {}
    for item in fields(cls):  # type: ignore[arg-type]
        value = source.get(item.name, item.default)
        if value is MISSING:
            factory = getattr(item, "default_factory", MISSING)
            value = factory() if factory is not MISSING else None
        if item.type in {float, "float"}:
            kwargs[item.name] = _clamp(value)
        elif item.type in {str, "str"}:
            kwargs[item.name] = _text(value, limit=256)
        elif str(item.type).startswith("list[str]"):
            kwargs[item.name] = _strings(value)
        elif str(item.type).startswith("list[dict"):
            kwargs[item.name] = [
                dict(entry)
                for entry in (value if isinstance(value, list) else [])
                if isinstance(entry, Mapping)
            ][:6]
        else:
            kwargs[item.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


@dataclass(frozen=True)
class TaskState:
    explicit_request: str
    inferred_need: str
    current_goal: str
    task_phase: str
    success_criteria: list[str]
    urgency: float


@dataclass(frozen=True)
class MemoryState:
    activated_memories: list[dict[str, object]]
    reusable_patterns: list[str]
    memory_conflicts: list[str]
    abstraction_candidates: list[str]
    memory_helpfulness: float


@dataclass(frozen=True)
class GapState:
    epistemic_gaps: list[str]
    contextual_gaps: list[str]
    instrumental_gaps: list[str]
    resource_gaps: list[str]
    social_gaps: list[str]
    blocking_gaps: list[str]


@dataclass(frozen=True)
class AffectiveState:
    mood_valence: float
    arousal: float
    social_safety: float
    irritation: float
    warmth: float
    fatigue_pressure: float
    repair_need: float
    decay_rate: float
    affective_notes: list[str]


@dataclass(frozen=True)
class MetaControlState:
    lambda_energy: float
    lambda_attention: float
    lambda_memory: float
    lambda_control: float
    beta_efe: float
    exploration_temperature: float
    control_gain: float
    memory_retrieval_gain: float
    abstraction_gain: float


@dataclass(frozen=True)
class CognitiveStateMVP:
    task: TaskState
    memory: MemoryState
    gaps: GapState
    affect: AffectiveState
    meta_control: MetaControlState

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CognitiveStateMVP":
        return cls(
            task=_dataclass_from_dict(TaskState, _mapping(payload.get("task"))),
            memory=_dataclass_from_dict(MemoryState, _mapping(payload.get("memory"))),
            gaps=_dataclass_from_dict(GapState, _mapping(payload.get("gaps"))),
            affect=_dataclass_from_dict(AffectiveState, _mapping(payload.get("affect"))),
            meta_control=_dataclass_from_dict(
                MetaControlState,
                _mapping(payload.get("meta_control")),
            ),
        )


def default_cognitive_state() -> CognitiveStateMVP:
    return CognitiveStateMVP(
        task=TaskState(
            explicit_request="",
            inferred_need="none",
            current_goal="",
            task_phase="observe",
            success_criteria=[],
            urgency=0.25,
        ),
        memory=MemoryState(
            activated_memories=[],
            reusable_patterns=[],
            memory_conflicts=[],
            abstraction_candidates=[],
            memory_helpfulness=0.0,
        ),
        gaps=GapState(
            epistemic_gaps=[],
            contextual_gaps=[],
            instrumental_gaps=[],
            resource_gaps=[],
            social_gaps=[],
            blocking_gaps=[],
        ),
        affect=AffectiveState(
            mood_valence=0.5,
            arousal=0.25,
            social_safety=0.75,
            irritation=0.0,
            warmth=0.5,
            fatigue_pressure=0.0,
            repair_need=0.0,
            decay_rate=0.18,
            affective_notes=[],
        ),
        meta_control=MetaControlState(
            lambda_energy=0.25,
            lambda_attention=0.35,
            lambda_memory=0.25,
            lambda_control=0.35,
            beta_efe=0.5,
            exploration_temperature=0.35,
            control_gain=0.35,
            memory_retrieval_gain=0.25,
            abstraction_gain=0.2,
        ),
    )


def _event_payloads(events: Sequence[CognitiveEvent]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for event in events:
        payload = getattr(event, "payload", {})
        if isinstance(payload, Mapping):
            payloads.append(dict(payload))
    return payloads


def _event_value(events: Sequence[CognitiveEvent], key: str) -> str:
    for payload in reversed(_event_payloads(events)):
        if key in payload:
            return _text(payload.get(key), limit=96)
    return ""


def _policy_margin(diagnostics: DecisionDiagnostics | None) -> float:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if len(ranked) < 2:
        return 1.0
    first = float(getattr(ranked[0], "policy_score", 0.0))
    second = float(getattr(ranked[1], "policy_score", 0.0))
    return _clamp(abs(first - second), 0.0, 1.0)


def _efe_margin(diagnostics: DecisionDiagnostics | None) -> float:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if len(ranked) < 2:
        return 1.0
    values = sorted(float(getattr(item, "expected_free_energy", 0.0)) for item in ranked[:2])
    return _clamp(abs(values[1] - values[0]), 0.0, 1.0)


def _outcome_negative(outcome: str) -> bool:
    lowered = outcome.lower()
    return any(token in lowered for token in ("fail", "negative", "rupture", "worse", "miss"))


def _outcome_positive(outcome: str) -> bool:
    lowered = outcome.lower()
    return any(token in lowered for token in ("success", "positive", "repair", "resolved", "good"))


def _body_pressure(observation: Mapping[str, float]) -> float:
    keys = ("fatigue", "fatigue_pressure", "stress", "danger", "threat", "urgency")
    return _clamp(max(float(observation.get(key, 0.0) or 0.0) for key in keys))


def _activated_memories(diagnostics: DecisionDiagnostics | None) -> list[dict[str, object]]:
    memories = list(getattr(diagnostics, "retrieved_memories", []) or [])
    ids = [str(item) for item in getattr(diagnostics, "retrieved_episode_ids", []) or []]
    activated: list[dict[str, object]] = []
    for index, memory in enumerate(memories[:4]):
        item = dict(memory) if isinstance(memory, Mapping) else {"summary": _text(memory)}
        compact = {
            "episode_id": _text(item.get("episode_id") or (ids[index] if index < len(ids) else "")),
            "summary": _text(
                item.get("summary")
                or item.get("content")
                or item.get("description")
                or item.get("action_taken")
            ),
        }
        compact = {key: value for key, value in compact.items() if value}
        if compact:
            activated.append(compact)
    if not activated:
        activated = [{"episode_id": item} for item in ids[:4] if item]
    return activated


def _self_prior_notes(self_prior_summary: Mapping[str, object] | str | None) -> list[str]:
    if self_prior_summary is None:
        return []
    if isinstance(self_prior_summary, str):
        return [_text(self_prior_summary)]
    notes: list[str] = []
    for key in ("summary", "stable_patterns", "reusable_patterns", "current_prior"):
        value = self_prior_summary.get(key)
        if isinstance(value, str):
            notes.append(_text(value))
        else:
            notes.extend(_strings(value, limit=3))
    return [note for note in notes if note][:4]


def _derive_memory(
    diagnostics: DecisionDiagnostics | None,
    self_prior_summary: Mapping[str, object] | str | None,
) -> MemoryState:
    activated = _activated_memories(diagnostics)
    summary = _text(getattr(diagnostics, "memory_context_summary", ""), limit=160)
    reusable = []
    if summary:
        reusable.append(summary)
    reusable.extend(_self_prior_notes(self_prior_summary))
    prediction_delta = dict(getattr(diagnostics, "prediction_delta", {}) or {})
    conflicts = [
        f"memory prediction conflict: {key}"
        for key, value in sorted(prediction_delta.items())
        if abs(float(value)) >= 0.35
    ][:4]
    abstraction = [
        str(key)
        for key, value in sorted(prediction_delta.items())
        if abs(float(value)) >= 0.2
    ][:4]
    memory_hit = bool(getattr(diagnostics, "memory_hit", False))
    helpfulness = 0.0
    if memory_hit:
        helpfulness += 0.35
    if activated:
        helpfulness += min(0.35, len(activated) * 0.08)
    if summary:
        helpfulness += 0.15
    if reusable:
        helpfulness += 0.10
    if conflicts:
        helpfulness -= 0.20
    return MemoryState(
        activated_memories=activated,
        reusable_patterns=_strings(reusable, limit=6, item_limit=128),
        memory_conflicts=conflicts,
        abstraction_candidates=abstraction,
        memory_helpfulness=_clamp(helpfulness),
    )


def _derive_gaps(
    *,
    diagnostics: DecisionDiagnostics | None,
    observation: Mapping[str, float],
    previous_outcome: str,
) -> GapState:
    policy_margin = _policy_margin(diagnostics)
    efe_margin = _efe_margin(diagnostics)
    prediction_error = _clamp(getattr(diagnostics, "prediction_error", 0.0))
    conflict = _clamp(observation.get("conflict_tension", 0.0))
    hidden_intent = _clamp(observation.get("hidden_intent", 0.5))
    missing_context = _clamp(observation.get("missing_context", 0.0))
    contextual_uncertainty = _clamp(observation.get("contextual_uncertainty", 0.0))
    social_depth = _clamp(observation.get("relationship_depth", 0.0))

    epistemic: list[str] = []
    contextual: list[str] = []
    instrumental: list[str] = []
    resource: list[str] = []
    social: list[str] = []
    blocking: list[str] = []

    if policy_margin < 0.12 or efe_margin < 0.12:
        epistemic.append("low decision margin between candidate actions")
    if prediction_error >= 0.55:
        epistemic.append("high prediction error needs verification")
    if hidden_intent >= 0.72:
        contextual.append("user intent signal is ambiguous")
    if missing_context >= 0.5 or contextual_uncertainty >= 0.5:
        contextual.append("missing context for confident response")
    if conflict >= 0.6:
        social.append("high interpersonal conflict tension")
    if hidden_intent >= 0.78 and social_depth <= 0.25:
        social.append("low relational context for hidden-intent signal")
    if _body_pressure(observation) >= 0.65:
        resource.append("body/resource pressure is elevated")
    if _outcome_negative(previous_outcome):
        instrumental.append("previous outcome indicates failed or incomplete strategy")
        blocking.append("prior failure should be repaired before escalation")
    if bool(getattr(diagnostics, "repair_triggered", False)):
        blocking.append("identity or commitment repair is active")

    return GapState(
        epistemic_gaps=epistemic[:5],
        contextual_gaps=contextual[:5],
        instrumental_gaps=instrumental[:5],
        resource_gaps=resource[:5],
        social_gaps=social[:5],
        blocking_gaps=blocking[:5],
    )


def _derive_task(
    *,
    events: Sequence[CognitiveEvent],
    diagnostics: DecisionDiagnostics | None,
    observation: Mapping[str, float],
    gaps: GapState,
    previous_outcome: str,
) -> TaskState:
    chosen = getattr(getattr(diagnostics, "chosen", None), "choice", "")
    active_goal = _text(getattr(diagnostics, "active_goal", ""), limit=96)
    explicit_request = _event_value(events, "current_turn") or _event_value(events, "selected_action")
    if not explicit_request:
        explicit_request = _text(chosen or active_goal)
    if _outcome_negative(previous_outcome) or gaps.blocking_gaps:
        phase = "repair"
    elif gaps.epistemic_gaps or gaps.contextual_gaps:
        phase = "clarify"
    elif chosen:
        phase = "act"
    else:
        phase = "observe"
    inferred_need = "repair" if phase == "repair" else "reduce uncertainty" if phase == "clarify" else "respond"
    criteria = [
        "address selected action",
        "reduce blocking gaps" if gaps.blocking_gaps else "",
        "resolve uncertainty" if gaps.epistemic_gaps or gaps.contextual_gaps else "",
        "maintain social safety" if gaps.social_gaps else "",
    ]
    urgency = max(
        _clamp(observation.get("urgency", 0.0)),
        _clamp(getattr(diagnostics, "prediction_error", 0.0)) * 0.7,
        0.65 if gaps.blocking_gaps else 0.0,
        0.55 if gaps.social_gaps else 0.0,
    )
    return TaskState(
        explicit_request=explicit_request,
        inferred_need=inferred_need,
        current_goal=active_goal or _text(chosen),
        task_phase=phase,
        success_criteria=[item for item in criteria if item][:5],
        urgency=_clamp(urgency),
    )


def _decay(previous_value: float, target: float, rate: float) -> float:
    return _clamp(previous_value + ((target - previous_value) * rate))


def _derive_affect(
    *,
    previous: CognitiveStateMVP | None,
    observation: Mapping[str, float],
    gaps: GapState,
    previous_outcome: str,
) -> AffectiveState:
    prior = previous.affect if previous is not None else default_cognitive_state().affect
    decay_rate = 0.18
    emotional_tone = _clamp(observation.get("emotional_tone", 0.5))
    conflict = _clamp(observation.get("conflict_tension", 0.0))
    pressure = _body_pressure(observation)
    positive = _outcome_positive(previous_outcome)
    negative = _outcome_negative(previous_outcome)

    target_valence = _clamp(emotional_tone + (0.08 if positive else 0.0) - (0.12 if negative else 0.0))
    target_arousal = _clamp((conflict * 0.55) + (pressure * 0.35) + (0.20 if negative else 0.0))
    target_social_safety = _clamp(0.82 - (conflict * 0.45) - (0.18 if gaps.social_gaps else 0.0) + (0.10 if positive else 0.0))
    target_irritation = _clamp((conflict * 0.45) + (pressure * 0.25) + (0.12 if negative else 0.0))
    target_warmth = _clamp(0.45 + ((emotional_tone - 0.5) * 0.7) - (conflict * 0.25) + (0.16 if positive else 0.0))
    target_repair = _clamp((conflict * 0.45) + (0.25 if gaps.social_gaps else 0.0) + (0.25 if negative else 0.0))

    notes = []
    if conflict >= 0.6:
        notes.append("conflict tension elevated")
    if pressure >= 0.65:
        notes.append("body pressure elevated")
    if positive:
        notes.append("prior outcome supports recovery")
    if negative:
        notes.append("prior outcome increases repair pressure")

    return AffectiveState(
        mood_valence=_decay(prior.mood_valence, target_valence, decay_rate),
        arousal=_decay(prior.arousal, target_arousal, decay_rate),
        social_safety=_decay(prior.social_safety, target_social_safety, decay_rate),
        irritation=_decay(prior.irritation, target_irritation, decay_rate),
        warmth=_decay(prior.warmth, target_warmth, decay_rate),
        fatigue_pressure=_decay(prior.fatigue_pressure, pressure, decay_rate),
        repair_need=_decay(prior.repair_need, target_repair, decay_rate),
        decay_rate=decay_rate,
        affective_notes=notes[:5],
    )


def _derive_meta_control(
    *,
    diagnostics: DecisionDiagnostics | None,
    memory: MemoryState,
    gaps: GapState,
    affect: AffectiveState,
    previous_outcome: str,
) -> MetaControlState:
    prediction_error = _clamp(getattr(diagnostics, "prediction_error", 0.0))
    low_margin_pressure = 1.0 - min(_policy_margin(diagnostics), _efe_margin(diagnostics))
    gap_pressure = _clamp(
        (
            len(gaps.epistemic_gaps)
            + len(gaps.contextual_gaps)
            + len(gaps.instrumental_gaps)
            + len(gaps.social_gaps)
            + len(gaps.blocking_gaps)
        )
        / 8.0
    )
    negative = _outcome_negative(previous_outcome)
    return MetaControlState(
        lambda_energy=_clamp(0.20 + affect.fatigue_pressure * 0.55),
        lambda_attention=_clamp(0.30 + prediction_error * 0.30 + gap_pressure * 0.25),
        lambda_memory=_clamp(0.20 + memory.memory_helpfulness * 0.35 + gap_pressure * 0.20),
        lambda_control=_clamp(0.30 + affect.repair_need * 0.25 + (0.15 if negative else 0.0)),
        beta_efe=_clamp(0.45 + low_margin_pressure * 0.25 + (0.10 if negative else 0.0)),
        exploration_temperature=_clamp(0.25 + gap_pressure * 0.30 + prediction_error * 0.20),
        control_gain=_clamp(0.30 + affect.repair_need * 0.35 + (0.12 if gaps.blocking_gaps else 0.0)),
        memory_retrieval_gain=_clamp(0.20 + memory.memory_helpfulness * 0.45 + len(gaps.epistemic_gaps) * 0.06),
        abstraction_gain=_clamp(0.18 + len(memory.abstraction_candidates) * 0.08 + gap_pressure * 0.22),
    )


def update_cognitive_state(
    previous: CognitiveStateMVP | None,
    *,
    events: Sequence[CognitiveEvent],
    diagnostics: DecisionDiagnostics | None,
    observation: Mapping[str, float],
    previous_outcome: str = "",
    self_prior_summary: Mapping[str, object] | str | None = None,
) -> CognitiveStateMVP:
    """Derive a compact state snapshot without mutating diagnostics or policy."""
    safe_observation = {
        str(key): float(value)
        for key, value in dict(observation or {}).items()
        if isinstance(value, (int, float))
    }
    memory = _derive_memory(diagnostics, self_prior_summary)
    gaps = _derive_gaps(
        diagnostics=diagnostics,
        observation=safe_observation,
        previous_outcome=previous_outcome,
    )
    task = _derive_task(
        events=events,
        diagnostics=diagnostics,
        observation=safe_observation,
        gaps=gaps,
        previous_outcome=previous_outcome,
    )
    affect = _derive_affect(
        previous=previous,
        observation=safe_observation,
        gaps=gaps,
        previous_outcome=previous_outcome,
    )
    meta_control = _derive_meta_control(
        diagnostics=diagnostics,
        memory=memory,
        gaps=gaps,
        affect=affect,
        previous_outcome=previous_outcome,
    )
    state = CognitiveStateMVP(
        task=task,
        memory=memory,
        gaps=gaps,
        affect=affect,
        meta_control=meta_control,
    )
    if is_dataclass(state):
        return state
    return default_cognitive_state()
