"""Compact turn-level dialogue traces and conscious session artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..cognitive_paths import (
    cognitive_paths_from_diagnostics,
    path_competition_summary,
)
from ..meta_control_guidance import summarize_affective_maintenance
from ..tracing import JsonlTraceWriter


_SENSITIVE_KEY_FRAGMENTS = (
    "api",
    "authorization",
    "body",
    "conversation_history",
    "content",
    "history",
    "key",
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

_FEP_CAPSULE_ALLOWED_KEYS = {
    "chosen_action",
    "chosen_predicted_outcome",
    "chosen_risk",
    "chosen_risk_label",
    "chosen_expected_free_energy",
    "chosen_policy_score",
    "chosen_dominant_component",
    "cognitive_paths",
    "top_alternatives",
    "path_competition",
    "policy_margin",
    "efe_margin",
    "decision_uncertainty",
    "prediction_error",
    "prediction_error_label",
    "workspace_focus",
    "workspace_suppressed",
    "previous_outcome",
    "hidden_intent_score",
    "hidden_intent_label",
    "persona_id",
    "session_id",
    "self_prior_summary",
    "selected_path_summary",
    "path_competition_summary",
    "active_gaps",
    "affective_state_summary",
    "meta_control_guidance",
    "affective_guidance",
    "memory_use_guidance",
    "omitted_signals",
    "prompt_budget_summary",
    "observation_channels",
}

_GENERATION_ALLOWED_KEYS = {
    "calibration_policy_source",
    "conditional_policy_frequency",
    "conditional_policy_strategy_frequency",
    "conditional_policy_support",
    "conditional_policy_top_strategy",
    "diagnostic_keys",
    "llm_error",
    "llm_latency_ms",
    "llm_model",
    "llm_tokens_completion",
    "llm_tokens_prompt",
    "llm_tokens_total",
    "meta_control_guidance",
    "partner_anchor_used",
    "policy_action_selection_lift_applied",
    "policy_context_bucket",
    "policy_evidence_weight",
    "policy_lift_applied",
    "policy_strategy_confidence",
    "prompt_capsule_guidance",
    "profile_anchor_match",
    "profile_confidence",
    "profile_degraded_reason",
    "profile_expression_source",
    "profile_expression_sources",
    "profile_length_bucket",
    "profile_opening_used",
    "profile_phrase_used",
    "rhetorical_move",
    "selected_action",
    "surface_shortcut_suppressed",
    "surface_source",
    "template_id",
    "topic_anchor_source",
    "topic_anchor_used",
    "affective_maintenance_summary",
}


def _round_float(value: object, default: float = 0.0) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return _json_safe(converted)
    return str(value)


def _redact_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in payload.items():
        key_text = str(key)
        lower_key = key_text.lower()
        if any(fragment in lower_key for fragment in _SENSITIVE_KEY_FRAGMENTS):
            redacted[key_text] = "[redacted]"
        elif isinstance(value, Mapping):
            redacted[key_text] = _redact_mapping(value)
        elif isinstance(value, (list, tuple)):
            redacted[key_text] = [
                _redact_mapping(item) if isinstance(item, Mapping) else _json_safe(item)
                for item in value[:8]
            ]
        else:
            redacted[key_text] = _json_safe(value)
    return redacted


def redacted_fep_prompt_capsule(capsule: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(capsule, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key in sorted(_FEP_CAPSULE_ALLOWED_KEYS):
        if key not in capsule:
            continue
        value = capsule[key]
        if isinstance(value, Mapping):
            if key in {
                "path_competition",
                "path_competition_summary",
                "self_prior_summary",
                "selected_path_summary",
                "active_gaps",
                "affective_state_summary",
                "meta_control_guidance",
                "affective_guidance",
                "memory_use_guidance",
                "prompt_budget_summary",
            }:
                safe[key] = _redact_mapping(value)
            else:
                safe[key] = {
                    str(k): _round_float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in value.items()
                }
        elif isinstance(value, list):
            safe[key] = [
                _redact_mapping(item) if isinstance(item, Mapping) else _json_safe(item)
                for item in value[:5]
            ]
        else:
            safe[key] = _round_float(value) if isinstance(value, (int, float)) else str(value)
    return safe


def summarize_generation_diagnostics(
    diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(diagnostics, Mapping):
        return {}
    summary: dict[str, object] = {
        "diagnostic_keys": sorted(str(key) for key in diagnostics.keys())
    }
    for key in sorted(str(item) for item in diagnostics.keys()):
        lower_key = key.lower()
        if any(fragment in lower_key for fragment in _SENSITIVE_KEY_FRAGMENTS):
            summary[key] = "[redacted]"
    for key in sorted(_GENERATION_ALLOWED_KEYS):
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        if isinstance(value, Mapping):
            summary[key] = _redact_mapping(value)
        elif isinstance(value, list):
            summary[key] = [_json_safe(item) for item in value[:8]]
        elif isinstance(value, float):
            summary[key] = _round_float(value)
        else:
            summary[key] = _json_safe(value)
    llm_generation = diagnostics.get("llm_generation")
    if isinstance(llm_generation, Mapping):
        for key in sorted(_GENERATION_ALLOWED_KEYS):
            if key in llm_generation:
                summary[key] = _json_safe(llm_generation[key])
    return summary


def _option_summary(option: object) -> dict[str, object]:
    return {
        "action": str(getattr(option, "choice", "")),
        "policy_score": _round_float(getattr(option, "policy_score", 0.0)),
        "expected_free_energy": _round_float(
            getattr(option, "expected_free_energy", 0.0)
        ),
        "risk": _round_float(getattr(option, "risk", 0.0)),
        "dominant_component": str(getattr(option, "dominant_component", "")),
        "predicted_outcome": str(getattr(option, "predicted_outcome", "")),
    }


def summarize_ranked_options(diagnostics: object, *, limit: int = 5) -> list[dict[str, object]]:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    return [_option_summary(option) for option in ranked[:limit]]


def summarize_affective_state(
    observation_channels: Mapping[str, float],
    diagnostics: object,
) -> dict[str, object]:
    emotional_tone = _round_float(observation_channels.get("emotional_tone", 0.5), 0.5)
    conflict_tension = _round_float(observation_channels.get("conflict_tension", 0.0))
    hidden_intent = _round_float(observation_channels.get("hidden_intent", 0.5), 0.5)
    relationship_depth = _round_float(observation_channels.get("relationship_depth", 0.0))
    prediction_error = _round_float(getattr(diagnostics, "prediction_error", 0.0))
    if emotional_tone >= 0.6:
        tone_label = "positive"
    elif emotional_tone <= 0.4:
        tone_label = "strained"
    else:
        tone_label = "neutral"
    if conflict_tension >= 0.65:
        conflict_label = "high"
    elif conflict_tension >= 0.35:
        conflict_label = "moderate"
    else:
        conflict_label = "low"
    return {
        "emotional_tone": emotional_tone,
        "tone_label": tone_label,
        "conflict_tension": conflict_tension,
        "conflict_label": conflict_label,
        "hidden_intent": hidden_intent,
        "relationship_depth": relationship_depth,
        "prediction_error": prediction_error,
    }


def summarize_retrieved_memory(diagnostics: object) -> dict[str, object]:
    retrieved_ids = [
        str(item) for item in getattr(diagnostics, "retrieved_episode_ids", []) or []
    ]
    retrieved_memories = list(getattr(diagnostics, "retrieved_memories", []) or [])
    return {
        "memory_hit": bool(getattr(diagnostics, "memory_hit", False)),
        "retrieved_episode_count": len(retrieved_memories),
        "retrieved_episode_ids": retrieved_ids[:5],
        "memory_context_summary_present": bool(
            str(getattr(diagnostics, "memory_context_summary", "") or "").strip()
        ),
    }


def _summarize_event(event: object) -> dict[str, object]:
    payload = getattr(event, "payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    payload_keys = sorted(str(key) for key in payload.keys())
    summary: dict[str, object] = {
        "event_type": str(getattr(event, "event_type", "")),
        "source": str(getattr(event, "source", "")),
        "salience": _round_float(getattr(event, "salience", 0.0)),
        "priority": _round_float(getattr(event, "priority", 0.0)),
        "payload_keys": payload_keys,
    }
    for key in (
        "channel_count",
        "memory_hit",
        "outcome",
        "ranked_option_count",
        "selected_action",
        "template_id",
        "turn_index",
    ):
        if key in payload:
            summary[key] = _json_safe(payload[key])
    return summary


def split_event_summaries(
    events: Iterable[object],
    *,
    limit: int = 8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected: list[dict[str, object]] = []
    suppressed: list[dict[str, object]] = []
    for event in events:
        target = (
            selected
            if _round_float(getattr(event, "salience", 0.0)) >= 0.5
            or _round_float(getattr(event, "priority", 0.0)) >= 0.7
            else suppressed
        )
        if len(target) < limit:
            target.append(_summarize_event(event))
    return selected, suppressed


def _safe_id(value: object, *, fallback: str) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


@dataclass
class TurnTrace:
    persona_id: str
    session_id: str
    turn_id: str
    turn_index: int
    cycle: int
    observation_channels: dict[str, float]
    observation_source: str
    selected_events: list[dict[str, object]]
    suppressed_events: list[dict[str, object]]
    attention_selected_channels: list[str]
    attention_dropped_channels: list[str]
    workspace_focus: list[str]
    workspace_suppressed: list[str]
    affective_state_summary: dict[str, object]
    cognitive_state: dict[str, object]
    retrieved_memory_summary: dict[str, object]
    ranked_options: list[dict[str, object]]
    cognitive_paths: list[dict[str, object]]
    path_competition: dict[str, object]
    meta_control_guidance: dict[str, object]
    affective_maintenance_summary: dict[str, object]
    chosen_action: str
    policy_margin: float
    efe_margin: float
    fep_prompt_capsule: dict[str, object]
    generation_diagnostics: dict[str, object]
    outcome_label: str
    memory_update_signal: dict[str, object]
    debug_fields: dict[str, object] = field(default_factory=dict, repr=False)

    def to_dict(self, *, debug: bool = False) -> dict[str, object]:
        payload = asdict(self)
        debug_payload = payload.pop("debug_fields", {})
        if debug and debug_payload:
            payload["debug"] = _redact_mapping(debug_payload)
        return _json_safe(payload)  # type: ignore[return-value]

    def write_jsonl(self, writer: JsonlTraceWriter, *, debug: bool = False) -> None:
        writer.append(self.to_dict(debug=debug))

    @classmethod
    def from_runtime(
        cls,
        *,
        persona_id: str,
        session_id: str,
        turn_id: str,
        turn_index: int,
        cycle: int,
        observation_channels: Mapping[str, float],
        diagnostics: object,
        fep_prompt_capsule: Mapping[str, object] | None,
        generation_diagnostics: Mapping[str, object] | None,
        outcome_label: str,
        memory_update_signal: Mapping[str, object],
        meta_control_guidance: Mapping[str, object] | None = None,
        cognitive_state: object | None = None,
        events: Sequence[object] = (),
        debug: bool = False,
    ) -> "TurnTrace":
        selected_events, suppressed_events = split_event_summaries(events)
        safe_capsule = redacted_fep_prompt_capsule(fep_prompt_capsule)
        ranked_options = summarize_ranked_options(diagnostics)
        paths = (
            cognitive_paths_from_diagnostics(diagnostics)
            if diagnostics is not None
            else []
        )
        cognitive_paths = [path.to_dict() for path in paths]
        path_competition = path_competition_summary(paths)
        safe_guidance = _redact_mapping(dict(meta_control_guidance or {}))
        affective_maintenance_summary = summarize_affective_maintenance(
            safe_guidance
        )
        policy_margin = _round_float(safe_capsule.get("policy_margin", 1.0), 1.0)
        efe_margin = _round_float(safe_capsule.get("efe_margin", 1.0), 1.0)
        chosen_action = str(
            safe_capsule.get("chosen_action")
            or getattr(getattr(diagnostics, "chosen", None), "choice", "")
            or "ask_question"
        )
        debug_fields: dict[str, object] = {}
        if debug:
            debug_fields = {
                "diagnostic_keys": sorted(vars(diagnostics).keys())
                if hasattr(diagnostics, "__dict__")
                else [],
                "generation_diagnostic_keys": sorted(
                    str(key) for key in (generation_diagnostics or {}).keys()
                ),
                "event_count": len(events),
            }
        return cls(
            persona_id=str(persona_id or "default"),
            session_id=str(session_id or "live"),
            turn_id=str(turn_id),
            turn_index=int(turn_index),
            cycle=int(cycle),
            observation_channels={
                str(key): _round_float(value)
                for key, value in dict(observation_channels or {}).items()
            },
            observation_source="DialogueObserver.observe",
            selected_events=selected_events,
            suppressed_events=suppressed_events,
            attention_selected_channels=[
                str(item)
                for item in getattr(diagnostics, "attention_selected_channels", []) or []
            ],
            attention_dropped_channels=[
                str(item)
                for item in getattr(diagnostics, "attention_dropped_channels", []) or []
            ],
            workspace_focus=[
                str(item)
                for item in getattr(diagnostics, "workspace_broadcast_channels", []) or []
            ],
            workspace_suppressed=[
                str(item)
                for item in getattr(diagnostics, "workspace_suppressed_channels", []) or []
            ],
            affective_state_summary=summarize_affective_state(
                observation_channels, diagnostics
            ),
            cognitive_state=(
                _redact_mapping(cognitive_state.to_dict())
                if hasattr(cognitive_state, "to_dict")
                else {}
            ),
            retrieved_memory_summary=summarize_retrieved_memory(diagnostics),
            ranked_options=ranked_options,
            cognitive_paths=cognitive_paths,
            path_competition=path_competition,
            meta_control_guidance=safe_guidance,
            affective_maintenance_summary=affective_maintenance_summary,
            chosen_action=chosen_action,
            policy_margin=policy_margin,
            efe_margin=efe_margin,
            fep_prompt_capsule=safe_capsule,
            generation_diagnostics=summarize_generation_diagnostics(
                generation_diagnostics
            ),
            outcome_label=str(outcome_label or "neutral"),
            memory_update_signal=_redact_mapping(dict(memory_update_signal)),
            debug_fields=debug_fields,
        )


class ConsciousMarkdownWriter:
    """Write session-scoped conscious artifacts derived from TurnTrace rows."""

    def __init__(self, root: str | Path = "artifacts/conscious", *, window: int = 3) -> None:
        self.root = Path(root)
        self.window = max(1, int(window))
        self._recent: dict[tuple[str, str], list[dict[str, object]]] = {}

    def session_dir(self, persona_id: str, session_id: str) -> Path:
        persona_part = _safe_id(persona_id, fallback="default")
        session_part = _safe_id(session_id, fallback="live")
        return self.root / "personas" / persona_part / "sessions" / session_part

    def write(self, trace: TurnTrace, *, debug: bool = False) -> Path:
        payload = trace.to_dict(debug=debug)
        session_dir = self.session_dir(trace.persona_id, trace.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        trace_writer = JsonlTraceWriter(session_dir / "conscious_trace.jsonl")
        trace_writer.append(payload)

        key = (trace.persona_id, trace.session_id)
        recent = self._recent.setdefault(key, [])
        recent.append(payload)
        del recent[:-self.window]

        markdown_path = session_dir / "Conscious.md"
        markdown_path.write_text(self._render_markdown(recent), encoding="utf-8")
        return markdown_path

    def _render_markdown(self, traces: Sequence[Mapping[str, object]]) -> str:
        latest = traces[-1]
        lines = [
            "# 当前意识投影",
            "",
            "此文件由 TurnTrace 派生，用于人类审阅；它不是认知状态的来源。",
            "",
            f"- 人格: {latest.get('persona_id', '')}",
            f"- 会话: {latest.get('session_id', '')}",
            f"- 当前轮次: {latest.get('turn_id', '')}",
            "",
            "## 当前观察",
            self._format_observation(latest),
            "",
            "## 注意与工作空间",
            self._format_attention(latest),
            "",
            "## 状态摘要",
            self._format_affect(latest),
            "",
            "## 候选路径",
            self._format_candidates(latest),
            "",
            "## 选择与理由",
            self._format_choice(latest),
            "",
            "## 提示引导",
            self._format_prompt_guidance(latest),
            "",
            "## Meta-control Guidance",
            self._format_meta_control_guidance(latest),
            "",
            "## 结果反馈",
            self._format_outcome(latest),
        ]
        if len(traces) > 1:
            lines.extend(["", "## 最近轮次"])
            for item in traces:
                lines.append(
                    f"- {item.get('turn_id', '')}: {item.get('chosen_action', '')} / "
                    f"{item.get('outcome_label', 'neutral')}"
                )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _format_observation(trace: Mapping[str, object]) -> str:
        channels = trace.get("observation_channels", {})
        if not isinstance(channels, Mapping):
            return "- 暂无观察通道。"
        items = ", ".join(
            f"{key}={_round_float(value):.3f}" for key, value in sorted(channels.items())
        )
        return f"- 通道: {items}"

    @staticmethod
    def _format_attention(trace: Mapping[str, object]) -> str:
        selected = trace.get("attention_selected_channels", [])
        dropped = trace.get("attention_dropped_channels", [])
        focus = trace.get("workspace_focus", [])
        suppressed = trace.get("workspace_suppressed", [])
        return (
            f"- 已选择: {', '.join(map(str, selected)) or '无'}\n"
            f"- 已抑制: {', '.join(map(str, dropped)) or '无'}\n"
            f"- 工作空间焦点: {', '.join(map(str, focus)) or '无'}\n"
            f"- 工作空间压制: {', '.join(map(str, suppressed)) or '无'}"
        )

    @staticmethod
    def _format_affect(trace: Mapping[str, object]) -> str:
        affect = trace.get("affective_state_summary", {})
        if not isinstance(affect, Mapping):
            return "- 暂无状态摘要。"
        return (
            f"- 情绪调性: {affect.get('tone_label', 'neutral')} "
            f"({affect.get('emotional_tone', 0.5)})\n"
            f"- 冲突张力: {affect.get('conflict_label', 'low')} "
            f"({affect.get('conflict_tension', 0.0)})\n"
            f"- 隐含意图信号: {affect.get('hidden_intent', 0.5)}"
        )

    @staticmethod
    def _format_candidates(trace: Mapping[str, object]) -> str:
        ranked = trace.get("ranked_options", [])
        if not isinstance(ranked, list) or not ranked:
            return "- 暂无候选路径。"
        lines = []
        for option in ranked[:3]:
            if not isinstance(option, Mapping):
                continue
            lines.append(
                f"- {option.get('action', '')}: policy={option.get('policy_score', 0.0)}, "
                f"efe={option.get('expected_free_energy', 0.0)}, "
                f"主导项={option.get('dominant_component', '')}"
            )
        return "\n".join(lines) if lines else "- 暂无候选路径。"

    @staticmethod
    def _format_choice(trace: Mapping[str, object]) -> str:
        return (
            f"- 已选择动作: {trace.get('chosen_action', '')}\n"
            f"- policy margin: {trace.get('policy_margin', 0.0)}\n"
            f"- EFE margin: {trace.get('efe_margin', 0.0)}"
        )

    @staticmethod
    def _format_prompt_guidance(trace: Mapping[str, object]) -> str:
        capsule = trace.get("fep_prompt_capsule", {})
        if not isinstance(capsule, Mapping):
            return "- 暂无提示胶囊摘要。"
        return (
            f"- 决策不确定性: {capsule.get('decision_uncertainty', '')}\n"
            f"- 预测误差: {capsule.get('prediction_error_label', '')}\n"
            f"- 上一反馈: {capsule.get('previous_outcome', 'neutral')}"
        )

    @staticmethod
    def _format_meta_control_guidance(trace: Mapping[str, object]) -> str:
        guidance = trace.get("meta_control_guidance", {})
        affective = trace.get("affective_maintenance_summary", {})
        if not isinstance(guidance, Mapping) or not guidance:
            return "- 暂无 meta-control guidance。"
        flags = [
            key
            for key, value in sorted(guidance.items())
            if isinstance(value, bool) and value
        ]
        reasons = guidance.get("trigger_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        affect_summary = ""
        if isinstance(affective, Mapping):
            affect_summary = str(affective.get("summary", ""))
        return (
            "- 来源: TurnTrace 派生提示，不是认知状态的来源\n"
            f"- 强度: {guidance.get('intensity', 0.0)}\n"
            f"- 标记: {', '.join(flags) or 'none'}\n"
            f"- 触发: {', '.join(map(str, reasons[:4])) or 'none'}\n"
            f"- 情感维护: {affect_summary or '维持默认表达强度'}"
        )

    @staticmethod
    def _format_outcome(trace: Mapping[str, object]) -> str:
        memory = trace.get("memory_update_signal", {})
        if not isinstance(memory, Mapping):
            memory = {}
        return (
            f"- 反馈标签: {trace.get('outcome_label', 'neutral')}\n"
            f"- 记忆更新: integrated={memory.get('integrated', False)}, "
            f"episodes={memory.get('episodic_episode_count_before', 0)}"
            f"->{memory.get('episodic_episode_count_after', 0)}"
        )
