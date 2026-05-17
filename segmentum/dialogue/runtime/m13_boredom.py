"""MVP-local M13.1 boredom and exploration bias for the UI chat path.

Engineering proxies only; diagnostics label them as such.
User-text semantics (task pressure, gain hint, salience) use a small LLM assessor — not regex cues.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from segmentum.dialogue.runtime.m13_drive import (
    M13EvaluationResult,
    M13_THRESHOLDS,
    build_topic_fingerprint,
    normalize_m13_drive_state,
)
from segmentum.dialogue.runtime.m13_reward import normalize_affective_reward_proxy_state

MAX_STALE_TURN_COUNT = 12
MAX_EXPLORATION_COOLDOWN = 5
MAX_PROMPT_GUIDANCE_LINES = 6
MAX_PROMPT_LINE_LENGTH = 160
MAX_SEMANTIC_INFORMATION_GAIN_HINT = 0.12
MAX_SEMANTIC_EXPLICIT_TASK_PRESSURE = 0.45
MAX_SEMANTIC_USER_NEED_SALIENCE = 0.35
MIN_BOREDOM_USER_TEXT_ASSESSMENT_CONFIDENCE = 0.5
MAX_ULTRA_SHORT_USER_CHARS = 6
OPPONENT_EXPLORATION_BIAS_BOOST = 0.12
OPPONENT_EXPLORATION_THRESHOLD = 0.35

BOREDOM_USER_TEXT_ASSESSOR_MARKER = "M13 无聊代理用户话语语义评估"

# Engineering fields live on events/state only; never in thinking-prompt drive_guidance.
_PROMPT_FORBIDDEN_DRIVE_GUIDANCE_KEYS: frozenset[str] = frozenset(
    {
        "exploration_bias",
        "exploration_target",
        "suppressed_repetition_actions",
        "exploration_cooldown",
        "exploration_suppressed",
        "preferred_exploration_mode",
        "boredom_level",
        "novelty_proxy",
        "information_gain_proxy",
        "repetition_pressure",
        "progress_signal",
        "stale_turn_count",
        "engineering_proxy_label",
    }
)

_EXPLORATION_MODES: tuple[str, ...] = (
    "ask_clarifying_question",
    "offer_new_angle",
    "summarize_and_choose_next_target",
    "retrieve_specific_memory",
    "mark_uncertainty",
    "propose_small_goal",
    "shift_from_repetition_to_progress",
)

class BoredomUserTextLLM(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()[:240]]
    if isinstance(value, list):
        return [str(item).strip()[:240] for item in value[:limit] if str(item).strip()]
    return []


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_boredom_user_text_assessment(raw: Any) -> dict[str, Any]:
    base = {
        "explicit_task_pressure": 0.0,
        "information_gain_hint": 0.0,
        "user_need_salience": 0.0,
        "low_information_utterance": False,
        "confidence": 0.0,
        "reason_codes": [],
    }
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    return {
        "explicit_task_pressure": round(
            min(
                MAX_SEMANTIC_EXPLICIT_TASK_PRESSURE,
                _bounded_float(raw.get("explicit_task_pressure")),
            ),
            6,
        ),
        "information_gain_hint": round(
            min(MAX_SEMANTIC_INFORMATION_GAIN_HINT, _bounded_float(raw.get("information_gain_hint"))),
            6,
        ),
        "user_need_salience": round(
            min(MAX_SEMANTIC_USER_NEED_SALIENCE, _bounded_float(raw.get("user_need_salience"))),
            6,
        ),
        "low_information_utterance": bool(raw.get("low_information_utterance")),
        "confidence": round(_bounded_float(raw.get("confidence")), 6),
        "reason_codes": _string_list(raw.get("reason_codes"), limit=6),
    }


def build_boredom_user_text_assessor_prompt(
    *,
    user_text: str,
    turn_index: int,
    topic_fingerprint: str,
) -> tuple[str, str]:
    system_prompt = f"""你是数字人格 MVP 路径的「{BOREDOM_USER_TEXT_ASSESSOR_MARKER}」模块。
根据用户本轮话语的语义（不是关键词表）估计无聊代理用的工程标量：

- explicit_task_pressure: 用户是否在要直接实现/步骤/修 bug/部署等可执行任务（0~{MAX_SEMANTIC_EXPLICIT_TASK_PRESSURE}）
- information_gain_hint: 是否在求分析/设计/权衡/根因等高密度信息（0~{MAX_SEMANTIC_INFORMATION_GAIN_HINT}）
- user_need_salience: 需求是否实质或带合理紧迫感（0~{MAX_SEMANTIC_USER_NEED_SALIENCE}）
- low_information_utterance: 是否仅为极短附和/回 channel（如单字嗯、好）几乎无新信息

这是工程代理，不是情绪模拟。只输出 JSON，不要 Markdown。"""
    user_prompt = f"""turn_index: {turn_index}
topic_fingerprint: {topic_fingerprint[:160]}

用户本轮发言:
{str(user_text or "")[:480]}

请输出 JSON:
{{
  "explicit_task_pressure": 0.0,
  "information_gain_hint": 0.0,
  "user_need_salience": 0.0,
  "low_information_utterance": false,
  "confidence": 0.0,
  "reason_codes": ["简短依据，最多4个"]
}}"""
    return system_prompt, user_prompt


def assess_boredom_user_text_semantics(
    llm: BoredomUserTextLLM,
    *,
    user_text: str,
    turn_index: int,
    topic_fingerprint: str,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_boredom_user_text_assessor_prompt(
        user_text=user_text,
        turn_index=turn_index,
        topic_fingerprint=topic_fingerprint,
    )
    try:
        raw = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return normalize_boredom_user_text_assessment({})
    return normalize_boredom_user_text_assessment(raw)


def _user_text_assessment_active(assessment: Mapping[str, Any] | None) -> bool:
    if not assessment:
        return False
    return _bounded_float(assessment.get("confidence")) >= MIN_BOREDOM_USER_TEXT_ASSESSMENT_CONFIDENCE


def _normalize_term(term: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff+#.-]+", "", str(term or "").casefold().strip())
    return cleaned[:80]


def default_boredom_state() -> dict[str, Any]:
    return {
        "boredom_level": 0.0,
        "novelty_baseline": 0.55,
        "last_exploration_target": "",
        "exploration_cooldown": 0,
        "stale_turn_count": 0,
        "last_progress_signal": 0.0,
        "recent_retrieval_ids": [],
        "recent_plan_terms": [],
        "engineering_proxy_label": "mvp_local_boredom_proxy",
    }


def normalize_boredom_state(raw: Any) -> dict[str, Any]:
    base = default_boredom_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    merged["boredom_level"] = _bounded_float(merged.get("boredom_level"))
    merged["novelty_baseline"] = _bounded_float(merged.get("novelty_baseline"), default=0.55)
    merged["exploration_cooldown"] = min(
        MAX_EXPLORATION_COOLDOWN,
        max(0, int(merged.get("exploration_cooldown", 0) or 0)),
    )
    merged["stale_turn_count"] = min(
        MAX_STALE_TURN_COUNT,
        max(0, int(merged.get("stale_turn_count", 0) or 0)),
    )
    merged["last_progress_signal"] = _bounded_float(merged.get("last_progress_signal"))
    merged["last_exploration_target"] = str(merged.get("last_exploration_target", "") or "")[:160]
    merged["recent_retrieval_ids"] = _string_list(merged.get("recent_retrieval_ids"), limit=12)
    merged["recent_plan_terms"] = _string_list(merged.get("recent_plan_terms"), limit=16)
    return merged


def boredom_band(level: float) -> str:
    if level >= 0.67:
        return "high"
    if level >= 0.35:
        return "medium"
    return "low"


def _topic_terms(topic_fingerprint: str) -> set[str]:
    return {t for t in topic_fingerprint.split("|") if t and t != "topic:unknown"}


def _recent_topics(state: Mapping[str, Any]) -> list[str]:
    rows = state.get("recent_topic_fingerprints", [])
    if not isinstance(rows, list):
        return []
    topics: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            topic = str(row.get("topic", "")).strip()
            if topic:
                topics.append(topic)
    return topics


def _novelty_proxy(
    *,
    topic_fingerprint: str,
    recent_topics: list[str],
    recent_trace: list[Mapping[str, Any]],
    user_text: str,
    retrieved_ids: list[str],
    prior_retrieval_ids: list[str],
    user_text_assessment: Mapping[str, Any] | None = None,
) -> float:
    """Higher means more novelty (less boring)."""
    current_terms = _topic_terms(topic_fingerprint)
    overlap_scores: list[float] = []
    for prior in recent_topics[-8:]:
        prior_terms = _topic_terms(prior)
        if not current_terms or not prior_terms:
            continue
        overlap = len(current_terms & prior_terms) / max(1, len(current_terms | prior_terms))
        overlap_scores.append(overlap)
    topic_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    low_topic_novelty = _bounded_float(topic_overlap)

    same_action = sum(
        1
        for row in recent_trace[-6:]
        if isinstance(row, Mapping) and str(row.get("topic_fingerprint", "")) == topic_fingerprint
    )
    action_repeat = _bounded_float(max(0.0, (same_action - 1) * 0.22))

    if retrieved_ids and prior_retrieval_ids:
        shared = len(set(retrieved_ids) & set(prior_retrieval_ids))
        retrieval_overlap = shared / max(1, len(set(retrieved_ids) | set(prior_retrieval_ids)))
    else:
        retrieval_overlap = 0.0

    stripped = str(user_text or "").strip()
    short_low_cue = 0.0
    if _user_text_assessment_active(user_text_assessment) and bool(
        user_text_assessment.get("low_information_utterance")
    ):
        short_low_cue = 0.35
    elif not user_text_assessment and len(stripped) <= MAX_ULTRA_SHORT_USER_CHARS and stripped:
        short_low_cue = 0.35

    novelty = 1.0 - (
        low_topic_novelty * 0.42
        + action_repeat * 0.28
        + _bounded_float(retrieval_overlap) * 0.20
        + short_low_cue * 0.10
    )
    return _bounded_float(novelty)


def _information_gain_proxy(
    *,
    conscious_plan: Mapping[str, Any],
    evidence_judgment: Mapping[str, Any] | None,
    entity_binding: Mapping[str, Any] | None,
    m11_result: Mapping[str, Any] | None,
    m12_payload: Mapping[str, Any] | None,
    m12_2_result: Mapping[str, Any] | None,
    user_text_assessment: Mapping[str, Any] | None = None,
    prior_plan_terms: list[str],
) -> float:
    gain = 0.0
    conscious = _mapping(conscious_plan)
    keywords = {_normalize_term(t) for t in _string_list(conscious.get("memory_search_keywords"), limit=16)}
    prior = {_normalize_term(t) for t in prior_plan_terms}
    new_terms = keywords - prior
    if new_terms:
        gain += min(0.35, 0.08 * len(new_terms))

    evidence = _mapping(evidence_judgment)
    stance = str(evidence.get("epistemic_stance", "")).strip().lower()
    candidates = evidence.get("candidate_evidence") or evidence.get("evidence_candidates") or []
    if stance in {"uncertain", "known_with_caveat"} and isinstance(candidates, list) and candidates:
        gain += 0.22

    binding = _mapping(entity_binding)
    if str(binding.get("binding_correction", "")).strip():
        gain += 0.18
    if str(binding.get("needs_clarification", "")).strip().lower() in {"true", "yes", "1"}:
        gain += 0.12

    m11 = _mapping(m11_result)
    if m11:
        m11_after = _mapping(m11.get("state_after"))
        m11_user = _mapping(m11_after.get("user_model"))
        m11_open = _string_list(m11_user.get("open_uncertainties"), limit=6)
        if m11_open:
            gain += min(0.24, 0.07 * len(m11_open))
        quarantined = _string_list(m11.get("quarantined_hypotheses"), limit=4)
        if quarantined:
            gain += 0.14

    m12 = _mapping(m12_payload)
    if m12:
        m12_after = _mapping(m12.get("state_after"))
        conflicts = m12_after.get("conflict_records") or []
        if isinstance(conflicts, list):
            open_conflicts = [
                row
                for row in conflicts
                if isinstance(row, Mapping)
                and str(row.get("resolution_status", "open")).lower() in {"open", "probed"}
            ]
            if open_conflicts:
                gain += min(0.26, 0.08 * len(open_conflicts))

    m12_2 = _mapping(m12_2_result)
    if m12_2:
        extractor = _mapping(m12_2.get("extractor_output"))
        for key in ("unresolved_uncertainty_points", "clarifying_reply_candidates"):
            items = extractor.get(key) or []
            if isinstance(items, list) and items:
                gain += min(0.22, 0.06 * len(items))

    if _user_text_assessment_active(user_text_assessment):
        gain += _bounded_float(user_text_assessment.get("information_gain_hint"))

    return _bounded_float(gain)


def _closed_expectation_progress(impact: Mapping[str, Any]) -> float:
    closed_ids = impact.get("closed_expectation_ids")
    if isinstance(closed_ids, list) and closed_ids:
        return 0.2
    if isinstance(closed_ids, str) and closed_ids.strip():
        return 0.2
    closed_count = impact.get("closed_count")
    try:
        if int(closed_count or 0) > 0:
            return 0.2
    except (TypeError, ValueError):
        pass
    return 0.0


def _progress_signal(
    *,
    conscious_plan: Mapping[str, Any],
    evidence_judgment: Mapping[str, Any] | None,
    memory_dynamics: Mapping[str, Any],
) -> float:
    conscious = _mapping(conscious_plan)
    progress = 0.0
    for item in conscious.get("expectation_results", []) or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status == "confirmed":
            progress += 0.28
        elif status in {"resolved", "closed"}:
            progress += 0.22

    dynamics = _mapping(memory_dynamics)
    impact = _mapping(dynamics.get("expectation_impact"))
    progress += _closed_expectation_progress(impact)

    evidence = _mapping(evidence_judgment)
    stance = str(evidence.get("epistemic_stance", "")).strip().lower()
    if stance in {"known", "supported"}:
        progress += 0.18
    elif stance == "known_with_caveat":
        progress += 0.10

    open_items = dynamics.get("open_item_updates") or []
    if isinstance(open_items, list):
        for row in open_items:
            if isinstance(row, Mapping) and str(row.get("status", "")).lower() in {"closed", "resolved"}:
                progress += 0.2
                break

    return _bounded_float(progress)


def _repetition_pressure(
    *,
    recent_trace: list[Mapping[str, Any]],
    topic_fingerprint: str,
    user_id: str,
    selected_action: str = "",
) -> float:
    recent = [row for row in recent_trace[-8:] if isinstance(row, Mapping)]
    if not recent:
        return 0.0
    same_topic = sum(1 for row in recent if str(row.get("topic_fingerprint", "")) == topic_fingerprint)
    same_action = sum(
        1
        for row in recent
        if str(row.get("action", "")) == selected_action and str(row.get("user_id", "")) == user_id
    ) if selected_action else 0
    topic_pressure = _bounded_float(max(0.0, (same_topic - 1) * 0.18))
    action_pressure = _bounded_float(max(0.0, (same_action - 1) * 0.22))
    return _bounded_float(max(topic_pressure, action_pressure))


def _predictability(
    *,
    topic_fingerprint: str,
    recent_topics: list[str],
    memory_dynamics: Mapping[str, Any],
) -> float:
    if not recent_topics:
        return 0.0
    repeats = sum(1 for topic in recent_topics[-6:] if topic == topic_fingerprint)
    recall = _mapping(_mapping(memory_dynamics.get("recall_query")))
    recall_terms = {_normalize_term(t) for t in _string_list(recall.get("semantic_terms"), limit=12)}
    topic_terms = _topic_terms(topic_fingerprint)
    recall_stable = 0.0
    if recall_terms and topic_terms:
        recall_stable = len(recall_terms & topic_terms) / max(1, len(recall_terms))
    return _bounded_float(min(1.0, repeats * 0.16 + recall_stable * 0.35))


def _explicit_task_pressure_heuristic(user_text: str) -> float:
    """Length/question heuristic only when no LLM assessor is available."""
    text = str(user_text or "").strip()
    if not text:
        return 0.0
    if "?" in text or "？" in text:
        if len(text) >= 18:
            return 0.25
    return 0.0


def _explicit_task_pressure(
    user_text: str,
    user_text_assessment: Mapping[str, Any] | None = None,
) -> float:
    if _user_text_assessment_active(user_text_assessment):
        return _bounded_float(user_text_assessment.get("explicit_task_pressure"))
    return _explicit_task_pressure_heuristic(user_text)


def _conflict_or_repair_pressure(memory_dynamics: Mapping[str, Any]) -> float:
    control = _mapping(memory_dynamics.get("control_guidance"))
    conflict = _bounded_float(control.get("conflict_level"))
    repair = _bounded_float(control.get("repair_bias"))
    clarification = _bounded_float(control.get("clarification_bias"))
    return _bounded_float(max(conflict, repair * 0.9, clarification * 0.85))


def _user_need_salience(
    user_text: str,
    user_text_assessment: Mapping[str, Any] | None = None,
) -> float:
    if _user_text_assessment_active(user_text_assessment):
        return _bounded_float(user_text_assessment.get("user_need_salience"))
    text = str(user_text or "").strip()
    if len(text) >= 48:
        return 0.35
    if len(text) >= 24:
        return 0.18
    return 0.0


def _identity_clarification_pressure(
    entity_binding: Mapping[str, Any] | None,
    evidence_judgment: Mapping[str, Any] | None,
) -> float:
    binding = _mapping(entity_binding)
    if str(binding.get("needs_clarification", "")).strip().lower() in {"true", "yes", "1"}:
        return 0.5
    evidence = _mapping(evidence_judgment)
    if str(evidence.get("epistemic_stance", "")).lower() == "forbidden_assumption":
        return 0.65
    contract = _mapping(_mapping(evidence.get("reply_contract")))
    if bool(contract.get("enforce_identity_verification")):
        return 0.45
    return 0.0


def _compute_boredom_level(
    *,
    predictability: float,
    repetition_pressure: float,
    novelty_proxy: float,
    information_gain_proxy: float,
    progress_signal: float,
    explicit_task_pressure: float,
    conflict_or_repair_pressure: float,
    user_need_salience: float,
    weak_pull_margin: bool,
) -> float:
    low_novelty = 1.0 - novelty_proxy
    low_information_gain = 1.0 - information_gain_proxy
    low_progress = 1.0 - progress_signal
    raw = (
        predictability * 0.22
        + repetition_pressure * 0.24
        + low_novelty * 0.22
        + low_information_gain * 0.18
        + low_progress * 0.14
        + (0.08 if weak_pull_margin else 0.0)
        - explicit_task_pressure
        - conflict_or_repair_pressure
        - user_need_salience
    )
    return _bounded_float(raw)


def sanitize_drive_guidance_for_prompt(guidance: Mapping[str, Any]) -> dict[str, Any]:
    """Strip M13 engineering scalars before control_guidance reaches the thinking prompt."""
    cleaned = dict(_mapping(guidance))
    for key in _PROMPT_FORBIDDEN_DRIVE_GUIDANCE_KEYS:
        cleaned.pop(key, None)
    return cleaned


def prompt_safe_control_guidance_for_thinking(control: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_mapping(control))
    drive = _mapping(result.get("drive_guidance"))
    if drive:
        result["drive_guidance"] = sanitize_drive_guidance_for_prompt(drive)
    affective = _mapping(result.get("affective_drive_guidance"))
    if affective:
        from segmentum.dialogue.runtime.m13_reward import sanitize_affective_drive_guidance_for_prompt

        result["affective_drive_guidance"] = sanitize_affective_drive_guidance_for_prompt(affective)
    return result


def _choose_exploration_mode(
    *,
    boredom_level: float,
    repetition_pressure: float,
    information_gain_proxy: float,
    progress_signal: float,
    evidence_judgment: Mapping[str, Any] | None,
    has_retrieved_evidence: bool = False,
) -> str:
    evidence = _mapping(evidence_judgment)
    stance = str(evidence.get("epistemic_stance", "")).lower()
    if stance in {"uncertain", "known_with_caveat", "forbidden_assumption"}:
        return "mark_uncertainty"
    if (
        has_retrieved_evidence
        and boredom_level >= 0.35
        and progress_signal < 0.35
        and information_gain_proxy < 0.35
        and 0.2 <= repetition_pressure < 0.35
    ):
        return "retrieve_specific_memory"
    if repetition_pressure >= 0.35 and progress_signal < 0.35:
        return "shift_from_repetition_to_progress"
    if information_gain_proxy < 0.25 and boredom_level >= 0.5:
        return "ask_clarifying_question"
    if progress_signal >= 0.35:
        return "summarize_and_choose_next_target"
    if boredom_level >= 0.67:
        return "offer_new_angle"
    return "propose_small_goal"


def _ordinary_language_hint(mode: str, *, suppressed: bool) -> str:
    if suppressed:
        return ""
    hints = {
        "ask_clarifying_question": (
            "The exchange is getting repetitive. If it still fits the user's request, "
            "prefer one useful clarifying question over another similar answer."
        ),
        "offer_new_angle": (
            "The current path lacks fresh gradient. Prefer a small new angle or concrete "
            "next step if it still serves the user's request."
        ),
        "summarize_and_choose_next_target": (
            "Briefly anchor what is settled, then suggest one concrete next target if helpful."
        ),
        "retrieve_specific_memory": (
            "A specific recalled detail may unblock progress; use it only if evidence-backed."
        ),
        "mark_uncertainty": (
            "Keep claims within evidence boundaries; state uncertainty plainly without diagnostics."
        ),
        "propose_small_goal": (
            "Offer one small, actionable next step when the user's goal is still open-ended."
        ),
        "shift_from_repetition_to_progress": (
            "The current exchange is starting to repeat. Prefer progress or reframing over "
            "another similar reply when it still fits the user's request."
        ),
    }
    return hints.get(mode, hints["shift_from_repetition_to_progress"])


def _compact_line(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:MAX_PROMPT_LINE_LENGTH]


def build_prompt_safe_guidance_lines(
    *,
    drive_summary: str = "",
    drive_caution: str = "",
    exploration_hint: str = "",
    exploration_mode: str = "",
) -> list[str]:
    lines: list[str] = []
    for candidate in (drive_summary, drive_caution, exploration_hint):
        line = _compact_line(candidate)
        if line and line not in lines:
            lines.append(line)
    if exploration_mode and len(lines) < MAX_PROMPT_GUIDANCE_LINES:
        mode_line = _compact_line(f"Exploration tendency: {exploration_mode.replace('_', ' ')}.")
        if mode_line not in lines:
            lines.append(mode_line)
    return lines[:MAX_PROMPT_GUIDANCE_LINES]


@dataclass
class M13BoredomEvaluationResult:
    event_id: str
    turn_id: str
    turn_index: int
    boredom_level: float
    boredom_band: str
    novelty_proxy: float
    information_gain_proxy: float
    repetition_pressure: float
    progress_signal: float
    exploration_bias: float
    preferred_exploration_mode: str
    exploration_target: str
    suppressed_repetition_actions: list[str]
    cooldown: int
    ordinary_language_hint: str
    evidence_refs: list[str]
    prompt_safe_summary: str
    exploration_suppressed: bool
    events: list[dict[str, Any]] = field(default_factory=list)


class M13BoredomEvaluator:
    """Deterministic boredom / exploration evaluator; no visible diagnostic text."""

    def evaluate(
        self,
        *,
        user_text: str,
        user_id: str,
        turn_id: str,
        turn_index: int,
        conscious_plan: Mapping[str, Any],
        memory_dynamics: Mapping[str, Any],
        retrieved_memories: list[Mapping[str, Any]],
        m13_state: Mapping[str, Any],
        m13_drive_evaluation: M13EvaluationResult | None = None,
        entity_binding: Mapping[str, Any] | None = None,
        evidence_judgment: Mapping[str, Any] | None = None,
        m11_result: Mapping[str, Any] | None = None,
        m12_payload: Mapping[str, Any] | None = None,
        m12_2_result: Mapping[str, Any] | None = None,
        llm: BoredomUserTextLLM | None = None,
    ) -> M13BoredomEvaluationResult:
        state = normalize_m13_drive_state(m13_state)
        boredom_state = normalize_boredom_state(state.get("boredom"))
        topic = build_topic_fingerprint(
            conscious_plan=conscious_plan,
            memory_dynamics=memory_dynamics,
            entity_binding=entity_binding,
            retrieved_memories=retrieved_memories,
            user_text=user_text,
        )
        retrieved_ids = [
            str(item.get("id", "")).strip()
            for item in retrieved_memories
            if isinstance(item, Mapping) and str(item.get("id", "")).strip()
        ]
        trace = [row for row in state.get("recent_action_trace", []) if isinstance(row, Mapping)]
        recent_topics = _recent_topics(state)
        top_action = ""
        pull_margin = 1.0
        if m13_drive_evaluation is not None:
            top_action = str(m13_drive_evaluation.top_behavioral_pull_action or "")
            pull_margin = float(m13_drive_evaluation.pull_margin or 0.0)

        user_text_assessment: dict[str, Any] | None = None
        assessment_events: list[dict[str, Any]] = []
        if llm is not None and str(user_text or "").strip():
            user_text_assessment = assess_boredom_user_text_semantics(
                llm,
                user_text=user_text,
                turn_index=turn_index,
                topic_fingerprint=topic,
            )
            assessment_events.append(
                {
                    "type": "M13BoredomUserTextAssessmentEvent",
                    "turn_id": turn_id,
                    "turn_index": turn_index,
                    "assessment": user_text_assessment,
                    "engineering_proxy_label": "mvp_local_boredom_proxy",
                }
            )

        novelty_proxy = _novelty_proxy(
            topic_fingerprint=topic,
            recent_topics=recent_topics,
            recent_trace=trace,
            user_text=user_text,
            retrieved_ids=retrieved_ids,
            prior_retrieval_ids=boredom_state.get("recent_retrieval_ids", []),
            user_text_assessment=user_text_assessment,
        )
        information_gain_proxy = _information_gain_proxy(
            conscious_plan=conscious_plan,
            evidence_judgment=evidence_judgment,
            entity_binding=entity_binding,
            m11_result=m11_result,
            m12_payload=m12_payload,
            m12_2_result=m12_2_result,
            user_text_assessment=user_text_assessment,
            prior_plan_terms=boredom_state.get("recent_plan_terms", []),
        )
        progress_signal = _progress_signal(
            conscious_plan=conscious_plan,
            evidence_judgment=evidence_judgment,
            memory_dynamics=memory_dynamics,
        )
        repetition_pressure = _repetition_pressure(
            recent_trace=trace,
            topic_fingerprint=topic,
            user_id=user_id,
            selected_action=top_action,
        )
        predictability = _predictability(
            topic_fingerprint=topic,
            recent_topics=recent_topics,
            memory_dynamics=memory_dynamics,
        )
        explicit_task = _explicit_task_pressure(user_text, user_text_assessment)
        conflict_repair = _conflict_or_repair_pressure(memory_dynamics)
        user_salience = _user_need_salience(user_text, user_text_assessment)
        identity_pressure = _identity_clarification_pressure(entity_binding, evidence_judgment)
        weak_margin = pull_margin < M13_THRESHOLDS.weak_pull_margin

        stale_boost = 0.0
        if information_gain_proxy < 0.4 and explicit_task < 0.25:
            stale_boost = _bounded_float(
                int(boredom_state.get("stale_turn_count", 0) or 0)
                / max(1, MAX_STALE_TURN_COUNT)
                * 0.22
            )
        boredom_level = _bounded_float(
            _compute_boredom_level(
                predictability=predictability,
                repetition_pressure=repetition_pressure,
                novelty_proxy=novelty_proxy,
                information_gain_proxy=information_gain_proxy,
                progress_signal=progress_signal,
                explicit_task_pressure=explicit_task,
                conflict_or_repair_pressure=max(conflict_repair, identity_pressure),
                user_need_salience=user_salience,
                weak_pull_margin=weak_margin,
            )
            + stale_boost
        )

        hard_suppressed = (
            turn_index <= 0
            or explicit_task >= 0.45
            or conflict_repair >= 0.55
            or identity_pressure >= 0.45
        )
        cooldown_active = int(boredom_state.get("exploration_cooldown", 0) or 0) > 0
        suppressed = hard_suppressed or cooldown_active
        if hard_suppressed:
            boredom_level = _bounded_float(boredom_level * 0.35)

        band = boredom_band(boredom_level)
        cooldown = int(boredom_state.get("exploration_cooldown", 0) or 0)
        exploration_bias = _bounded_float(boredom_level * (0.65 + repetition_pressure * 0.35))
        if hard_suppressed:
            exploration_bias = 0.0
        elif cooldown_active:
            exploration_bias = _bounded_float(exploration_bias * 0.45)
        reward_proxy = normalize_affective_reward_proxy_state(
            normalize_m13_drive_state(m13_state).get("affective_reward_proxy")
        )
        opponent_strength = _bounded_float(reward_proxy.get("opponent_strength"))
        if (
            not hard_suppressed
            and opponent_strength >= OPPONENT_EXPLORATION_THRESHOLD
        ):
            exploration_bias = _bounded_float(
                exploration_bias + OPPONENT_EXPLORATION_BIAS_BOOST * opponent_strength
            )
        mode = _choose_exploration_mode(
            boredom_level=boredom_level,
            repetition_pressure=repetition_pressure,
            information_gain_proxy=information_gain_proxy,
            progress_signal=progress_signal,
            evidence_judgment=evidence_judgment,
            has_retrieved_evidence=bool(retrieved_ids),
        )
        exploration_target = topic if exploration_bias >= 0.35 and not suppressed else ""
        suppressed_actions: list[str] = []
        if repetition_pressure >= 0.3 and top_action:
            suppressed_actions.append(top_action)

        hint = _ordinary_language_hint(
            mode,
            suppressed=hard_suppressed or cooldown_active or exploration_bias < 0.35,
        )
        if (
            not hint
            and not hard_suppressed
            and opponent_strength >= OPPONENT_EXPLORATION_THRESHOLD
            and exploration_bias >= 0.35
        ):
            hint = (
                "Recent repair pressure suggests checking understanding before "
                "repeating the same approach."
            )
        if exploration_bias < 0.35 or cooldown_active or hard_suppressed:
            hint = ""

        evidence_refs = _string_list(
            [str(item.get("id", "")) for item in retrieved_memories if isinstance(item, Mapping)],
            limit=8,
        )
        event_id = _new_id("m13_boredom")
        summary = hint or "No exploration nudge; follow the live user request and evidence boundaries."
        eval_event = {
            "type": "M13BoredomEvaluationEvent",
            "event_id": event_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "source": "m13_boredom_evaluator",
            "boredom_level": round(boredom_level, 6),
            "boredom_band": band,
            "novelty_proxy": round(novelty_proxy, 6),
            "information_gain_proxy": round(information_gain_proxy, 6),
            "repetition_pressure": round(repetition_pressure, 6),
            "progress_signal": round(progress_signal, 6),
            "exploration_bias": round(exploration_bias, 6),
            "exploration_target": exploration_target,
            "preferred_exploration_mode": mode,
            "suppressed_repetition_actions": suppressed_actions,
            "cooldown": cooldown,
            "evidence_refs": evidence_refs,
            "prompt_safe_summary": summary,
            "engineering_proxy_label": "mvp_local_boredom_proxy",
            "exploration_suppressed": suppressed,
        }
        return M13BoredomEvaluationResult(
            event_id=event_id,
            turn_id=turn_id,
            turn_index=turn_index,
            boredom_level=boredom_level,
            boredom_band=band,
            novelty_proxy=novelty_proxy,
            information_gain_proxy=information_gain_proxy,
            repetition_pressure=repetition_pressure,
            progress_signal=progress_signal,
            exploration_bias=exploration_bias,
            preferred_exploration_mode=mode,
            exploration_target=exploration_target,
            suppressed_repetition_actions=suppressed_actions,
            cooldown=cooldown,
            ordinary_language_hint=hint,
            evidence_refs=evidence_refs,
            prompt_safe_summary=summary,
            exploration_suppressed=suppressed,
            events=[*assessment_events, eval_event],
        )


def merge_exploration_guidance_into_control(
    memory_dynamics: dict[str, Any],
    boredom: M13BoredomEvaluationResult,
    *,
    drive_guidance: Mapping[str, Any] | None = None,
) -> None:
    """Patch prompt-safe exploration lines into drive_guidance; scalars stay on events/state."""
    control = _mapping(memory_dynamics.get("control_guidance"))
    guidance = dict(_mapping(control.get("drive_guidance")))
    if drive_guidance:
        guidance.update(dict(drive_guidance))
    exploration_mode = (
        boredom.preferred_exploration_mode if boredom.ordinary_language_hint else ""
    )
    lines = build_prompt_safe_guidance_lines(
        drive_summary=str(guidance.get("action_tendency_reason", "") or ""),
        drive_caution=str(guidance.get("caution", "") or ""),
        exploration_hint=boredom.ordinary_language_hint,
        exploration_mode=exploration_mode,
    )
    guidance["prompt_safe_lines"] = lines
    control["drive_guidance"] = sanitize_drive_guidance_for_prompt(guidance)
    memory_dynamics["control_guidance"] = control


def prompt_safe_m13_boredom_diagnostics(boredom: M13BoredomEvaluationResult) -> dict[str, Any]:
    return {
        "event_id": boredom.event_id,
        "boredom_band": boredom.boredom_band,
        "preferred_exploration_mode": boredom.preferred_exploration_mode,
        "exploration_suppressed": boredom.exploration_suppressed,
        "prompt_safe_summary": boredom.prompt_safe_summary,
        "engineering_proxy_label": "mvp_local_boredom_proxy",
    }


def apply_post_turn_boredom_state(
    m13_state: dict[str, Any],
    *,
    boredom: M13BoredomEvaluationResult,
    conscious_plan: Mapping[str, Any],
    retrieved_memories: list[Mapping[str, Any]],
    turn_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compact boredom patch only; returns state and audit events."""
    state = normalize_m13_drive_state(m13_state)
    boredom_state = normalize_boredom_state(state.get("boredom"))
    events: list[dict[str, Any]] = []

    stale = int(boredom_state.get("stale_turn_count", 0) or 0)
    if boredom.boredom_band in {"medium", "high"} and boredom.repetition_pressure >= 0.2:
        stale = min(MAX_STALE_TURN_COUNT, stale + 1)
    else:
        stale = max(0, stale - 1)

    cooldown = int(boredom_state.get("exploration_cooldown", 0) or 0)
    if cooldown > 0:
        cooldown = max(0, cooldown - 1)
    fired_exploration = (
        not boredom.exploration_suppressed
        and boredom.exploration_bias >= 0.35
        and boredom.boredom_band in {"medium", "high"}
    )
    if fired_exploration:
        cooldown = MAX_EXPLORATION_COOLDOWN

    retrieved_ids = [
        str(item.get("id", "")).strip()
        for item in retrieved_memories
        if isinstance(item, Mapping) and str(item.get("id", "")).strip()
    ]
    conscious = _mapping(conscious_plan)
    plan_terms = _string_list(conscious.get("memory_search_keywords"), limit=16)

    boredom_state.update(
        {
            "boredom_level": round(boredom.boredom_level, 6),
            "novelty_baseline": round(
                (float(boredom_state.get("novelty_baseline", 0.55)) * 0.7)
                + boredom.novelty_proxy * 0.3,
                6,
            ),
            "last_exploration_target": boredom.exploration_target or boredom_state.get("last_exploration_target", ""),
            "exploration_cooldown": cooldown,
            "stale_turn_count": stale,
            "last_progress_signal": round(boredom.progress_signal, 6),
            "recent_retrieval_ids": retrieved_ids[-12:],
            "recent_plan_terms": plan_terms,
        }
    )
    state["boredom"] = boredom_state

    if fired_exploration or boredom.boredom_level >= 0.35:
        patch_id = _new_id("m13_boredom_patch")
        events.append(
            {
                "type": "M13BoredomPatchProposal",
                "patch_id": patch_id,
                "target": "m13_drive_state.boredom",
                "operation": "update",
                "field_path": "boredom",
                "previous_summary": f"stale={boredom_state.get('stale_turn_count')}",
                "new_summary": (
                    f"level={boredom.boredom_level:.2f} band={boredom.boredom_band} "
                    f"cooldown={cooldown}"
                ),
                "source_event_id": boredom.event_id,
                "reason": "boredom_proxy_turn_update",
                "confidence": 0.7,
                "ttl": 5,
                "engineering_proxy_label": "mvp_local_boredom_proxy",
            }
        )
        events.append(
            {
                "type": "M13BoredomPatchCommit",
                "commit_id": _new_id("m13_boredom_commit"),
                "patch_id": patch_id,
                "accepted": True,
                "owner": "MVPDialogueRuntime",
                "reason": "boredom_proxy_turn_update",
                "committed_summary": f"stale={stale} cooldown={cooldown}",
            }
        )

    return state, events
