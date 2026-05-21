"""M14.0 conscious idle reflector: context, plan schema, drive rules (Path B)."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_boredom import boredom_band, normalize_boredom_state
from segmentum.dialogue.runtime.m13_drive import _bounded_float, _mapping, _string_list, normalize_m13_drive_state
from segmentum.dialogue.runtime.m13_memory_efe import normalize_memory_efe_state
from segmentum.dialogue.runtime.m13_reward import (
    list_assessable_pending_rows,
    normalize_affective_reward_proxy_state,
)
from segmentum.dialogue.runtime.m14_3_proactive_alignment import build_traceable_proactive_intent, is_traceable_open_item

M14_ENGINEERING_PROXY_LABEL = "mvp_local_conscious_idle_reflector"
IDLE_INTROSPECTION_MARKER = "M14 空闲内省意识主循环"
MIN_PATCH_CONFIDENCE = 0.6
UNCERTAIN_EXPECTATION_TURNS = 3
SALIENCE_RECALL_THRESHOLD = 0.5
MAX_IDLE_CONTEXT_OPEN_ITEMS = 5
MAX_IDLE_CONTEXT_MEMORIES = 5
MAX_IDLE_CONTEXT_EXPECTATIONS = 5

_REFLECTION_KINDS = frozenset({"open_item", "user_model", "self_consistency", "habit_calibration", "none"})
_OUTREACH_REASONS = frozenset(
    {
        "user_active",
        "reflection_only",
        "low_value",
        "already_recent",
        "none",
        "open_item_followup",
        "traceable_focus",
    }
)
_MEMORY_TARGETS = frozenset({"short_term", "long_term"})
_MEMORY_KINDS = frozenset({"episode", "habit", "preference", "fact"})
_OPEN_ITEM_OPS = frozenset({"update", "close", "defer"})

_SUBJECTIVE_TERMS = (
    "lonely",
    "loneliness",
    "addicted",
    "addiction",
    "bored",
    "i needed to reach out",
    "寂寞",
    "孤独",
    "上瘾",
    "成瘾",
    "我很无聊",
    "我需要找你",
)

_REPLY_LIKE_KEYS = frozenset({"reply", "user_reply", "assistant_reply", "message", "chat_reply"})


def _json_text(payload: Any, *, limit: int = 12000) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return text[:limit] if len(text) > limit else text


def subjective_language_violations(text: str) -> list[str]:
    lowered = str(text or "").casefold()
    codes: list[str] = []
    for term in _SUBJECTIVE_TERMS:
        if term.casefold() in lowered:
            codes.append(f"subjective_term:{term[:32]}")
    return codes[:8]


def _looks_like_user_reply(payload: Mapping[str, Any]) -> bool:
    for key in _REPLY_LIKE_KEYS:
        if key in payload and str(payload.get(key, "")).strip():
            return True
    return False


def top_traction_actions(m13_state: Mapping[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    traction = _mapping(m13_state.get("traction_by_action"))
    ranked = sorted(
        ((str(action), _bounded_float(score)) for action, score in traction.items()),
        key=lambda row: row[1],
        reverse=True,
    )
    return [{"action": action, "traction": round(score, 4)} for action, score in ranked[:limit] if action]


def build_idle_context(
    state: Mapping[str, Any],
    *,
    m13_state: Mapping[str, Any],
    structural_signals: Any,
    turn_index: int,
    now: int,
) -> dict[str, Any]:
    """Compact prompt-safe idle summary; no raw state dumps."""
    open_rows: list[dict[str, Any]] = []
    for item in state.get("open_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "open")).strip().lower()
        if status not in {"open", "pending", "active", ""}:
            continue
        next_check = str(item.get("next_check", item.get("next_step", "")) or "").strip()
        if not next_check:
            continue
        open_rows.append(
            {
                "id": str(item.get("id", ""))[:64],
                "title": str(item.get("title", item.get("summary", "")) or "")[:120],
                "next_check": next_check[:140],
                "status": status[:32],
                "evidence_refs": _string_list(item.get("evidence_refs"), limit=8),
                "bound_memory_ids": _string_list(item.get("bound_memory_ids"), limit=8),
                "scheduled_intent_id": str(item.get("scheduled_intent_id", item.get("intent_id", "")) or "")[:120],
                "due_at_epoch": int(item.get("due_at_epoch", 0) or 0),
            }
        )
        if len(open_rows) >= MAX_IDLE_CONTEXT_OPEN_ITEMS:
            break

    salient_memories: list[dict[str, Any]] = []
    for key in ("short_term_memory", "long_term_memory"):
        rows = state.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            salience = _bounded_float(row.get("salience"), default=0.0)
            if salience < SALIENCE_RECALL_THRESHOLD:
                continue
            last_recalled = int(row.get("last_recalled_at", 0) or 0)
            if last_recalled > 0 and (now - last_recalled) < 3600:
                continue
            salient_memories.append(
                {
                    "id": str(row.get("id", ""))[:64],
                    "kind": str(row.get("kind", "episode"))[:32],
                    "salience": round(salience, 3),
                    "summary": str(row.get("content", row.get("text", "")) or "")[:160],
                    "source": key,
                }
            )
            if len(salient_memories) >= MAX_IDLE_CONTEXT_MEMORIES:
                break
        if len(salient_memories) >= MAX_IDLE_CONTEXT_MEMORIES:
            break

    uncertain_expectations: list[dict[str, Any]] = []
    pending = state.get("pending_expectations", [])
    if isinstance(pending, list):
        for row in pending:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("status", "")).strip().lower() != "uncertain":
                continue
            created_turn = int(row.get("created_turn_index", row.get("turn_index", 0)) or 0)
            age = max(0, turn_index - created_turn) if created_turn > 0 else UNCERTAIN_EXPECTATION_TURNS
            if age < UNCERTAIN_EXPECTATION_TURNS:
                continue
            uncertain_expectations.append(
                {
                    "id": str(row.get("id", ""))[:64],
                    "content": str(row.get("content", ""))[:160],
                    "uncertain_turns": age,
                }
            )
            if len(uncertain_expectations) >= MAX_IDLE_CONTEXT_EXPECTATIONS:
                break

    boredom = normalize_boredom_state(m13_state.get("boredom"))
    reward = normalize_affective_reward_proxy_state(m13_state.get("affective_reward_proxy"))
    assessable_pending = list_assessable_pending_rows(reward, turn_index=turn_index)

    temporal = _mapping(state.get("temporal_state"))
    last_user_at = int(temporal.get("last_user_turn_at", 0) or temporal.get("last_turn_at", 0) or 0)
    elapsed_user = float(max(0, now - last_user_at)) if last_user_at > 0 else 0.0

    initiative = _mapping(_mapping(m13_state.get("initiative")))
    idle = _mapping(initiative.get("idle_introspection"))
    memory_efe = normalize_memory_efe_state(m13_state.get("memory_efe"))

    sig = (
        structural_signals.to_dict()
        if hasattr(structural_signals, "to_dict")
        else dict(structural_signals)
        if isinstance(structural_signals, Mapping)
        else {}
    )
    return {
        "turn_index": turn_index,
        "open_items": open_rows,
        "salient_unrecalled_memories": salient_memories,
        "uncertain_expectations": uncertain_expectations,
        "boredom": {
            "level": round(_bounded_float(boredom.get("boredom_level")), 4),
            "band": boredom_band(_bounded_float(boredom.get("boredom_level"))),
            "exploration_target": str(boredom.get("last_exploration_target", "") or "")[:120],
        },
        "affective_reward_proxy": {
            "path_feels_stale": bool(reward.get("path_feels_stale_proxy")),
            "caution_about_repeating_easy_path": bool(reward.get("caution_about_repeating_easy_path")),
            "reward_baseline": round(_bounded_float(reward.get("reward_baseline")), 4),
            "pending_settlements_count": len(assessable_pending),
        },
        "behavioral_pull_top": top_traction_actions(m13_state),
        "temporal": {
            "seconds_since_last_user_turn": round(elapsed_user, 1),
            "last_reply_summary": str(temporal.get("last_reply", ""))[:200],
        },
        "session_counters": {
            "reflection_count_this_session": int(idle.get("reflection_count_this_session", 0) or 0),
            "outreach_count_this_session": int(initiative.get("proactive_count_this_session", 0) or 0),
            "outreach_via_introspection_count": int(
                idle.get("outreach_via_introspection_count_this_session", 0) or 0
            ),
        },
        "memory_efe": {
            "phase": str(memory_efe.get("phase", "")),
            "eligible_count": len(memory_efe.get("eligible_for_efe", []) or []),
            "diagnostic_only_count": len(memory_efe.get("diagnostic_only", []) or []),
            "selected_policy": str(memory_efe.get("selected_policy", "")),
            "reply_angle_bias": str(memory_efe.get("reply_angle_bias", "none")),
            "should_outreach": bool(memory_efe.get("should_outreach")),
            "suppression_reasons": _string_list(memory_efe.get("suppression_reasons"), limit=6),
            "traceable_expectation_id": str(memory_efe.get("traceable_expectation_id", ""))[:120],
            "social_prediction_error": round(_bounded_float(memory_efe.get("social_prediction_error")), 4),
            "epistemic_prediction_error": round(_bounded_float(memory_efe.get("epistemic_prediction_error")), 4),
        },
        "structural_signals": sig,
        "engineering_proxy_label": M14_ENGINEERING_PROXY_LABEL,
    }


def idle_retrieval_keywords(idle_context: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in idle_context.get("open_items", []) or []:
        if isinstance(item, Mapping):
            terms.extend(_string_list(item.get("title"), limit=4))
            terms.extend(_string_list(item.get("next_check"), limit=4))
    boredom = _mapping(idle_context.get("boredom"))
    terms.extend(_string_list(boredom.get("exploration_target"), limit=4))
    for mem in idle_context.get("salient_unrecalled_memories", []) or []:
        if isinstance(mem, Mapping):
            terms.extend(_string_list(mem.get("summary"), limit=3))
    return _string_list(terms, limit=16)


def normalize_conscious_idle_plan(raw: Any) -> dict[str, Any]:
    empty = empty_conscious_idle_plan()
    if not isinstance(raw, Mapping):
        return empty
    if _looks_like_user_reply(raw):
        return empty

    focus_raw = raw.get("reflection_focus")
    focus: dict[str, Any] | None = None
    if isinstance(focus_raw, Mapping) and focus_raw:
        kind = str(focus_raw.get("reflection_kind", "none") or "none").strip()
        if kind not in _REFLECTION_KINDS:
            kind = "none"
        topic = str(focus_raw.get("topic", "") or "")[:120]
        refs = _string_list(focus_raw.get("evidence_refs"), limit=8)
        if kind != "none" or topic or refs:
            focus = {"topic": topic, "evidence_refs": refs, "reflection_kind": kind}

    patch_raw = raw.get("self_cognition_patch_proposal")
    patch = copy.deepcopy(empty["self_cognition_patch_proposal"])
    if isinstance(patch_raw, Mapping):
        patch["apply"] = bool(patch_raw.get("apply"))
        patch["summary_delta"] = str(patch_raw.get("summary_delta", "") or "")[:400]
        patch["new_identity_tensions"] = _string_list(patch_raw.get("new_identity_tensions"), limit=6)
        patch["new_known_limits"] = _string_list(patch_raw.get("new_known_limits"), limit=6)
        patch["evidence_refs"] = _string_list(patch_raw.get("evidence_refs"), limit=8)
        patch["confidence"] = round(_bounded_float(patch_raw.get("confidence")), 4)
        patch["reason"] = str(patch_raw.get("reason", "") or "")[:240]

    memory_props: list[dict[str, Any]] = []
    for row in raw.get("memory_consolidation_proposals", []) or []:
        if not isinstance(row, Mapping):
            continue
        target = str(row.get("target", "short_term") or "short_term")
        if target not in _MEMORY_TARGETS:
            continue
        kind = str(row.get("kind", "episode") or "episode")
        if kind not in _MEMORY_KINDS:
            continue
        content = str(row.get("content", "") or "")[:400]
        if not content.strip():
            continue
        memory_props.append(
            {
                "target": target,
                "kind": kind,
                "content": content,
                "confidence": round(_bounded_float(row.get("confidence")), 4),
                "evidence_refs": _string_list(row.get("evidence_refs"), limit=8),
            }
        )
        if len(memory_props) >= 5:
            break

    open_props: list[dict[str, Any]] = []
    for row in raw.get("open_item_proposals", []) or []:
        if not isinstance(row, Mapping):
            continue
        op = str(row.get("op", "update") or "update")
        if op not in _OPEN_ITEM_OPS:
            continue
        item_id = str(row.get("id", "") or "")[:64]
        if not item_id:
            continue
        open_props.append(
            {
                "op": op,
                "id": item_id,
                "rationale": str(row.get("rationale", "") or "")[:240],
            }
        )
        if len(open_props) >= 5:
            break

    outreach_raw = raw.get("outreach_recommendation")
    outreach = copy.deepcopy(empty["outreach_recommendation"])
    if isinstance(outreach_raw, Mapping):
        outreach["should_outreach"] = bool(outreach_raw.get("should_outreach"))
        reason = str(outreach_raw.get("reason", "reflection_only") or "reflection_only")
        outreach["reason"] = reason if reason in _OUTREACH_REASONS else "reflection_only"
        outreach["suggested_intent"] = str(outreach_raw.get("suggested_intent", "") or "")[:240]
        outreach["trigger"] = "reflection_outreach"

    hint = str(raw.get("thought_intensity_hint", "short") or "short").strip().lower()
    if hint not in {"short", "long", "none"}:
        hint = "short"

    plan = {
        "mode": "idle_introspection",
        "reflection_focus": focus,
        "self_cognition_patch_proposal": patch,
        "memory_consolidation_proposals": memory_props,
        "open_item_proposals": open_props,
        "outreach_recommendation": outreach,
        "thought_intensity_hint": hint,
    }
    violations = subjective_language_violations(json.dumps(plan, ensure_ascii=False))
    if violations:
        plan = empty_conscious_idle_plan()
        plan["outreach_recommendation"]["reason"] = "low_value"
    return plan


def empty_conscious_idle_plan() -> dict[str, Any]:
    return {
        "mode": "idle_introspection",
        "reflection_focus": None,
        "self_cognition_patch_proposal": {
            "apply": False,
            "summary_delta": "",
            "new_identity_tensions": [],
            "new_known_limits": [],
            "evidence_refs": [],
            "confidence": 0.0,
            "reason": "",
        },
        "memory_consolidation_proposals": [],
        "open_item_proposals": [],
        "outreach_recommendation": {
            "should_outreach": False,
            "reason": "reflection_only",
            "suggested_intent": "",
            "trigger": "reflection_outreach",
        },
        "thought_intensity_hint": "short",
    }


def build_structural_idle_plan(
    idle_context: Mapping[str, Any],
    *,
    retrieved_ids: set[str],
) -> dict[str, Any]:
    """Deterministic fallback when LLM unavailable or returns empty."""
    plan = empty_conscious_idle_plan()
    open_items = [row for row in idle_context.get("open_items", []) if isinstance(row, Mapping)]
    traceable_open_items = [row for row in open_items if is_traceable_open_item(row)]
    boredom = _mapping(idle_context.get("boredom"))
    reward = _mapping(idle_context.get("affective_reward_proxy"))
    band = str(boredom.get("band", "low"))

    focus_open_items = traceable_open_items or [
        row for row in open_items if str(row.get("id", "")) in retrieved_ids
    ]
    if focus_open_items:
        first = focus_open_items[0]
        item_id = str(first.get("id", ""))
        refs = [item_id] if item_id in retrieved_ids else [rid for rid in retrieved_ids][:1]
        if refs:
            plan["reflection_focus"] = {
                "topic": str(first.get("title", first.get("content", first.get("summary", ""))) or "")[:120],
                "evidence_refs": refs,
                "reflection_kind": "open_item",
            }
    elif band in {"medium", "high"} and bool(reward.get("path_feels_stale")):
        target = str(boredom.get("exploration_target", "") or "")[:120]
        refs = [rid for rid in retrieved_ids][:2]
        if target and refs:
            plan["reflection_focus"] = {
                "topic": target,
                "evidence_refs": refs,
                "reflection_kind": "habit_calibration",
            }

    if traceable_open_items and not bool(reward.get("path_feels_stale")):
        first = traceable_open_items[0]
        item_id = str(first.get("id", ""))
        if item_id in retrieved_ids and is_traceable_open_item(first):
            plan["outreach_recommendation"] = {
                "should_outreach": True,
                "reason": "open_item_followup",
                "suggested_intent": build_traceable_proactive_intent(first),
                "trigger": "reflection_outreach",
            }
    return plan


def apply_idle_drive_rules(
    plan: Mapping[str, Any],
    *,
    idle_context: Mapping[str, Any],
    structural_signals: Any,
) -> dict[str, Any]:
    """Deterministic post-process; not LLM hints."""
    merged = normalize_conscious_idle_plan(plan)
    boredom = _mapping(idle_context.get("boredom"))
    reward = _mapping(idle_context.get("affective_reward_proxy"))
    memory_efe = _mapping(idle_context.get("memory_efe"))
    band = str(boredom.get("band", "low"))
    outreach = dict(merged["outreach_recommendation"])
    hard_reflection_only = False

    if bool(reward.get("path_feels_stale")):
        outreach["should_outreach"] = False
        outreach["reason"] = "reflection_only"
        hard_reflection_only = True

    pending_count = int(reward.get("pending_settlements_count", 0) or 0)
    if pending_count > 0:
        outreach["should_outreach"] = False
        outreach["reason"] = "reflection_only"
        hard_reflection_only = True

    sig = (
        structural_signals.to_dict()
        if hasattr(structural_signals, "to_dict")
        else dict(structural_signals)
        if isinstance(structural_signals, Mapping)
        else {}
    )
    scheduled_owned_outreach = any(
        isinstance(row, Mapping) and str(row.get("kind", "")) == "scheduled_outreach"
        for row in sig.get("scheduled_intents", []) or []
    )
    if bool(sig.get("just_outreached_recently")):
        outreach["should_outreach"] = False
        outreach["reason"] = "already_recent"
        hard_reflection_only = True

    open_items = [row for row in idle_context.get("open_items", []) if isinstance(row, Mapping)]
    has_traceable_open_item = any(is_traceable_open_item(row) for row in open_items)
    if band == "low" and outreach.get("should_outreach") and not has_traceable_open_item:
        outreach["should_outreach"] = False
        outreach["reason"] = "low_value"

    memory_efe_allows_outreach = bool(memory_efe.get("should_outreach"))
    if outreach.get("should_outreach") and not memory_efe_allows_outreach and not scheduled_owned_outreach:
        outreach["should_outreach"] = False
        outreach["reason"] = "reflection_only"
    elif memory_efe_allows_outreach and not outreach.get("should_outreach") and not hard_reflection_only:
        outreach["should_outreach"] = True
        outreach["reason"] = "traceable_focus"
        trace_id = str(memory_efe.get("traceable_expectation_id", "") or "")[:120]
        intent = ""
        for row in open_items:
            if str(row.get("id", "")) == trace_id:
                intent = build_traceable_proactive_intent(row)
                break
        if not intent and not outreach.get("suggested_intent"):
            intent = f"Follow up on traceable expectation: {trace_id}" if trace_id else ""
        if intent:
            outreach["suggested_intent"] = intent

    focus = merged.get("reflection_focus")
    if isinstance(focus, Mapping) and band == "high":
        kind = str(focus.get("reflection_kind", ""))
        if kind not in {"open_item", "habit_calibration"}:
            focus = dict(focus)
            focus["reflection_kind"] = "habit_calibration"
            merged["reflection_focus"] = focus

    pull_top = idle_context.get("behavioral_pull_top", []) or []
    if isinstance(pull_top, list) and pull_top:
        top_score = 0.0
        if isinstance(pull_top[0], Mapping):
            top_score = _bounded_float(pull_top[0].get("traction"))
        if top_score < 0.12:
            merged["thought_intensity_hint"] = "short"
        elif outreach.get("should_outreach") and not outreach.get("suggested_intent"):
            action = str(pull_top[0].get("action", "answer"))
            outreach["suggested_intent"] = f"Offer a concise next step aligned with prior {action} thread."

    merged["outreach_recommendation"] = outreach
    return merged


def build_conscious_idle_prompt(
    *,
    idle_context: Mapping[str, Any],
    retrieved_memories: list[Mapping[str, Any]],
    turn_index: int,
    self_continuity_snapshot: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    system_prompt = f"""你是数字人格 MVP 路径的「{IDLE_INTROSPECTION_MARKER}」模块。
本轮没有用户输入；不要生成 reply、对话台词、独白或 Markdown。
只输出 JSON 形式的 conscious_idle_plan；使用中性工程语言，不要声称孤独、无聊感受、上瘾或主观需求。
每个 patch 提案必须引用本轮检索到的 evidence id（evidence_refs）。
若无法从材料中证明具体焦点，返回空计划（reflection_focus=null，各 apply=false/空列表）。
"""
    continuity_block = ""
    if self_continuity_snapshot:
        continuity_block = f"""
运营性自我基线快照 (self_continuity_snapshot):
{_json_text(dict(self_continuity_snapshot))}
"""
    user_prompt = f"""turn_index: {turn_index}
engineering_proxy_label: {M14_ENGINEERING_PROXY_LABEL}
{continuity_block}
空闲上下文摘要:
{_json_text(dict(idle_context))}

本轮检索证据卡:
{_json_text([dict(item) for item in retrieved_memories[:8]])}

请输出 JSON（字段名与类型必须匹配）:
{{
  "mode": "idle_introspection",
  "reflection_focus": {{"topic": "", "evidence_refs": [], "reflection_kind": "open_item|user_model|self_consistency|habit_calibration|none"}} ,
  "self_cognition_patch_proposal": {{
    "apply": false,
    "summary_delta": "",
    "new_identity_tensions": [],
    "new_known_limits": [],
    "evidence_refs": [],
    "confidence": 0.0,
    "reason": ""
  }},
  "memory_consolidation_proposals": [],
  "open_item_proposals": [],
  "outreach_recommendation": {{
    "should_outreach": false,
    "reason": "reflection_only|low_value|already_recent|open_item_followup|traceable_focus|none",
    "suggested_intent": "",
    "trigger": "reflection_outreach"
  }},
  "thought_intensity_hint": "short"
}}
"""
    return system_prompt, user_prompt


@dataclass(frozen=True)
class IdlePlanRunResult:
    plan: dict[str, Any]
    retrieved_ids: set[str]
    retrieved_memories: list[dict[str, Any]]
    ran_llm: bool
    llm_error: str = ""
