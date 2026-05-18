"""MVP-local M13.0 behavioral pull for the UI chat path.

Technical debt: M13 is MVP-local and intentionally parallel to Path A
drive/process-valence surfaces in ``segmentum/drives.py``. Future architecture
work should reconcile names and state ownership rather than treating both as
independent permanent systems.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import re
import uuid
from typing import Any, Mapping

M13_STATE_VERSION = 1

CANDIDATE_REPLY_ACTIONS: tuple[str, ...] = (
    "answer",
    "ask_question",
    "empathize",
    "clarify",
    "disagree",
    "deflect",
    "self_disclose",
    "abstract_share",
    "truthful_refusal",
)

DISCLOSURE_TO_REPLY_ACTIONS: dict[str, frozenset[str]] = {
    "direct_share": frozenset({"answer", "ask_question", "empathize", "clarify", "disagree", "self_disclose"}),
    "abstract_share": frozenset({"abstract_share", "deflect", "answer", "clarify"}),
    "truthful_refusal": frozenset({"truthful_refusal", "clarify", "deflect"}),
    "deflect": frozenset({"deflect", "clarify", "abstract_share"}),
    "deny_knowledge": frozenset({"clarify", "deflect", "truthful_refusal", "ask_question"}),
}

# Bounded state limits (M13.0 contract).
MAX_RECENT_ACTION_TRACE = 30
MAX_PATH_PATTERNS = 200
MAX_RECENT_TOPIC_FINGERPRINTS = 24
MAX_PENDING_SETTLEMENTS = 8
MAX_TOLERANCE_BY_PATH = 64
MAX_SOURCE_EVIDENCE_IDS = 8

# Proxy and update constraints.
MINIMUM_SUCCESS_PROXY_CONFIDENCE = 0.60
MAX_HABIT_PRECISION_DELTA = 0.05
MAX_CONTROL_DISCOUNT_DELTA = 0.04
PATTERN_DECAY_PER_TURN = 0.002
SETTLEMENT_ROLLBACK_PREDICTION_ERROR_THRESHOLD = 0.35


class M13PullWeights:
    """Single place for behavioral-pull weights; avoid magic numbers in mvp_loop."""

    # Habit evidence from prior successful paths for this user/action.
    habit: float = 0.28
    # Topic cue overlap with conscious/recall terms.
    cue: float = 0.18
    # Relationship-specific path precision (per user_id).
    relation: float = 0.16
    # Retrieved memory support for the topic/action class.
    memory: float = 0.14
    # Recent same-turn / short-horizon success proxy.
    success: float = 0.14
    # Penalty for repeating the same action too often in trace.
    repetition: float = 0.12
    # Learned control-cost discount (higher => easier to reuse path).
    control_discount: float = 0.10


M13_WEIGHTS = M13PullWeights()


class M13Thresholds:
    """Non-weight tuning constants; keep magic numbers out of call sites."""

    preferred_band: float = 0.08
    discouraged_gap: float = 0.05
    weak_pull_margin: float = 0.04
    repetition_penalty_high: float = 0.15
    repetition_step: float = 0.15
    repetition_trace_window: int = 6
    cue_overlap_base: float = 0.2
    cue_overlap_scale: float = 0.6
    memory_hit_cap: int = 3
    success_proxy_confidence_base: float = 0.60
    success_proxy_confidence_step: float = 0.10
    success_proxy_confidence_cap: float = 0.95
    uncertain_confidence: float = 0.55
    weak_uncertain_confidence: float = 0.45
    min_pull_margin_for_action_match: float = 0.03
    min_habit_for_action_match: float = 0.05
    cue_map_from_habit_scale: float = 0.5
    relation_map_from_habit_scale: float = 0.4


M13_THRESHOLDS = M13Thresholds()

# Memory kinds that better support specific reply actions.
_ACTION_MEMORY_KINDS: dict[str, frozenset[str]] = {
    "answer": frozenset({"fact", "preference", "episode", "open_item"}),
    "ask_question": frozenset({"open_item", "episode", "fact"}),
    "empathize": frozenset({"relationship", "episode"}),
    "clarify": frozenset({"fact", "open_item", "episode"}),
    "disagree": frozenset({"fact", "preference"}),
    "deflect": frozenset({"episode", "relationship"}),
    "self_disclose": frozenset({"identity", "preference", "episode"}),
    "abstract_share": frozenset({"fact", "preference", "relationship"}),
    "truthful_refusal": frozenset({"fact", "identity"}),
}

_TOPIC_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "的", "了", "吗",
    "呢", "啊", "吧", "在", "是", "有", "和", "就", "也", "都", "还", "很",
    "不", "没", "这", "那", "什么", "怎么", "为什么", "可以", "会", "要",
})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()[:240]]
    if isinstance(value, list):
        return [str(item).strip()[:240] for item in value[:limit] if str(item).strip()]
    return []


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_m13_drive_state() -> dict[str, Any]:
    from segmentum.dialogue.runtime.m13_boredom import default_boredom_state
    from segmentum.dialogue.runtime.m13_initiative import default_initiative_state
    from segmentum.dialogue.runtime.m13_reward import default_affective_reward_proxy_state

    return copy.deepcopy(
        {
            "version": M13_STATE_VERSION,
            "path_patterns_by_action": [],
            "recent_action_trace": [],
            "traction_by_action": {},
            "cue_precision_by_topic": {},
            "relation_path_precision": {},
            "recent_topic_fingerprints": [],
            "pending_settlements": [],
            "tolerance_by_path": [],
            "last_patch_id": "",
            "rollback_window": [],
            "boredom": default_boredom_state(),
            "affective_reward_proxy": default_affective_reward_proxy_state(),
            "initiative": default_initiative_state(),
        }
    )


def normalize_m13_drive_state(raw: Any) -> dict[str, Any]:
    base = default_m13_drive_state()
    if not isinstance(raw, Mapping):
        return base
    merged = {**base, **dict(raw)}
    merged["version"] = int(merged.get("version", M13_STATE_VERSION) or M13_STATE_VERSION)
    for key in (
        "path_patterns_by_action",
        "recent_action_trace",
        "recent_topic_fingerprints",
        "pending_settlements",
        "tolerance_by_path",
        "rollback_window",
    ):
        rows = merged.get(key)
        merged[key] = [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []
    for key in ("traction_by_action", "cue_precision_by_topic", "relation_path_precision"):
        value = merged.get(key)
        merged[key] = dict(value) if isinstance(value, Mapping) else {}
    merged["last_patch_id"] = str(merged.get("last_patch_id", "") or "")
    from segmentum.dialogue.runtime.m13_boredom import normalize_boredom_state
    from segmentum.dialogue.runtime.m13_initiative import normalize_initiative_state
    from segmentum.dialogue.runtime.m13_reward import normalize_affective_reward_proxy_state

    merged["boredom"] = normalize_boredom_state(merged.get("boredom"))
    merged["initiative"] = normalize_initiative_state(merged.get("initiative"))
    reward_proxy = normalize_affective_reward_proxy_state(merged.get("affective_reward_proxy"))
    legacy_pending = merged.get("pending_settlements") or []
    legacy_tolerance = merged.get("tolerance_by_path") or []
    if legacy_pending and not reward_proxy.get("pending_settlements"):
        reward_proxy["pending_settlements"] = [
            dict(item) for item in legacy_pending if isinstance(item, Mapping)
        ][-MAX_PENDING_SETTLEMENTS:]
    if legacy_tolerance and not reward_proxy.get("tolerance_by_path"):
        reward_proxy["tolerance_by_path"] = [
            dict(item) for item in legacy_tolerance if isinstance(item, Mapping)
        ][-MAX_TOLERANCE_BY_PATH:]
    merged["affective_reward_proxy"] = reward_proxy
    merged["pending_settlements"] = []
    merged["tolerance_by_path"] = []
    return copy.deepcopy(merged)


def prompt_safe_m13_state_summary(
    m13_state: Mapping[str, Any] | None,
    *,
    user_id: str = "",
) -> dict[str, Any]:
    """Compact summary for thinking prompts; never expose raw pattern tables."""
    normalized = normalize_m13_drive_state(m13_state)
    traction = _mapping(normalized.get("traction_by_action"))
    summaries: list[str] = []
    suffix = f"|{user_id}" if user_id else ""
    for key, value in traction.items():
        if suffix and not str(key).endswith(suffix):
            continue
        summaries.append(f"{key}: {_bounded_float(value):.2f}")
    return {
        "version": normalized.get("version"),
        "last_patch_id": str(normalized.get("last_patch_id", "") or ""),
        "traction_summary_for_user": summaries[:6],
        "recent_trace_count": len(normalized.get("recent_action_trace", []) or []),
        "visible_policy": (
            "Use control_guidance.drive_guidance for reply tendencies; "
            "do not treat internal habit scores as facts or compulsion."
        ),
    }


def normalize_recorded_reply_action(
    action: str,
    *,
    allowed: set[str] | None = None,
) -> str:
    """Map LLM reply_action to a known candidate before persisting M13 patterns."""
    cleaned = str(action or "").strip().lower()
    if cleaned in CANDIDATE_REPLY_ACTIONS and (allowed is None or cleaned in allowed):
        return cleaned
    if allowed:
        for fallback in ("clarify", "deflect", "answer"):
            if fallback in allowed:
                return fallback
        return sorted(allowed)[0]
    return "answer"


def _normalize_term(term: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff+#.-]+", "", str(term or "").casefold().strip())
    return cleaned[:80]


def _rough_user_terms(user_text: str, *, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", str(user_text or ""))
    return [_normalize_term(token) for token in tokens[:limit] if _normalize_term(token)]


def build_topic_fingerprint(
    *,
    conscious_plan: Mapping[str, Any] | None = None,
    memory_dynamics: Mapping[str, Any] | None = None,
    entity_binding: Mapping[str, Any] | None = None,
    retrieved_memories: list[Mapping[str, Any]] | None = None,
    user_text: str = "",
) -> str:
    """Deterministic topic key shared by all M13 milestones; no LLM."""
    sources: list[list[str]] = []
    conscious = _mapping(conscious_plan)
    dynamics = _mapping(memory_dynamics)
    binding = _mapping(entity_binding)
    recall_query = _mapping(dynamics.get("recall_query"))

    sources.append(_string_list(conscious.get("memory_search_keywords"), limit=16))
    sources.append(_string_list(recall_query.get("semantic_terms"), limit=16))
    target = str(binding.get("target_person", "")).strip()
    if target:
        sources.append([target])
    pronoun_bindings = _mapping(binding.get("pronoun_bindings"))
    sources.append([str(v).strip() for v in pronoun_bindings.values() if str(v).strip()][:8])
    memory_terms: list[str] = []
    for item in retrieved_memories or []:
        if not isinstance(item, Mapping):
            continue
        memory_terms.extend(_string_list(item.get("keywords"), limit=4))
    sources.append(memory_terms[:12])
    sources.append(_rough_user_terms(user_text, limit=8))

    seen: set[str] = set()
    ordered: list[str] = []
    for group in sources:
        for raw in group:
            term = _normalize_term(raw)
            if not term or term in _TOPIC_STOPWORDS or term in seen:
                continue
            seen.add(term)
            ordered.append(term)
            if len(ordered) >= 8:
                return "|".join(ordered)
    return "|".join(ordered) if ordered else "topic:unknown"


def _pattern_lookup(
    patterns: list[dict[str, Any]],
    *,
    action: str,
    user_id: str,
    topic_fingerprint: str = "",
    allow_topic_fallback: bool = False,
) -> dict[str, Any] | None:
    topic = str(topic_fingerprint or "").strip()
    fallback: dict[str, Any] | None = None
    for row in reversed(patterns):
        if str(row.get("action", "")) != action or str(row.get("user_id", "")) != user_id:
            continue
        row_topic = str(row.get("topic_fingerprint", "")).strip()
        if topic and row_topic == topic:
            return row
        if allow_topic_fallback and fallback is None:
            fallback = row
    if allow_topic_fallback:
        return fallback
    return None


def _traction_key(action: str, user_id: str) -> str:
    return f"{action}|{user_id}"


def collect_allowed_reply_actions(
    *,
    evidence_judgment: Mapping[str, Any] | None = None,
    memory_dynamics: Mapping[str, Any] | None = None,
) -> set[str]:
    """Actions permitted after safety, evidence, and sharing-policy gates."""
    allowed = set(CANDIDATE_REPLY_ACTIONS)
    evidence = _mapping(evidence_judgment)
    disclosure_allowed = _string_list(evidence.get("allowed_reply_actions"), limit=8)
    if disclosure_allowed:
        from_disclosure: set[str] = set()
        for disclosure in disclosure_allowed:
            mapped = DISCLOSURE_TO_REPLY_ACTIONS.get(disclosure)
            if mapped:
                from_disclosure.update(mapped)
        if from_disclosure:
            allowed &= from_disclosure
    dynamics = _mapping(memory_dynamics)
    control = _mapping(dynamics.get("control_guidance"))
    sharing = _mapping(control.get("sharing_policy"))
    contract = _mapping(control.get("reply_contract"))
    if not bool(sharing.get("allow_direct_disclosure", True)):
        allowed -= {"self_disclose", "answer", "empathize", "disagree"}
        allowed.add("abstract_share")
        allowed.add("deflect")
        allowed.add("clarify")
    if not bool(sharing.get("allow_abstract_sharing", True)):
        allowed -= {"abstract_share"}
    if bool(contract.get("deny_identity_anchored_action")):
        allowed -= {"self_disclose", "disagree"}
    if bool(contract.get("prefer_clarification")):
        if "clarify" in allowed:
            pass
        else:
            allowed.add("clarify")
    if bool(contract.get("enforce_identity_verification")):
        allowed &= {"clarify", "ask_question", "deflect", "truthful_refusal", "abstract_share"}
    if not allowed:
        allowed = {"clarify", "deflect"}
    return allowed


def _habit_precision_for_action(
    patterns: list[dict[str, Any]],
    *,
    action: str,
    user_id: str,
    topic_fingerprint: str,
) -> float:
    row = _pattern_lookup(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic_fingerprint,
    )
    if not row:
        return 0.0
    return _bounded_float(row.get("habit_precision"), default=0.0)


def _cue_precision(
    state: Mapping[str, Any],
    *,
    topic_fingerprint: str,
    conscious_plan: Mapping[str, Any],
) -> float:
    by_topic = _mapping(state.get("cue_precision_by_topic"))
    if topic_fingerprint in by_topic:
        return _bounded_float(by_topic[topic_fingerprint], default=0.0)
    keywords = {_normalize_term(t) for t in _string_list(conscious_plan.get("memory_search_keywords"), limit=12)}
    topic_terms = {t for t in topic_fingerprint.split("|") if t and t != "topic:unknown"}
    if not keywords or not topic_terms:
        return 0.0
    overlap = len(keywords & topic_terms) / max(1, len(keywords))
    return _bounded_float(
        M13_THRESHOLDS.cue_overlap_base + overlap * M13_THRESHOLDS.cue_overlap_scale
    )


def _relation_precision(state: Mapping[str, Any], *, user_id: str) -> float:
    rel = _mapping(state.get("relation_path_precision"))
    if user_id in rel:
        return _bounded_float(rel[user_id], default=0.0)
    return 0.0


def _relationship_action_bias(
    relationship_value_context: Mapping[str, Any] | None,
    *,
    action: str,
) -> float:
    ctx = _mapping(relationship_value_context)
    active = ctx.get("active_relationship_value_memories", [])
    if not isinstance(active, list) or not active:
        return 0.0
    if action not in {"empathize", "clarify", "ask_question"}:
        return 0.0
    return _bounded_float(0.1 + 0.03 * min(3, len(active)))


def _memory_support(
    retrieved_memories: list[Mapping[str, Any]],
    *,
    topic_fingerprint: str,
    action: str,
) -> float:
    topic_terms = {t for t in topic_fingerprint.split("|") if t and t != "topic:unknown"}
    if not retrieved_memories:
        return 0.0
    allowed_kinds = _ACTION_MEMORY_KINDS.get(action, frozenset())
    weighted_hits = 0.0
    for item in retrieved_memories:
        if not isinstance(item, Mapping):
            continue
        keywords = {_normalize_term(k) for k in _string_list(item.get("keywords"), limit=8)}
        content = _normalize_term(str(item.get("content", ""))[:120])
        if not topic_terms or not (keywords & topic_terms or any(t in content for t in topic_terms)):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        weighted_hits += 1.0 if not allowed_kinds or kind in allowed_kinds else 0.65
    cap = M13_THRESHOLDS.memory_hit_cap
    return _bounded_float(min(1.0, weighted_hits / max(1, min(cap, len(retrieved_memories)))))


def _recent_success_proxy(pattern: dict[str, Any] | None) -> float:
    if not pattern:
        return 0.0
    support = max(1, int(pattern.get("support_count", 0) or 0))
    success = int(pattern.get("success_proxy_count", 0) or 0)
    return _bounded_float(success / support)


def _repetition_penalty(
    trace: list[Mapping[str, Any]],
    *,
    action: str,
    user_id: str,
    window: int = M13_THRESHOLDS.repetition_trace_window,
) -> float:
    recent = [row for row in trace[-window:] if isinstance(row, Mapping)]
    same = sum(
        1
        for row in recent
        if str(row.get("action", "")) == action and str(row.get("user_id", "")) == user_id
    )
    if same <= 1:
        return 0.0
    return _bounded_float(M13_THRESHOLDS.repetition_step * (same - 1))


def _control_cost_discount(pattern: dict[str, Any] | None) -> float:
    if not pattern:
        return 0.0
    return _bounded_float(pattern.get("mean_control_cost_discount"), default=0.0)


def _score_candidate(
    *,
    action: str,
    user_id: str,
    m13_state: Mapping[str, Any],
    conscious_plan: Mapping[str, Any],
    topic_fingerprint: str,
    retrieved_memories: list[Mapping[str, Any]],
    habit_traits: Mapping[str, Any] | None = None,
    relationship_value_context: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    patterns = [
        dict(row)
        for row in m13_state.get("path_patterns_by_action", [])
        if isinstance(row, Mapping)
    ]
    pattern = _pattern_lookup(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic_fingerprint,
    )
    trace = [
        row for row in m13_state.get("recent_action_trace", []) if isinstance(row, Mapping)
    ]
    habit_precision = _habit_precision_for_action(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic_fingerprint,
    )
    relation_precision = max(
        _relation_precision(m13_state, user_id=user_id),
        _relationship_action_bias(relationship_value_context, action=action),
    )
    components = {
        "habit_precision": habit_precision,
        "cue_precision": _cue_precision(m13_state, topic_fingerprint=topic_fingerprint, conscious_plan=conscious_plan),
        "relation_precision": relation_precision,
        "memory_support": _memory_support(retrieved_memories, topic_fingerprint=topic_fingerprint, action=action),
        "recent_success_proxy": _recent_success_proxy(pattern),
        "repetition_penalty": _repetition_penalty(trace, action=action, user_id=user_id),
        "control_cost_discount": _control_cost_discount(pattern),
    }
    pull = (
        components["habit_precision"] * M13_WEIGHTS.habit
        + components["cue_precision"] * M13_WEIGHTS.cue
        + components["relation_precision"] * M13_WEIGHTS.relation
        + components["memory_support"] * M13_WEIGHTS.memory
        + components["recent_success_proxy"] * M13_WEIGHTS.success
        - components["repetition_penalty"] * M13_WEIGHTS.repetition
        + components["control_cost_discount"] * M13_WEIGHTS.control_discount
    )
    components["behavioral_pull"] = round(_bounded_float(pull), 6)
    return components


def _prompt_safe_summary(top_action: str, margin: float, *, advisory: bool = True) -> str:
    del margin
    if not top_action:
        return "No strong path habit; follow current user need and evidence."
    lead = (
        "A recently useful response pattern is available, but do not repeat it mechanically "
        "if the current user need differs."
    )
    if advisory:
        return f"{lead} A mild tendency favors {top_action.replace('_', ' ')} when it still fits the user need."
    return lead


@dataclass
class M13EvaluationResult:
    event_id: str
    turn_id: str
    turn_index: int
    topic_fingerprint: str
    candidate_actions: list[str]
    selected_action: str
    top_behavioral_pull_action: str
    pull_margin: float
    scores_by_action: dict[str, dict[str, float]]
    evidence_refs: list[str]
    prompt_safe_summary: str
    preferred_reply_actions: list[str]
    discouraged_reply_actions: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)


class M13DriveEvaluator:
    """Deterministic behavioral-pull evaluator for the MVP chat path."""

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
        response_style_prior: Mapping[str, Any] | None,
        habit_traits: Mapping[str, Any],
        relationship_value_context: Mapping[str, Any] | None,
        m13_state: Mapping[str, Any],
        entity_binding: Mapping[str, Any] | None = None,
        evidence_judgment: Mapping[str, Any] | None = None,
    ) -> M13EvaluationResult:
        del response_style_prior
        topic = build_topic_fingerprint(
            conscious_plan=conscious_plan,
            memory_dynamics=memory_dynamics,
            entity_binding=entity_binding,
            retrieved_memories=retrieved_memories,
            user_text=user_text,
        )
        allowed = collect_allowed_reply_actions(
            evidence_judgment=evidence_judgment,
            memory_dynamics=memory_dynamics,
        )
        candidates = [action for action in CANDIDATE_REPLY_ACTIONS if action in allowed]
        if not candidates:
            candidates = ["clarify", "deflect"]

        scores: dict[str, dict[str, float]] = {}
        for action in candidates:
            scores[action] = _score_candidate(
                action=action,
                user_id=user_id,
                m13_state=m13_state,
                conscious_plan=conscious_plan,
                topic_fingerprint=topic,
                retrieved_memories=retrieved_memories,
                habit_traits=habit_traits,
                relationship_value_context=relationship_value_context,
            )

        ranked = sorted(
            candidates,
            key=lambda action: scores[action]["behavioral_pull"],
            reverse=True,
        )
        top_action = ranked[0] if ranked else "answer"
        second_pull = scores[ranked[1]]["behavioral_pull"] if len(ranked) > 1 else 0.0
        margin = round(scores[top_action]["behavioral_pull"] - second_pull, 6)

        preferred: list[str] = []
        discouraged: list[str] = []
        threshold = scores[top_action]["behavioral_pull"] - M13_THRESHOLDS.preferred_band
        for action in ranked[:3]:
            if scores[action]["behavioral_pull"] >= threshold:
                preferred.append(action)
        for action in reversed(ranked):
            if (
                scores[action]["behavioral_pull"] < threshold - M13_THRESHOLDS.discouraged_gap
                and action not in preferred
            ):
                discouraged.append(action)
            if len(discouraged) >= 2:
                break

        preferred = [action for action in preferred if action in allowed][:3]
        discouraged = [action for action in discouraged if action in allowed and action not in preferred][:2]

        evidence_refs = _string_list(
            [str(item.get("id", "")) for item in retrieved_memories if item.get("id")],
            limit=8,
        )
        event_id = _new_id("m13_eval")
        eval_event = {
            "type": "M13DriveEvaluationEvent",
            "event_id": event_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "source": "m13_drive_evaluator",
            "candidate_actions": candidates,
            "top_behavioral_pull_action": top_action,
            "selected_action": top_action,
            "pull_margin": margin,
            "topic_fingerprint": topic,
            "evidence_refs": evidence_refs,
            "prompt_safe_summary": _prompt_safe_summary(top_action, margin),
            "scores_summary": {
                "top": top_action,
                "runner_up": ranked[1] if len(ranked) > 1 else "",
            },
        }
        return M13EvaluationResult(
            event_id=event_id,
            turn_id=turn_id,
            turn_index=turn_index,
            topic_fingerprint=topic,
            candidate_actions=candidates,
            selected_action=top_action,
            top_behavioral_pull_action=top_action,
            pull_margin=margin,
            scores_by_action=scores,
            evidence_refs=evidence_refs,
            prompt_safe_summary=eval_event["prompt_safe_summary"],
            preferred_reply_actions=preferred,
            discouraged_reply_actions=discouraged,
            events=[eval_event],
        )


def merge_drive_guidance_into_control(
    memory_dynamics: dict[str, Any],
    evaluation: M13EvaluationResult,
    *,
    evidence_judgment: Mapping[str, Any] | None = None,
    boredom_evaluation: Any | None = None,
) -> None:
    """Patch prompt-safe drive_guidance; intersect with allowed actions."""
    allowed = collect_allowed_reply_actions(
        evidence_judgment=evidence_judgment,
        memory_dynamics=memory_dynamics,
    )
    preferred = [action for action in evaluation.preferred_reply_actions if action in allowed]
    discouraged = [action for action in evaluation.discouraged_reply_actions if action in allowed]
    caution = ""
    if evaluation.pull_margin < M13_THRESHOLDS.weak_pull_margin:
        caution = "Path habits are weak; prioritize the live user request over style inertia."
    repetition_high = any(
        evaluation.scores_by_action.get(action, {}).get("repetition_penalty", 0.0)
        >= M13_THRESHOLDS.repetition_penalty_high
        for action in preferred
    )
    if repetition_high:
        caution = (
            "A similar reply pattern was used recently; vary approach if the user's need changed."
        )
    control = _mapping(memory_dynamics.get("control_guidance"))
    drive_guidance = {
        "preferred_reply_actions": preferred,
        "discouraged_reply_actions": discouraged,
        "action_tendency_reason": evaluation.prompt_safe_summary,
        "caution": caution,
        "ordinary_language_policy": (
            "Treat these as response tendencies, not obligations or visible inner states."
        ),
        "advisory_only": True,
    }
    if boredom_evaluation is not None:
        from segmentum.dialogue.runtime.m13_boredom import merge_exploration_guidance_into_control

        merge_exploration_guidance_into_control(
            memory_dynamics,
            boredom_evaluation,
            drive_guidance=drive_guidance,
        )
        return
    control["drive_guidance"] = drive_guidance
    memory_dynamics["control_guidance"] = control


def resolve_m13_safety_repair(
    *,
    reply_validation: Mapping[str, Any] | None = None,
    post_reply_observer: Mapping[str, Any] | None = None,
) -> bool:
    """Same repair signals as rollback, excluding expectation violations (handled separately)."""
    validation = _mapping(reply_validation)
    observer = _mapping(post_reply_observer)
    if bool(validation.get("changed")):
        return True
    followup = str(observer.get("followup_type", "")).strip().lower()
    if bool(observer.get("needs_followup")) and followup not in {"", "none"}:
        return True
    return str(observer.get("self_correction_requested", "")).strip().lower() in {"true", "yes", "1"}


def should_trigger_m13_rollback(
    *,
    reply_validation: Mapping[str, Any] | None = None,
    post_reply_observer: Mapping[str, Any] | None = None,
    conscious_plan: Mapping[str, Any] | None = None,
    safety_repair: bool = False,
) -> tuple[bool, str]:
    """Single trigger surface; M13.2 next-turn settlement can replace internals."""
    validation = _mapping(reply_validation)
    observer = _mapping(post_reply_observer)
    if safety_repair:
        return True, "rollback_safety_repair"
    if bool(validation.get("changed")):
        return True, "rollback_reply_validation_rewrite"
    followup = str(observer.get("followup_type", "")).strip().lower()
    if bool(observer.get("needs_followup")) and followup not in {"", "none"}:
        return True, "rollback_observer_repair"
    if str(observer.get("self_correction_requested", "")).strip().lower() in {"true", "yes", "1"}:
        return True, "rollback_observer_repair"
    for item in _mapping(conscious_plan).get("expectation_results", []) or []:
        if isinstance(item, Mapping) and str(item.get("status", "")) == "violated":
            return True, "rollback_expectation_violated"
    return False, ""


def _apply_pattern_decay(patterns: list[dict[str, Any]]) -> None:
    for row in patterns:
        hp = _bounded_float(row.get("habit_precision"), default=0.0)
        row["habit_precision"] = round(max(0.0, hp - PATTERN_DECAY_PER_TURN), 6)
        discount = _bounded_float(row.get("mean_control_cost_discount"), default=0.0)
        row["mean_control_cost_discount"] = round(max(0.0, discount - PATTERN_DECAY_PER_TURN * 0.5), 6)


def _upsert_pattern(
    patterns: list[dict[str, Any]],
    *,
    action: str,
    user_id: str,
    topic_fingerprint: str,
    turn_index: int,
    evidence_id: str,
) -> dict[str, Any]:
    row = _pattern_lookup(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic_fingerprint,
    )
    if (
        row is not None
        and topic_fingerprint
        and str(row.get("topic_fingerprint", "")).strip() not in {"", topic_fingerprint}
    ):
        row = None
    if row is None:
        row = {
            "action": action,
            "user_id": user_id,
            "topic_fingerprint": topic_fingerprint,
            "support_count": 0,
            "success_proxy_count": 0,
            "failure_proxy_count": 0,
            "habit_precision": 0.0,
            "mean_control_cost_discount": 0.0,
            "last_seen_turn": turn_index,
            "source_evidence_ids": [],
            "status": "active",
        }
        patterns.append(row)
    row["support_count"] = int(row.get("support_count", 0) or 0) + 1
    row["last_seen_turn"] = turn_index
    row["topic_fingerprint"] = topic_fingerprint
    refs = _string_list(row.get("source_evidence_ids"), limit=MAX_SOURCE_EVIDENCE_IDS)
    if evidence_id and evidence_id not in refs:
        refs.append(evidence_id)
    row["source_evidence_ids"] = refs[-MAX_SOURCE_EVIDENCE_IDS:]
    return row


def evict_path_patterns(patterns: list[dict[str, Any]]) -> None:
    """Drop oldest patterns when over MAX_PATH_PATTERNS (M13.0 state bound)."""
    if len(patterns) <= MAX_PATH_PATTERNS:
        return
    patterns.sort(key=lambda row: int(row.get("last_seen_turn", 0) or 0))
    del patterns[: len(patterns) - MAX_PATH_PATTERNS]


def _evict_patterns(patterns: list[dict[str, Any]]) -> None:
    evict_path_patterns(patterns)


def prompt_safe_m13_turn_diagnostics(evaluation: M13EvaluationResult) -> dict[str, Any]:
    """Conversation-log safe summary; omit internal score field names."""
    return {
        "event_id": evaluation.event_id,
        "topic_fingerprint": evaluation.topic_fingerprint,
        "preferred_reply_actions": list(evaluation.preferred_reply_actions),
        "discouraged_reply_actions": list(evaluation.discouraged_reply_actions),
        "prompt_safe_summary": evaluation.prompt_safe_summary,
        "advisory_only": True,
    }


def _record_topic_fingerprint(state: dict[str, Any], topic: str, *, turn_index: int) -> None:
    rows = state.setdefault("recent_topic_fingerprints", [])
    if not isinstance(rows, list):
        rows = []
        state["recent_topic_fingerprints"] = rows
    rows.append({"topic": topic, "turn_index": turn_index})
    state["recent_topic_fingerprints"] = rows[-MAX_RECENT_TOPIC_FINGERPRINTS:]


def _append_trace(
    state: dict[str, Any],
    *,
    turn_id: str,
    turn_index: int,
    user_id: str,
    action: str,
    topic_fingerprint: str,
    outcome_band: str,
) -> None:
    trace = state.setdefault("recent_action_trace", [])
    if not isinstance(trace, list):
        trace = []
        state["recent_action_trace"] = trace
    trace.append(
        {
            "turn_id": turn_id,
            "turn_index": turn_index,
            "user_id": user_id,
            "action": action,
            "topic_fingerprint": topic_fingerprint,
            "outcome_band": outcome_band,
        }
    )
    state["recent_action_trace"] = trace[-MAX_RECENT_ACTION_TRACE:]


def _build_proxy_outcome(
    *,
    reply_validation: Mapping[str, Any],
    post_reply_observer: Mapping[str, Any],
    conscious_plan: Mapping[str, Any],
    selected_action: str,
    top_pull_action: str,
    memory_candidates_applied: list[Any],
    pull_margin: float = 0.0,
    pattern: Mapping[str, Any] | None = None,
    safety_repair: bool = False,
) -> tuple[str, float]:
    rollback, _ = should_trigger_m13_rollback(
        reply_validation=reply_validation,
        post_reply_observer=post_reply_observer,
        conscious_plan=conscious_plan,
        safety_repair=safety_repair,
    )
    if rollback:
        return "negative", 0.85
    validation = _mapping(reply_validation)
    pattern_row = _mapping(pattern)
    habit_on_topic = _bounded_float(pattern_row.get("habit_precision"), default=0.0)
    positive_signals = 0
    if not bool(validation.get("changed")):
        positive_signals += 1
    if memory_candidates_applied:
        positive_signals += 1
    action_match = (
        selected_action == top_pull_action
        and bool(top_pull_action)
        and (
            pull_margin >= M13_THRESHOLDS.min_pull_margin_for_action_match
            or habit_on_topic >= M13_THRESHOLDS.min_habit_for_action_match
        )
    )
    if action_match:
        positive_signals += 1
    confirmed = [
        item
        for item in _mapping(conscious_plan).get("expectation_results", []) or []
        if isinstance(item, Mapping) and str(item.get("status", "")) == "confirmed"
    ]
    if confirmed:
        positive_signals += 1
    anchor_signal = bool(memory_candidates_applied) or bool(confirmed)
    if positive_signals >= 2 and anchor_signal:
        return (
            "positive",
            min(
                M13_THRESHOLDS.success_proxy_confidence_cap,
                M13_THRESHOLDS.success_proxy_confidence_base
                + M13_THRESHOLDS.success_proxy_confidence_step * positive_signals,
            ),
        )
    if positive_signals >= 1:
        return "uncertain", M13_THRESHOLDS.uncertain_confidence
    return "uncertain", M13_THRESHOLDS.weak_uncertain_confidence


def apply_post_turn_m13_state(
    m13_state: dict[str, Any],
    *,
    evaluation: M13EvaluationResult,
    user_id: str,
    turn_id: str,
    turn_index: int,
    selected_action: str,
    reply_validation: Mapping[str, Any],
    post_reply_observer: Mapping[str, Any],
    conscious_plan: Mapping[str, Any],
    memory_candidates_applied: list[Any],
    safety_repair: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply bounded post-turn patches; returns updated state and audit events."""
    state = normalize_m13_drive_state(m13_state)
    events: list[dict[str, Any]] = []
    patterns = [dict(row) for row in state.get("path_patterns_by_action", []) if isinstance(row, Mapping)]
    _apply_pattern_decay(patterns)

    action = normalize_recorded_reply_action(
        selected_action,
        allowed=set(evaluation.candidate_actions),
    )
    topic = evaluation.topic_fingerprint
    existing_pattern = _pattern_lookup(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic,
        allow_topic_fallback=False,
    )

    outcome_band, confidence = _build_proxy_outcome(
        reply_validation=reply_validation,
        post_reply_observer=post_reply_observer,
        conscious_plan=conscious_plan,
        selected_action=action,
        top_pull_action=evaluation.top_behavioral_pull_action,
        memory_candidates_applied=memory_candidates_applied,
        pull_margin=evaluation.pull_margin,
        pattern=existing_pattern,
        safety_repair=safety_repair,
    )
    evidence_id = evaluation.evidence_refs[0] if evaluation.evidence_refs else ""
    pattern = _upsert_pattern(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic,
        turn_index=turn_index,
        evidence_id=evidence_id,
    )

    rollback, rollback_reason = should_trigger_m13_rollback(
        reply_validation=reply_validation,
        post_reply_observer=post_reply_observer,
        conscious_plan=conscious_plan,
        safety_repair=safety_repair,
    )
    previous_hp = _bounded_float(pattern.get("habit_precision"), default=0.0)
    previous_discount = _bounded_float(pattern.get("mean_control_cost_discount"), default=0.0)

    if rollback:
        pattern["failure_proxy_count"] = int(pattern.get("failure_proxy_count", 0) or 0) + 1
        window = state.setdefault("rollback_window", [])
        if isinstance(window, list) and window:
            last = window[-1]
            if isinstance(last, Mapping):
                rev_hp = _bounded_float(last.get("previous_habit_precision"), default=previous_hp)
                rev_discount = _bounded_float(last.get("previous_control_discount"), default=previous_discount)
                pattern["habit_precision"] = round(max(rev_hp, previous_hp - MAX_HABIT_PRECISION_DELTA), 6)
                pattern["mean_control_cost_discount"] = round(
                    max(rev_discount, previous_discount - MAX_CONTROL_DISCOUNT_DELTA),
                    6,
                )
        proposal_id = _new_id("m13_patch")
        events.append(
            {
                "type": "M13DrivePatchProposal",
                "patch_id": proposal_id,
                "target": "m13_drive_state",
                "operation": "rollback",
                "field_path": f"path_patterns_by_action/{action}/{user_id}",
                "previous_summary": f"habit={previous_hp:.3f}",
                "new_summary": f"habit={pattern['habit_precision']:.3f}",
                "source_event_id": evaluation.event_id,
                "reason": rollback_reason,
                "confidence": confidence,
                "ttl": 1,
            }
        )
        events.append(
            {
                "type": "M13DrivePatchCommit",
                "commit_id": _new_id("m13_commit"),
                "patch_id": proposal_id,
                "accepted": True,
                "owner": "MVPDialogueRuntime",
                "reason": rollback_reason,
                "committed_summary": f"rolled back {action} for {user_id}",
            }
        )
        state["last_patch_id"] = proposal_id
    elif outcome_band == "positive" and confidence >= MINIMUM_SUCCESS_PROXY_CONFIDENCE:
        pattern["success_proxy_count"] = int(pattern.get("success_proxy_count", 0) or 0) + 1
        new_hp = round(min(1.0, previous_hp + MAX_HABIT_PRECISION_DELTA), 6)
        new_discount = round(min(1.0, previous_discount + MAX_CONTROL_DISCOUNT_DELTA), 6)
        pattern["habit_precision"] = new_hp
        pattern["mean_control_cost_discount"] = new_discount
        proposal_id = _new_id("m13_patch")
        window = state.setdefault("rollback_window", [])
        if not isinstance(window, list):
            window = []
            state["rollback_window"] = window
        window.append(
            {
                "patch_id": proposal_id,
                "action": action,
                "user_id": user_id,
                "topic_fingerprint": topic,
                "previous_habit_precision": previous_hp,
                "previous_control_discount": previous_discount,
                "confidence": confidence,
            }
        )
        state["rollback_window"] = window[-8:]
        events.append(
            {
                "type": "M13DrivePatchProposal",
                "patch_id": proposal_id,
                "target": "m13_drive_state",
                "operation": "increment",
                "field_path": f"path_patterns_by_action/{action}/{user_id}",
                "previous_summary": f"habit={previous_hp:.3f}",
                "new_summary": f"habit={new_hp:.3f}",
                "source_event_id": evaluation.event_id,
                "reason": "success_proxy_current_turn",
                "confidence": confidence,
                "ttl": 8,
            }
        )
        events.append(
            {
                "type": "M13DrivePatchCommit",
                "commit_id": _new_id("m13_commit"),
                "patch_id": proposal_id,
                "accepted": True,
                "owner": "MVPDialogueRuntime",
                "reason": "success_proxy_current_turn",
                "committed_summary": f"strengthened {action} for {user_id}",
            }
        )
        state["last_patch_id"] = proposal_id
    elif outcome_band == "negative":
        pattern["failure_proxy_count"] = int(pattern.get("failure_proxy_count", 0) or 0) + 1

    _evict_patterns(patterns)
    state["path_patterns_by_action"] = patterns
    traction = _mapping(state.get("traction_by_action"))
    key = _traction_key(action, user_id)
    traction[key] = round(_bounded_float(pattern.get("habit_precision")), 6)
    state["traction_by_action"] = traction
    cue_map = _mapping(state.get("cue_precision_by_topic"))
    cue_map[topic] = round(
        max(
            _bounded_float(cue_map.get(topic)),
            _bounded_float(pattern.get("habit_precision")) * M13_THRESHOLDS.cue_map_from_habit_scale,
        ),
        6,
    )
    state["cue_precision_by_topic"] = cue_map
    rel_map = _mapping(state.get("relation_path_precision"))
    rel_map[user_id] = round(
        max(
            _bounded_float(rel_map.get(user_id)),
            _bounded_float(pattern.get("habit_precision")) * M13_THRESHOLDS.relation_map_from_habit_scale,
        ),
        6,
    )
    state["relation_path_precision"] = rel_map
    _record_topic_fingerprint(state, topic, turn_index=turn_index)
    _append_trace(
        state,
        turn_id=turn_id,
        turn_index=turn_index,
        user_id=user_id,
        action=action,
        topic_fingerprint=topic,
        outcome_band=outcome_band,
    )
    return state, events


def apply_settlement_habit_rollback(
    m13_state: dict[str, Any],
    *,
    action: str,
    user_id: str,
    topic_fingerprint: str,
    settlement_id: str,
    reason: str,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reverse recent habit increments when M13.2 settlement disconfirms a prior success proxy."""
    state = normalize_m13_drive_state(m13_state)
    events: list[dict[str, Any]] = []
    patterns = [dict(row) for row in state.get("path_patterns_by_action", []) if isinstance(row, Mapping)]
    pattern = _pattern_lookup(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic_fingerprint,
        allow_topic_fallback=True,
    )
    if pattern is None:
        return state, events

    current_hp = _bounded_float(pattern.get("habit_precision"), default=0.0)
    current_discount = _bounded_float(pattern.get("mean_control_cost_discount"), default=0.0)
    window = state.get("rollback_window", [])
    if not isinstance(window, list):
        window = []

    matched_entry: Mapping[str, Any] | None = None
    for entry in reversed(window):
        if not isinstance(entry, Mapping):
            continue
        if (
            str(entry.get("action", "")) == action
            and str(entry.get("user_id", "")) == user_id
            and str(entry.get("topic_fingerprint", "")).strip() == str(topic_fingerprint or "").strip()
        ):
            matched_entry = entry
            break

    scale = _bounded_float(confidence, default=0.7)
    if matched_entry is not None:
        rev_hp = _bounded_float(matched_entry.get("previous_habit_precision"), default=current_hp)
        rev_discount = _bounded_float(matched_entry.get("previous_control_discount"), default=current_discount)
        pattern["habit_precision"] = round(
            max(rev_hp, current_hp - MAX_HABIT_PRECISION_DELTA * scale),
            6,
        )
        pattern["mean_control_cost_discount"] = round(
            max(rev_discount, current_discount - MAX_CONTROL_DISCOUNT_DELTA * scale),
            6,
        )
    else:
        pattern["habit_precision"] = round(max(0.0, current_hp - MAX_HABIT_PRECISION_DELTA * scale), 6)
        pattern["mean_control_cost_discount"] = round(
            max(0.0, current_discount - MAX_CONTROL_DISCOUNT_DELTA * scale),
            6,
        )

    pattern["failure_proxy_count"] = int(pattern.get("failure_proxy_count", 0) or 0) + 1
    _evict_patterns(patterns)
    state["path_patterns_by_action"] = patterns
    traction = _mapping(state.get("traction_by_action"))
    traction[_traction_key(action, user_id)] = round(_bounded_float(pattern.get("habit_precision")), 6)
    state["traction_by_action"] = traction

    proposal_id = _new_id("m13_patch")
    events.append(
        {
            "type": "M13DrivePatchProposal",
            "patch_id": proposal_id,
            "target": "m13_drive_state",
            "operation": "rollback",
            "field_path": f"path_patterns_by_action/{action}/{user_id}",
            "previous_summary": f"habit={current_hp:.3f}",
            "new_summary": f"habit={pattern['habit_precision']:.3f}",
            "source_event_id": settlement_id,
            "reason": reason,
            "confidence": round(scale, 6),
            "ttl": 1,
        }
    )
    events.append(
        {
            "type": "M13DrivePatchCommit",
            "commit_id": _new_id("m13_commit"),
            "patch_id": proposal_id,
            "accepted": True,
            "owner": "MVPDialogueRuntime",
            "reason": reason,
            "committed_summary": f"settlement rollback {action} for {user_id}",
        }
    )
    state["last_patch_id"] = proposal_id
    return state, events
