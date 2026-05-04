"""Memory dynamics for reusable cognitive paths, decay, and interference.

M6 Stage 6: records outcome-backed reusable path patterns and derives bounded
interference signals.

M9.0: adds value-based retention pressure, decay state computation, cue-based
recall requirements, and interference feedback routing to control surfaces.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
_SUCCESS_OUTCOMES = {
    "dialogue_reward",
    "dialogue_epistemic_gain",
    "epistemic_gain",
    "identity_affirm",
    "resource_gain",
    "social_reward",
}


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(lo, min(hi, numeric)), 6)


def _text(value: object, *, limit: int = 96) -> str:
    return " ".join(str(value or "").split())[:limit]


def _chosen_action(diagnostics: object | None) -> str:
    chosen = getattr(diagnostics, "chosen", None)
    return _text(getattr(chosen, "choice", ""), limit=64)


def _chosen_outcome(diagnostics: object | None) -> str:
    chosen = getattr(diagnostics, "chosen", None)
    return _text(getattr(chosen, "predicted_outcome", ""), limit=96)


def _is_successful_outcome(outcome_label: object, outcome: Mapping[str, object] | None = None) -> bool:
    label = _text(outcome_label).lower()
    if label in _SUCCESS_OUTCOMES:
        return True
    if any(token in label for token in ("reward", "gain", "affirm", "success")):
        return True
    if isinstance(outcome, Mapping):
        return _clamp(outcome.get("free_energy_drop", 0.0), -1.0, 1.0) > 0.02
    return False


def _successful_expected_outcome(
    outcome_label: object,
    diagnostics: object | None,
    outcome: Mapping[str, object] | None,
) -> str:
    label = _text(outcome_label)
    if label and _is_successful_outcome(label, None):
        return label
    chosen = _chosen_outcome(diagnostics)
    if chosen and chosen.lower() not in {"neutral", "none"}:
        return chosen
    if isinstance(outcome, Mapping) and _clamp(outcome.get("free_energy_drop", 0.0), -1.0, 1.0) > 0.02:
        return "free_energy_reduction"
    return label or "positive_outcome"


@dataclass(frozen=True)
class MemoryInterferenceSignal:
    detected: bool
    kind: str = ""
    severity: float = 0.0
    conflicting_episode_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["conflicting_episode_ids"] = list(self.conflicting_episode_ids)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class ReusableCognitivePathPattern:
    pattern_id: str
    action: str
    expected_outcome: str
    support_count: int
    success_count: int
    failure_count: int
    confidence: float
    first_seen_cycle: int
    last_seen_cycle: int
    source: str = "outcome_driven_consolidation"
    outcome_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_memory_interference(
    *,
    diagnostics: object | None = None,
    retrieved_memories: Sequence[Mapping[str, object]] | None = None,
    prediction_delta: Mapping[str, object] | None = None,
) -> MemoryInterferenceSignal:
    """Detect small, testable memory interference signals."""
    reasons: list[str] = []
    episode_ids: list[str] = []
    severity = 0.0
    memories = list(retrieved_memories or getattr(diagnostics, "retrieved_memories", []) or [])
    deltas = dict(prediction_delta or getattr(diagnostics, "prediction_delta", {}) or {})

    large_deltas = [
        key
        for key, value in sorted(deltas.items())
        if abs(_clamp(value, -1.0, 1.0)) >= 0.35
    ]
    if large_deltas:
        reasons.append("memory_prediction_conflict")
        severity = max(severity, min(1.0, 0.35 + (0.12 * len(large_deltas))))

    outcomes: Counter[str] = Counter()
    action_outcomes: dict[str, set[str]] = {}
    for memory in memories:
        episode_id = _text(memory.get("episode_id"), limit=80)
        if episode_id:
            episode_ids.append(episode_id)
        outcome = _text(
            memory.get("predicted_outcome")
            or memory.get("dialogue_outcome_semantic")
            or memory.get("value_label")
            or memory.get("outcome_label"),
            limit=96,
        )
        action = _text(memory.get("action") or memory.get("action_taken"), limit=64)
        if outcome:
            outcomes[outcome] += 1
            if action:
                action_outcomes.setdefault(action, set()).add(outcome)

    if len(outcomes) >= 2:
        has_positive = any(
            outcome in _SUCCESS_OUTCOMES or any(token in outcome for token in ("reward", "gain", "affirm"))
            for outcome in outcomes
        )
        has_negative = any(
            any(token in outcome for token in ("threat", "loss", "fail", "negative"))
            for outcome in outcomes
        )
        if has_positive and has_negative:
            reasons.append("retrieved_outcome_conflict")
            severity = max(severity, 0.55)
    if any(len(values) >= 2 for values in action_outcomes.values()):
        reasons.append("same_action_conflicting_memory")
        severity = max(severity, 0.62)

    detected = bool(reasons)
    return MemoryInterferenceSignal(
        detected=detected,
        kind="memory_interference" if detected else "",
        severity=_clamp(severity),
        conflicting_episode_ids=tuple(dict.fromkeys(episode_ids) if detected else ()),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def memory_overdominance_detected_from_retrieval(
    retrieval_result: Mapping[str, object] | None,
    *,
    max_top_weight: float = 0.72,
) -> bool:
    if not isinstance(retrieval_result, Mapping):
        return False
    hypothesis = retrieval_result.get("recall_hypothesis")
    if isinstance(hypothesis, Mapping):
        weights = hypothesis.get("winner_take_most_weights")
        if isinstance(weights, Mapping):
            try:
                if max(float(value) for value in weights.values()) >= max_top_weight:
                    return True
            except (TypeError, ValueError):
                pass
    candidates = retrieval_result.get("candidates")
    if isinstance(candidates, Sequence) and len(candidates) == 1:
        return True
    return False


def reusable_path_summary(pattern: Mapping[str, object]) -> str:
    action = _text(pattern.get("action"), limit=64)
    outcome = _text(pattern.get("expected_outcome"), limit=96)
    support = int(pattern.get("support_count", 0) or 0)
    confidence = _clamp(pattern.get("confidence", 0.0))
    return f"{action}->{outcome} support={support} confidence={confidence:.2f}"


def consolidate_successful_path_pattern(
    existing_patterns: Sequence[Mapping[str, object]],
    *,
    diagnostics: object | None,
    outcome_label: object,
    cycle: int,
    outcome: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Upsert a reusable cognitive path pattern when an outcome is successful."""
    if not _is_successful_outcome(outcome_label, outcome):
        return [dict(item) for item in existing_patterns], None

    action = _chosen_action(diagnostics)
    if not action:
        return [dict(item) for item in existing_patterns], None
    expected_outcome = _successful_expected_outcome(outcome_label, diagnostics, outcome)
    pattern_id = f"path:{action}:{expected_outcome}"
    cycle_int = int(cycle)
    updated: list[dict[str, object]] = []
    target: dict[str, object] | None = None

    for item in existing_patterns:
        payload = dict(item)
        if str(payload.get("pattern_id", "")) == pattern_id:
            target = payload
        else:
            updated.append(payload)

    if target is None:
        target = ReusableCognitivePathPattern(
            pattern_id=pattern_id,
            action=action,
            expected_outcome=expected_outcome,
            support_count=1,
            success_count=1,
            failure_count=0,
            confidence=0.6,
            first_seen_cycle=cycle_int,
            last_seen_cycle=cycle_int,
            outcome_counts={expected_outcome: 1},
        ).to_dict()
    else:
        success_count = int(target.get("success_count", 0) or 0) + 1
        failure_count = int(target.get("failure_count", 0) or 0)
        support_count = success_count + failure_count
        outcome_counts = dict(target.get("outcome_counts", {}) or {})
        outcome_counts[expected_outcome] = int(outcome_counts.get(expected_outcome, 0) or 0) + 1
        target.update(
            {
                "support_count": support_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "confidence": _clamp(success_count / max(1, support_count)),
                "last_seen_cycle": cycle_int,
                "outcome_counts": outcome_counts,
            }
        )

    updated.append(target)
    updated.sort(
        key=lambda item: (
            -int(item.get("support_count", 0) or 0),
            str(item.get("pattern_id", "")),
        )
    )
    return updated[:32], target


def record_failed_path_outcome(
    existing_patterns: Sequence[Mapping[str, object]],
    *,
    diagnostics: object | None,
    outcome_label: object,
    cycle: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Update an existing reusable pattern after a negative outcome."""
    label = _text(outcome_label)
    if not label or _is_successful_outcome(label):
        return [dict(item) for item in existing_patterns], None
    action = _chosen_action(diagnostics)
    if not action:
        return [dict(item) for item in existing_patterns], None
    updated: list[dict[str, object]] = []
    changed: dict[str, object] | None = None
    for item in existing_patterns:
        payload = dict(item)
        if payload.get("action") == action:
            success_count = int(payload.get("success_count", 0) or 0)
            failure_count = int(payload.get("failure_count", 0) or 0) + 1
            support_count = success_count + failure_count
            payload.update(
                {
                    "support_count": support_count,
                    "failure_count": failure_count,
                    "confidence": _clamp(success_count / max(1, support_count)),
                    "last_seen_cycle": int(cycle),
                }
            )
            changed = payload
        updated.append(payload)
    return updated, changed


# ═══════════════════════════════════════════════════════════════════════════
# M9.0: Retention, Decay, Cue-Based Recall, Interference Feedback
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RetentionPressure:
    """Value-based retention pressure for a memory item."""

    total_pressure: float = 0.0
    identity_continuity_value: float = 0.0
    relationship_continuity_value: float = 0.0
    future_prediction_value: float = 0.0
    affective_salience: float = 0.0
    user_emphasis: float = 0.0
    privacy_or_safety_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    low_confidence_penalty: float = 0.0
    maintenance_cost: float = 0.0
    decay_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "total_pressure": self.total_pressure,
            "identity_continuity_value": self.identity_continuity_value,
            "relationship_continuity_value": self.relationship_continuity_value,
            "future_prediction_value": self.future_prediction_value,
            "affective_salience": self.affective_salience,
            "user_emphasis": self.user_emphasis,
            "privacy_or_safety_penalty": self.privacy_or_safety_penalty,
            "contradiction_penalty": self.contradiction_penalty,
            "low_confidence_penalty": self.low_confidence_penalty,
            "maintenance_cost": self.maintenance_cost,
            "decay_reason": self.decay_reason,
        }


def compute_retention_pressure(
    *,
    identity_continuity_value: float = 0.0,
    relationship_continuity_value: float = 0.0,
    future_prediction_value: float = 0.0,
    affective_salience: float = 0.0,
    user_emphasis: float = 0.0,
    privacy_or_safety_penalty: float = 0.0,
    contradiction_penalty: float = 0.0,
    low_confidence_penalty: float = 0.0,
    maintenance_cost: float = 0.0,
    confidence: float = 0.5,
    has_conflict: bool = False,
    tags: Sequence[str] | None = None,
    memory_type: str = "",
) -> RetentionPressure:
    """Compute retention pressure from value and cost components.

    High-value memories get positive pressure (protected, slower decay).
    Low-value memories get negative pressure (faster decay, excluded from
    retrieval).
    """

    def _clamp_component(value: float) -> float:
        return round(max(-1.0, min(1.0, float(value or 0.0))), 6)

    id_val = _clamp_component(identity_continuity_value)
    rel_val = _clamp_component(relationship_continuity_value)
    fut_val = _clamp_component(future_prediction_value)
    aff_val = _clamp_component(affective_salience)
    emp_val = _clamp_component(user_emphasis)

    priv_pen = _clamp_component(privacy_or_safety_penalty)
    con_pen = _clamp_component(contradiction_penalty)
    low_pen = _clamp_component(low_confidence_penalty)
    maint = _clamp_component(maintenance_cost)

    # auto-derive penalties from context when not explicitly provided
    if confidence < 0.5 and low_pen == 0.0:
        low_pen = _clamp_component(0.5 - confidence)
    if has_conflict and con_pen == 0.0:
        con_pen = 0.3
    if confidence < 0.3:
        low_pen = max(low_pen, 0.5)

    positive = id_val + rel_val + fut_val + aff_val + emp_val
    negative = priv_pen + con_pen + low_pen + maint
    total = _clamp_component(positive - negative)

    reasons: list[str] = []
    if id_val >= 0.5:
        reasons.append("high_identity_value")
    if rel_val >= 0.5:
        reasons.append("high_relationship_value")
    if aff_val >= 0.5:
        reasons.append("high_affect")
    if emp_val >= 0.5:
        reasons.append("user_emphasized")
    if con_pen >= 0.3:
        reasons.append("contradiction_detected")
    if low_pen >= 0.3:
        reasons.append("low_confidence")
    if priv_pen >= 0.3:
        reasons.append("privacy_safety_risk")
    if total >= 0.5:
        reasons.append("protected")
    elif total <= -0.3:
        reasons.append("decay_candidate")

    return RetentionPressure(
        total_pressure=total,
        identity_continuity_value=id_val,
        relationship_continuity_value=rel_val,
        future_prediction_value=fut_val,
        affective_salience=aff_val,
        user_emphasis=emp_val,
        privacy_or_safety_penalty=priv_pen,
        contradiction_penalty=con_pen,
        low_confidence_penalty=low_pen,
        maintenance_cost=maint,
        decay_reason="; ".join(reasons) if reasons else "neutral",
    )


def compute_decay_state(
    *,
    retention_pressure: RetentionPressure | None = None,
    total_pressure: float | None = None,
    last_access_cycles_ago: int = 0,
    access_frequency: int = 1,
    cycle: int = 0,
    created_at_cycle: int = 0,
    max_cycles_fresh: int = 5,
    max_cycles_active: int = 20,
    max_cycles_fading: int = 50,
) -> str:
    """Map retention pressure and access patterns to a decay state.

    Returns one of: fresh, active, fading, dormant, pruned.
    """
    pressure = (
        retention_pressure.total_pressure
        if retention_pressure is not None
        else (total_pressure or 0.0)
    )

    age = max(0, cycle - created_at_cycle) if cycle else last_access_cycles_ago

    # High-value memories get extended lifetimes
    if pressure >= 0.5:
        max_fresh = max_cycles_fresh * 3
        max_active = max_cycles_active * 3
        max_fading = max_cycles_fading * 2
    elif pressure >= 0.2:
        max_fresh = max_cycles_fresh * 2
        max_active = max_cycles_active * 2
        max_fading = max_cycles_fading
    elif pressure <= -0.3:
        max_fresh = max(1, max_cycles_fresh // 2)
        max_active = max(2, max_cycles_active // 2)
        max_fading = max(3, max_cycles_fading // 3)
    else:
        max_fresh = max_cycles_fresh
        max_active = max_cycles_active
        max_fading = max_cycles_fading

    # Frequently accessed memories stay fresh longer
    if access_frequency >= 5:
        max_fresh = max_fresh * 2
        max_active = max_active * 2
    elif access_frequency >= 2:
        max_fresh = int(max_fresh * 1.5)

    if age <= max_fresh:
        return "fresh"
    elif age <= max_active:
        return "active"
    elif age <= max_fading:
        return "fading"
    elif pressure >= 0.5:
        return "dormant"
    else:
        return "pruned"


@dataclass(frozen=True)
class CueMatchResult:
    """Result of matching a cue against memory content."""

    matched: bool = False
    cue: str = ""
    score: float = 0.0
    matched_fields: tuple[str, ...] = ()
    trace: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "matched": self.matched,
            "cue": self.cue,
            "score": self.score,
            "matched_fields": list(self.matched_fields),
            "trace": self.trace,
        }


def compute_cue_match(
    *,
    cue: str,
    memory_content: str,
    memory_tags: Sequence[str] | None = None,
    memory_type: str = "",
    speaker: str = "",
) -> CueMatchResult:
    """Compute how well a cue matches a memory item.

    A cue may come from:
    - current user input
    - current task/gap
    - self-thought event
    - outcome feedback
    - active relationship/entity
    - explicit user request

    Returns a score in [0, 1] and indicates which fields matched.
    """
    if not cue or not cue.strip():
        return CueMatchResult(matched=False, cue="", score=0.0, trace="empty_cue")

    cue_lower = cue.strip().lower()
    content_lower = memory_content.strip().lower()
    tags_lower = [t.lower() for t in (memory_tags or [])]
    matched_fields: list[str] = []
    total_score = 0.0

    # 1. Exact content match (strongest signal)
    if cue_lower in content_lower:
        ratio = len(cue_lower) / max(1, len(content_lower))
        total_score += 0.5 + 0.5 * ratio
        matched_fields.append("content_exact")

    # 2. Word-level overlap in content (handles both space-separated and CJK)
    cue_words = set(cue_lower.split())
    content_words = set(content_lower.split())
    has_space_overlap = bool(cue_words and content_words and (cue_words & content_words))
    if has_space_overlap:
        overlap = cue_words & content_words
        word_score = 0.3 * (len(overlap) / max(1, len(cue_words)))
        total_score += word_score
        if "content_exact" not in matched_fields:
            matched_fields.append("content_words")
    else:
        # CJK text lacks spaces: check each cue word as a substring
        cjk_matches = 0
        for cw in cue_words:
            if len(cw) >= 1 and cw in content_lower:
                cjk_matches += 1
        if cjk_matches > 0:
            word_score = 0.3 * (cjk_matches / max(1, len(cue_words)))
            total_score += word_score
            if "content_exact" not in matched_fields:
                matched_fields.append("content_words")

    # 3. Tag match
    if tags_lower:
        for tag in tags_lower:
            if tag in cue_lower or cue_lower in tag:
                total_score += 0.15
                matched_fields.append(f"tag:{tag}")
                break
        for tag in tags_lower:
            tag_words = set(tag.split())
            if tag_words and cue_words:
                if tag_words & cue_words:
                    if "tag:" not in str(matched_fields):
                        total_score += 0.1
                        matched_fields.append("tag_words")
                        break

    # 4. Entity / speaker match
    if speaker and speaker.lower() in cue_lower:
        total_score += 0.15
        matched_fields.append(f"speaker:{speaker}")

    score = round(min(1.0, total_score), 4)
    matched = score >= 0.15

    trace_parts = []
    if matched:
        trace_parts.append(f"cue='{cue[:60]}'")
        trace_parts.append(f"score={score:.2f}")
        trace_parts.append(f"fields={','.join(matched_fields)}")
    else:
        trace_parts.append(f"cue='{cue[:60]}' no_match")

    return CueMatchResult(
        matched=matched,
        cue=cue,
        score=score,
        matched_fields=tuple(matched_fields),
        trace="; ".join(trace_parts),
    )


def require_cue_for_recall(
    *,
    cue: str = "",
    memory_items: Sequence[Mapping[str, object]] | None = None,
    min_cue_score: float = 0.15,
) -> tuple[list[dict[str, object]], str]:
    """Filter memory items to only those matched by a cue.

    Without a cue, returns empty list and 'unknown' stance.
    With a cue, returns matched items and stance.
    """
    if not cue or not cue.strip():
        return [], "unknown: no cue provided, long-term memory not searched"

    matched: list[dict[str, object]] = []
    for item in (memory_items or []):
        content = str(
            item.get("proposition")
            or item.get("content_summary")
            or item.get("summary")
            or item.get("content")
            or ""
        )
        tags = item.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        speaker = str(item.get("speaker", ""))

        result = compute_cue_match(
            cue=cue,
            memory_content=content,
            memory_tags=list(tags) if tags else None,
            speaker=speaker,
        )
        if result.matched and result.score >= min_cue_score:
            entry = dict(item)
            entry["cue_match"] = result.trace
            entry["cue_score"] = result.score
            matched.append(entry)

    matched.sort(key=lambda e: float(e.get("cue_score", 0)), reverse=True)
    stance = (
        f"cued: {len(matched)} items matched cue '{cue[:80]}'"
        if matched
        else f"unknown: cue '{cue[:80]}' did not match any stored memory"
    )
    return matched, stance


@dataclass(frozen=True)
class InterferenceFeedback:
    """How memory conflict should affect generation and control surfaces."""

    reduce_assertiveness: bool = False
    reduce_memory_reliance: bool = False
    increase_caution: bool = False
    increase_clarification_bias: bool = False
    severity: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "reduce_assertiveness": self.reduce_assertiveness,
            "reduce_memory_reliance": self.reduce_memory_reliance,
            "increase_caution": self.increase_caution,
            "increase_clarification_bias": self.increase_clarification_bias,
            "severity": self.severity,
            "reason": self.reason,
        }


def derive_interference_feedback(
    interference: MemoryInterferenceSignal | None = None,
    *,
    overdominance_detected: bool = False,
    memory_retrieval_gain: float = 0.5,
) -> InterferenceFeedback:
    """Derive generation control guidance from memory interference signals.

    Conflicting memories reduce overconfident generation.
    Memory overdominance does not silently steer the reply without trace.
    """
    if interference is None and not overdominance_detected:
        return InterferenceFeedback(
            reduce_assertiveness=False,
            reduce_memory_reliance=False,
            increase_caution=False,
            increase_clarification_bias=False,
            severity=0.0,
            reason="no_interference_detected",
        )

    sev = interference.severity if interference else 0.0
    if overdominance_detected:
        sev = max(sev, 0.5)

    reduce_assert = sev >= 0.35
    reduce_mem = sev >= 0.45 or overdominance_detected
    increase_caut = sev >= 0.3
    increase_clarify = sev >= 0.5

    reason_parts = []
    if interference and interference.detected:
        reason_parts.append(f"interference:{interference.kind}")
    if overdominance_detected:
        reason_parts.append("overdominance")
    reason_parts.append(f"severity={sev:.2f}")

    return InterferenceFeedback(
        reduce_assertiveness=reduce_assert,
        reduce_memory_reliance=reduce_mem,
        increase_caution=increase_caut,
        increase_clarification_bias=increase_clarify,
        severity=sev,
        reason="; ".join(reason_parts),
    )


def apply_interference_to_evidence_contract(
    feedback: InterferenceFeedback,
    *,
    current_caution_level: float = 0.5,
    current_assertiveness: float = 0.5,
    memory_retrieval_gain: float = 0.5,
) -> dict[str, float]:
    """Apply interference feedback to produce updated control parameters.

    Returns a dict of updated control values for meta-control consumption.
    The response evidence contract can use these to adjust generation constraints.
    """
    caution = current_caution_level
    assertiveness = current_assertiveness
    mem_gain = memory_retrieval_gain

    if feedback.increase_caution:
        caution = min(1.0, caution + 0.2 * feedback.severity)
    if feedback.reduce_assertiveness:
        assertiveness = max(0.1, assertiveness - 0.25 * feedback.severity)
    if feedback.reduce_memory_reliance:
        mem_gain = max(0.1, mem_gain - 0.3 * feedback.severity)

    return {
        "caution_level": round(caution, 4),
        "assertiveness": round(assertiveness, 4),
        "memory_retrieval_gain": round(mem_gain, 4),
        "interference_severity": round(feedback.severity, 4),
        "interference_reason": feedback.reason,
    }
