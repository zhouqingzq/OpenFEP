"""Minimal memory dynamics for reusable cognitive paths.

Stage 6 deliberately keeps this layer small: it records outcome-backed reusable
path patterns and derives bounded interference signals. It does not replace the
episodic memory store or the existing policy/memory bias machinery.
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
