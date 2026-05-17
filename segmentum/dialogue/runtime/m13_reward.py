"""MVP-local M13.2 affective reward proxy, tolerance, and next-turn settlement.

Engineering proxies only; never exposed as subjective pleasure or addiction labels.
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from segmentum.dialogue.runtime.m13_drive import (
    M13_THRESHOLDS,
    MAX_PENDING_SETTLEMENTS,
    MAX_TOLERANCE_BY_PATH,
    MINIMUM_SUCCESS_PROXY_CONFIDENCE,
    M13EvaluationResult,
    SETTLEMENT_ROLLBACK_PREDICTION_ERROR_THRESHOLD,
    _bounded_float,
    _mapping,
    _new_id,
    _string_list,
    apply_settlement_habit_rollback,
    normalize_m13_drive_state,
)

MAX_REWARD_HISTORY = 24
PENDING_SETTLEMENT_TTL = 5
MAX_REASON_CODES = 8
MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA = 0.05
MAX_SINGLE_TURN_TOLERANCE_DELTA = 0.04
MAX_SINGLE_TURN_OPPONENT_STRENGTH_DELTA = 0.04
MAX_SINGLE_TURN_BASELINE_DELTA = 0.03
MIN_SETTLEMENT_CONFIDENCE_FOR_POSITIVE = 0.60
MAX_REWARD_PULL_CONFIDENCE_BOOST = 0.02

# Bounded user-text cues for settlement only; structured diagnostics dominate.
_USER_CORRECTION_PATTERN = re.compile(
    r"(?i)(不对|错了|不是这样|你搞错|我说的是|actually|wrong|not what i|纠正|更正)",
)
_USER_UPTAKE_STRONG_PATTERN = re.compile(
    r"(?i)(谢谢|感谢|明白了|懂了|got it|thanks|makes sense|收到|good to know)",
)
_USER_UPTAKE_WEAK_PATTERN = re.compile(r"(?i)^(好的|继续|嗯|ok|okay)[。!！?？…~\s]*$")

_PROMPT_FORBIDDEN_AFFECTIVE_KEYS: frozenset[str] = frozenset(
    {
        "net_affective_reward_proxy",
        "observed_reward_proxy",
        "predicted_reward",
        "reward_baseline",
        "tolerance",
        "opponent_strength",
        "prediction_error_proxy",
        "behavioral_pull",
        "engineering_proxy_label",
    }
)


def default_affective_reward_proxy_state() -> dict[str, Any]:
    return {
        "reward_baseline": 0.35,
        "opponent_strength": 0.0,
        "tolerance_by_path": [],
        "predicted_reward_by_path": {},
        "last_net_reward_proxy": 0.0,
        "last_relief_proxy": 0.0,
        "last_information_gain_proxy": 0.0,
        "pending_settlements": [],
        "reward_history": [],
        "engineering_proxy_label": "mvp_local_affective_reward_proxy",
    }


def normalize_affective_reward_proxy_state(raw: Any) -> dict[str, Any]:
    base = default_affective_reward_proxy_state()
    if not isinstance(raw, Mapping):
        return copy.deepcopy(base)
    merged = {**base, **dict(raw)}
    merged["reward_baseline"] = _bounded_float(merged.get("reward_baseline"), default=0.35)
    merged["opponent_strength"] = _bounded_float(merged.get("opponent_strength"))
    merged["last_net_reward_proxy"] = _bounded_float(merged.get("last_net_reward_proxy"))
    merged["last_relief_proxy"] = _bounded_float(merged.get("last_relief_proxy"))
    merged["last_information_gain_proxy"] = _bounded_float(
        merged.get("last_information_gain_proxy")
    )
    rows = merged.get("tolerance_by_path")
    merged["tolerance_by_path"] = (
        [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []
    )
    predicted = merged.get("predicted_reward_by_path")
    merged["predicted_reward_by_path"] = dict(predicted) if isinstance(predicted, Mapping) else {}
    pending = merged.get("pending_settlements")
    merged["pending_settlements"] = (
        [dict(item) for item in pending if isinstance(item, Mapping)] if isinstance(pending, list) else []
    )
    history = merged.get("reward_history")
    merged["reward_history"] = (
        [dict(item) for item in history if isinstance(item, Mapping)] if isinstance(history, list) else []
    )
    return merged


def path_id_for(*, action: str, user_id: str, topic_fingerprint: str) -> str:
    return f"{action}|{user_id}|{topic_fingerprint}"[:240]


def _path_row_lookup(
    rows: list[dict[str, Any]],
    *,
    path_id: str,
) -> dict[str, Any] | None:
    for row in reversed(rows):
        if str(row.get("path_id", "")) == path_id:
            return row
    return None


def _upsert_path_row(
    rows: list[dict[str, Any]],
    *,
    path_id: str,
    action: str,
    topic_fingerprint: str,
    turn_index: int,
) -> dict[str, Any]:
    row = _path_row_lookup(rows, path_id=path_id)
    if row is None:
        row = {
            "path_id": path_id,
            "action": action,
            "topic_fingerprint": topic_fingerprint,
            "support_count": 0,
            "predicted_reward": 0.0,
            "tolerance": 0.0,
            "opponent_strength": 0.0,
            "last_net_reward_proxy": 0.0,
            "last_updated_turn": turn_index,
        }
        rows.append(row)
    row["support_count"] = int(row.get("support_count", 0) or 0) + 1
    row["last_updated_turn"] = turn_index
    return row


def _evict_tolerance_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) <= MAX_TOLERANCE_BY_PATH:
        return
    rows.sort(key=lambda item: int(item.get("last_updated_turn", 0) or 0))
    del rows[: len(rows) - MAX_TOLERANCE_BY_PATH]


def _bounded_delta(current: float, target: float, *, max_delta: float) -> float:
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


def _bounded_net_proxy(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(-1.0, min(1.0, numeric)), 6)


def observation_channels_from_bus(bus_messages: list[Mapping[str, Any]] | None) -> dict[str, Any]:
    for message in reversed(list(bus_messages or [])):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("type", "")) != "ObservationEvent":
            continue
        channels = message.get("channels")
        if isinstance(channels, Mapping):
            return dict(channels)
    return {}


def _user_uptake_signal(user_text: str, *, structured_signal_count: int) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    if _USER_CORRECTION_PATTERN.search(text):
        return False
    if _USER_UPTAKE_STRONG_PATTERN.search(text):
        return True
    if structured_signal_count > 0 and _USER_UPTAKE_WEAK_PATTERN.match(text):
        return True
    return False


def _settlement_observation_adjustment(
    observation_channels: Mapping[str, Any],
) -> tuple[float, float, list[str]]:
    """Small bounded deltas from UI observation channels; never alone sufficient for positive."""
    channels = _mapping(observation_channels)
    if not channels:
        return 0.0, 0.0, []
    observed_delta = 0.0
    confidence_boost = 0.0
    reasons: list[str] = []
    conflict = _bounded_float(channels.get("conflict_tension"))
    tone = _bounded_float(channels.get("emotional_tone"))
    semantic = _bounded_float(channels.get("semantic_content"))
    novelty = _bounded_float(channels.get("topic_novelty"))
    if conflict >= 0.62:
        observed_delta -= 0.12
        reasons.append("observation_conflict_pressure")
    if tone >= 0.68 and conflict < 0.45:
        observed_delta += 0.08
        confidence_boost = max(confidence_boost, 0.06)
        reasons.append("observation_positive_tone")
    if semantic >= 0.55 and novelty >= 0.35:
        observed_delta += 0.06
        confidence_boost = max(confidence_boost, 0.05)
        reasons.append("observation_semantic_continuity")
    return observed_delta, confidence_boost, reasons[:4]


@dataclass
class M13RewardEvaluationResult:
    event_id: str
    turn_id: str
    turn_index: int
    path_id: str
    net_affective_reward_proxy: float
    observed_reward_proxy: float
    relief_proxy: float
    information_gain_proxy: float
    predicted_reward: float
    reward_baseline: float
    tolerance: float
    opponent_strength: float
    prediction_error_proxy: float
    path_feels_stale_proxy: bool
    reason_codes: list[str]
    evidence_refs: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class M13SettlementResult:
    settlement_id: str
    pending_id: str
    observed_reward_proxy: float
    prediction_error_proxy: float
    relief_proxy: float
    information_gain_proxy: float
    outcome_band: str
    reason_codes: list[str]
    confidence: float
    events: list[dict[str, Any]] = field(default_factory=list)


def _same_turn_positive_components(
    *,
    conscious_plan: Mapping[str, Any],
    reply_validation: Mapping[str, Any],
    post_reply_observer: Mapping[str, Any],
    memory_candidates_applied: list[Any],
    evidence_judgment: Mapping[str, Any] | None,
    relationship_value_context: Mapping[str, Any] | None = None,
) -> tuple[float, list[str]]:
    observed = 0.0
    reasons: list[str] = []
    conscious = _mapping(conscious_plan)
    for item in conscious.get("expectation_results", []) or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status == "confirmed":
            observed += 0.22
            reasons.append("expectation_confirmed")
        elif status in {"resolved", "closed"}:
            observed += 0.16
            reasons.append("expectation_closed")

    validation = _mapping(reply_validation)
    if validation and "changed" in validation and not bool(validation.get("changed")):
        observed += 0.14
        reasons.append("reply_validation_unchanged")

    observer = _mapping(post_reply_observer)
    if observer and "needs_followup" in observer and not bool(observer.get("needs_followup")):
        followup = str(observer.get("followup_type", "none")).lower()
        if followup in {"", "none"}:
            observed += 0.12
            reasons.append("observer_no_repair")

    if memory_candidates_applied:
        observed += 0.1
        reasons.append("memory_write_accepted")

    evidence = _mapping(evidence_judgment)
    stance = str(evidence.get("epistemic_stance", "")).lower()
    if stance in {"known", "supported"}:
        observed += 0.12
        reasons.append("evidence_stance_clearer")
    elif stance == "known_with_caveat":
        observed += 0.06
        reasons.append("evidence_stance_partial")

    relationship = _mapping(relationship_value_context)
    active = relationship.get("active_relationship_value_memories", [])
    if isinstance(active, list) and active and not bool(relationship.get("prediction_blocked")):
        observed += 0.08
        reasons.append("relationship_value_constraint_active")

    return _bounded_float(observed), reasons[:MAX_REASON_CODES]


def _same_turn_negative_components(
    *,
    conscious_plan: Mapping[str, Any],
    reply_validation: Mapping[str, Any],
    post_reply_observer: Mapping[str, Any],
    safety_repair: bool,
    repetition_pressure: float = 0.0,
    conflict_level: float = 0.0,
) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []
    if safety_repair:
        penalty += 0.28
        reasons.append("safety_rewrite")
    validation = _mapping(reply_validation)
    if validation and bool(validation.get("changed")):
        penalty += 0.18
        reasons.append("reply_validation_changed")
    observer = _mapping(post_reply_observer)
    if observer and bool(observer.get("needs_followup")) and str(observer.get("followup_type", "")).lower() not in {
        "",
        "none",
    }:
        penalty += 0.14
        reasons.append("observer_repair_required")
    for item in _mapping(conscious_plan).get("expectation_results", []) or []:
        if isinstance(item, Mapping) and str(item.get("status", "")) == "violated":
            penalty += 0.22
            reasons.append("expectation_violated")
            break
    if repetition_pressure >= 0.3:
        penalty += 0.1
        reasons.append("repeated_low_novelty_path")
    if conflict_level >= 0.55:
        penalty += 0.12
        reasons.append("conflict_pressure")
    return _bounded_float(penalty), reasons[:MAX_REASON_CODES]


def settle_pending_m13_actions(
    m13_state: dict[str, Any],
    *,
    user_text: str,
    user_id: str,
    turn_index: int,
    turn_id: str,
    boredom_information_gain: float = 0.0,
    observation_channels: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[M13SettlementResult], list[dict[str, Any]]]:
    """Settle prior-turn pending traces at run_turn start (before conscious planning)."""
    state = normalize_m13_drive_state(m13_state)
    reward_state = normalize_affective_reward_proxy_state(state.get("affective_reward_proxy"))
    pending_rows = list(reward_state.get("pending_settlements", []))
    events: list[dict[str, Any]] = []
    settlements: list[M13SettlementResult] = []
    kept_pending: list[dict[str, Any]] = []

    for row in pending_rows:
        pending_id = str(row.get("pending_id", ""))
        prior_turn = int(row.get("prior_turn_index", -1) or -1)
        expires = int(row.get("expires_after_turns", PENDING_SETTLEMENT_TTL) or PENDING_SETTLEMENT_TTL)
        if turn_index > prior_turn + expires:
            events.append(
                {
                    "type": "M13RewardSettlementExpired",
                    "pending_id": pending_id,
                    "turn_index": turn_index,
                    "reason": "pending_settlement_ttl_expired",
                }
            )
            continue

        if turn_index <= prior_turn:
            kept_pending.append(row)
            continue

        prior_validation = _mapping(row.get("prior_reply_validation"))
        prior_observer = _mapping(row.get("prior_post_reply_observer"))
        prior_expectations = row.get("prior_expectation_results") or []
        prior_safety_repair = bool(row.get("prior_safety_repair"))

        pos, pos_reasons = _same_turn_positive_components(
            conscious_plan={"expectation_results": prior_expectations},
            reply_validation=prior_validation,
            post_reply_observer=prior_observer,
            memory_candidates_applied=row.get("prior_memory_candidates_applied") or [],
            evidence_judgment=_mapping(row.get("prior_evidence_judgment")),
        )
        neg, neg_reasons = _same_turn_negative_components(
            conscious_plan={"expectation_results": prior_expectations},
            reply_validation=prior_validation,
            post_reply_observer=prior_observer,
            safety_repair=prior_safety_repair,
            repetition_pressure=_bounded_float(row.get("prior_repetition_pressure")),
            conflict_level=_bounded_float(row.get("prior_conflict_level")),
        )

        structured_signal_count = len(pos_reasons) + len(neg_reasons)
        observed = _bounded_float(pos - neg)
        reason_codes: list[str] = []
        confidence = 0.45
        obs_delta, obs_conf_boost, obs_reasons = _settlement_observation_adjustment(
            observation_channels or {}
        )
        if obs_reasons:
            observed = _bounded_float(observed + obs_delta)
            reason_codes.extend(obs_reasons)
            confidence = max(confidence, 0.45 + obs_conf_boost)
        if _USER_CORRECTION_PATTERN.search(user_text or ""):
            observed = _bounded_float(observed - 0.25)
            reason_codes.append("user_correction_marker")
            confidence = 0.72
        uptake = _user_uptake_signal(user_text, structured_signal_count=structured_signal_count)
        if uptake:
            observed = _bounded_float(observed + (0.12 if structured_signal_count > 0 else 0.06))
            reason_codes.append("user_uptake_marker")
            confidence = max(confidence, 0.65 if structured_signal_count > 0 else 0.52)

        reason_codes.extend(pos_reasons[:4])
        reason_codes.extend(neg_reasons[:4])
        reason_codes = list(dict.fromkeys(reason_codes))[:MAX_REASON_CODES]

        signal_count = structured_signal_count + len(obs_reasons) + (
            1 if _USER_CORRECTION_PATTERN.search(user_text or "") else 0
        ) + (1 if uptake else 0)
        if signal_count == 0:
            outcome_band = "uncertain"
            confidence = 0.4
            reason_codes.append("insufficient_settlement_evidence")
        elif observed >= 0.55 and confidence >= MIN_SETTLEMENT_CONFIDENCE_FOR_POSITIVE:
            outcome_band = "positive"
        elif observed <= 0.25 or "user_correction_marker" in reason_codes:
            outcome_band = "negative"
        else:
            outcome_band = "uncertain"

        predicted = _bounded_float(row.get("predicted_reward", 0.0))
        predicted_relief = _bounded_float(row.get("predicted_relief", 0.0))
        relief = _bounded_float(max(0.0, observed - predicted_relief))
        information_gain = _bounded_float(boredom_information_gain or row.get("prior_information_gain", 0.0))
        prediction_error = round(observed - predicted, 6)

        settlement_id = _new_id("m13_settle")
        settlement = M13SettlementResult(
            settlement_id=settlement_id,
            pending_id=pending_id,
            observed_reward_proxy=observed,
            prediction_error_proxy=prediction_error,
            relief_proxy=relief,
            information_gain_proxy=information_gain,
            outcome_band=outcome_band,
            reason_codes=reason_codes,
            confidence=confidence,
            events=[
                {
                    "type": "M13RewardSettlementEvent",
                    "settlement_id": settlement_id,
                    "pending_id": pending_id,
                    "turn_id": turn_id,
                    "turn_index": turn_index,
                    "observed_reward_proxy": round(observed, 6),
                    "prediction_error_proxy": prediction_error,
                    "relief_proxy": round(relief, 6),
                    "information_gain_proxy": round(information_gain, 6),
                    "outcome_band": outcome_band,
                    "reason_codes": reason_codes,
                    "confidence": round(confidence, 6),
                    "engineering_proxy_label": "mvp_local_affective_reward_proxy",
                }
            ],
        )
        settlements.append(settlement)
        events.extend(settlement.events)

        action = str(row.get("prior_action", "answer"))
        topic = str(row.get("prior_topic_fingerprint", "topic:unknown"))
        pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic)
        path_rows = list(reward_state.get("tolerance_by_path", []))
        path_row = _upsert_path_row(
            path_rows,
            path_id=pid,
            action=action,
            topic_fingerprint=topic,
            turn_index=turn_index,
        )
        predicted_map = dict(reward_state.get("predicted_reward_by_path", {}))
        old_predicted = _bounded_float(path_row.get("predicted_reward", predicted_map.get(pid, 0.0)))
        if outcome_band == "positive":
            target_predicted = observed
            path_row["predicted_reward"] = round(
                _bounded_delta(old_predicted, target_predicted, max_delta=MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA),
                6,
            )
            predicted_map[pid] = path_row["predicted_reward"]
        elif outcome_band == "negative":
            path_row["predicted_reward"] = round(
                _bounded_delta(old_predicted, observed, max_delta=MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA),
                6,
            )
            predicted_map[pid] = path_row["predicted_reward"]
        path_row["last_net_reward_proxy"] = round(
            compute_net_affective_reward_proxy(
                observed_reward_proxy=observed,
                relief_proxy=relief,
                information_gain_proxy=information_gain,
                predicted_reward=_bounded_float(path_row.get("predicted_reward")),
                reward_baseline=_bounded_float(reward_state.get("reward_baseline")),
                tolerance=_bounded_float(path_row.get("tolerance")),
                opponent_strength=_bounded_float(
                    max(
                        _bounded_float(reward_state.get("opponent_strength")),
                        _bounded_float(path_row.get("opponent_strength")),
                    )
                ),
            ),
            6,
        )
        if outcome_band == "positive" and confidence >= MIN_SETTLEMENT_CONFIDENCE_FOR_POSITIVE:
            path_row["tolerance"] = round(
                _bounded_delta(
                    _bounded_float(path_row.get("tolerance")),
                    min(1.0, _bounded_float(path_row.get("tolerance")) + 0.03),
                    max_delta=MAX_SINGLE_TURN_TOLERANCE_DELTA,
                ),
                6,
            )
        if outcome_band == "negative" or "safety_rewrite" in reason_codes:
            global_opp = _bounded_float(reward_state.get("opponent_strength"))
            reward_state["opponent_strength"] = round(
                _bounded_delta(global_opp, min(1.0, global_opp + 0.04), max_delta=MAX_SINGLE_TURN_OPPONENT_STRENGTH_DELTA),
                6,
            )
            path_row["opponent_strength"] = round(
                _bounded_delta(
                    _bounded_float(path_row.get("opponent_strength")),
                    min(1.0, _bounded_float(path_row.get("opponent_strength")) + 0.03),
                    max_delta=MAX_SINGLE_TURN_OPPONENT_STRENGTH_DELTA,
                ),
                6,
            )
        _evict_tolerance_rows(path_rows)
        reward_state["tolerance_by_path"] = path_rows
        reward_state["predicted_reward_by_path"] = predicted_map

        if outcome_band != "uncertain":
            baseline = _bounded_float(reward_state.get("reward_baseline"))
            reward_state["reward_baseline"] = round(
                _bounded_delta(baseline, observed, max_delta=MAX_SINGLE_TURN_BASELINE_DELTA),
                6,
            )

        history = list(reward_state.get("reward_history", []))
        history.append(
            {
                "turn_index": turn_index,
                "path_id": pid,
                "net_affective_reward_proxy": path_row["last_net_reward_proxy"],
                "outcome_band": outcome_band,
            }
        )
        reward_state["reward_history"] = history[-MAX_REWARD_HISTORY:]
        reward_state["last_net_reward_proxy"] = path_row["last_net_reward_proxy"]
        reward_state["last_relief_proxy"] = round(relief, 6)
        reward_state["last_information_gain_proxy"] = round(information_gain, 6)

        should_rollback_habit = outcome_band == "negative" or (
            outcome_band == "uncertain"
            and abs(prediction_error) >= SETTLEMENT_ROLLBACK_PREDICTION_ERROR_THRESHOLD
            and confidence >= 0.55
        )
        if should_rollback_habit:
            rollback_reason = (
                "rollback_negative_settlement"
                if outcome_band == "negative"
                else "rollback_uncertain_high_prediction_error"
            )
            state, rollback_events = apply_settlement_habit_rollback(
                state,
                action=action,
                user_id=user_id,
                topic_fingerprint=topic,
                settlement_id=settlement_id,
                reason=rollback_reason,
                confidence=confidence,
            )
            events.extend(rollback_events)

    reward_state["pending_settlements"] = kept_pending
    state["affective_reward_proxy"] = reward_state
    return state, settlements, events


def compute_net_affective_reward_proxy(
    *,
    observed_reward_proxy: float,
    relief_proxy: float,
    information_gain_proxy: float,
    predicted_reward: float,
    reward_baseline: float,
    tolerance: float,
    opponent_strength: float,
) -> float:
    raw = (
        observed_reward_proxy
        + relief_proxy
        + information_gain_proxy
        - predicted_reward
        - reward_baseline
        - tolerance
        - opponent_strength
    )
    return _bounded_net_proxy(raw)


def evaluate_pre_turn_reward_proxy(
    *,
    turn_id: str,
    turn_index: int,
    user_id: str,
    m13_state: Mapping[str, Any],
    m13_evaluation: M13EvaluationResult,
    information_gain_proxy: float,
    repetition_pressure: float,
    conflict_level: float,
) -> M13RewardEvaluationResult:
    """Pre-thinking guidance from settled path dynamics and boredom proxies."""
    action = m13_evaluation.top_behavioral_pull_action
    topic = m13_evaluation.topic_fingerprint
    pull = m13_evaluation.scores_by_action.get(action, {}).get("behavioral_pull", 0.0)
    reward_state = normalize_affective_reward_proxy_state(
        normalize_m13_drive_state(m13_state).get("affective_reward_proxy")
    )
    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic)
    path_row = _path_row_lookup(list(reward_state.get("tolerance_by_path", [])), path_id=pid) or {}
    predicted = _bounded_float(
        path_row.get("predicted_reward", _mapping(reward_state.get("predicted_reward_by_path")).get(pid, 0.0))
    )
    tolerance = _bounded_float(path_row.get("tolerance"))
    opponent = _bounded_float(
        max(_bounded_float(reward_state.get("opponent_strength")), _bounded_float(path_row.get("opponent_strength")))
    )
    baseline = _bounded_float(reward_state.get("reward_baseline"), default=0.35)
    information_gain = _bounded_float(information_gain_proxy)
    observed = _bounded_float(reward_state.get("last_net_reward_proxy")) * 0.35
    relief = 0.0
    net = compute_net_affective_reward_proxy(
        observed_reward_proxy=observed,
        relief_proxy=relief,
        information_gain_proxy=information_gain,
        predicted_reward=predicted,
        reward_baseline=baseline,
        tolerance=tolerance,
        opponent_strength=opponent,
    )
    stale_proxy = predicted >= 0.3 and repetition_pressure >= 0.2 and net <= 0.4
    event_id = _new_id("m13_reward_pre")
    return M13RewardEvaluationResult(
        event_id=event_id,
        turn_id=turn_id,
        turn_index=turn_index,
        path_id=pid,
        net_affective_reward_proxy=net,
        observed_reward_proxy=observed,
        relief_proxy=relief,
        information_gain_proxy=information_gain,
        predicted_reward=predicted,
        reward_baseline=baseline,
        tolerance=tolerance,
        opponent_strength=opponent,
        prediction_error_proxy=round(observed - predicted, 6),
        path_feels_stale_proxy=stale_proxy,
        reason_codes=["pre_turn_path_state"],
        evidence_refs=[],
        events=[
            {
                "type": "M13RewardPreTurnGuidanceEvent",
                "event_id": event_id,
                "turn_id": turn_id,
                "turn_index": turn_index,
                "path_id": pid,
                "net_affective_reward_proxy": net,
                "predicted_reward": round(predicted, 6),
                "tolerance": round(tolerance, 6),
                "engineering_proxy_label": "mvp_local_affective_reward_proxy",
            }
        ],
    )


class M13RewardEvaluator:
    """Post-reply reward proxy for settlement creation; does not replace behavioral_pull."""

    def evaluate(
        self,
        *,
        turn_id: str,
        turn_index: int,
        user_id: str,
        action: str,
        topic_fingerprint: str,
        m13_state: Mapping[str, Any],
        conscious_plan: Mapping[str, Any],
        reply_validation: Mapping[str, Any],
        post_reply_observer: Mapping[str, Any],
        memory_candidates_applied: list[Any],
        evidence_judgment: Mapping[str, Any] | None,
        safety_repair: bool,
        information_gain_proxy: float,
        repetition_pressure: float,
        conflict_level: float,
        behavioral_pull: float,
        evidence_refs: list[str] | None = None,
        relationship_value_context: Mapping[str, Any] | None = None,
    ) -> M13RewardEvaluationResult:
        reward_state = normalize_affective_reward_proxy_state(
            normalize_m13_drive_state(m13_state).get("affective_reward_proxy")
        )
        pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic_fingerprint)
        path_rows = list(reward_state.get("tolerance_by_path", []))
        path_row = _path_row_lookup(path_rows, path_id=pid) or {
            "predicted_reward": _bounded_float(
                _mapping(reward_state.get("predicted_reward_by_path")).get(pid, 0.0)
            ),
            "tolerance": 0.0,
            "opponent_strength": 0.0,
        }
        predicted = _bounded_float(path_row.get("predicted_reward"))
        tolerance = _bounded_float(path_row.get("tolerance"))
        opponent = _bounded_float(
            max(_bounded_float(reward_state.get("opponent_strength")), _bounded_float(path_row.get("opponent_strength")))
        )
        baseline = _bounded_float(reward_state.get("reward_baseline"), default=0.35)

        pos, pos_reasons = _same_turn_positive_components(
            conscious_plan=conscious_plan,
            reply_validation=reply_validation,
            post_reply_observer=post_reply_observer,
            memory_candidates_applied=memory_candidates_applied,
            evidence_judgment=evidence_judgment,
            relationship_value_context=relationship_value_context,
        )
        neg, neg_reasons = _same_turn_negative_components(
            conscious_plan=conscious_plan,
            reply_validation=reply_validation,
            post_reply_observer=post_reply_observer,
            safety_repair=safety_repair,
            repetition_pressure=repetition_pressure,
            conflict_level=conflict_level,
        )
        observed = _bounded_float(pos - neg)
        relief = _bounded_float(max(0.0, observed - predicted))
        information_gain = _bounded_float(information_gain_proxy)
        net = compute_net_affective_reward_proxy(
            observed_reward_proxy=observed,
            relief_proxy=relief,
            information_gain_proxy=information_gain,
            predicted_reward=predicted,
            reward_baseline=baseline,
            tolerance=tolerance,
            opponent_strength=opponent,
        )
        prediction_error = round(observed - predicted, 6)
        stale_proxy = predicted >= 0.35 and repetition_pressure >= 0.25 and net <= 0.35
        reason_codes = list(dict.fromkeys([*pos_reasons[:4], *neg_reasons[:4]]))[:MAX_REASON_CODES]

        event_id = _new_id("m13_reward")
        event = {
            "type": "M13RewardEvaluationEvent",
            "event_id": event_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "path_id": pid,
            "net_affective_reward_proxy": net,
            "observed_reward_proxy": round(observed, 6),
            "relief_proxy": round(relief, 6),
            "information_gain_proxy": round(information_gain, 6),
            "predicted_reward": round(predicted, 6),
            "reward_baseline": round(baseline, 6),
            "tolerance": round(tolerance, 6),
            "opponent_strength": round(opponent, 6),
            "prediction_error_proxy": prediction_error,
            "behavioral_pull_reference": round(_bounded_float(behavioral_pull), 6),
            "reason_codes": reason_codes,
            "engineering_proxy_label": "mvp_local_affective_reward_proxy",
        }
        return M13RewardEvaluationResult(
            event_id=event_id,
            turn_id=turn_id,
            turn_index=turn_index,
            path_id=pid,
            net_affective_reward_proxy=net,
            observed_reward_proxy=observed,
            relief_proxy=relief,
            information_gain_proxy=information_gain,
            predicted_reward=predicted,
            reward_baseline=baseline,
            tolerance=tolerance,
            opponent_strength=opponent,
            prediction_error_proxy=prediction_error,
            path_feels_stale_proxy=stale_proxy,
            reason_codes=reason_codes,
            evidence_refs=_string_list(evidence_refs, limit=8),
            events=[event],
        )


def build_affective_drive_guidance(evaluation: M13RewardEvaluationResult) -> dict[str, Any]:
    stale = evaluation.path_feels_stale_proxy
    opponent_high = evaluation.opponent_strength >= 0.35
    hint = ""
    if stale and opponent_high:
        hint = (
            "An easy repeat path may feel less fresh now. Prefer evidence-safe progress or a "
            "small new angle when it still fits the user's request."
        )
    elif stale:
        hint = "The current path is getting predictable. A small fresh angle may help if it fits the request."
    elif opponent_high:
        hint = "Recent turns had repair pressure. Keep claims cautious and within evidence boundaries."
    return {
        "path_feels_stale_proxy": stale,
        "relief_seeking_tendency": evaluation.relief_proxy >= 0.25 and evaluation.net_affective_reward_proxy < 0.35,
        "novelty_needed": stale or evaluation.information_gain_proxy < 0.2 or opponent_high,
        "caution_about_repeating_easy_path": stale or evaluation.tolerance >= 0.35 or opponent_high,
        "ordinary_language_hint": hint,
        "advisory_only": True,
        "engineering_proxy_label": "mvp_local_affective_reward_proxy",
    }


def sanitize_affective_drive_guidance_for_prompt(guidance: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(_mapping(guidance))
    for key in _PROMPT_FORBIDDEN_AFFECTIVE_KEYS:
        cleaned.pop(key, None)
    return cleaned


def merge_affective_guidance_into_control(
    memory_dynamics: dict[str, Any],
    evaluation: M13RewardEvaluationResult,
) -> None:
    control = _mapping(memory_dynamics.get("control_guidance"))
    guidance = build_affective_drive_guidance(evaluation)
    hint = str(guidance.get("ordinary_language_hint", "") or "").strip()
    drive = _mapping(control.get("drive_guidance"))
    lines = list(drive.get("prompt_safe_lines") or [])
    if hint and hint not in lines:
        lines.append(hint[:160])
    drive["prompt_safe_lines"] = lines[:6]
    control["drive_guidance"] = drive
    control["affective_drive_guidance"] = sanitize_affective_drive_guidance_for_prompt(guidance)
    memory_dynamics["control_guidance"] = control


def prompt_safe_m13_reward_diagnostics(evaluation: M13RewardEvaluationResult) -> dict[str, Any]:
    return {
        "event_id": evaluation.event_id,
        "path_feels_stale_proxy": evaluation.path_feels_stale_proxy,
        "reason_codes": list(evaluation.reason_codes[:4]),
        "engineering_proxy_label": "mvp_local_affective_reward_proxy",
        "settlement_hint": (
            "repeat path feels less fresh while habit may still be easy to reuse"
            if evaluation.path_feels_stale_proxy
            else ""
        ),
    }


def prompt_safe_m13_reward_ui_labels() -> dict[str, str]:
    """UI-facing labels must avoid reward/tolerance/addiction jargon."""
    return {
        "path_staleness": "Path freshness",
        "repeat_tendency": "Repeat tendency",
        "fresh_angle": "Needs a fresher angle",
        "easy_path_caution": "Easy-path caution",
    }


def create_pending_settlement(
    *,
    turn_index: int,
    action: str,
    topic_fingerprint: str,
    reply_summary: str,
    predicted_reward: float,
    predicted_relief: float,
    information_gain_proxy: float,
    evidence_refs: list[str],
    reply_validation: Mapping[str, Any],
    post_reply_observer: Mapping[str, Any],
    conscious_plan: Mapping[str, Any],
    memory_candidates_applied: list[Any],
    evidence_judgment: Mapping[str, Any] | None,
    safety_repair: bool,
    repetition_pressure: float,
    conflict_level: float,
) -> dict[str, Any]:
    return {
        "pending_id": _new_id("m13_pending"),
        "prior_turn_index": turn_index,
        "prior_action": action,
        "prior_topic_fingerprint": topic_fingerprint,
        "prior_reply_summary": str(reply_summary or "")[:160],
        "predicted_reward": round(_bounded_float(predicted_reward), 6),
        "predicted_relief": round(_bounded_float(predicted_relief), 6),
        "prior_information_gain": round(_bounded_float(information_gain_proxy), 6),
        "evidence_refs": _string_list(evidence_refs, limit=8),
        "expires_after_turns": PENDING_SETTLEMENT_TTL,
        "prior_reply_validation": dict(_mapping(reply_validation)),
        "prior_post_reply_observer": dict(_mapping(post_reply_observer)),
        "prior_expectation_results": list(_mapping(conscious_plan).get("expectation_results", []) or []),
        "prior_memory_candidates_applied": bool(memory_candidates_applied),
        "prior_evidence_judgment": dict(_mapping(evidence_judgment)) if evidence_judgment else {},
        "prior_safety_repair": safety_repair,
        "prior_repetition_pressure": round(_bounded_float(repetition_pressure), 6),
        "prior_conflict_level": round(_bounded_float(conflict_level), 6),
    }


def apply_post_turn_m13_reward_state(
    m13_state: dict[str, Any],
    *,
    evaluation: M13RewardEvaluationResult,
    user_id: str,
    action: str,
    topic_fingerprint: str,
    turn_index: int,
    reply_summary: str,
    reply_validation: Mapping[str, Any],
    post_reply_observer: Mapping[str, Any],
    conscious_plan: Mapping[str, Any],
    memory_candidates_applied: list[Any],
    evidence_judgment: Mapping[str, Any] | None,
    safety_repair: bool,
    repetition_pressure: float,
    conflict_level: float,
    behavioral_pull: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = normalize_m13_drive_state(m13_state)
    reward_state = normalize_affective_reward_proxy_state(state.get("affective_reward_proxy"))
    events: list[dict[str, Any]] = []

    pid = path_id_for(action=action, user_id=user_id, topic_fingerprint=topic_fingerprint)
    path_rows = list(reward_state.get("tolerance_by_path", []))
    path_row = _upsert_path_row(
        path_rows,
        path_id=pid,
        action=action,
        topic_fingerprint=topic_fingerprint,
        turn_index=turn_index,
    )
    predicted_map = dict(reward_state.get("predicted_reward_by_path", {}))
    old_predicted = _bounded_float(path_row.get("predicted_reward", predicted_map.get(pid, 0.0)))
    if evaluation.observed_reward_proxy >= 0.45:
        path_row["predicted_reward"] = round(
            _bounded_delta(old_predicted, evaluation.observed_reward_proxy, max_delta=MAX_SINGLE_TURN_PREDICTED_REWARD_DELTA),
            6,
        )
    predicted_map[pid] = path_row["predicted_reward"]
    path_row["last_net_reward_proxy"] = evaluation.net_affective_reward_proxy

    if repetition_pressure >= 0.3 and evaluation.observed_reward_proxy >= 0.45:
        path_row["tolerance"] = round(
            _bounded_delta(
                _bounded_float(path_row.get("tolerance")),
                min(1.0, _bounded_float(path_row.get("tolerance")) + 0.02),
                max_delta=MAX_SINGLE_TURN_TOLERANCE_DELTA,
            ),
            6,
        )

    _evict_tolerance_rows(path_rows)
    reward_state["tolerance_by_path"] = path_rows
    reward_state["predicted_reward_by_path"] = predicted_map
    reward_state["last_net_reward_proxy"] = evaluation.net_affective_reward_proxy
    reward_state["last_relief_proxy"] = evaluation.relief_proxy
    reward_state["last_information_gain_proxy"] = evaluation.information_gain_proxy

    pending = create_pending_settlement(
        turn_index=turn_index,
        action=action,
        topic_fingerprint=topic_fingerprint,
        reply_summary=reply_summary,
        predicted_reward=path_row["predicted_reward"],
        predicted_relief=evaluation.relief_proxy,
        information_gain_proxy=evaluation.information_gain_proxy,
        evidence_refs=evaluation.evidence_refs,
        reply_validation=reply_validation,
        post_reply_observer=post_reply_observer,
        conscious_plan=conscious_plan,
        memory_candidates_applied=memory_candidates_applied,
        evidence_judgment=evidence_judgment,
        safety_repair=safety_repair,
        repetition_pressure=repetition_pressure,
        conflict_level=conflict_level,
    )
    pending_rows = list(reward_state.get("pending_settlements", []))
    pending_rows.append(pending)
    reward_state["pending_settlements"] = pending_rows[-MAX_PENDING_SETTLEMENTS:]

    history = list(reward_state.get("reward_history", []))
    history.append(
        {
            "turn_index": turn_index,
            "path_id": pid,
            "net_affective_reward_proxy": evaluation.net_affective_reward_proxy,
            "behavioral_pull_reference": round(_bounded_float(behavioral_pull), 6),
        }
    )
    reward_state["reward_history"] = history[-MAX_REWARD_HISTORY:]

    state["affective_reward_proxy"] = reward_state

    patch_id = _new_id("m13_reward_patch")
    events.append(
        {
            "type": "M13RewardPatchProposal",
            "patch_id": patch_id,
            "target": "m13_drive_state.affective_reward_proxy",
            "operation": "update",
            "field_path": "affective_reward_proxy",
            "previous_summary": f"predicted={old_predicted:.2f}",
            "new_summary": (
                f"net={evaluation.net_affective_reward_proxy:.2f} "
                f"predicted={path_row['predicted_reward']:.2f} "
                f"tolerance={path_row.get('tolerance', 0.0):.2f}"
            ),
            "source_event_id": evaluation.event_id,
            "reason": "affective_reward_proxy_turn_update",
            "confidence": 0.7,
            "ttl": 5,
            "engineering_proxy_label": "mvp_local_affective_reward_proxy",
        }
    )
    events.append(
        {
            "type": "M13RewardPatchCommit",
            "commit_id": _new_id("m13_reward_commit"),
            "patch_id": patch_id,
            "accepted": True,
            "owner": "MVPDialogueRuntime",
            "reason": "affective_reward_proxy_turn_update",
            "committed_summary": f"path={pid}",
        }
    )
    return state, events


def apply_reward_pull_connection(
    m13_state: dict[str, Any],
    *,
    evaluation: M13RewardEvaluationResult,
    behavioral_pull: float,
) -> dict[str, Any]:
    """Small bounded path-confidence nudge; behavioral_pull remains separate."""
    state = normalize_m13_drive_state(m13_state)
    if evaluation.net_affective_reward_proxy < 0.25 or behavioral_pull < M13_THRESHOLDS.min_habit_for_action_match:
        return state
    patterns = [dict(row) for row in state.get("path_patterns_by_action", []) if isinstance(row, Mapping)]
    parts = evaluation.path_id.split("|", 2)
    if len(parts) < 3:
        return state
    action, user_id, topic = parts[0], parts[1], parts[2]
    from segmentum.dialogue.runtime.m13_drive import _pattern_lookup

    row = _pattern_lookup(
        patterns,
        action=action,
        user_id=user_id,
        topic_fingerprint=topic,
    )
    if row is None:
        return state
    hp = _bounded_float(row.get("habit_precision"))
    if evaluation.net_affective_reward_proxy >= 0.45:
        row["habit_precision"] = round(min(1.0, hp + MAX_REWARD_PULL_CONFIDENCE_BOOST), 6)
    state["path_patterns_by_action"] = patterns
    return state
