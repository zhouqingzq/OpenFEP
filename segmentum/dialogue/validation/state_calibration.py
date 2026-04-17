from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, Mapping, Protocol

from ...agent import SegmentAgent
from ...self_model import NarrativePriors, PreferredPolicies
from ...slow_learning import SlowTraitState
from ..actions import DIALOGUE_ACTION_NAMES
from .surface_profile import tokenize_surface


class _Predictor(Protocol):
    def predict(self, text: str) -> object: ...


_EXPLOIT_ACTIONS = {"agree", "empathize", "elaborate", "joke"}
_ESCAPE_ACTIONS = {"deflect", "minimal_response", "disengage", "disagree"}
_EXPLORE_ACTIONS = {"ask_question", "introduce_topic", "share_opinion"}


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: object, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _iter_reply_pairs(
    sessions: Iterable[Mapping[str, object]],
    *,
    user_uid: int,
) -> Iterable[tuple[str, str]]:
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            continue
        pending_partner = ""
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            body = str(turn.get("body", "")).strip()
            if not body:
                continue
            sender_uid = _safe_int(turn.get("sender_uid"), -1)
            if sender_uid == int(user_uid):
                if pending_partner:
                    yield pending_partner, body
                    pending_partner = ""
                continue
            pending_partner = body


def _fallback_action(text: str) -> tuple[str, str]:
    compact = text.strip()
    if len(compact) <= 3:
        return "minimal_response", "escape"
    if "?" in compact or "锛?" in compact or "？" in compact:
        return "ask_question", "explore"
    return "elaborate", "exploit"


def _predict_action(text: str, classifier: _Predictor | None) -> tuple[str, str]:
    if classifier is None:
        return _fallback_action(text)
    try:
        pred = classifier.predict(text)
    except Exception:
        return _fallback_action(text)
    action = str(getattr(pred, "label_11", "") or "")
    strategy = str(getattr(pred, "label_3", "") or "")
    if not action or not strategy:
        return _fallback_action(text)
    return action, strategy


def _profile_numeric(user_dataset: Mapping[str, object], key: str, default: float = 0.5) -> float:
    profile = user_dataset.get("profile", {})
    if not isinstance(profile, Mapping):
        return default
    return _safe_float(profile.get(key), default)


def _target_traits(
    user_dataset: Mapping[str, object],
    *,
    reply_count: int,
    action_counts: Counter[str],
    strategy_counts: Counter[str],
    avg_reply_chars: float,
    ultra_short_ratio: float,
    token_variety: float,
    partner_question_ratio: float,
) -> SlowTraitState:
    total = float(max(1, reply_count))
    exploit_ratio = float(strategy_counts.get("exploit", 0)) / total
    explore_ratio = float(strategy_counts.get("explore", 0)) / total
    escape_ratio = float(strategy_counts.get("escape", 0)) / total
    supportive_ratio = sum(action_counts.get(name, 0) for name in _EXPLOIT_ACTIONS) / total
    guarded_ratio = sum(action_counts.get(name, 0) for name in _ESCAPE_ACTIONS) / total
    question_ratio = float(action_counts.get("ask_question", 0)) / total

    agreeableness = _profile_numeric(user_dataset, "agreeableness", 0.5)
    extraversion = _profile_numeric(user_dataset, "extraversion", 0.5)
    openness = _profile_numeric(user_dataset, "openness", 0.5)
    neuroticism = _profile_numeric(user_dataset, "neuroticism", 0.5)
    profile_trust = _profile_numeric(user_dataset, "trust_prior", 0.0)

    length_signal = _clamp((avg_reply_chars - 8.0) / 48.0)
    trust_from_profile = _clamp(0.5 + profile_trust * 0.5)
    social = _clamp(
        0.36
        + supportive_ratio * 0.26
        + exploit_ratio * 0.12
        + length_signal * 0.10
        + (agreeableness - 0.5) * 0.18
        + (extraversion - 0.5) * 0.12
        - ultra_short_ratio * 0.16
    )
    trust = _clamp(
        0.38
        + supportive_ratio * 0.20
        + exploit_ratio * 0.12
        + trust_from_profile * 0.18
        - guarded_ratio * 0.20
        - partner_question_ratio * 0.06
    )
    caution = _clamp(
        0.34
        + guarded_ratio * 0.28
        + escape_ratio * 0.16
        + ultra_short_ratio * 0.18
        + partner_question_ratio * 0.08
        + (neuroticism - 0.5) * 0.16
        - supportive_ratio * 0.10
    )
    exploration = _clamp(
        0.34
        + explore_ratio * 0.28
        + question_ratio * 0.18
        + token_variety * 0.16
        + (openness - 0.5) * 0.18
        - guarded_ratio * 0.08
    )
    threat = _clamp(0.42 + caution * 0.36 + (neuroticism - 0.5) * 0.16 - trust * 0.12)
    return SlowTraitState(
        caution_bias=caution,
        threat_sensitivity=threat,
        trust_stance=trust,
        exploration_posture=exploration,
        social_approach=social,
    )


def _target_priors(traits: SlowTraitState, token_variety: float) -> NarrativePriors:
    return NarrativePriors(
        trust_prior=round((traits.trust_stance - 0.5) * 1.2, 6),
        controllability_prior=round((traits.exploration_posture - traits.caution_bias) * 0.8, 6),
        trauma_bias=round(max(0.0, traits.threat_sensitivity - 0.5) * 0.9, 6),
        contamination_sensitivity=0.0,
        meaning_stability=round((traits.social_approach + token_variety - 0.8) * 0.45, 6),
    )


def _target_policies(
    action_counts: Counter[str],
    strategy_counts: Counter[str],
    *,
    reply_count: int,
) -> PreferredPolicies:
    total = float(max(1, reply_count))
    action_distribution = {
        action: round(float(action_counts.get(action, 0)) / total, 6)
        for action in DIALOGUE_ACTION_NAMES
        if action_counts.get(action, 0) > 0
    }
    if strategy_counts:
        dominant_strategy = max(
            strategy_counts.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )[0]
    else:
        dominant_strategy = "expected_free_energy"
    ranked_actions = sorted(
        action_distribution.items(),
        key=lambda item: (-float(item[1]), DIALOGUE_ACTION_NAMES.index(item[0])),
    )
    preference_floor = 0.12 if reply_count >= 12 else 0.20
    learned_preferences = [
        action
        for action, frequency in ranked_actions
        if float(frequency) >= preference_floor
    ][:4]
    if not learned_preferences and ranked_actions:
        learned_preferences = [ranked_actions[0][0]]

    learned_avoidances: list[str] = []
    if reply_count >= 12:
        for action in ("agree", "elaborate", "share_opinion", "minimal_response"):
            frequency = float(action_distribution.get(action, 0.0))
            if frequency <= 0.03:
                learned_avoidances.append(action)

    return PreferredPolicies(
        dominant_strategy=str(dominant_strategy),
        action_distribution=action_distribution,
        risk_profile="risk_neutral",
        learned_avoidances=learned_avoidances[:3],
        learned_preferences=learned_preferences,
        last_updated_tick=0,
    )


def apply_train_state_calibration(
    agent: SegmentAgent,
    user_dataset: Mapping[str, object],
    *,
    classifier: _Predictor | None = None,
    source: str = "train",
) -> dict[str, object]:
    user_uid = _safe_int(user_dataset.get("uid"), 0)
    sessions = user_dataset.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    pairs = list(_iter_reply_pairs(sessions, user_uid=user_uid))
    if not pairs:
        return {
            "source": source,
            "train_only": True,
            "reply_count": 0,
            "applied": False,
            "reason": "no_train_replies",
        }

    action_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    lengths: list[int] = []
    tokens: Counter[str] = Counter()
    partner_questions = 0
    for partner_text, reply_text in pairs:
        action, strategy = _predict_action(reply_text, classifier)
        action_counts[action] += 1
        strategy_counts[strategy] += 1
        lengths.append(len(reply_text))
        tokens.update(tokenize_surface(reply_text))
        if "?" in partner_text or "锛?" in partner_text or "？" in partner_text:
            partner_questions += 1

    reply_count = len(pairs)
    avg_reply_chars = float(mean(lengths)) if lengths else 0.0
    ultra_short_ratio = sum(1 for value in lengths if value <= 3) / float(max(1, len(lengths)))
    token_variety = min(1.0, len(tokens) / float(max(1, reply_count * 6)))
    partner_question_ratio = partner_questions / float(max(1, reply_count))

    before_traits = agent.slow_variable_learner.state.traits.to_dict()
    before_priors = agent.self_model.narrative_priors.to_dict()
    traits = _target_traits(
        user_dataset,
        reply_count=reply_count,
        action_counts=action_counts,
        strategy_counts=strategy_counts,
        avg_reply_chars=avg_reply_chars,
        ultra_short_ratio=ultra_short_ratio,
        token_variety=token_variety,
        partner_question_ratio=partner_question_ratio,
    )
    priors = _target_priors(traits, token_variety)
    policies = _target_policies(
        action_counts,
        strategy_counts,
        reply_count=reply_count,
    )
    agent.slow_variable_learner.state.traits = traits
    agent.self_model.narrative_priors = priors
    agent.self_model.preferred_policies = policies
    after_traits = traits.to_dict()
    after_priors = priors.to_dict()
    return {
        "source": source,
        "train_only": True,
        "applied": True,
        "reply_count": int(reply_count),
        "action_counts": {key: int(value) for key, value in sorted(action_counts.items())},
        "strategy_counts": {key: int(value) for key, value in sorted(strategy_counts.items())},
        "policy_action_distribution": dict(policies.action_distribution),
        "policy_dominant_strategy": policies.dominant_strategy,
        "policy_learned_preferences": list(policies.learned_preferences),
        "policy_learned_avoidances": list(policies.learned_avoidances),
        "avg_reply_chars": round(avg_reply_chars, 6),
        "ultra_short_ratio": round(float(ultra_short_ratio), 6),
        "token_variety": round(float(token_variety), 6),
        "partner_question_ratio": round(float(partner_question_ratio), 6),
        "trait_deltas": {
            key: round(float(after_traits.get(key, 0.0)) - float(before_traits.get(key, 0.0)), 6)
            for key in sorted(after_traits)
        },
        "prior_deltas": {
            key: round(float(after_priors.get(key, 0.0)) - float(before_priors.get(key, 0.0)), 6)
            for key in sorted(after_priors)
        },
    }
