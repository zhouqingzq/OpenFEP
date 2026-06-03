from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping

from .action_schema import action_name
from .fep_surrogate import CHANNEL_WEIGHTS, STATE_MODALITIES, build_free_energy_surrogate


def _clamp(value: object, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(low, min(high, numeric))


def _coerce_state(payload: Mapping[str, object] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _action_dimensions(action: str) -> tuple[str, ...]:
    token = action_name(action).strip().lower()
    if token in {"hide", "rest", "exploit_shelter"}:
        return ("danger", "shelter")
    if token == "scan":
        return ("danger", "novelty")
    if token == "forage":
        return ("food", "danger")
    if token == "seek_contact":
        return ("social", "danger")
    if token == "observe_world":
        return ("novelty",)
    return ()


def _effect_dimensions(predicted_effects: Mapping[str, object] | None) -> tuple[str, ...]:
    if not isinstance(predicted_effects, Mapping):
        return ()
    dims: list[str] = []
    energy_delta = float(predicted_effects.get("energy_delta", 0.0) or 0.0)
    stress_delta = float(predicted_effects.get("stress_delta", 0.0) or 0.0)
    if energy_delta > 0.0:
        dims.append("food")
    if stress_delta < 0.0:
        dims.extend(["danger", "shelter"])
    return tuple(dict.fromkeys(dims))


def _member_signature(payload: Mapping[str, object]) -> str:
    source_ids = tuple(sorted(str(item) for item in payload.get("source_episode_ids", []) if str(item)))
    seed = "|".join(
        [
            str(payload.get("path_id", "")),
            str(payload.get("dominant_action", "")),
            ",".join(source_ids),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LocalFieldMember:
    path_id: str
    dominant_action: str
    source_episode_ids: list[str]
    source_memory_ids: list[str]
    proposal_score: float
    path_quality: float
    path_polarity: str
    support_count: int
    cue_signature: dict[str, object]
    outcome_profile: dict[str, object]
    risk_profile: dict[str, object]
    expected_surprise_profile: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "dominant_action": self.dominant_action,
            "source_episode_ids": list(self.source_episode_ids),
            "source_memory_ids": list(self.source_memory_ids),
            "proposal_score": round(self.proposal_score, 6),
            "path_quality": round(self.path_quality, 6),
            "path_polarity": self.path_polarity,
            "support_count": int(self.support_count),
            "cue_signature": dict(self.cue_signature),
            "outcome_profile": dict(self.outcome_profile),
            "risk_profile": dict(self.risk_profile),
            "expected_surprise_profile": dict(self.expected_surprise_profile),
        }


@dataclass(frozen=True)
class LocalMemoryField:
    field_id: str
    member_path_ids: list[str]
    member_memory_ids: list[str]
    potential_by_channel: dict[str, float]
    gradient_by_channel: dict[str, float]
    basin_strength: float
    ridge_strength: float
    conflict_zones: list[str]
    field_confidence: float
    effective_member_count: float
    dominant_path_share: float
    synergy_margin: float
    gradient_magnitude: float
    field_flatness: float
    conflict_density: float
    action_influences: dict[str, dict[str, object]] = field(default_factory=dict)
    neighborhood_members: list[dict[str, object]] = field(default_factory=list)
    counterfactual_audit: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "member_path_ids": list(self.member_path_ids),
            "member_memory_ids": list(self.member_memory_ids),
            "potential_by_channel": {
                key: round(value, 6) for key, value in sorted(self.potential_by_channel.items())
            },
            "gradient_by_channel": {
                key: round(value, 6) for key, value in sorted(self.gradient_by_channel.items())
            },
            "basin_strength": round(self.basin_strength, 6),
            "ridge_strength": round(self.ridge_strength, 6),
            "conflict_zones": list(self.conflict_zones),
            "field_confidence": round(self.field_confidence, 6),
            "effective_member_count": round(self.effective_member_count, 6),
            "dominant_path_share": round(self.dominant_path_share, 6),
            "synergy_margin": round(self.synergy_margin, 6),
            "gradient_magnitude": round(self.gradient_magnitude, 6),
            "field_flatness": round(self.field_flatness, 6),
            "conflict_density": round(self.conflict_density, 6),
            "action_influences": {
                key: dict(value) for key, value in sorted(self.action_influences.items())
            },
            "neighborhood_members": [dict(item) for item in self.neighborhood_members],
            "counterfactual_audit": dict(self.counterfactual_audit),
        }


def _dedupe_members(paths: list[dict[str, object]]) -> list[LocalFieldMember]:
    deduped: dict[str, LocalFieldMember] = {}
    for payload in paths:
        signature = _member_signature(payload)
        current = LocalFieldMember(
            path_id=str(payload.get("path_id", "")),
            dominant_action=action_name(payload.get("dominant_action", "")),
            source_episode_ids=[str(item) for item in payload.get("source_episode_ids", []) if str(item)],
            source_memory_ids=[str(item) for item in payload.get("source_memory_ids", []) if str(item)],
            proposal_score=float(payload.get("retrieval_score", payload.get("proposal_score", 0.0)) or 0.0),
            path_quality=float(payload.get("path_quality", 0.0) or 0.0),
            path_polarity=str(payload.get("path_polarity", "positive")),
            support_count=int(payload.get("support_count", 0) or 0),
            cue_signature=dict(payload.get("cue_signature", {}))
            if isinstance(payload.get("cue_signature"), dict)
            else {},
            outcome_profile=dict(payload.get("outcome_profile", {}))
            if isinstance(payload.get("outcome_profile"), dict)
            else {},
            risk_profile=dict(payload.get("risk_profile", {}))
            if isinstance(payload.get("risk_profile"), dict)
            else {},
            expected_surprise_profile=dict(payload.get("expected_surprise_profile", {}))
            if isinstance(payload.get("expected_surprise_profile"), dict)
            else {},
        )
        previous = deduped.get(signature)
        if previous is None or current.proposal_score > previous.proposal_score:
            deduped[signature] = current
    members = list(deduped.values())
    members.sort(
        key=lambda item: (
            (item.proposal_score * 0.60) + (item.path_quality * 0.40),
            item.path_id,
        ),
        reverse=True,
    )
    return members


def _normalized_member_weights(members: list[LocalFieldMember]) -> dict[str, float]:
    raw: dict[str, float] = {}
    for member in members:
        support_bonus = min(1.0, float(member.support_count) / 4.0) * 0.12
        contradiction_penalty = _clamp(dict(member.risk_profile).get("contradiction_burden", 0.0)) * 0.18
        raw[member.path_id] = max(
            0.01,
            (member.proposal_score * 0.55)
            + (member.path_quality * 0.45)
            + support_bonus
            - contradiction_penalty,
        )
    total = sum(raw.values()) or 1.0
    return {
        key: round(value / total, 6)
        for key, value in raw.items()
    }


def _field_channels(member: LocalFieldMember) -> tuple[str, ...]:
    cue = dict(member.cue_signature)
    sensitive = [str(item) for item in cue.get("sensitive_channels", []) if str(item)]
    action_channels = list(_action_dimensions(member.dominant_action))
    effect_channels = list(_effect_dimensions(member.outcome_profile.get("predicted_effects")))
    channels = [*sensitive, *action_channels, *effect_channels]
    return tuple(dict.fromkeys(channel for channel in channels if channel in STATE_MODALITIES))


def _counterfactual_signature(action: str, channels: list[str]) -> tuple[str, tuple[str, ...]]:
    return (action_name(action), tuple(sorted(str(item) for item in channels[:2] if str(item))))


def build_local_memory_field(
    neighborhood_paths: list[dict[str, object]],
    *,
    baseline_prediction: Mapping[str, object] | None,
    errors: Mapping[str, object] | None,
    body_state: Mapping[str, object] | None,
) -> LocalMemoryField | None:
    members = _dedupe_members(list(neighborhood_paths))
    if not members:
        return None

    weights = _normalized_member_weights(members)
    member_ids = [member.path_id for member in members]
    memory_ids = sorted(
        {
            memory_id
            for member in members
            for memory_id in member.source_memory_ids
        }
    )
    baseline_fe = build_free_energy_surrogate(errors=errors, body_state=body_state)
    baseline_prediction_state = _coerce_state(baseline_prediction)
    potential_by_channel = {channel: float(baseline_fe.channel_costs.get(channel, 0.0)) for channel in STATE_MODALITIES}
    gradient_by_channel = {channel: 0.0 for channel in STATE_MODALITIES}
    utility_totals = {channel: 0.0 for channel in STATE_MODALITIES}
    caution_totals = {channel: 0.0 for channel in STATE_MODALITIES}
    action_totals: dict[str, dict[str, float]] = {}
    polarity_by_action: dict[str, set[str]] = {}

    for member in members:
        weight = float(weights.get(member.path_id, 0.0))
        outcome_profile = dict(member.outcome_profile)
        risk_profile = dict(member.risk_profile)
        surprise_profile = dict(member.expected_surprise_profile)
        future_utility = max(0.0, float(outcome_profile.get("future_path_utility", 0.0) or 0.0))
        contradiction = max(0.0, float(risk_profile.get("contradiction_burden", 0.0) or 0.0))
        mean_risk = max(0.0, float(risk_profile.get("mean_risk", 0.0) or 0.0))
        caution_score = max(
            mean_risk,
            contradiction,
            max(0.0, float(surprise_profile.get("mean_prediction_error", 0.0) or 0.0)),
        )
        channels = _field_channels(member)
        channel_gain = max(0.0, member.path_quality) * max(0.08, future_utility + 0.10)
        channel_cost = caution_score * 0.40
        if member.path_polarity == "negative":
            channel_cost += 0.12
        elif member.path_polarity == "cautionary":
            channel_cost += 0.06
        for channel in channels:
            utility_totals[channel] += weight * channel_gain
            caution_totals[channel] += weight * channel_cost
        action_key = action_name(member.dominant_action)
        bucket = action_totals.setdefault(
            action_key,
            {
                "support": 0.0,
                "quality": 0.0,
                "utility": 0.0,
                "risk": 0.0,
                "surprise": 0.0,
                "preferred_probability": 0.0,
            },
        )
        bucket["support"] += weight
        bucket["quality"] += weight * member.path_quality
        bucket["utility"] += weight * future_utility
        bucket["risk"] += weight * mean_risk
        bucket["surprise"] += weight * max(0.0, float(surprise_profile.get("mean_prediction_error", 0.0) or 0.0))
        bucket["preferred_probability"] += weight * max(0.0, float(outcome_profile.get("preferred_probability", 0.0) or 0.0))
        polarity_by_action.setdefault(action_key, set()).add(member.path_polarity)

    for channel in STATE_MODALITIES:
        potential = max(0.0, potential_by_channel[channel] + caution_totals[channel] - utility_totals[channel])
        potential_by_channel[channel] = round(potential, 6)
        gradient_by_channel[channel] = round(caution_totals[channel] - utility_totals[channel], 6)

    dominant_path_share = max(float(weights.get(member.path_id, 0.0)) for member in members)
    effective_member_count = 1.0 / max(1e-9, sum(value * value for value in weights.values()))
    support_values = sorted((bucket["support"] for bucket in action_totals.values()), reverse=True)
    top_support = support_values[0] if support_values else 0.0
    second_support = support_values[1] if len(support_values) > 1 else 0.0
    field_flatness = _clamp(1.0 - max(0.0, top_support - second_support))
    conflict_density = _clamp(
        sum(
            min(1.0, len(polarities) / 2.0) * max(0.0, bucket["support"])
            for action, polarities in polarity_by_action.items()
            for bucket in [action_totals[action]]
            if len(polarities) > 1
        )
        + sum(caution_totals.values()) * 0.20
    )
    synergy_margin = max(0.0, top_support - dominant_path_share)
    basin_strength = _clamp(sum(max(0.0, utility_totals[channel] - caution_totals[channel]) for channel in STATE_MODALITIES))
    ridge_strength = _clamp(sum(max(0.0, caution_totals[channel] - utility_totals[channel]) for channel in STATE_MODALITIES))
    gradient_magnitude = sum(abs(value) for value in gradient_by_channel.values()) / max(1, len(STATE_MODALITIES))
    field_confidence = _clamp(
        0.18
        + min(1.0, effective_member_count / 3.0) * 0.30
        + synergy_margin * 0.22
        + basin_strength * 0.18
        - field_flatness * 0.12
        - conflict_density * 0.20
    )
    conflict_zones = [
        channel
        for channel in STATE_MODALITIES
        if caution_totals[channel] > utility_totals[channel] and caution_totals[channel] >= 0.05
    ]

    action_influences: dict[str, dict[str, object]] = {}
    for action_key, bucket in sorted(action_totals.items()):
        support = max(1e-9, float(bucket["support"]))
        avg_quality = float(bucket["quality"]) / support
        avg_utility = float(bucket["utility"]) / support
        avg_risk = float(bucket["risk"]) / support
        avg_surprise = float(bucket["surprise"]) / support
        avg_probability = float(bucket["preferred_probability"]) / support
        dims = _action_dimensions(action_key)
        dominant_channels = [
            channel
            for channel in sorted(
                dims or STATE_MODALITIES,
                key=lambda item: (
                    abs(float(gradient_by_channel.get(item, 0.0))),
                    float(utility_totals.get(item, 0.0)) - float(caution_totals.get(item, 0.0)),
                ),
                reverse=True,
            )
            if channel in STATE_MODALITIES
        ][:2]
        ambiguity = (field_flatness * 0.18) + (conflict_density * 0.24)
        risk_adjustment = (
            (ridge_strength * 0.18)
            - (basin_strength * 0.14)
            + (avg_risk * 0.10)
        )
        surprise_adjustment = (
            (field_flatness * 0.12)
            + (conflict_density * 0.10)
            - (synergy_margin * 0.08)
            + (avg_surprise * 0.08)
        )
        field_score = (
            support
            + (avg_quality * 0.25)
            + (avg_utility * 0.20)
            + (synergy_margin * 0.22)
            - (avg_risk * 0.22)
            - (avg_surprise * 0.20)
            - (conflict_density * 0.18)
        )
        projected_fe = max(
            0.0,
            (avg_surprise + surprise_adjustment)
            + max(0.0, avg_risk + risk_adjustment)
            + ambiguity,
        )
        action_influences[action_key] = {
            "field_score": round(field_score, 6),
            "support": round(support, 6),
            "path_quality": round(avg_quality, 6),
            "future_path_utility": round(avg_utility, 6),
            "risk": round(max(0.0, avg_risk + risk_adjustment), 6),
            "expected_surprise": round(max(0.0, avg_surprise + surprise_adjustment), 6),
            "preferred_probability": round(_clamp(avg_probability + (basin_strength * 0.08) - (conflict_density * 0.05)), 6),
            "dominant_channels": list(dominant_channels),
            "ambiguity_cost": round(ambiguity, 6),
            "projected_expected_free_energy": round(projected_fe, 6),
            "field_adjustment": {
                "basin_strength": round(basin_strength, 6),
                "ridge_strength": round(ridge_strength, 6),
                "risk_adjustment": round(risk_adjustment, 6),
                "surprise_adjustment": round(surprise_adjustment, 6),
                "dominant_channels": list(dominant_channels),
                "gradient_magnitude": round(float(gradient_magnitude), 6),
            },
        }

    best_single_member = max(
        members,
        key=lambda item: (float(weights.get(item.path_id, 0.0)), item.path_id),
    )
    best_single_action = action_name(best_single_member.dominant_action)
    best_single_channels = list(_field_channels(best_single_member))[:2]
    naive_topk_action = max(
        action_influences.items(),
        key=lambda item: (float(item[1]["support"]), float(item[1]["path_quality"]), item[0]),
    )[0]
    naive_topk_channels = list(action_influences[naive_topk_action].get("dominant_channels", []))
    field_selected_action = max(
        action_influences.items(),
        key=lambda item: (float(item[1]["field_score"]), -float(item[1]["projected_expected_free_energy"]), item[0]),
    )[0]
    field_channels = list(action_influences[field_selected_action].get("dominant_channels", []))
    best_single_signature = _counterfactual_signature(best_single_action, best_single_channels)
    naive_topk_signature = _counterfactual_signature(naive_topk_action, naive_topk_channels)
    field_signature = _counterfactual_signature(field_selected_action, field_channels)
    best_single_counterfactual = best_single_signature == field_signature
    naive_topk_counterfactual = naive_topk_signature == field_signature
    field_fe = float(action_influences[field_selected_action]["projected_expected_free_energy"])
    best_single_fe = float(action_influences.get(best_single_action, {}).get("projected_expected_free_energy", field_fe))
    naive_topk_fe = float(action_influences.get(naive_topk_action, {}).get("projected_expected_free_energy", field_fe))
    fe_advantage_vs_best = best_single_fe - field_fe
    fe_advantage_vs_naive = naive_topk_fe - field_fe
    if best_single_counterfactual:
        status = "suppressed_best_single_equivalent"
    elif naive_topk_counterfactual:
        status = "suppressed_naive_topk_equivalent"
    elif max(fe_advantage_vs_best, fe_advantage_vs_naive) <= 0.0:
        status = "field_divergent_no_gain"
    else:
        status = "field_required"

    field_id = "field:" + hashlib.sha1(
        "|".join(member_ids).encode("utf-8")
    ).hexdigest()[:12]
    return LocalMemoryField(
        field_id=field_id,
        member_path_ids=member_ids,
        member_memory_ids=memory_ids,
        potential_by_channel=potential_by_channel,
        gradient_by_channel=gradient_by_channel,
        basin_strength=basin_strength,
        ridge_strength=ridge_strength,
        conflict_zones=conflict_zones,
        field_confidence=field_confidence,
        effective_member_count=effective_member_count,
        dominant_path_share=dominant_path_share,
        synergy_margin=synergy_margin,
        gradient_magnitude=gradient_magnitude,
        field_flatness=field_flatness,
        conflict_density=conflict_density,
        action_influences=action_influences,
        neighborhood_members=[member.to_dict() for member in members],
        counterfactual_audit={
            "best_single_action": best_single_action,
            "naive_topk_action": naive_topk_action,
            "field_selected_action": field_selected_action,
            "best_single_signature": [best_single_signature[0], list(best_single_signature[1])],
            "naive_topk_signature": [naive_topk_signature[0], list(naive_topk_signature[1])],
            "field_signature": [field_signature[0], list(field_signature[1])],
            "best_single_counterfactual": bool(best_single_counterfactual),
            "naive_topk_counterfactual": bool(naive_topk_counterfactual),
            "field_required": bool(status == "field_required"),
            "status": status,
            "chosen_decision_subsequent_fe": round(field_fe, 6),
            "best_single_counterfactual_fe": round(best_single_fe, 6),
            "naive_topk_counterfactual_fe": round(naive_topk_fe, 6),
            "fe_advantage_vs_best_single": round(fe_advantage_vs_best, 6),
            "fe_advantage_vs_naive_topk": round(fe_advantage_vs_naive, 6),
            "outcome_quantity": "m17_5_expected_free_energy_surrogate",
        },
    )
