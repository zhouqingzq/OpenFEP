from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from ...agent import SegmentAgent
from ...self_model import NarrativePriors
from ...slow_learning import SlowTraitState
from ..lifecycle import ImplantationConfig, implant_personality
from ..observer import DialogueObserver
from ..world import DialogueWorld

_BIG_FIVE_KEYS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def _flatten_numeric(prefix: str, value: object, out: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), inner, out)
        return
    try:
        out[prefix] = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return


def _profile_vector(profile_like: Mapping[str, object]) -> dict[str, float]:
    profile = profile_like.get("profile")
    if not isinstance(profile, Mapping):
        profile = profile_like
    out: dict[str, float] = {}
    _flatten_numeric("", profile, out)
    if out:
        return out
    return {"_default": 0.0}


def _cosine_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return 1.0
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    l_norm = sum(float(left.get(key, 0.0)) ** 2 for key in keys) ** 0.5
    r_norm = sum(float(right.get(key, 0.0)) ** 2 for key in keys) ** 0.5
    if l_norm <= 1e-12 or r_norm <= 1e-12:
        return 1.0
    cosine = max(-1.0, min(1.0, dot / (l_norm * r_norm)))
    return max(0.0, 1.0 - cosine)


def create_default_agent(seed: int = 42) -> SegmentAgent:
    return SegmentAgent(rng=random.Random(int(seed)))


def create_wrong_agent(
    wrong_user_data: dict,
    config: ImplantationConfig,
    seed: int = 42,
) -> SegmentAgent:
    agent = SegmentAgent(rng=random.Random(int(seed)))
    observer = DialogueObserver()
    world = DialogueWorld(wrong_user_data, observer, seed=int(seed))
    implant_personality(agent, world, config)
    return agent


def _mean_map(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    acc: dict[str, float] = {}
    for item in values:
        for key, value in item.items():
            acc[key] = acc.get(key, 0.0) + float(value)
    inv = 1.0 / float(len(values))
    return {key: value * inv for key, value in acc.items()}


def _assign_if_present(obj: object, payload: Mapping[str, float]) -> None:
    for key, value in payload.items():
        if hasattr(obj, key):
            try:
                setattr(obj, key, float(value))
            except (TypeError, ValueError):
                continue


def _inject_average_personality_profile(agent: SegmentAgent, mean_vector: dict[str, float]) -> None:
    profile_obj = getattr(agent.self_model, "personality_profile", None)
    if profile_obj is None:
        return
    injected = 0
    for key in _BIG_FIVE_KEYS:
        if key not in mean_vector or not hasattr(profile_obj, key):
            continue
        try:
            v = float(mean_vector[key])
            setattr(profile_obj, key, max(0.0, min(1.0, v)))
            injected += 1
        except (TypeError, ValueError):
            continue
    priors_obj = getattr(agent.self_model, "narrative_priors", None)
    if priors_obj is not None and "trust_prior" in mean_vector and hasattr(priors_obj, "trust_prior"):
        try:
            setattr(priors_obj, "trust_prior", float(mean_vector["trust_prior"]))
            injected += 1
        except (TypeError, ValueError):
            pass
    if injected == 0:
        _assign_if_present(profile_obj, mean_vector)
        if priors_obj is not None:
            _assign_if_present(priors_obj, mean_vector)


def create_average_agent(
    all_user_profiles: list[dict],
    seed: int = 42,
) -> SegmentAgent:
    """Baseline C fallback: mean Big Five / profile scalars only (no full implant)."""
    agent = SegmentAgent(rng=random.Random(int(seed)))
    if not all_user_profiles:
        return agent
    vectors = [_profile_vector(item) for item in all_user_profiles if isinstance(item, Mapping)]
    mean_vector = _mean_map(vectors)
    _inject_average_personality_profile(agent, mean_vector)
    traits_obj = getattr(getattr(agent, "slow_variable_learner", None), "state", None)
    traits_obj = getattr(traits_obj, "traits", None)
    if traits_obj is not None:
        _assign_if_present(traits_obj, mean_vector)
    return agent


def build_population_average_agent(
    user_datasets: list[dict],
    config: ImplantationConfig,
    *,
    seed: int = 42,
) -> SegmentAgent:
    """Baseline C: full-data implant per user, then mean SlowTraitState + NarrativePriors (+ profile)."""
    agent = SegmentAgent(rng=random.Random(int(seed)))
    if not user_datasets:
        return agent
    trait_dicts: list[dict[str, float]] = []
    prior_dicts: list[dict[str, float]] = []
    profile_vectors: list[dict[str, float]] = []
    for idx, ds in enumerate(user_datasets):
        if not isinstance(ds, Mapping):
            continue
        sub = SegmentAgent(rng=random.Random(int(seed) + idx + 1))
        world = DialogueWorld(ds, DialogueObserver(), seed=int(seed) + idx + 1)
        implant_personality(sub, world, config)
        traits = getattr(getattr(sub.slow_variable_learner, "state", None), "traits", None)
        if traits is not None and hasattr(traits, "to_dict"):
            trait_dicts.append(dict(traits.to_dict()))
        priors = getattr(sub.self_model, "narrative_priors", None)
        if priors is not None and hasattr(priors, "to_dict"):
            prior_dicts.append(dict(priors.to_dict()))
        profile_vectors.append(_profile_vector(ds))
    mean_traits = _mean_map(trait_dicts) if trait_dicts else {}
    mean_priors = _mean_map(prior_dicts) if prior_dicts else {}
    mean_profile = _mean_map(profile_vectors) if profile_vectors else {}
    _inject_average_personality_profile(agent, mean_profile)
    if mean_traits:
        agent.slow_variable_learner.state.traits = SlowTraitState.from_dict(mean_traits)
    if mean_priors:
        agent.self_model.narrative_priors = NarrativePriors.from_dict(mean_priors)
    return agent


def clone_agent_template(template: SegmentAgent, *, seed: int) -> SegmentAgent:
    """Fresh agent for evaluation (dialogue mutates state)."""
    payload = template.to_dict()
    return SegmentAgent.from_dict(payload, rng=random.Random(int(seed)))


def select_wrong_users(
    target_user_profile: dict,
    candidate_user_profiles: list[dict],
    *,
    k: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Select deterministic semi-hard / medium / far wrong users."""
    target_uid = int(target_user_profile.get("uid", -1))
    target_vec = _profile_vector(target_user_profile)
    scored: list[tuple[float, int, dict]] = []
    for item in candidate_user_profiles:
        if not isinstance(item, Mapping):
            continue
        uid = int(item.get("uid", -999999))
        if uid == target_uid:
            continue
        vec = _profile_vector(item)
        dist = _cosine_distance(target_vec, vec)
        scored.append((float(dist), uid, dict(item)))
    if not scored:
        return []
    scored.sort(key=lambda row: (row[0], row[1]))
    n = len(scored)

    def _pick_idx(lo: int, hi: int, *, rng: random.Random) -> int:
        lo = max(0, min(n - 1, lo))
        hi = max(lo, min(n - 1, hi))
        return rng.randint(lo, hi)

    rng = random.Random(int(seed) + target_uid)
    semi_hard_idx = _pick_idx(max(0, int(0.15 * n)), max(0, int(0.35 * n)), rng=rng)
    medium_idx = _pick_idx(max(0, int(0.40 * n)), max(0, int(0.65 * n)), rng=rng)
    far_idx = _pick_idx(max(0, int(0.70 * n)), n - 1, rng=rng)
    chosen_indices = [semi_hard_idx, medium_idx, far_idx]
    unique_order: list[int] = []
    for idx in chosen_indices:
        if idx not in unique_order:
            unique_order.append(idx)
    if len(unique_order) < k:
        for idx in range(n):
            if idx not in unique_order:
                unique_order.append(idx)
            if len(unique_order) >= k:
                break
    picks: list[tuple[float, int, dict]] = [scored[idx] for idx in unique_order[: max(1, int(k))]]
    picks.sort(key=lambda row: (row[0], row[1]))
    band_names = ["semi-hard", "medium", "far"]
    selected: list[dict] = []
    for i, (dist, uid, payload) in enumerate(picks):
        payload = dict(payload)
        payload["_wrong_user_distance"] = round(float(dist), 6)
        payload["_wrong_user_band"] = band_names[min(i, len(band_names) - 1)]
        payload["_wrong_user_uid"] = int(uid)
        payload["_wrong_user_distance_rank"] = int(i)
        selected.append(payload)
    return selected

