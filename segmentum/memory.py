from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from statistics import mean

from .preferences import PreferenceModel, ValueHierarchy


@dataclass
class Episode:
    timestamp: int
    state_vector: dict[str, float]
    action_taken: str
    outcome_state: dict[str, float]
    predicted_outcome: str
    prediction_error: float
    risk: float
    value_score: float
    total_surprise: float
    embedding: list[float]
    preferred_probability: float
    preference_log_value: float

    @property
    def value_label(self) -> str:
        return self.predicted_outcome

    def to_dict(self) -> dict[str, object]:
        state_snapshot = _split_state_vector(self.state_vector)
        return {
            "timestamp": self.timestamp,
            "cycle": self.timestamp,
            "state_vector": dict(self.state_vector),
            "state_snapshot": state_snapshot,
            "action_taken": self.action_taken,
            "action": self.action_taken,
            "outcome_state": dict(self.outcome_state),
            "outcome": dict(self.outcome_state),
            "predicted_outcome": self.predicted_outcome,
            "prediction_error": self.prediction_error,
            "risk": self.risk,
            "value_score": self.value_score,
            "total_surprise": self.total_surprise,
            "weighted_surprise": self.total_surprise,
            "embedding": list(self.embedding),
            "value_label": self.predicted_outcome,
            "preferred_probability": self.preferred_probability,
            "preference_log_value": self.preference_log_value,
            "observation": state_snapshot["observation"],
            "prediction": state_snapshot["prediction"],
            "errors": state_snapshot["errors"],
            "body_state": state_snapshot["body_state"],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Episode:
        timestamp = int(payload.get("timestamp", payload.get("cycle", 0)))
        state_vector = payload.get("state_vector")
        if not isinstance(state_vector, dict):
            state_snapshot = payload.get("state_snapshot")
            if not isinstance(state_snapshot, dict):
                state_snapshot = {
                    "observation": dict(payload.get("observation", {})),
                    "prediction": dict(payload.get("prediction", {})),
                    "errors": dict(payload.get("errors", {})),
                    "body_state": dict(payload.get("body_state", {})),
                }
            state_vector = _flatten_state_snapshot(state_snapshot)
        embedding = payload.get("embedding")
        return cls(
            timestamp=timestamp,
            state_vector=_coerce_float_dict(state_vector),
            action_taken=str(payload.get("action_taken", payload.get("action", ""))),
            outcome_state=_coerce_float_dict(
                payload.get("outcome_state", payload.get("outcome"))
            ),
            predicted_outcome=str(
                payload.get("predicted_outcome", payload.get("value_label", "neutral"))
            ),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            risk=float(payload.get("risk", 0.0)),
            value_score=float(payload.get("value_score", 0.0)),
            total_surprise=float(
                payload.get("total_surprise", payload.get("weighted_surprise", 0.0))
            ),
            embedding=list(embedding) if isinstance(embedding, list) else [],
            preferred_probability=float(payload.get("preferred_probability", 0.0)),
            preference_log_value=float(payload.get("preference_log_value", 0.0)),
        )


@dataclass(frozen=True)
class MemoryDecision:
    value_score: float
    prediction_error: float
    total_surprise: float
    episode_created: bool
    predicted_outcome: str = "neutral"
    preferred_probability: float = 0.0
    risk: float = 0.0
    preference_log_value: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "value_score": self.value_score,
            "prediction_error": self.prediction_error,
            "total_surprise": self.total_surprise,
            "weighted_surprise": self.total_surprise,
            "episode_created": self.episode_created,
            "predicted_outcome": self.predicted_outcome,
            "value_label": self.predicted_outcome,
            "preferred_probability": self.preferred_probability,
            "risk": self.risk,
            "preference_log_value": self.preference_log_value,
        }

    @property
    def weighted_surprise(self) -> float:
        return self.total_surprise

    @property
    def value_label(self) -> str:
        return self.predicted_outcome


def compute_prediction_error(
    observation: dict[str, float],
    predicted_state: dict[str, float],
) -> float:
    keys = sorted(set(observation) | set(predicted_state))
    if not keys:
        return 0.0
    mismatch = mean(
        abs(observation.get(key, 0.0) - predicted_state.get(key, 0.0))
        for key in keys
    )
    return _clamp(mismatch)


def compute_surprise(prediction_error: float) -> float:
    return abs(prediction_error)


RISK_WEIGHT = 0.7


def compute_total_surprise(
    prediction_error: float,
    risk: float,
) -> float:
    return compute_surprise(prediction_error) + (RISK_WEIGHT * max(0.0, risk))


def compute_weighted_surprise(
    prediction_error: float,
    risk: float,
) -> float:
    return compute_total_surprise(prediction_error, risk)


def _coerce_float_dict(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _flatten_state_snapshot(state_snapshot: dict[str, object]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for prefix, field in (
        ("obs", "observation"),
        ("pred", "prediction"),
        ("err", "errors"),
        ("body", "body_state"),
    ):
        values = _coerce_float_dict(state_snapshot.get(field))
        for key, value in values.items():
            vector[f"{prefix}_{key}"] = value
    return vector


def _split_state_vector(state_vector: dict[str, float]) -> dict[str, dict[str, float]]:
    state_snapshot = {
        "observation": {},
        "prediction": {},
        "errors": {},
        "body_state": {},
    }
    prefix_map = {
        "obs": "observation",
        "pred": "prediction",
        "err": "errors",
        "body": "body_state",
    }
    for key, value in state_vector.items():
        prefix, _, field = key.partition("_")
        bucket = prefix_map.get(prefix)
        if bucket and field:
            state_snapshot[bucket][field] = float(value)
    return state_snapshot


def _state_vector_order() -> list[str]:
    observation_keys = [
        "food",
        "danger",
        "novelty",
        "shelter",
        "temperature",
        "social",
    ]
    body_keys = ["cycle", "energy", "stress", "fatigue", "temperature", "dopamine"]
    ordered = [f"obs_{key}" for key in observation_keys]
    ordered.extend(f"pred_{key}" for key in observation_keys)
    ordered.extend(f"err_{key}" for key in observation_keys)
    ordered.extend(f"body_{key}" for key in body_keys)
    return ordered


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@dataclass
class LongTermMemory:
    """Episodic and semantic memory for pattern retrieval."""

    episodes: list[dict] = field(default_factory=list)
    semantic_patterns: list[dict] = field(default_factory=list)
    max_episodes: int = 50
    surprise_threshold: float = 0.40
    duplicate_similarity_threshold: float = 0.999
    preference_model: PreferenceModel = field(default_factory=PreferenceModel)

    @property
    def value_hierarchy(self) -> PreferenceModel:
        return self.preference_model

    @value_hierarchy.setter
    def value_hierarchy(self, model: PreferenceModel) -> None:
        self.preference_model = model

    def store_episode(
        self,
        cycle: int,
        observation: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        action: str,
        outcome: dict[str, float],
        body_state: dict[str, float] | None = None,
    ) -> dict[str, object]:
        """Store a full episode with context."""
        state_snapshot = {
            "observation": dict(observation),
            "prediction": dict(prediction),
            "errors": dict(errors),
            "body_state": dict(body_state or {}),
        }
        episode = self._build_episode(
            timestamp=cycle,
            state_snapshot=state_snapshot,
            action=action,
            outcome=outcome,
        )
        payload = episode.to_dict()
        self.episodes.append(payload)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]
        self._refresh_semantic_patterns()
        return payload

    def maybe_store_episode(
        self,
        cycle: int,
        observation: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        action: str,
        outcome: dict[str, float],
        body_state: dict[str, float] | None = None,
    ) -> MemoryDecision:
        state_snapshot = {
            "observation": dict(observation),
            "prediction": dict(prediction),
            "errors": dict(errors),
            "body_state": dict(body_state or {}),
        }
        episode = self._build_episode(
            timestamp=cycle,
            state_snapshot=state_snapshot,
            action=action,
            outcome=outcome,
        )
        if episode.total_surprise <= self.surprise_threshold:
            return MemoryDecision(
                value_score=episode.value_score,
                prediction_error=episode.prediction_error,
                total_surprise=episode.total_surprise,
                episode_created=False,
                predicted_outcome=episode.predicted_outcome,
                preferred_probability=episode.preferred_probability,
                risk=episode.risk,
                preference_log_value=episode.preference_log_value,
            )
        if self._is_duplicate_episode(episode):
            return MemoryDecision(
                value_score=episode.value_score,
                prediction_error=episode.prediction_error,
                total_surprise=episode.total_surprise,
                episode_created=False,
                predicted_outcome=episode.predicted_outcome,
                preferred_probability=episode.preferred_probability,
                risk=episode.risk,
                preference_log_value=episode.preference_log_value,
            )

        self.episodes.append(episode.to_dict())
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]
        self._refresh_semantic_patterns()
        return MemoryDecision(
            value_score=episode.value_score,
            prediction_error=episode.prediction_error,
            total_surprise=episode.total_surprise,
            episode_created=True,
            predicted_outcome=episode.predicted_outcome,
            preferred_probability=episode.preferred_probability,
            risk=episode.risk,
            preference_log_value=episode.preference_log_value,
        )

    def retrieve_similar_memories(
        self,
        current_state: dict[str, object],
        k: int = 3,
    ) -> list[dict[str, object]]:
        if not self.episodes:
            return []

        state_vector = _flatten_state_snapshot(
            {
                "observation": dict(current_state.get("observation", {})),
                "prediction": dict(current_state.get("prediction", {})),
                "errors": dict(current_state.get("errors", {})),
                "body_state": dict(current_state.get("body_state", {})),
            }
        )
        current_embedding = self._build_embedding(state_vector)
        reference_cycle = self._resolve_reference_cycle(
            current_observation=_coerce_float_dict(current_state.get("observation")),
            current_body_state=_coerce_float_dict(current_state.get("body_state")),
        )
        scored: list[tuple[float, float, float, Episode]] = []
        for payload in self.episodes:
            episode = Episode.from_dict(payload)
            vector_similarity = _cosine_similarity(
                current_embedding,
                episode.embedding or self._build_embedding(episode.state_vector),
            )
            recency = 1.0 / (1.0 + abs(self._episode_cycle(payload) - reference_cycle))
            scored.append(
                (
                    vector_similarity * recency,
                    vector_similarity,
                    float(episode.timestamp),
                    episode,
                )
            )

        scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
        results: list[dict[str, object]] = []
        for similarity, vector_similarity, _, episode in scored[:k]:
            item = episode.to_dict()
            item["similarity"] = similarity
            item["vector_similarity"] = vector_similarity
            results.append(item)
        return results

    def retrieve_similar(
        self,
        current_observation: dict[str, float],
        current_body_state: dict[str, float],
        k: int = 3,
    ) -> list[dict]:
        """Backward-compatible retrieval API used by older callers and tests."""
        return self.retrieve_similar_memories(
            {
                "observation": current_observation,
                "prediction": {},
                "errors": {},
                "body_state": current_body_state,
            },
            k=k,
        )

    def memory_bias(
        self,
        action: str,
        retrieved_memories: list[dict[str, object]],
    ) -> float:
        weighted_utilities: list[float] = []
        for payload in retrieved_memories:
            if str(payload.get("action_taken", payload.get("action", ""))) != action:
                continue
            similarity = float(payload.get("similarity", 0.0))
            if similarity <= 0.0:
                continue
            utility = self._episode_utility(payload)
            weighted_utilities.append(similarity * utility)
        if not weighted_utilities:
            return 0.0
        return max(-1.0, min(1.0, sum(weighted_utilities) / len(weighted_utilities)))

    def pattern_bias(
        self,
        action: str,
        action_history: list[str] | None = None,
    ) -> float:
        summary = next(
            (
                pattern
                for pattern in self.semantic_patterns
                if pattern.get("action") == action
            ),
            None,
        )
        if not summary:
            return 0.0

        recurrence = float(summary.get("recurrence", 0.0))
        mean_utility = float(summary.get("mean_utility", 0.0))
        bias = mean_utility * (0.40 + recurrence)
        if action_history:
            recent = action_history[-8:]
            if recent:
                repeat_ratio = recent.count(action) / len(recent)
                streak = 0
                for previous in reversed(recent):
                    if previous != action:
                        break
                    streak += 1
                if repeat_ratio > 0.50:
                    bias -= (repeat_ratio - 0.50) * 0.45
                if streak > 3:
                    bias -= (streak - 3) * 0.10
        return max(-1.0, min(1.0, bias))

    def _resolve_reference_cycle(
        self,
        current_observation: dict[str, float],
        current_body_state: dict[str, float],
    ) -> float:
        for source in (current_body_state, current_observation):
            cycle = source.get("cycle")
            if isinstance(cycle, (int, float)):
                return float(cycle)

        latest_cycle = max(
            self._episode_cycle(episode)
            for episode in self.episodes
        )
        # When the caller does not provide a cycle, treat retrieval as happening
        # immediately after the newest stored episode rather than at cycle 0.
        return latest_cycle + 1.0

    def extract_pattern(self, episodes: list[dict]) -> dict[str, float]:
        """Extract a semantic pattern from a set of episodes."""
        if not episodes:
            return {}

        avg_obs = {}
        for key in episodes[0]["observation"]:
            avg_obs[key] = mean(episode["observation"][key] for episode in episodes)
        return avg_obs

    def to_dict(self) -> dict:
        return {
            "episodes": list(self.episodes),
            "semantic_patterns": list(self.semantic_patterns),
            "max_episodes": self.max_episodes,
            "surprise_threshold": self.surprise_threshold,
            "duplicate_similarity_threshold": self.duplicate_similarity_threshold,
            "preference_model": self.preference_model.to_dict(),
            "value_hierarchy": self.preference_model.legacy_value_hierarchy_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> LongTermMemory:
        if not payload:
            return cls()

        preference_payload = payload.get("preference_model")
        if not isinstance(preference_payload, dict):
            preference_payload = payload.get("value_hierarchy")
        if not isinstance(preference_payload, dict):
            preference_payload = {}

        return cls(
            episodes=list(payload.get("episodes", [])),
            semantic_patterns=list(payload.get("semantic_patterns", [])),
            max_episodes=int(payload.get("max_episodes", 50)),
            surprise_threshold=float(payload.get("surprise_threshold", 0.40)),
            duplicate_similarity_threshold=float(
                payload.get("duplicate_similarity_threshold", 0.999)
            ),
            preference_model=PreferenceModel.from_dict(preference_payload),
        )

    def _build_episode(
        self,
        *,
        timestamp: int,
        state_snapshot: dict[str, object],
        action: str,
        outcome: dict[str, float],
    ) -> Episode:
        observation = _coerce_float_dict(state_snapshot.get("observation"))
        prediction = _coerce_float_dict(state_snapshot.get("prediction"))
        preference = self.preference_model.evaluate_state(
            {
                **state_snapshot,
                "predicted_outcome": dict(outcome),
            }
        )
        state_vector = _flatten_state_snapshot(state_snapshot)
        prediction_error = compute_prediction_error(observation, prediction)
        return Episode(
            timestamp=timestamp,
            state_vector=state_vector,
            action_taken=action,
            outcome_state=dict(outcome),
            predicted_outcome=preference.outcome,
            prediction_error=prediction_error,
            risk=preference.risk,
            value_score=preference.value_score,
            total_surprise=compute_total_surprise(
                prediction_error,
                preference.risk,
            ),
            embedding=self._build_embedding(state_vector),
            preferred_probability=preference.preferred_probability,
            preference_log_value=preference.log_preference,
        )

    def _build_embedding(
        self,
        state_vector: dict[str, float],
    ) -> list[float]:
        ordered_keys = _state_vector_order()
        return [float(state_vector.get(key, 0.0)) for key in ordered_keys]

    def _is_duplicate_episode(self, candidate: Episode) -> bool:
        for payload in self.episodes:
            existing = Episode.from_dict(payload)
            if not existing.embedding:
                continue
            if existing.action_taken != candidate.action_taken:
                continue
            similarity = _cosine_similarity(candidate.embedding, existing.embedding)
            if similarity >= self.duplicate_similarity_threshold:
                return True
        return False

    def _episode_utility(self, payload: dict[str, object]) -> float:
        outcome = _coerce_float_dict(payload.get("outcome_state", payload.get("outcome")))
        value_score = float(payload.get("value_score", 0.0))
        prediction_error = float(payload.get("prediction_error", 0.0))
        utility = outcome.get("free_energy_drop", 0.0) + value_score - prediction_error
        return max(-1.0, min(1.0, utility))

    def _refresh_semantic_patterns(self) -> None:
        by_action: dict[str, list[dict[str, object]]] = {}
        for payload in self.episodes:
            action = str(payload.get("action_taken", payload.get("action", "")))
            by_action.setdefault(action, []).append(payload)

        total = max(1, len(self.episodes))
        patterns: list[dict[str, object]] = []
        for action, payloads in sorted(by_action.items()):
            patterns.append(
                {
                    "action": action,
                    "count": len(payloads),
                    "recurrence": len(payloads) / total,
                    "mean_utility": mean(self._episode_utility(payload) for payload in payloads),
                    "mean_prediction_error": mean(
                        float(payload.get("prediction_error", 0.0))
                        for payload in payloads
                    ),
                    "mean_value_score": mean(
                        float(payload.get("value_score", 0.0))
                        for payload in payloads
                    ),
                }
            )
        self.semantic_patterns = patterns

    def _episode_observation(self, episode: dict[str, object]) -> dict[str, float]:
        direct = _coerce_float_dict(episode.get("observation"))
        if direct:
            return direct
        snapshot = episode.get("state_snapshot")
        if isinstance(snapshot, dict):
            return _coerce_float_dict(snapshot.get("observation"))
        return {}

    def _episode_cycle(self, episode: dict[str, object]) -> float:
        raw_cycle = episode.get("cycle", episode.get("timestamp", 0))
        if isinstance(raw_cycle, (int, float)):
            return float(raw_cycle)
        snapshot = episode.get("state_snapshot")
        if isinstance(snapshot, dict):
            body_state = snapshot.get("body_state")
            if isinstance(body_state, dict):
                cycle = body_state.get("cycle")
                if isinstance(cycle, (int, float)):
                    return float(cycle)
        return 0.0