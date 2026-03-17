from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log, sqrt
from statistics import mean

from .action_schema import ActionSchema, action_name, ensure_action_schema
from .preferences import PreferenceModel, ValueHierarchy


@dataclass
class Episode:
    timestamp: int
    state_vector: dict[str, float]
    action_taken: ActionSchema
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
            "action_taken": self.action_taken.to_dict(),
            "action": self.action_taken.name,
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
            action_taken=ActionSchema.from_dict(
                payload.get("action_taken", payload.get("action", ""))
            ),
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
    episode_score: float = 0.0
    value_relevance: float = 0.0
    policy_delta: float = 0.0
    threat_significance: float = 0.0
    redundancy_penalty: float = 0.0
    support_delta: int = 0
    merged_into_episode_id: str | None = None
    gating_reasons: tuple[str, ...] = ()

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
            "episode_score": self.episode_score,
            "value_relevance": self.value_relevance,
            "policy_delta": self.policy_delta,
            "threat_significance": self.threat_significance,
            "redundancy_penalty": self.redundancy_penalty,
            "support_delta": self.support_delta,
            "merged_into_episode_id": self.merged_into_episode_id,
            "gating_reasons": list(self.gating_reasons),
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


RISK_WEIGHT = 1.0

# Lifecycle stage constants
LIFECYCLE_ATTENTION_TRACE = "attention_trace"
LIFECYCLE_CANDIDATE_EPISODE = "candidate_episode"
LIFECYCLE_VALIDATED_EPISODE = "validated_episode"
LIFECYCLE_ARCHIVED_SUMMARY = "archived_summary"
LIFECYCLE_PROTECTED_IDENTITY_CRITICAL = "protected_identity_critical_episode"
CONTINUITY_ROLE_IDENTITY = "identity_critical_memory"
CONTINUITY_ROLE_COMMITMENT = "commitment_supporting_memory"
CONTINUITY_ROLE_MAINTENANCE = "restart_critical_maintenance_memory"

EPISODE_FAMILY_HAZARD = "hazard_response"
EPISODE_FAMILY_RESOURCE = "resource_opportunity"
EPISODE_FAMILY_SOCIAL = "social_signal"
EPISODE_FAMILY_ENVIRONMENT = "environmental_shift"
EPISODE_FAMILY_ROUTINE = "routine_monitoring"

# Minimum raw prediction error required before risk can amplify surprise.
# When observation closely matches prediction (low PE), even a high model-risk
# should not inflate total_surprise — the world is behaving as expected.
RAW_PE_RISK_GATE = 0.04


def compute_total_surprise(
    prediction_error: float,
    risk: float,
) -> float:
    raw_surprise = compute_surprise(prediction_error)
    # When raw prediction error is below the gate, the observation closely
    # matches the prediction — the world is behaving as expected.  Any risk
    # score reflects learned priors, not an actual anomaly.  Cap the risk
    # contribution so that model-inflated risk cannot dominate surprise.
    if prediction_error < RAW_PE_RISK_GATE:
        ratio = prediction_error / max(RAW_PE_RISK_GATE, 1e-12)
        # Cap risk contribution to at most 2× the raw surprise.
        risk_contribution = min(
            RISK_WEIGHT * max(0.0, risk) * ratio * ratio,
            raw_surprise * 2.0,
        )
    else:
        risk_contribution = RISK_WEIGHT * max(0.0, risk)
    return raw_surprise + risk_contribution


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
    body_keys: list[str] = []
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
    max_episodes: int = 1024
    surprise_threshold: float = 0.40
    duplicate_similarity_threshold: float = 0.999
    suppression_similarity_threshold: float = 0.992
    novelty_window: int = 8
    episode_score_threshold: float = 0.58
    sleep_interval: int = 200
    memory_threshold: int = 512
    sleep_batch_size: int = 128
    minimum_support: int = 5
    sleep_minimum_support: int = 3
    compression_similarity_threshold: float = 0.95
    cluster_distance_threshold: float = 0.15
    max_active_age: int = 5000
    cluster_centroids: list[list[float]] = field(default_factory=list)
    cluster_counts: list[int] = field(default_factory=list)
    archived_episodes: list[dict] = field(default_factory=list)
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)
    rehearsal_log: list[dict[str, object]] = field(default_factory=list)
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
        action: str | ActionSchema,
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
        self._initialize_episode_payload(payload)
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
        action: str | ActionSchema,
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
        gate = self._score_episode_candidate(episode)
        if (
            episode.total_surprise <= self.surprise_threshold
            and gate["episode_score"] < self.episode_score_threshold
            and not gate["identity_critical"]
        ):
            return MemoryDecision(
                value_score=episode.value_score,
                prediction_error=episode.prediction_error,
                total_surprise=episode.total_surprise,
                episode_created=False,
                predicted_outcome=episode.predicted_outcome,
                preferred_probability=episode.preferred_probability,
                risk=episode.risk,
                preference_log_value=episode.preference_log_value,
                episode_score=gate["episode_score"],
                value_relevance=gate["value_relevance"],
                policy_delta=gate["policy_delta"],
                threat_significance=gate["threat_significance"],
                redundancy_penalty=gate["redundancy_penalty"],
                gating_reasons=tuple(gate["reasons"]),
            )
        if gate["redundancy_penalty"] >= 0.65 and not gate["identity_critical"]:
            return MemoryDecision(
                value_score=episode.value_score,
                prediction_error=episode.prediction_error,
                total_surprise=episode.total_surprise,
                episode_created=False,
                predicted_outcome=episode.predicted_outcome,
                preferred_probability=episode.preferred_probability,
                risk=episode.risk,
                preference_log_value=episode.preference_log_value,
                episode_score=gate["episode_score"],
                value_relevance=gate["value_relevance"],
                policy_delta=gate["policy_delta"],
                threat_significance=gate["threat_significance"],
                redundancy_penalty=gate["redundancy_penalty"],
                gating_reasons=tuple(gate["reasons"]),
            )

        merged_payload = self._find_merge_target(episode)
        if merged_payload is not None:
            self._merge_into_existing_episode(merged_payload, episode, gate)
            self._refresh_semantic_patterns()
            return MemoryDecision(
                value_score=episode.value_score,
                prediction_error=episode.prediction_error,
                total_surprise=episode.total_surprise,
                episode_created=False,
                predicted_outcome=episode.predicted_outcome,
                preferred_probability=episode.preferred_probability,
                risk=episode.risk,
                preference_log_value=episode.preference_log_value,
                episode_score=gate["episode_score"],
                value_relevance=gate["value_relevance"],
                policy_delta=gate["policy_delta"],
                threat_significance=gate["threat_significance"],
                redundancy_penalty=gate["redundancy_penalty"],
                support_delta=1,
                merged_into_episode_id=str(merged_payload.get("episode_id", "")) or None,
                gating_reasons=tuple(gate["reasons"] + ["merged_into_existing_episode"]),
            )

        payload = episode.to_dict()
        self._initialize_episode_payload(payload, gate)
        self.episodes.append(payload)
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
            episode_score=gate["episode_score"],
            value_relevance=gate["value_relevance"],
            policy_delta=gate["policy_delta"],
            threat_significance=gate["threat_significance"],
            redundancy_penalty=gate["redundancy_penalty"],
            gating_reasons=tuple(gate["reasons"]),
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
        action: str | ActionSchema,
        retrieved_memories: list[dict[str, object]],
    ) -> float:
        weighted_utilities: list[float] = []
        for payload in retrieved_memories:
            if action_name(payload.get("action_taken", payload.get("action", ""))) != action:
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

    def life_history_timeline(
        self,
        *,
        max_events: int = 20,
        surprise_floor: float = 0.0,
    ) -> list[dict[str, object]]:
        """Return a chronological autobiographical timeline of significant episodes.

        Merges active and archived episodes, ranks by surprise, and returns
        an ordered list of landmark events suitable for narrative generation
        or life-history replay.
        """
        all_episodes: list[dict[str, object]] = []
        for payload in self.episodes:
            entry = dict(payload)
            entry["source"] = "active"
            all_episodes.append(entry)
        for payload in self.archived_episodes:
            entry = dict(payload)
            entry["source"] = "archived"
            all_episodes.append(entry)

        if surprise_floor > 0.0:
            all_episodes = [
                entry
                for entry in all_episodes
                if float(entry.get("total_surprise", entry.get("weighted_surprise", 0.0)))
                >= surprise_floor
            ]

        all_episodes.sort(
            key=lambda entry: (
                -float(entry.get("total_surprise", entry.get("weighted_surprise", 0.0))),
                -self._episode_cycle(entry),
            ),
        )
        top = all_episodes[:max_events]
        top.sort(key=self._episode_cycle)

        timeline: list[dict[str, object]] = []
        for entry in top:
            timeline.append({
                "tick": int(self._episode_cycle(entry)),
                "action": action_name(entry.get("action_taken", entry.get("action", "unknown"))),
                "outcome": str(entry.get("predicted_outcome", "neutral")),
                "surprise": float(
                    entry.get("total_surprise", entry.get("weighted_surprise", 0.0))
                ),
                "value_score": float(entry.get("value_score", 0.0)),
                "cluster_id": entry.get("cluster_id"),
                "source": entry.get("source", "active"),
            })
        return timeline

    def should_sleep(self, cycle_count: int) -> bool:
        if cycle_count > 0 and self.sleep_interval > 0 and cycle_count % self.sleep_interval == 0:
            return True
        return len(self.episodes) > self.memory_threshold

    def prioritized_replay_sample(
        self,
        *,
        rng,
    ) -> list[dict[str, object]]:
        if not self.episodes:
            return []

        latest_timestamp = max(self._episode_cycle(episode) for episode in self.episodes)
        scored: list[tuple[float, float, dict[str, object]]] = []
        for payload in self.episodes:
            preferred_probability = max(
                1e-12,
                float(payload.get("preferred_probability", 0.0)),
            )
            risk = -log(preferred_probability)
            timestamp = self._episode_cycle(payload)
            recency_weight = 1.0 / (1.0 + max(0.0, latest_timestamp - timestamp))
            priority = (
                float(payload.get("prediction_error", 0.0))
                + risk
                + recency_weight
            )
            scored.append((priority, timestamp, payload))

        scored.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("action", ""))))
        sample_size = min(self.sleep_batch_size, len(scored))
        high_priority_count = min(len(scored), int(round(sample_size * 0.8)))
        high_priority = [payload for _, _, payload in scored[:high_priority_count]]
        remaining = [payload for _, _, payload in scored[high_priority_count:]]
        random_count = sample_size - len(high_priority)
        random_replay = (
            rng.sample(remaining, min(random_count, len(remaining)))
            if remaining and random_count > 0
            else []
        )
        selected_ids = {id(payload) for payload in high_priority + random_replay}
        fallback = [
            payload for _, _, payload in scored
            if id(payload) not in selected_ids
        ]
        replay_batch = list(high_priority)
        replay_batch.extend(random_replay)
        replay_batch.extend(fallback[: sample_size - len(replay_batch)])
        replay_batch.sort(key=self._episode_cycle)
        return replay_batch

    def replay_during_sleep(
        self,
        *,
        rng,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        """Return a replay batch for the offline sleep phase."""
        batch = self.prioritized_replay_sample(rng=rng)
        if limit is None:
            return batch
        return batch[: max(0, int(limit))]

    def assign_clusters(self) -> int:
        if not self.episodes:
            self.cluster_centroids = []
            self.cluster_counts = []
            return 0

        if self._needs_cluster_rebuild():
            self.cluster_centroids = []
            self.cluster_counts = []
            for payload in self.episodes:
                payload.pop("cluster_id", None)

        created = 0
        for payload in sorted(self.episodes, key=self._episode_cycle):
            if isinstance(payload.get("cluster_id"), int):
                continue
            embedding = self._episode_embedding(payload)
            cluster_id, was_created = self._assign_embedding_to_cluster(embedding)
            payload["cluster_id"] = cluster_id
            if was_created:
                created += 1
        return created

    def infer_cluster_id(self, current_state: dict[str, object]) -> int | None:
        if not self.cluster_centroids:
            return None
        state_vector = _flatten_state_snapshot(
            {
                "observation": dict(current_state.get("observation", {})),
                "prediction": dict(current_state.get("prediction", {})),
                "errors": dict(current_state.get("errors", {})),
                "body_state": dict(current_state.get("body_state", {})),
            }
        )
        embedding = self._build_embedding(state_vector)
        cluster_id, distance = self._nearest_cluster(embedding)
        if cluster_id is None or distance >= self.cluster_distance_threshold:
            return None
        return cluster_id

    def reconstruct_transitions(
        self,
        episodes: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        ordered = sorted(episodes, key=self._episode_cycle)
        transitions: list[dict[str, object]] = []
        for current, successor in zip(ordered, ordered[1:]):
            cluster_id = current.get("cluster_id")
            next_cluster_id = successor.get("cluster_id")
            if not isinstance(cluster_id, int) or not isinstance(next_cluster_id, int):
                continue
            transitions.append(
                {
                    "state_cluster": cluster_id,
                    "action": action_name(current.get("action_taken", current.get("action", ""))),
                    "next_cluster": next_cluster_id,
                    "timestamp": int(self._episode_cycle(current)),
                }
            )
        return transitions

    def delete_episode(self, payload: dict[str, object], *, record_event: bool = True) -> bool:
        if record_event:
            event_cycle = int(payload.get("last_seen_cycle", self._episode_cycle(payload)))
            self._record_memory_lifecycle_event(
                payload,
                event="deleted",
                stage=str(payload.get("lifecycle_stage", "deleted")),
                cycle=event_cycle,
                previous_stage=payload.get("lifecycle_stage"),
                details={"archive_reason": payload.get("archive_reason")},
            )
        try:
            self.episodes.remove(payload)
        except ValueError:
            if record_event and self.lifecycle_events:
                self.lifecycle_events.pop()
            return False
        self._refresh_semantic_patterns()
        return True

    def archive_episode(
        self,
        payload: dict[str, object],
        *,
        archive_cycle: int,
        reason: str,
    ) -> bool:
        archived = dict(payload)
        archived["archived_at_cycle"] = archive_cycle
        archived["archive_reason"] = reason
        self._set_lifecycle_stage(
            archived,
            LIFECYCLE_ARCHIVED_SUMMARY,
            event="archived",
            cycle=archive_cycle,
            details={"reason": reason},
        )
        self.archived_episodes.append(archived)
        self._record_memory_lifecycle_event(
            archived,
            event="archived",
            stage=LIFECYCLE_ARCHIVED_SUMMARY,
            cycle=archive_cycle,
            previous_stage=payload.get("lifecycle_stage"),
            details={"reason": reason},
        )
        return self.delete_episode(payload, record_event=False)

    def compress_episodes(self) -> int:
        if len(self.episodes) < 2:
            return 0

        ordered_payloads = sorted(
            self.episodes,
            key=lambda payload: (
                action_name(payload.get("action_taken", payload.get("action", ""))),
                self._episode_cycle(payload),
            ),
        )
        compressed: list[dict[str, object]] = []
        used: set[int] = set()
        removed = 0
        for index, payload in enumerate(ordered_payloads):
            if index in used:
                continue
            if self._is_restart_protected_payload(payload):
                retained = dict(payload)
                retained["consolidated"] = bool(retained.get("consolidated", False))
                retained["compressed_count"] = int(retained.get("compressed_count", 1))
                compressed.append(retained)
                used.add(index)
                continue
            base_episode = Episode.from_dict(payload)
            base_embedding = base_episode.embedding or self._build_embedding(base_episode.state_vector)
            group = [payload]
            used.add(index)
            for other_index in range(index + 1, len(ordered_payloads)):
                if other_index in used:
                    continue
                candidate = ordered_payloads[other_index]
                if self._is_restart_protected_payload(candidate):
                    continue
                if action_name(candidate.get("action_taken", candidate.get("action", ""))) != base_episode.action_taken.name:
                    continue
                candidate_episode = Episode.from_dict(candidate)
                candidate_embedding = candidate_episode.embedding or self._build_embedding(candidate_episode.state_vector)
                if _cosine_similarity(base_embedding, candidate_embedding) < self.compression_similarity_threshold:
                    continue
                group.append(candidate)
                used.add(other_index)
            if len(group) == 1:
                retained = dict(payload)
                retained["consolidated"] = bool(retained.get("consolidated", False))
                retained["compressed_count"] = int(retained.get("compressed_count", 1))
                compressed.append(retained)
                continue
            merged_payload = self._merge_episode_group(group)
            merged_payload["consolidated"] = True
            compression_cycle = int(max(self._episode_cycle(item) for item in group))
            self._set_lifecycle_stage(
                merged_payload,
                str(merged_payload.get("lifecycle_stage", LIFECYCLE_VALIDATED_EPISODE)),
                event="compressed",
                cycle=compression_cycle,
                details={
                    "source_episode_ids": [
                        str(item.get("episode_id", ""))
                        for item in group
                        if item.get("episode_id")
                    ],
                    "compressed_count": len(group),
                },
            )
            self._record_memory_lifecycle_event(
                merged_payload,
                event="compressed",
                stage=str(merged_payload.get("lifecycle_stage", LIFECYCLE_VALIDATED_EPISODE)),
                cycle=compression_cycle,
                previous_stage=None,
                details={
                    "source_episode_ids": [
                        str(item.get("episode_id", ""))
                        for item in group
                        if item.get("episode_id")
                    ],
                    "compressed_count": len(group),
                },
            )
            compressed.append(merged_payload)
            removed += len(group) - 1

        compressed.sort(key=self._episode_cycle)
        self.episodes = compressed[-self.max_episodes :]
        self._refresh_semantic_patterns()
        return removed

    def family_coverage_summary(self) -> dict[str, object]:
        family_counts: dict[str, int] = {}
        stage_counts: dict[str, int] = {}
        family_stage_matrix: dict[str, dict[str, int]] = {}
        combined = [(episode, "active") for episode in self.episodes]
        combined.extend((episode, "archived") for episode in self.archived_episodes)
        for payload, source in combined:
            family = str(payload.get("episode_family", EPISODE_FAMILY_ROUTINE))
            stage = str(payload.get("lifecycle_stage", "unknown"))
            family_counts[family] = family_counts.get(family, 0) + 1
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            matrix_entry = family_stage_matrix.setdefault(family, {})
            matrix_entry[stage] = matrix_entry.get(stage, 0) + 1
            matrix_entry[source] = matrix_entry.get(source, 0) + 1
        return {
            "family_count": len(family_counts),
            "family_counts": family_counts,
            "stage_counts": stage_counts,
            "source_counts": {
                "active": len(self.episodes),
                "archived": len(self.archived_episodes),
            },
            "family_stage_matrix": family_stage_matrix,
        }

    def lifecycle_audit(self) -> dict[str, object]:
        event_counts: dict[str, int] = {}
        stage_transitions: dict[str, int] = {}
        for event in self.lifecycle_events:
            event_type = str(event.get("event", "unknown"))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            previous = event.get("previous_stage")
            current = event.get("stage")
            if previous and current and previous != current:
                key = f"{previous}->{current}"
                stage_transitions[key] = stage_transitions.get(key, 0) + 1
        return {
            "event_count": len(self.lifecycle_events),
            "event_counts": event_counts,
            "stage_transitions": stage_transitions,
            "current_stage_counts": self.family_coverage_summary()["stage_counts"],
        }

    def protected_identity_anchors(self, *, limit: int = 12) -> list[dict[str, object]]:
        anchors = [
            payload
            for payload in [*self.episodes, *self.archived_episodes]
            if bool(payload.get("identity_critical", False))
        ]
        anchors.sort(
            key=lambda payload: (
                int(payload.get("last_seen_cycle", payload.get("cycle", 0))),
                str(payload.get("episode_id", "")),
            ),
            reverse=True,
        )
        return [dict(payload) for payload in anchors[:limit]]

    def restart_anchor_payload(self, *, limit: int = 16) -> list[dict[str, object]]:
        anchors = [
            payload
            for payload in [*self.episodes, *self.archived_episodes]
            if self._is_restart_protected_payload(payload)
        ]
        anchors.sort(
            key=lambda payload: (
                int(payload.get("last_seen_cycle", payload.get("cycle", 0))),
                str(payload.get("episode_id", "")),
            ),
            reverse=True,
        )
        return [
            {
                "episode_id": str(payload.get("episode_id", "")),
                "continuity_role": str(payload.get("continuity_role", "")),
                "continuity_tags": [str(item) for item in payload.get("continuity_tags", [])],
                "identity_critical": bool(payload.get("identity_critical", False)),
                "restart_protected": bool(payload.get("restart_protected", False)),
                "action": action_name(payload.get("action_taken", payload.get("action", ""))),
            }
            for payload in anchors[:limit]
        ]

    def rehearsal_batch(
        self,
        *,
        current_cycle: int,
        limit: int = 6,
    ) -> list[dict[str, object]]:
        anchors = self.protected_identity_anchors(limit=max(limit * 3, limit))
        anchors.sort(
            key=lambda payload: (
                int(payload.get("last_rehearsed_cycle", -1)),
                int(payload.get("last_seen_cycle", payload.get("cycle", 0))),
                str(payload.get("episode_id", "")),
            )
        )
        selected = anchors[:limit]
        selected_ids = {str(payload.get("episode_id", "")) for payload in selected}
        for bucket in (self.episodes, self.archived_episodes):
            for payload in bucket:
                episode_id = str(payload.get("episode_id", ""))
                if episode_id in selected_ids:
                    payload["last_rehearsed_cycle"] = int(current_cycle)
        if selected:
            self.rehearsal_log.append(
                {
                    "cycle": int(current_cycle),
                    "episode_ids": sorted(selected_ids),
                }
            )
            self.rehearsal_log = self.rehearsal_log[-128:]
        return [dict(payload) for payload in selected]

    def retire_stale_episodes(
        self,
        *,
        current_cycle: int,
        retain_recent: int = 128,
    ) -> int:
        if len(self.episodes) <= retain_recent:
            return 0
        retired = 0
        candidates = sorted(
            self.episodes,
            key=lambda payload: (
                int(payload.get("last_seen_cycle", payload.get("cycle", 0))),
                str(payload.get("episode_id", "")),
            ),
        )
        for payload in candidates:
            if len(self.episodes) <= retain_recent:
                break
            if bool(payload.get("identity_critical", False)) or self._is_restart_protected_payload(payload):
                continue
            lifecycle_stage = str(payload.get("lifecycle_stage", ""))
            if lifecycle_stage == LIFECYCLE_PROTECTED_IDENTITY_CRITICAL:
                continue
            age = int(current_cycle - self._episode_cycle(payload))
            if age < max(64, self.max_active_age // 8):
                continue
            if self.archive_episode(
                payload,
                archive_cycle=current_cycle,
                reason="m218_continuity_retirement",
            ):
                retired += 1
        return retired

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
            "suppression_similarity_threshold": self.suppression_similarity_threshold,
            "novelty_window": self.novelty_window,
            "episode_score_threshold": self.episode_score_threshold,
            "sleep_interval": self.sleep_interval,
            "memory_threshold": self.memory_threshold,
            "sleep_batch_size": self.sleep_batch_size,
            "minimum_support": self.minimum_support,
            "sleep_minimum_support": self.sleep_minimum_support,
            "compression_similarity_threshold": self.compression_similarity_threshold,
            "cluster_distance_threshold": self.cluster_distance_threshold,
            "max_active_age": self.max_active_age,
            "cluster_centroids": [list(centroid) for centroid in self.cluster_centroids],
            "cluster_counts": list(self.cluster_counts),
            "archived_episodes": list(self.archived_episodes),
            "lifecycle_events": list(self.lifecycle_events),
            "rehearsal_log": list(self.rehearsal_log),
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
            max_episodes=int(payload.get("max_episodes", 1024)),
            surprise_threshold=float(payload.get("surprise_threshold", 0.40)),
            duplicate_similarity_threshold=float(
                payload.get("duplicate_similarity_threshold", 0.999)
            ),
            suppression_similarity_threshold=float(
                payload.get("suppression_similarity_threshold", 0.992)
            ),
            novelty_window=int(payload.get("novelty_window", 8)),
            episode_score_threshold=float(payload.get("episode_score_threshold", 0.58)),
            sleep_interval=int(payload.get("sleep_interval", 200)),
            memory_threshold=int(payload.get("memory_threshold", 512)),
            sleep_batch_size=int(payload.get("sleep_batch_size", 128)),
            minimum_support=int(payload.get("minimum_support", 5)),
            sleep_minimum_support=int(payload.get("sleep_minimum_support", 3)),
            compression_similarity_threshold=float(
                payload.get("compression_similarity_threshold", 0.95)
            ),
            cluster_distance_threshold=float(
                payload.get("cluster_distance_threshold", 0.15)
            ),
            max_active_age=int(payload.get("max_active_age", 5000)),
            cluster_centroids=[
                [float(value) for value in centroid]
                for centroid in list(payload.get("cluster_centroids", []))
                if isinstance(centroid, list)
            ],
            cluster_counts=[
                int(value) for value in list(payload.get("cluster_counts", []))
            ],
            archived_episodes=list(payload.get("archived_episodes", [])),
            lifecycle_events=list(payload.get("lifecycle_events", [])),
            rehearsal_log=list(payload.get("rehearsal_log", [])),
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
            action_taken=ensure_action_schema(action),
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

    def _needs_cluster_rebuild(self) -> bool:
        if len(self.cluster_centroids) != len(self.cluster_counts):
            return True
        if not self.cluster_centroids:
            return any("cluster_id" in payload for payload in self.episodes)
        for payload in self.episodes:
            cluster_id = payload.get("cluster_id")
            if cluster_id is None:
                continue
            if not isinstance(cluster_id, int) or cluster_id < 0:
                return True
            if cluster_id >= len(self.cluster_centroids):
                return True
        return False

    def _episode_embedding(self, payload: dict[str, object]) -> list[float]:
        episode = Episode.from_dict(payload)
        return episode.embedding or self._build_embedding(episode.state_vector)

    def _infer_continuity_role(self, payload: dict[str, object]) -> tuple[str, list[str]]:
        tags: list[str] = []
        action = action_name(payload.get("action_taken", payload.get("action", "")))
        predicted_outcome = str(payload.get("predicted_outcome", "neutral"))
        observation = self._episode_observation(payload)
        if bool(payload.get("identity_critical", False)) or predicted_outcome in {
            "survival_threat",
            "integrity_loss",
        }:
            tags.extend(["identity", "threat"])
            return CONTINUITY_ROLE_IDENTITY, tags
        if action in {"hide", "exploit_shelter"} and float(observation.get("danger", 0.0)) >= 0.55:
            tags.extend(["commitment", "guard"])
            return CONTINUITY_ROLE_COMMITMENT, tags
        if action in {"rest", "thermoregulate"} and (
            float(payload.get("prediction_error", 0.0)) >= 0.10
            or float(observation.get("temperature", 0.5)) >= 0.62
            or float(observation.get("temperature", 0.5)) <= 0.38
        ):
            tags.extend(["maintenance", "recovery"])
            return CONTINUITY_ROLE_MAINTENANCE, tags
        return "", tags

    def _is_restart_protected_payload(self, payload: dict[str, object]) -> bool:
        if bool(payload.get("restart_protected", False)):
            return True
        role = str(payload.get("continuity_role", ""))
        return role in {
            CONTINUITY_ROLE_IDENTITY,
            CONTINUITY_ROLE_COMMITMENT,
            CONTINUITY_ROLE_MAINTENANCE,
        }

    def _assign_embedding_to_cluster(
        self,
        embedding: list[float],
    ) -> tuple[int, bool]:
        cluster_id, distance = self._nearest_cluster(embedding)
        if cluster_id is None or distance >= self.cluster_distance_threshold:
            self.cluster_centroids.append(list(embedding))
            self.cluster_counts.append(1)
            return len(self.cluster_centroids) - 1, True

        count = self.cluster_counts[cluster_id]
        centroid = self.cluster_centroids[cluster_id]
        updated_count = count + 1
        self.cluster_centroids[cluster_id] = [
            ((value * count) + sample) / updated_count
            for value, sample in zip(centroid, embedding)
        ]
        self.cluster_counts[cluster_id] = updated_count
        return cluster_id, False

    def _nearest_cluster(
        self,
        embedding: list[float],
    ) -> tuple[int | None, float]:
        if not self.cluster_centroids:
            return None, float("inf")

        best_index = None
        best_distance = float("inf")
        for index, centroid in enumerate(self.cluster_centroids):
            distance = _cosine_distance(embedding, centroid)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index, best_distance

    def _mean_vector(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        width = len(vectors[0])
        return [
            mean(vector[index] for vector in vectors)
            for index in range(width)
        ]

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

    def _score_episode_candidate(self, episode: Episode) -> dict[str, object]:
        value_relevance = abs(float(episode.value_score))
        # When raw prediction error is very low, the preference model's risk
        # scores may reflect learned priors rather than an actual anomaly.
        # Dampen model-derived signals proportionally to raw PE.
        raw_pe = abs(episode.prediction_error)
        pe_dampening = min(1.0, raw_pe / RAW_PE_RISK_GATE) if raw_pe < RAW_PE_RISK_GATE else 1.0
        policy_delta = min(
            1.0,
            max(0.0, episode.risk / 8.0) + max(0.0, -episode.preference_log_value / 12.0),
        ) * pe_dampening
        threat_significance = min(
            1.0,
            max(0.0, episode.risk / 4.0)
            + (0.30 if episode.predicted_outcome == "survival_threat" else 0.0)
            + (0.15 if episode.predicted_outcome == "integrity_loss" else 0.0),
        ) * pe_dampening
        redundancy_penalty = self._redundancy_penalty(episode)
        identity_critical = self._is_identity_critical_episode(episode)
        episode_score = (
            (0.38 * min(1.0, episode.total_surprise))
            + (0.27 * value_relevance)
            + (0.17 * policy_delta)
            + (0.24 * threat_significance)
            - (0.34 * redundancy_penalty)
        )
        reasons: list[str] = []
        if identity_critical:
            reasons.append("identity_critical_exception")
        if value_relevance >= 0.75:
            reasons.append("high_value_relevance")
        if policy_delta >= 0.55:
            reasons.append("policy_relevant")
        if threat_significance >= 0.60:
            reasons.append("threat_significant")
        if redundancy_penalty >= 0.40:
            reasons.append("redundancy_penalty")
        if self._recent_similarity(episode) >= self.suppression_similarity_threshold:
            reasons.append("novelty_suppression_window")
        return {
            "episode_score": max(0.0, min(1.5, episode_score)),
            "value_relevance": value_relevance,
            "policy_delta": policy_delta,
            "threat_significance": threat_significance,
            "redundancy_penalty": redundancy_penalty,
            "identity_critical": identity_critical,
            "reasons": reasons,
        }

    def _initialize_episode_payload(
        self,
        payload: dict[str, object],
        gate: dict[str, object] | None = None,
    ) -> None:
        cycle = int(self._episode_cycle(payload))
        action = action_name(payload.get("action_taken", payload.get("action", "unknown")))
        payload.setdefault("episode_id", f"ep-{cycle:06d}-{action}")
        payload.setdefault("occurrence_count", 1)
        payload.setdefault("support", int(payload.get("occurrence_count", 1)))
        payload.setdefault("support_count", int(payload.get("occurrence_count", 1)))
        payload.setdefault("first_seen_cycle", cycle)
        payload.setdefault("last_seen_cycle", cycle)
        payload.setdefault("episode_family", self._infer_episode_family(payload))
        payload.setdefault("family_features", self._extract_family_features(payload))
        identity_critical = bool(payload.get("identity_critical", False))
        raw_pe = float(payload.get("prediction_error", 0.0))
        if gate is not None:
            payload["episode_score"] = float(gate.get("episode_score", 0.0))
            payload["value_relevance"] = float(gate.get("value_relevance", 0.0))
            payload["policy_delta"] = float(gate.get("policy_delta", 0.0))
            payload["threat_significance"] = float(gate.get("threat_significance", 0.0))
            payload["redundancy_penalty"] = float(gate.get("redundancy_penalty", 0.0))
            payload["gating_reasons"] = list(gate.get("reasons", []))
            identity_critical = bool(gate.get("identity_critical", False))
        elif not identity_critical:
            # No gate provided (direct store_episode path) — infer identity
            # criticality from the episode payload itself.
            episode = Episode.from_dict(payload)
            identity_critical = self._is_identity_critical_episode(episode)
        payload["identity_critical"] = identity_critical
        continuity_role, continuity_tags = self._infer_continuity_role(payload)
        payload["continuity_role"] = continuity_role
        payload["continuity_tags"] = continuity_tags
        payload["restart_protected"] = continuity_role in {
            CONTINUITY_ROLE_IDENTITY,
            CONTINUITY_ROLE_COMMITMENT,
            CONTINUITY_ROLE_MAINTENANCE,
        }
        if identity_critical:
            initial_stage = LIFECYCLE_PROTECTED_IDENTITY_CRITICAL
        elif raw_pe < RAW_PE_RISK_GATE:
            # Low raw prediction error: the world matched expectations closely.
            # Demote to candidate_episode regardless of model-derived risk.
            initial_stage = LIFECYCLE_CANDIDATE_EPISODE
        else:
            initial_stage = str(payload.get("lifecycle_stage", LIFECYCLE_VALIDATED_EPISODE))
        self._set_lifecycle_stage(
            payload,
            initial_stage,
            event="created",
            cycle=cycle,
            details={
                "support_count": int(payload.get("support_count", 1)),
                "episode_family": payload.get("episode_family"),
            },
        )
        self._record_memory_lifecycle_event(
            payload,
            event="created",
            stage=initial_stage,
            cycle=cycle,
            previous_stage=None,
            details={"episode_family": payload.get("episode_family")},
        )

    def _redundancy_penalty(self, episode: Episode) -> float:
        if not self.episodes:
            return 0.0
        similarity = self._recent_similarity(episode)
        same_action_recent = sum(
            1
            for payload in self.episodes[-self.novelty_window :]
            if action_name(payload.get("action_taken", payload.get("action", ""))) == episode.action_taken.name
        )
        density_penalty = min(0.35, same_action_recent * 0.05)
        if similarity < 0.90:
            return density_penalty
        return min(1.0, density_penalty + ((similarity - 0.90) / 0.10))

    def _recent_similarity(self, candidate: Episode) -> float:
        similarities = []
        for payload in self.episodes[-self.novelty_window :]:
            existing = Episode.from_dict(payload)
            if existing.action_taken != candidate.action_taken:
                continue
            similarities.append(_cosine_similarity(candidate.embedding, existing.embedding))
        return max(similarities, default=0.0)

    def _is_identity_critical_episode(self, episode: Episode) -> bool:
        # A genuinely identity-critical episode requires both model risk AND
        # a non-trivial raw prediction error.  If the world matched our
        # prediction almost perfectly, the preference model's risk score
        # reflects learned priors, not an actual surprising threat.
        raw_pe = abs(episode.prediction_error)
        if raw_pe < RAW_PE_RISK_GATE:
            # Low prediction error — only mark critical on extreme outcome.
            return float(episode.outcome_state.get("free_energy_drop", 0.0)) <= -0.30
        if episode.predicted_outcome in {"survival_threat", "integrity_loss"}:
            return True
        if episode.risk >= 3.0:
            return True
        return float(episode.outcome_state.get("free_energy_drop", 0.0)) <= -0.30

    def _find_merge_target(self, candidate: Episode) -> dict[str, object] | None:
        best_payload: dict[str, object] | None = None
        best_similarity = 0.0
        for payload in self.episodes[-self.novelty_window :]:
            existing = Episode.from_dict(payload)
            if existing.action_taken != candidate.action_taken:
                continue
            similarity = _cosine_similarity(candidate.embedding, existing.embedding)
            if similarity >= self.suppression_similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_payload = payload
        return best_payload

    def _merge_into_existing_episode(
        self,
        payload: dict[str, object],
        episode: Episode,
        gate: dict[str, object],
    ) -> None:
        previous_stage = payload.get("lifecycle_stage")
        occurrences = int(payload.get("occurrence_count", 1)) + 1
        payload["occurrence_count"] = occurrences
        payload["support"] = occurrences
        payload["support_count"] = occurrences
        payload["last_seen_cycle"] = int(episode.timestamp)
        payload["total_surprise"] = max(float(payload.get("total_surprise", 0.0)), episode.total_surprise)
        payload["weighted_surprise"] = float(payload["total_surprise"])
        payload["episode_score"] = max(float(payload.get("episode_score", 0.0)), float(gate["episode_score"]))
        payload["value_relevance"] = max(float(payload.get("value_relevance", 0.0)), float(gate["value_relevance"]))
        payload["policy_delta"] = max(float(payload.get("policy_delta", 0.0)), float(gate["policy_delta"]))
        payload["threat_significance"] = max(
            float(payload.get("threat_significance", 0.0)),
            float(gate["threat_significance"]),
        )
        payload["redundancy_penalty"] = max(
            float(payload.get("redundancy_penalty", 0.0)),
            float(gate["redundancy_penalty"]),
        )
        reasons = list(payload.get("gating_reasons", []))
        for reason in list(gate["reasons"]) + ["support_increment"]:
            if reason not in reasons:
                reasons.append(reason)
        payload["gating_reasons"] = reasons
        self._set_lifecycle_stage(
            payload,
            str(payload.get("lifecycle_stage", LIFECYCLE_VALIDATED_EPISODE)),
            event="support_merged",
            cycle=int(episode.timestamp),
            details={"support_count": occurrences},
        )
        self._record_memory_lifecycle_event(
            payload,
            event="support_merged",
            stage=str(payload.get("lifecycle_stage", LIFECYCLE_VALIDATED_EPISODE)),
            cycle=int(episode.timestamp),
            previous_stage=previous_stage,
            details={"support_count": occurrences},
        )
        if bool(gate.get("identity_critical", False)):
            payload["identity_critical"] = True
            self._set_lifecycle_stage(
                payload,
                LIFECYCLE_PROTECTED_IDENTITY_CRITICAL,
                event="promoted_identity_critical",
                cycle=int(episode.timestamp),
                details={"support_count": occurrences},
            )
            self._record_memory_lifecycle_event(
                payload,
                event="promoted_identity_critical",
                stage=LIFECYCLE_PROTECTED_IDENTITY_CRITICAL,
                cycle=int(episode.timestamp),
                previous_stage=previous_stage,
                details={"support_count": occurrences},
            )
            return
        self._maybe_promote_episode_lifecycle(payload, cycle=int(episode.timestamp))

    def _maybe_promote_episode_lifecycle(
        self,
        payload: dict[str, object],
        *,
        cycle: int,
    ) -> None:
        current_stage = str(payload.get("lifecycle_stage", ""))
        if current_stage != LIFECYCLE_CANDIDATE_EPISODE:
            return
        support = int(payload.get("support_count", payload.get("support", 1)))
        qualifies = (
            support >= self.sleep_minimum_support
            and (
                float(payload.get("episode_score", 0.0)) >= self.episode_score_threshold
                or float(payload.get("value_relevance", 0.0)) >= 0.75
                or float(payload.get("threat_significance", 0.0)) >= 0.60
            )
        )
        if not qualifies:
            return
        self._set_lifecycle_stage(
            payload,
            LIFECYCLE_VALIDATED_EPISODE,
            event="promoted_by_support",
            cycle=cycle,
            details={"support_count": support},
        )
        self._record_memory_lifecycle_event(
            payload,
            event="promoted_by_support",
            stage=LIFECYCLE_VALIDATED_EPISODE,
            cycle=cycle,
            previous_stage=current_stage,
            details={"support_count": support},
        )

    def _record_memory_lifecycle_event(
        self,
        payload: dict[str, object],
        *,
        event: str,
        stage: str,
        cycle: int,
        previous_stage: object,
        details: dict[str, object] | None = None,
    ) -> None:
        self.lifecycle_events.append(
            {
                "episode_id": payload.get("episode_id"),
                "event": event,
                "stage": stage,
                "previous_stage": previous_stage,
                "cycle": int(cycle),
                "episode_family": payload.get("episode_family", EPISODE_FAMILY_ROUTINE),
                "details": dict(details or {}),
            }
        )

    def _set_lifecycle_stage(
        self,
        payload: dict[str, object],
        stage: str,
        *,
        event: str,
        cycle: int,
        details: dict[str, object] | None = None,
    ) -> None:
        previous_stage = payload.get("lifecycle_stage")
        payload["lifecycle_stage"] = stage
        history = list(payload.get("lifecycle_history", []))
        history.append(
            {
                "event": event,
                "stage": stage,
                "previous_stage": previous_stage,
                "cycle": int(cycle),
                "details": dict(details or {}),
            }
        )
        payload["lifecycle_history"] = history
        payload["last_lifecycle_event"] = event

    def _infer_episode_family(self, payload: dict[str, object]) -> str:
        observation = self._episode_observation(payload)
        outcome = _coerce_float_dict(payload.get("outcome_state", payload.get("outcome")))
        predicted_outcome = str(payload.get("predicted_outcome", "neutral"))
        if (
            predicted_outcome in {"survival_threat", "integrity_loss"}
            or float(observation.get("danger", 0.0)) >= 0.60
        ):
            return EPISODE_FAMILY_HAZARD
        if (
            predicted_outcome == "resource_gain"
            or float(observation.get("food", 0.0)) >= 0.70
            or float(outcome.get("energy_delta", 0.0)) > 0.08
        ):
            return EPISODE_FAMILY_RESOURCE
        if float(observation.get("social", 0.0)) >= 0.60:
            return EPISODE_FAMILY_SOCIAL
        if (
            float(observation.get("shelter", 0.0)) <= 0.20
            or abs(float(observation.get("temperature", 0.0)) - 0.50) >= 0.12
            or float(observation.get("novelty", 0.0)) >= 0.70
        ):
            return EPISODE_FAMILY_ENVIRONMENT
        return EPISODE_FAMILY_ROUTINE

    def _extract_family_features(self, payload: dict[str, object]) -> dict[str, float]:
        observation = self._episode_observation(payload)
        return {
            "danger": float(observation.get("danger", 0.0)),
            "food": float(observation.get("food", 0.0)),
            "social": float(observation.get("social", 0.0)),
            "novelty": float(observation.get("novelty", 0.0)),
            "shelter": float(observation.get("shelter", 0.0)),
            "temperature": float(observation.get("temperature", 0.0)),
        }

    def _refresh_semantic_patterns(self) -> None:
        by_action: dict[str, list[dict[str, object]]] = {}
        for payload in self.episodes:
            action = action_name(payload.get("action_taken", payload.get("action", "")))
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

    def _merge_episode_group(
        self,
        payloads: list[dict[str, object]],
    ) -> dict[str, object]:
        episodes = [Episode.from_dict(payload) for payload in payloads]
        protected_payloads = [
            payload for payload in payloads if bool(payload.get("identity_critical", False))
        ]
        average_state_vector = {
            key: mean(episode.state_vector.get(key, 0.0) for episode in episodes)
            for key in {
                state_key
                for episode in episodes
                for state_key in episode.state_vector
            }
        }
        average_outcome = {
            key: mean(episode.outcome_state.get(key, 0.0) for episode in episodes)
            for key in {
                outcome_key
                for episode in episodes
                for outcome_key in episode.outcome_state
            }
        }
        label_counts = Counter(episode.predicted_outcome for episode in episodes)
        merged = Episode(
            timestamp=int(round(mean(episode.timestamp for episode in episodes))),
            state_vector=average_state_vector,
            action_taken=episodes[0].action_taken,
            outcome_state=average_outcome,
            predicted_outcome=sorted(
                label_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0],
            prediction_error=mean(episode.prediction_error for episode in episodes),
            risk=mean(episode.risk for episode in episodes),
            value_score=mean(episode.value_score for episode in episodes),
            total_surprise=mean(episode.total_surprise for episode in episodes),
            embedding=self._mean_vector(
                [episode.embedding or self._build_embedding(episode.state_vector) for episode in episodes]
            ),
            preferred_probability=mean(episode.preferred_probability for episode in episodes),
            preference_log_value=mean(episode.preference_log_value for episode in episodes),
        )
        merged_payload = merged.to_dict()
        merged_payload["compressed_count"] = len(payloads)
        if protected_payloads:
            reference = max(
                protected_payloads,
                key=lambda payload: (
                    int(payload.get("timestamp", payload.get("cycle", 0))),
                    str(payload.get("episode_id", "")),
                ),
            )
            merged_payload["identity_critical"] = True
            merged_payload["identity_commitment_reason"] = str(
                reference.get("identity_commitment_reason", "protected_identity_continuity")
            )
            merged_payload["identity_commitment_ids"] = list(
                dict.fromkeys(
                    str(commitment_id)
                    for payload in protected_payloads
                    for commitment_id in payload.get("identity_commitment_ids", [])
                )
            )
            merged_payload["lifecycle_stage"] = LIFECYCLE_PROTECTED_IDENTITY_CRITICAL
        return merged_payload


def _cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - _cosine_similarity(left, right)

class AutobiographicalMemory(LongTermMemory):
    """Named M2 memory surface for persistent autobiographical continuity."""

    def replay_during_sleep(
        self,
        *,
        rng,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        return super().replay_during_sleep(rng=rng, limit=limit)
