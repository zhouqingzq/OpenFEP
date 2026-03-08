from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class LongTermMemory:
    """Episodic and semantic memory for pattern retrieval."""

    episodes: list[dict] = field(default_factory=list)
    semantic_patterns: list[dict] = field(default_factory=list)
    max_episodes: int = 50

    def store_episode(
        self,
        cycle: int,
        observation: dict[str, float],
        prediction: dict[str, float],
        errors: dict[str, float],
        action: str,
        outcome: dict[str, float],
    ) -> None:
        """Store a full episode with context."""
        episode = {
            "cycle": cycle,
            "observation": dict(observation),
            "prediction": dict(prediction),
            "errors": dict(errors),
            "action": action,
            "outcome": outcome,
        }
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]

    def retrieve_similar(
        self,
        current_observation: dict[str, float],
        current_body_state: dict[str, float],
        k: int = 3,
    ) -> list[dict]:
        """Retrieve the most similar past episodes based on current observation."""
        if not self.episodes:
            return []

        reference_cycle = self._resolve_reference_cycle(
            current_observation=current_observation,
            current_body_state=current_body_state,
        )

        def similarity(episode: dict) -> float:
            obs_sim = sum(
                abs(
                    episode["observation"].get(key, 0.5)
                    - current_observation.get(key, 0.5)
                )
                for key in current_observation
            ) / len(current_observation)
            recency = 1.0 / (1.0 + abs(episode["cycle"] - reference_cycle))
            return 1.0 / (1.0 + obs_sim) * recency

        scored = [(similarity(episode), episode) for episode in self.episodes]
        scored.sort(reverse=True, key=lambda item: item[0])
        return [episode for _, episode in scored[:k]]

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
            float(episode.get("cycle", 0))
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
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> LongTermMemory:
        if not payload:
            return cls()

        return cls(
            episodes=list(payload.get("episodes", [])),
            semantic_patterns=list(payload.get("semantic_patterns", [])),
            max_episodes=int(payload.get("max_episodes", 50)),
        )