from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from .m4_cognitive_style import CognitiveStyleParameters

if TYPE_CHECKING:
    from .memory_model import MemoryEntry
    from .memory_store import MemoryStore
    from .memory_retrieval import RetrievalQuery, RetrievalResult
    from .memory_encoding import SalienceConfig


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_token(value: str) -> str:
    return str(value).strip().lower()


def _tokenize(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                tokens.update(_tokenize(item))
            continue
        if isinstance(value, dict):
            for item in value.values():
                tokens.update(_tokenize(item))
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        tokens.update(token for token in re.split(r"[^a-z0-9_]+", text) if token)
    return tokens


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    shared = left & right
    return _clamp(len(shared) / max(1, min(len(left), len(right))))


def _style_value(
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None,
    key: str,
    default: float = 0.0,
) -> float:
    if cognitive_style is None:
        return default
    if hasattr(cognitive_style, key):
        try:
            return float(getattr(cognitive_style, key))
        except (TypeError, ValueError):
            return default
    if isinstance(cognitive_style, dict):
        try:
            return float(cognitive_style.get(key, default))
        except (TypeError, ValueError):
            return default
    return default


def _coerce_cycle(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool:
    return bool(value)


def _recent_entries(
    entries: Iterable["MemoryEntry"],
    *,
    window_size: int,
) -> list["MemoryEntry"]:
    ordered = sorted(
        list(entries),
        key=lambda entry: (
            max(
                _coerce_cycle(getattr(entry, "last_accessed", 0)),
                _coerce_cycle(getattr(entry, "created_at", 0)),
            ),
            str(getattr(entry, "id", "")),
        ),
        reverse=True,
    )
    return ordered[: max(1, int(window_size))]


def _recent_mood_baseline(entries: list["MemoryEntry"]) -> str:
    moods = [str(entry.mood_context).strip() for entry in entries if str(entry.mood_context).strip()]
    if moods:
        return Counter(moods).most_common(1)[0][0]
    if not entries:
        return "neutral"
    average_valence = sum(float(entry.valence) for entry in entries) / len(entries)
    if average_valence < -0.15:
        return "threatened"
    if average_valence > 0.15:
        return "steady"
    return "neutral"


def _dominant_tags(entries: list["MemoryEntry"], *, limit: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        for tag in [*getattr(entry, "semantic_tags", []), *getattr(entry, "context_tags", [])]:
            normalized = _normalize_token(tag)
            if normalized:
                counter[normalized] += 1
    return [tag for tag, _ in counter.most_common(limit)]


def _identity_themes(entries: list["MemoryEntry"], *, limit: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        metadata = dict(getattr(entry, "compression_metadata", {}) or {})
        encoding_audit = dict(metadata.get("encoding_audit", {}))
        self_evidence = _string_list(encoding_audit.get("self_evidence"))
        is_identity_entry = (
            float(getattr(entry, "relevance_self", 0.0)) >= 0.35
            or str(metadata.get("lineage_type", "")) == "identity_consolidation"
            or any("identity" in item or "commitment" in item or "continuity" in item for item in self_evidence)
        )
        if not is_identity_entry:
            continue
        for tag in getattr(entry, "semantic_tags", []):
            normalized = _normalize_token(tag)
            if normalized:
                counter[normalized] += 2
        for item in self_evidence:
            if item.startswith("guard:") or item.endswith(":none"):
                continue
            _, _, suffix = item.partition(":")
            for token in _tokenize(suffix or item):
                counter[token] += 1
    return [tag for tag, _ in counter.most_common(limit)]


def _threat_level(entries: list["MemoryEntry"]) -> float:
    if not entries:
        return 0.0
    pressures: list[float] = []
    for entry in entries:
        negative_valence = max(0.0, -float(entry.valence))
        arousal = _clamp(float(entry.arousal))
        threat_relevance = _clamp(float(getattr(entry, "relevance_threat", 0.0)))
        pressures.append(_clamp((negative_valence * 0.45) + (arousal * 0.35) + (threat_relevance * 0.35)))
    average = sum(pressures) / len(pressures)
    return _clamp(average * 1.1)


@dataclass
class AgentStateVector:
    active_goals: list[str] = field(default_factory=list)
    recent_mood_baseline: str = "neutral"
    recent_dominant_tags: list[str] = field(default_factory=list)
    identity_active_themes: list[str] = field(default_factory=list)
    threat_level: float = 0.0
    reward_context_active: bool = False
    social_context_active: bool = False
    last_updated: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "active_goals": list(self.active_goals),
            "recent_mood_baseline": self.recent_mood_baseline,
            "recent_dominant_tags": list(self.recent_dominant_tags),
            "identity_active_themes": list(self.identity_active_themes),
            "threat_level": round(self.threat_level, 6),
            "reward_context_active": bool(self.reward_context_active),
            "social_context_active": bool(self.social_context_active),
            "last_updated": int(self.last_updated),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | dict[str, object] | None) -> "AgentStateVector":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            active_goals=_string_list(payload.get("active_goals")),
            recent_mood_baseline=str(payload.get("recent_mood_baseline", "neutral") or "neutral"),
            recent_dominant_tags=_string_list(payload.get("recent_dominant_tags")),
            identity_active_themes=_string_list(payload.get("identity_active_themes")),
            threat_level=_clamp(float(payload.get("threat_level", 0.0))),
            reward_context_active=_coerce_bool(payload.get("reward_context_active")),
            social_context_active=_coerce_bool(payload.get("social_context_active")),
            last_updated=_coerce_cycle(payload.get("last_updated", 0)),
        )

    def to_context(self) -> dict[str, object]:
        return {
            "active_goals": list(self.active_goals),
            "goal_keywords": list(self.active_goals),
            "recent_mood_baseline": self.recent_mood_baseline,
            "recent_dominant_tags": list(self.recent_dominant_tags),
            "identity_active_themes": list(self.identity_active_themes),
            "identity_themes": list(self.identity_active_themes),
            "self_narrative_keywords": list(self.identity_active_themes),
            "threat_level": self.threat_level,
            "reward_context_active": self.reward_context_active,
            "social_context_active": self.social_context_active,
            "last_updated": self.last_updated,
        }

    def snapshot_for_consolidation(self) -> dict[str, object]:
        snapshot = self.to_dict()
        snapshot["snapshot_kind"] = "pre_consolidation_state_vector"
        return snapshot

    @classmethod
    def update_from_recent_entries(
        cls,
        store: "MemoryStore | Iterable[MemoryEntry]",
        *,
        window_size: int = 30,
        cycle: int = 0,
        active_goals: list[str] | None = None,
        previous_state: "AgentStateVector | None" = None,
    ) -> "AgentStateVector":
        source_entries = list(store.entries) if hasattr(store, "entries") else list(store)
        recent = _recent_entries(source_entries, window_size=max(20, min(50, int(window_size))))
        reward_context_active = any(
            float(getattr(entry, "relevance_reward", 0.0)) >= 0.45 or float(entry.valence) >= 0.35
            for entry in recent
        )
        social_context_active = any(
            float(getattr(entry, "relevance_social", 0.0)) >= 0.35
            or bool(getattr(entry, "anchor_slots", {}).get("agents"))
            for entry in recent
        )
        return cls(
            active_goals=list(active_goals or (previous_state.active_goals if previous_state else [])),
            recent_mood_baseline=_recent_mood_baseline(recent),
            recent_dominant_tags=_dominant_tags(recent),
            identity_active_themes=_identity_themes(recent),
            threat_level=_threat_level(recent),
            reward_context_active=reward_context_active,
            social_context_active=social_context_active,
            last_updated=int(cycle),
        )


@dataclass(frozen=True)
class DynamicSalienceAudit:
    base_weights: dict[str, float]
    effective_weights: dict[str, float]
    learning_mode_weight: float
    goal_match_ratio: float
    goal_match_boost: float
    identity_match_ratio: float
    identity_match_boost: float
    effective_relevance_self_multiplier: float
    contributions: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "base_weights": dict(self.base_weights),
            "effective_weights": dict(self.effective_weights),
            "learning_mode_weight": self.learning_mode_weight,
            "goal_match_ratio": self.goal_match_ratio,
            "goal_match_boost": self.goal_match_boost,
            "identity_match_ratio": self.identity_match_ratio,
            "identity_match_boost": self.identity_match_boost,
            "effective_relevance_self_multiplier": self.effective_relevance_self_multiplier,
            "contributions": dict(self.contributions),
        }


@dataclass(frozen=True)
class DynamicSalienceRegulation:
    effective_w_arousal: float
    effective_w_attention: float
    effective_w_novelty: float
    effective_w_relevance: float
    effective_relevance_self_multiplier: float
    goal_match_ratio: float
    identity_match_ratio: float
    audit: DynamicSalienceAudit


def normalize_agent_state(agent_state: AgentStateVector | dict[str, object] | None) -> AgentStateVector:
    if isinstance(agent_state, AgentStateVector):
        return agent_state
    if isinstance(agent_state, dict):
        return AgentStateVector.from_dict(agent_state)
    return AgentStateVector()


def merge_state_context(
    current_state: dict[str, object] | None,
    *,
    agent_state: AgentStateVector | dict[str, object] | None = None,
    cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
) -> dict[str, object]:
    merged = dict(current_state or {})
    normalized = normalize_agent_state(agent_state)
    merged.update(normalized.to_context())
    if cognitive_style is not None:
        merged["cognitive_style"] = (
            cognitive_style.to_dict() if hasattr(cognitive_style, "to_dict") else dict(cognitive_style)
        )
    return merged


def resolve_dynamic_salience(
    raw_input: dict[str, Any],
    current_state: dict[str, object] | None,
    config: "SalienceConfig",
    *,
    agent_state: AgentStateVector | dict[str, object] | None = None,
) -> DynamicSalienceRegulation:
    normalized = normalize_agent_state(agent_state)
    state_context = merge_state_context(current_state, agent_state=normalized)
    input_tokens = _tokenize(
        raw_input.get("content"),
        raw_input.get("semantic_tags"),
        raw_input.get("context_tags"),
        raw_input.get("goal_cues"),
        raw_input.get("identity_themes"),
        raw_input.get("narrative_nodes"),
        raw_input.get("roles"),
        raw_input.get("relationships"),
        raw_input.get("commitments"),
    )
    goal_tokens = _tokenize(
        state_context.get("active_goals"),
        state_context.get("goal_keywords"),
    )
    identity_tokens = _tokenize(
        state_context.get("identity_active_themes"),
        state_context.get("identity_themes"),
        state_context.get("self_narrative_keywords"),
        state_context.get("active_commitments"),
    )
    goal_match_ratio = _overlap_ratio(input_tokens, goal_tokens)
    identity_match_ratio = _overlap_ratio(input_tokens, identity_tokens)
    learning_mode_weight = 0.3 if normalized.reward_context_active else 0.0
    goal_match_boost = goal_match_ratio * 0.5
    identity_match_boost = identity_match_ratio * 0.5
    effective_w_arousal = float(config.w_arousal) * (1.0 + (normalized.threat_level * 0.5))
    effective_w_novelty = float(config.w_novelty) * (1.0 + learning_mode_weight)
    effective_w_relevance = float(config.w_relevance) * (
        1.0 + goal_match_boost + (identity_match_boost * 0.35)
    )
    effective_relevance_self_multiplier = 1.0 + identity_match_boost
    audit = DynamicSalienceAudit(
        base_weights={
            "w_arousal": float(config.w_arousal),
            "w_attention": float(config.w_attention),
            "w_novelty": float(config.w_novelty),
            "w_relevance": float(config.w_relevance),
        },
        effective_weights={
            "w_arousal": round(effective_w_arousal, 6),
            "w_attention": float(config.w_attention),
            "w_novelty": round(effective_w_novelty, 6),
            "w_relevance": round(effective_w_relevance, 6),
        },
        learning_mode_weight=round(learning_mode_weight, 6),
        goal_match_ratio=round(goal_match_ratio, 6),
        goal_match_boost=round(goal_match_boost, 6),
        identity_match_ratio=round(identity_match_ratio, 6),
        identity_match_boost=round(identity_match_boost, 6),
        effective_relevance_self_multiplier=round(effective_relevance_self_multiplier, 6),
        contributions={
            "active_goals": list(normalized.active_goals),
            "identity_active_themes": list(normalized.identity_active_themes),
            "reward_context_active": normalized.reward_context_active,
            "threat_level": round(normalized.threat_level, 6),
            "identity_self_contribution_printable": bool(identity_match_ratio > 0.0),
        },
    )
    return DynamicSalienceRegulation(
        effective_w_arousal=effective_w_arousal,
        effective_w_attention=float(config.w_attention),
        effective_w_novelty=effective_w_novelty,
        effective_w_relevance=effective_w_relevance,
        effective_relevance_self_multiplier=effective_relevance_self_multiplier,
        goal_match_ratio=goal_match_ratio,
        identity_match_ratio=identity_match_ratio,
        audit=audit,
    )


def update_agent_state_vector(
    store: "MemoryStore | Iterable[MemoryEntry]",
    *,
    window_size: int = 30,
    cycle: int = 0,
    active_goals: list[str] | None = None,
    previous_state: AgentStateVector | None = None,
) -> AgentStateVector:
    return AgentStateVector.update_from_recent_entries(
        store,
        window_size=window_size,
        cycle=cycle,
        active_goals=active_goals,
        previous_state=previous_state,
    )


def identity_match_ratio_for_entry(
    entry: "MemoryEntry",
    agent_state: AgentStateVector | dict[str, object] | None,
) -> float:
    normalized = normalize_agent_state(agent_state)
    entry_tokens = _tokenize(entry.semantic_tags, entry.context_tags, entry.content)
    identity_tokens = _tokenize(normalized.identity_active_themes)
    return _overlap_ratio(entry_tokens, identity_tokens)


class MemoryAwareAgentMixin:
    memory_store: "MemoryStore"
    agent_state_vector: AgentStateVector
    memory_cognitive_style: CognitiveStyleParameters
    memory_cycle_interval: int
    memory_salience_config: "SalienceConfig"

    def init_memory_awareness(
        self,
        *,
        memory_store: "MemoryStore | None" = None,
        state_vector: AgentStateVector | dict[str, object] | None = None,
        cognitive_style: CognitiveStyleParameters | dict[str, object] | None = None,
        memory_cycle_interval: int = 5,
        salience_config: "SalienceConfig | None" = None,
    ) -> None:
        from .memory_encoding import SalienceConfig
        from .memory_store import MemoryStore

        self.memory_store = memory_store or getattr(self, "memory_store", None) or MemoryStore()
        self.agent_state_vector = normalize_agent_state(state_vector)
        if cognitive_style is None:
            self.memory_cognitive_style = CognitiveStyleParameters()
        elif isinstance(cognitive_style, CognitiveStyleParameters):
            self.memory_cognitive_style = cognitive_style
        else:
            self.memory_cognitive_style = CognitiveStyleParameters.from_dict(dict(cognitive_style))
        self.memory_cycle_interval = max(1, int(memory_cycle_interval))
        self.memory_salience_config = salience_config or getattr(self, "memory_salience_config", SalienceConfig())
        self.memory_store.agent_state_vector = self.agent_state_vector.to_dict()

    def _memory_active_goals(self) -> list[str]:
        active_goal = getattr(getattr(self, "goal_stack", None), "active_goal", None)
        goals: list[str] = []
        if active_goal is not None:
            name = str(getattr(active_goal, "name", "") or "").strip()
            if name:
                goals.append(name)
        goals.extend(self.agent_state_vector.active_goals)
        return list(dict.fromkeys(goals))

    def _memory_identity_themes(self) -> list[str]:
        themes = list(self.agent_state_vector.identity_active_themes)
        narrative = getattr(getattr(self, "self_model", None), "identity_narrative", None)
        if narrative is not None:
            core_identity = str(getattr(narrative, "core_identity", "") or "").strip()
            if core_identity:
                themes.extend(sorted(_tokenize(core_identity)))
            commitments = getattr(narrative, "commitments", [])
            for commitment in commitments[:4]:
                statement = str(getattr(commitment, "statement", "") or "").strip()
                if statement:
                    themes.extend(sorted(_tokenize(statement)))
        return list(dict.fromkeys(themes))

    def build_memory_state_context(self) -> dict[str, object]:
        context = self.agent_state_vector.to_context()
        context["active_goals"] = self._memory_active_goals()
        context["goal_keywords"] = list(context["active_goals"])
        identity_themes = self._memory_identity_themes()
        context["identity_active_themes"] = identity_themes
        context["identity_themes"] = identity_themes
        context["self_narrative_keywords"] = identity_themes
        context["cognitive_style"] = self.memory_cognitive_style.to_dict()
        return context

    def encode_cycle_memory(self, raw_input: dict[str, Any], cycle: int):
        from .memory_encoding import encode_memory

        current_state = self.build_memory_state_context()
        payload = dict(raw_input)
        payload.setdefault("created_at", cycle)
        payload.setdefault("cycle", cycle)
        entry = encode_memory(
            payload,
            current_state,
            self.memory_salience_config,
            agent_state=self.agent_state_vector,
            cognitive_style=self.memory_cognitive_style,
        )
        self.memory_store.add(
            entry,
            current_state=current_state,
            cognitive_style=self.memory_cognitive_style,
        )
        self.agent_state_vector = update_agent_state_vector(
            self.memory_store,
            window_size=getattr(self.memory_store, "state_window_size", 30),
            cycle=cycle,
            active_goals=self._memory_active_goals(),
            previous_state=self.agent_state_vector,
        )
        self.memory_store.agent_state_vector = self.agent_state_vector.to_dict()
        return entry

    def retrieve_for_decision(
        self,
        query: "RetrievalQuery",
        cycle: int,
        *,
        current_mood: str | None = None,
        k: int = 5,
    ) -> "RetrievalResult":
        result = self.memory_store.retrieve(
            query,
            current_mood=current_mood,
            k=k,
            agent_state=self.agent_state_vector,
            cognitive_style=self.memory_cognitive_style,
        )
        return result

    def reconsolidate_after_recall(
        self,
        entry_id: str,
        *,
        current_mood: str | None = None,
        current_context_tags: list[str] | None = None,
        current_cycle: int | None = None,
        recall_artifact: Any = None,
        conflict_type: Any = None,
    ):
        return self.memory_store.reconsolidate_entry(
            entry_id,
            current_mood=current_mood,
            current_context_tags=current_context_tags,
            current_cycle=current_cycle,
            current_state=self.build_memory_state_context(),
            recall_artifact=recall_artifact,
            conflict_type=conflict_type,
            cognitive_style=self.memory_cognitive_style,
        )

    def run_memory_consolidation_if_due(self, cycle: int, *, rng: Any | None = None):
        import random

        if cycle % self.memory_cycle_interval != 0:
            return None
        report = self.memory_store.run_consolidation_cycle(
            current_cycle=cycle,
            rng=rng or random.Random(cycle),
            current_state=self.build_memory_state_context(),
            cognitive_style=self.memory_cognitive_style,
        )
        self.agent_state_vector = update_agent_state_vector(
            self.memory_store,
            window_size=getattr(self.memory_store, "state_window_size", 30),
            cycle=cycle,
            active_goals=self._memory_active_goals(),
            previous_state=self.agent_state_vector,
        )
        self.memory_store.agent_state_vector = self.agent_state_vector.to_dict()
        return report
