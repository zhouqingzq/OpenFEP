from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from types import MappingProxyType
from typing import Callable, Mapping

from .action_schema import ActionSchema, action_name, ensure_action_schema


SELF_ERROR = "self_error"
WORLD_ERROR = "world_error"
EXISTENTIAL_THREAT = "existential_threat"

CORE_ACTIONS = (
    "observe_world",
    "update_beliefs",
    "act_on_world",
    "persist_state",
    "emit_trace",
    "sample_host_telemetry",
)

CORE_API_LIMITS = {
    "requests_per_minute": 60.0,
    "context_window_tokens": 256.0,
    "file_ops_per_window": 8.0,
    "network_ops_per_window": 2.0,
    "external_failures_before_lockout": 3.0,
}
HIGH_SURPRISE_THRESHOLD = 3.0
MAX_CHAPTER_TICKS = 500
MAX_CHAPTERS = 50


@dataclass(slots=True)
class ThreatProfile:
    hard_limits: dict[str, dict[str, object]] = field(
        default_factory=lambda: {
            "energy": {"critical_low": 0.05, "source": "self"},
            "stress": {"critical_high": 0.95, "source": "self"},
            "fatigue": {"critical_high": 0.95, "source": "self"},
            "temperature": {"critical_low": 0.05, "critical_high": 0.95, "source": "self"},
        }
    )
    learned_threats: list[dict[str, object]] = field(default_factory=list)

    def add_learned_threat(
        self,
        pattern: str,
        risk_level: float,
        tick: int,
        source: str = "world",
    ) -> None:
        self.learned_threats.append(
            {
                "pattern": pattern,
                "risk_level": float(risk_level),
                "learned_at_tick": int(tick),
                "source": source,
            }
        )
        self.learned_threats.sort(
            key=lambda item: (
                int(item.get("learned_at_tick", 0)),
                str(item.get("pattern", "")),
            )
        )

    def get(self, modality: str, default=None):
        return self.hard_limits.get(modality, default)

    def to_dict(self) -> dict[str, object]:
        return {
            "hard_limits": {str(key): dict(value) for key, value in self.hard_limits.items()},
            "learned_threats": [dict(item) for item in self.learned_threats],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ThreatProfile":
        if not payload:
            return cls()
        if "hard_limits" in payload:
            hard_limits = payload.get("hard_limits")
            learned = payload.get("learned_threats", [])
            return cls(
                hard_limits={
                    str(key): dict(value)
                    for key, value in dict(hard_limits or {}).items()
                    if isinstance(value, Mapping)
                },
                learned_threats=[dict(item) for item in learned if isinstance(item, Mapping)],
            )
        return cls(
            hard_limits={
                str(key): dict(value)
                for key, value in dict(payload).items()
                if isinstance(value, Mapping)
            }
        )


def _event_name(event: object) -> str:
    if isinstance(event, RuntimeFailureEvent):
        return event.name
    if isinstance(event, BaseException):
        return type(event).__name__
    name = getattr(event, "__name__", None)
    if isinstance(name, str):
        return name
    return str(event)


def _normalize_event_name(event: object) -> str:
    aliases = {
        "tokenlimitexceeded": "token_exhaustion",
        "tokenexhaustion": "token_exhaustion",
        "outofmemory": "out_of_memory",
        "memoryerror": "out_of_memory",
        "httptimeout": "http_timeout",
        "timeouterror": "http_timeout",
        "networkfailure": "network_failure",
        "fatalexception": "fatal_exception",
        "memoryindexcorruption": "memory_index_corruption",
        "toolcapabilitydowngrade": "tool_capability_downgrade",
        "readonlyfilesystem": "read_only_filesystem",
        "domstructurechanged": "dom_structure_changed",
    }
    key = _event_name(event).strip().casefold()
    return aliases.get(key, key)


@dataclass(frozen=True, slots=True)
class RuntimeFailureEvent:
    name: str
    stage: str
    category: str = ""
    origin_hint: str = ""
    details: Mapping[str, object] = field(default_factory=dict)
    resource_state: Mapping[str, float | int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        if self.resource_state is not None:
            object.__setattr__(
                self,
                "resource_state",
                MappingProxyType(dict(self.resource_state)),
            )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "name": self.name,
            "stage": self.stage,
            "category": self.category,
            "origin_hint": self.origin_hint,
            "details": dict(self.details),
        }
        if self.resource_state is not None:
            payload["resource_state"] = dict(self.resource_state)
        return payload


@dataclass(frozen=True, slots=True)
class BodySchema:
    """Immutable structural priors describing the agent's internal body."""

    energy: float
    token_budget: int
    memory_usage: float
    compute_load: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "energy": self.energy,
            "token_budget": self.token_budget,
            "memory_usage": self.memory_usage,
            "compute_load": self.compute_load,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> BodySchema:
        default = cls(energy=0.85, token_budget=256, memory_usage=128.0, compute_load=0.25)
        if not payload:
            return default
        return cls(
            energy=float(payload.get("energy", default.energy)),
            token_budget=int(payload.get("token_budget", default.token_budget)),
            memory_usage=float(payload.get("memory_usage", default.memory_usage)),
            compute_load=float(payload.get("compute_load", default.compute_load)),
        )


@dataclass(frozen=True, slots=True)
class CapabilityModel:
    """Immutable prior over what the agent can do and where it is constrained."""

    available_actions: tuple[str, ...] = ()
    action_schemas: tuple[ActionSchema, ...] = ()
    api_limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action_schemas = tuple(
            ensure_action_schema(action) for action in self.action_schemas
        )
        if not action_schemas and self.available_actions:
            action_schemas = tuple(
                ActionSchema(name=str(action)) for action in self.available_actions
            )
        object.__setattr__(
            self,
            "action_schemas",
            action_schemas,
        )
        object.__setattr__(
            self,
            "available_actions",
            tuple(schema.name for schema in action_schemas),
        )
        object.__setattr__(
            self,
            "api_limits",
            MappingProxyType(dict(self.api_limits)),
        )

    def descriptor_for(self, action: str | ActionSchema) -> ActionSchema | None:
        action_key = action_name(action)
        for schema in self.action_schemas:
            if schema.name == action_key:
                return schema
        return None

    def supports(self, action: str | ActionSchema) -> bool:
        return self.descriptor_for(action) is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "action_schemas": [schema.to_dict() for schema in self.action_schemas],
            "available_actions": list(self.available_actions),
            "api_limits": dict(self.api_limits),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> CapabilityModel:
        if not payload:
            return cls()
        api_limits = payload.get("api_limits")
        if not isinstance(api_limits, dict):
            api_limits = {}
        action_schemas_payload = payload.get("action_schemas", [])
        if isinstance(action_schemas_payload, list) and action_schemas_payload:
            action_schemas = tuple(
                ActionSchema.from_dict(item) for item in action_schemas_payload
            )
        else:
            action_schemas = tuple(
                ActionSchema(name=str(action))
                for action in payload.get("available_actions", [])
            )
        return cls(
            action_schemas=action_schemas,
            api_limits={str(key): float(value) for key, value in api_limits.items()},
        )


@dataclass(slots=True)
class ResourceState:
    """Mutable runtime resource estimates."""

    tokens_remaining: int
    cpu_budget: float
    memory_free: float

    def snapshot(self) -> dict[str, float | int]:
        return {
            "tokens_remaining": self.tokens_remaining,
            "cpu_budget": self.cpu_budget,
            "memory_free": self.memory_free,
        }

    def to_dict(self) -> dict[str, float | int]:
        return self.snapshot()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ResourceState:
        default = cls(tokens_remaining=256, cpu_budget=1.0, memory_free=1024.0)
        if not payload:
            return default
        return cls(
            tokens_remaining=int(payload.get("tokens_remaining", default.tokens_remaining)),
            cpu_budget=float(payload.get("cpu_budget", default.cpu_budget)),
            memory_free=float(payload.get("memory_free", default.memory_free)),
        )

    def predict(self, body_schema: BodySchema, threat_model: ThreatModel | None = None) -> dict[str, bool]:
        token_threshold = threat_model.token_exhaustion_threshold if threat_model else 0
        memory_threshold = threat_model.memory_overflow_threshold if threat_model else 0.0
        return {
            "token_exhaustion": self.tokens_remaining <= token_threshold,
            "memory_overflow": self.memory_free <= memory_threshold
            or body_schema.memory_usage > self.memory_free,
            "cpu_overload": body_schema.compute_load > self.cpu_budget,
        }


@dataclass(frozen=True, slots=True)
class ThreatModel:
    """Immutable fatal-condition priors for the agent."""

    token_exhaustion_threshold: int = 0
    memory_overflow_threshold: float = 0.0
    fatal_exceptions: tuple[str, ...] = ("FatalException",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fatal_exceptions",
            tuple(_normalize_event_name(name) for name in self.fatal_exceptions),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "token_exhaustion_threshold": self.token_exhaustion_threshold,
            "memory_overflow_threshold": self.memory_overflow_threshold,
            "fatal_exceptions": list(self.fatal_exceptions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ThreatModel:
        if not payload:
            return cls()
        return cls(
            token_exhaustion_threshold=int(payload.get("token_exhaustion_threshold", 0)),
            memory_overflow_threshold=float(payload.get("memory_overflow_threshold", 0.0)),
            fatal_exceptions=tuple(str(name) for name in payload.get("fatal_exceptions", ["FatalException"])),
        )

    def detect(
        self,
        event: object,
        resource_state: ResourceState,
        body_schema: BodySchema,
    ) -> tuple[str, ...]:
        predicted = resource_state.predict(body_schema, self)
        threats: list[str] = []
        if predicted["token_exhaustion"]:
            threats.append("token_exhaustion")
        if predicted["memory_overflow"]:
            threats.append("memory_overflow")
        if _normalize_event_name(event) in self.fatal_exceptions:
            threats.append("fatal_exception")
        return tuple(threats)


@dataclass(frozen=True, slots=True)
class ErrorClassifier:
    """Deterministic classifier for surprise source."""

    self_error_events: tuple[str, ...] = (
        "token_exhaustion",
        "out_of_memory",
        "memory_index_corruption",
        "tool_capability_downgrade",
    )
    world_error_events: tuple[str, ...] = (
        "http_timeout",
        "network_failure",
        "read_only_filesystem",
        "dom_structure_changed",
    )
    existential_events: tuple[str, ...] = ("fatal_exception",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "self_error_events",
            tuple(_normalize_event_name(name) for name in self.self_error_events),
        )
        object.__setattr__(
            self,
            "world_error_events",
            tuple(_normalize_event_name(name) for name in self.world_error_events),
        )
        object.__setattr__(
            self,
            "existential_events",
            tuple(_normalize_event_name(name) for name in self.existential_events),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "self_error_events": list(self.self_error_events),
            "world_error_events": list(self.world_error_events),
            "existential_events": list(self.existential_events),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> ErrorClassifier:
        if not payload:
            return cls()
        return cls(
            self_error_events=tuple(str(name) for name in payload.get("self_error_events", cls().self_error_events)),
            world_error_events=tuple(str(name) for name in payload.get("world_error_events", cls().world_error_events)),
            existential_events=tuple(str(name) for name in payload.get("existential_events", cls().existential_events)),
        )

    def classify(self, event: object) -> str:
        if isinstance(event, RuntimeFailureEvent):
            origin_hint = str(event.origin_hint).strip().casefold()
            if origin_hint in {"self", SELF_ERROR}:
                return SELF_ERROR
            if origin_hint in {"world", WORLD_ERROR}:
                return WORLD_ERROR
            if origin_hint in {"existential", EXISTENTIAL_THREAT}:
                return EXISTENTIAL_THREAT
            category = str(event.category).strip().casefold()
            if category in {"resource_exhaustion", "memory_budget", "context_budget"}:
                return SELF_ERROR
            if category in {"external_failure", "environment_shift", "timeout", "tool_failure"}:
                return WORLD_ERROR
        normalized = _normalize_event_name(event)
        if normalized in self.existential_events:
            return EXISTENTIAL_THREAT
        if normalized in self.self_error_events:
            return SELF_ERROR
        if normalized in self.world_error_events:
            return WORLD_ERROR
        return WORLD_ERROR

    @staticmethod
    def attribution(classification: str) -> str:
        if classification == SELF_ERROR:
            return "self"
        if classification == WORLD_ERROR:
            return "world"
        return "existential"

    @staticmethod
    def surprise_source(classification: str) -> str:
        if classification == SELF_ERROR:
            return "interoceptive"
        if classification == WORLD_ERROR:
            return "exteroceptive"
        return "existential"

    def evidence(
        self,
        event: object,
        *,
        resource_state: Mapping[str, float | int],
        body_schema: BodySchema,
    ) -> dict[str, object]:
        predicted = ResourceState(
            tokens_remaining=int(resource_state.get("tokens_remaining", 0)),
            cpu_budget=float(resource_state.get("cpu_budget", 0.0)),
            memory_free=float(resource_state.get("memory_free", 0.0)),
        ).predict(body_schema, None)
        evidence = {
            "event_name": _event_name(event),
            "normalized_event": _normalize_event_name(event),
            "resource_state": dict(resource_state),
            "resource_flags": predicted,
        }
        if isinstance(event, RuntimeFailureEvent):
            evidence.update(
                {
                    "stage": event.stage,
                    "category": event.category,
                    "origin_hint": event.origin_hint,
                    "details": dict(event.details),
                }
            )
            if event.resource_state is not None:
                evidence["resource_state"] = dict(event.resource_state)
        return evidence


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    event: str
    classification: str
    attribution: str
    resource_state: Mapping[str, float | int]
    surprise_source: str
    detected_threats: tuple[str, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_state",
            MappingProxyType(dict(self.resource_state)),
        )
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )

    def to_log_string(self) -> str:
        return "\n".join(
            [
                "[SelfModel]",
                f"event={self.event}",
                f"classification={self.classification}",
                f"attribution={self.attribution}",
                f"resource_state={dict(self.resource_state)}",
                f"surprise_source={self.surprise_source}",
                f"detected_threats={list(self.detected_threats)}",
                f"evidence={dict(self.evidence)}",
            ]
        )


@dataclass(slots=True)
class PreferredPolicies:
    dominant_strategy: str = "expected_free_energy"
    action_distribution: dict[str, float] = field(default_factory=dict)
    risk_profile: str = "risk_neutral"
    learned_avoidances: list[str] = field(default_factory=list)
    learned_preferences: list[str] = field(default_factory=list)
    last_updated_tick: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "dominant_strategy": self.dominant_strategy,
            "action_distribution": dict(self.action_distribution),
            "risk_profile": self.risk_profile,
            "learned_avoidances": list(self.learned_avoidances),
            "learned_preferences": list(self.learned_preferences),
            "last_updated_tick": self.last_updated_tick,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> PreferredPolicies:
        if not data:
            return cls()
        distribution = data.get("action_distribution")
        if not isinstance(distribution, dict):
            distribution = {}
        return cls(
            dominant_strategy=str(data.get("dominant_strategy", "expected_free_energy")),
            action_distribution={str(key): float(value) for key, value in distribution.items()},
            risk_profile=str(data.get("risk_profile", "risk_neutral")),
            learned_avoidances=[str(item) for item in data.get("learned_avoidances", [])],
            learned_preferences=[str(item) for item in data.get("learned_preferences", [])],
            last_updated_tick=int(data.get("last_updated_tick", 0)),
        )


@dataclass(slots=True)
class NarrativeChapter:
    """A time-bounded chapter in the agent's autobiographical narrative."""

    chapter_id: int
    tick_range: tuple[int, int]
    dominant_theme: str
    key_events: list[str] = field(default_factory=list)
    behavioral_shift: str | None = None
    state_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter_id": self.chapter_id,
            "tick_range": [int(self.tick_range[0]), int(self.tick_range[1])],
            "dominant_theme": self.dominant_theme,
            "key_events": list(self.key_events),
            "behavioral_shift": self.behavioral_shift,
            "state_summary": dict(self.state_summary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> NarrativeChapter:
        if not data:
            return cls(chapter_id=0, tick_range=(0, 0), dominant_theme="consolidation")
        tick_range = data.get("tick_range", (0, 0))
        if not isinstance(tick_range, (list, tuple)) or len(tick_range) != 2:
            tick_range = (0, 0)
        state_summary = data.get("state_summary")
        if not isinstance(state_summary, dict):
            state_summary = {}
        behavioral_shift = data.get("behavioral_shift")
        return cls(
            chapter_id=int(data.get("chapter_id", 0)),
            tick_range=(int(tick_range[0]), int(tick_range[1])),
            dominant_theme=str(data.get("dominant_theme", "consolidation")),
            key_events=[str(item) for item in data.get("key_events", [])][:5],
            behavioral_shift=(
                str(behavioral_shift) if isinstance(behavioral_shift, str) and behavioral_shift else None
            ),
            state_summary=dict(state_summary),
        )


@dataclass(slots=True)
class NarrativeClaim:
    claim_id: str
    claim_type: str
    text: str
    claim_key: str
    supported_by: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)
    support_score: float = 0.0
    contradiction_score: float = 0.0
    support_count: int = 0
    contradict_count: int = 0
    confidence: float = 0.0
    stale_since: int | None = None
    last_validated_at: int = 0
    source_sleep_session_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "text": self.text,
            "claim_key": self.claim_key,
            "supported_by": list(self.supported_by),
            "contradicted_by": list(self.contradicted_by),
            "support_score": self.support_score,
            "contradiction_score": self.contradiction_score,
            "support_count": self.support_count,
            "contradict_count": self.contradict_count,
            "confidence": self.confidence,
            "stale_since": self.stale_since,
            "last_validated_at": self.last_validated_at,
            "source_sleep_session_id": self.source_sleep_session_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> NarrativeClaim:
        if not data:
            return cls(claim_id="", claim_type="trait", text="", claim_key="")
        stale_since = data.get("stale_since")
        source_sleep_session_id = data.get("source_sleep_session_id")
        return cls(
            claim_id=str(data.get("claim_id", "")),
            claim_type=str(data.get("claim_type", "trait")),
            text=str(data.get("text", "")),
            claim_key=str(data.get("claim_key", "")),
            supported_by=[str(item) for item in data.get("supported_by", [])],
            contradicted_by=[str(item) for item in data.get("contradicted_by", [])],
            support_score=float(data.get("support_score", 0.0)),
            contradiction_score=float(data.get("contradiction_score", 0.0)),
            support_count=int(data.get("support_count", 0)),
            contradict_count=int(data.get("contradict_count", 0)),
            confidence=float(data.get("confidence", 0.0)),
            stale_since=int(stale_since) if isinstance(stale_since, (int, float)) else None,
            last_validated_at=int(data.get("last_validated_at", 0)),
            source_sleep_session_id=(
                int(source_sleep_session_id)
                if isinstance(source_sleep_session_id, (int, float))
                else None
            ),
        )


@dataclass(slots=True)
class IdentityCommitment:
    commitment_id: str
    commitment_type: str
    statement: str
    target_actions: list[str] = field(default_factory=list)
    discouraged_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    priority: float = 0.0
    source_claim_ids: list[str] = field(default_factory=list)
    source_chapter_ids: list[int] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    active: bool = True
    last_reaffirmed_tick: int = 0
    last_violated_tick: int | None = None
    violation_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "commitment_id": self.commitment_id,
            "commitment_type": self.commitment_type,
            "statement": self.statement,
            "target_actions": list(self.target_actions),
            "discouraged_actions": list(self.discouraged_actions),
            "confidence": self.confidence,
            "priority": self.priority,
            "source_claim_ids": list(self.source_claim_ids),
            "source_chapter_ids": list(self.source_chapter_ids),
            "evidence_ids": list(self.evidence_ids),
            "active": self.active,
            "last_reaffirmed_tick": self.last_reaffirmed_tick,
            "last_violated_tick": self.last_violated_tick,
            "violation_count": self.violation_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "IdentityCommitment":
        if not data:
            return cls(commitment_id="", commitment_type="identity", statement="")
        last_violated_tick = data.get("last_violated_tick")
        return cls(
            commitment_id=str(data.get("commitment_id", "")),
            commitment_type=str(data.get("commitment_type", "identity")),
            statement=str(data.get("statement", "")),
            target_actions=[str(item) for item in data.get("target_actions", [])],
            discouraged_actions=[str(item) for item in data.get("discouraged_actions", [])],
            confidence=float(data.get("confidence", 0.0)),
            priority=float(data.get("priority", 0.0)),
            source_claim_ids=[str(item) for item in data.get("source_claim_ids", [])],
            source_chapter_ids=[int(item) for item in data.get("source_chapter_ids", [])],
            evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
            active=bool(data.get("active", True)),
            last_reaffirmed_tick=int(data.get("last_reaffirmed_tick", 0)),
            last_violated_tick=(
                int(last_violated_tick)
                if isinstance(last_violated_tick, (int, float))
                else None
            ),
            violation_count=int(data.get("violation_count", 0)),
        )


@dataclass(slots=True)
class IdentityNarrative:
    chapters: list[NarrativeChapter] = field(default_factory=list)
    current_chapter: NarrativeChapter | None = None
    core_identity: str = ""
    core_summary: str = ""
    autobiographical_summary: str = ""
    trait_self_model: dict[str, object] = field(default_factory=dict)
    behavioral_patterns: list[str] = field(default_factory=list)
    significant_events: list[str] = field(default_factory=list)
    values_statement: str = ""
    claims: list[NarrativeClaim] = field(default_factory=list)
    commitments: list[IdentityCommitment] = field(default_factory=list)
    chapter_transition_evidence: list[dict[str, object]] = field(default_factory=list)
    contradiction_summary: dict[str, object] = field(default_factory=dict)
    evidence_provenance: dict[str, object] = field(default_factory=dict)
    last_updated_tick: int = 0
    version: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "current_chapter": (
                self.current_chapter.to_dict() if self.current_chapter is not None else None
            ),
            "core_identity": self.core_identity,
            "core_summary": self.core_summary,
            "autobiographical_summary": self.autobiographical_summary,
            "trait_self_model": dict(self.trait_self_model),
            "behavioral_patterns": list(self.behavioral_patterns),
            "significant_events": list(self.significant_events),
            "values_statement": self.values_statement,
            "claims": [claim.to_dict() for claim in self.claims],
            "commitments": [commitment.to_dict() for commitment in self.commitments],
            "chapter_transition_evidence": [dict(item) for item in self.chapter_transition_evidence],
            "contradiction_summary": dict(self.contradiction_summary),
            "evidence_provenance": dict(self.evidence_provenance),
            "last_updated_tick": self.last_updated_tick,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> IdentityNarrative:
        if not data:
            return cls()
        chapters_payload = data.get("chapters", [])
        current_chapter_payload = data.get("current_chapter")
        trait_self_model = data.get("trait_self_model")
        contradiction_summary = data.get("contradiction_summary")
        evidence_provenance = data.get("evidence_provenance")
        chapter_transition_evidence = data.get("chapter_transition_evidence")
        if not isinstance(trait_self_model, dict):
            trait_self_model = {}
        if not isinstance(contradiction_summary, dict):
            contradiction_summary = {}
        if not isinstance(evidence_provenance, dict):
            evidence_provenance = {}
        if not isinstance(chapter_transition_evidence, list):
            chapter_transition_evidence = []
        return cls(
            chapters=[
                NarrativeChapter.from_dict(item)
                for item in chapters_payload
                if isinstance(item, Mapping)
            ],
            current_chapter=(
                NarrativeChapter.from_dict(current_chapter_payload)
                if isinstance(current_chapter_payload, Mapping)
                else None
            ),
            core_identity=str(data.get("core_identity", "")),
            core_summary=str(data.get("core_summary", "")),
            autobiographical_summary=str(
                data.get("autobiographical_summary", data.get("core_summary", ""))
            ),
            trait_self_model=dict(trait_self_model),
            behavioral_patterns=[str(item) for item in data.get("behavioral_patterns", [])],
            significant_events=[str(item) for item in data.get("significant_events", [])],
            values_statement=str(data.get("values_statement", "")),
            claims=[
                NarrativeClaim.from_dict(item)
                for item in data.get("claims", [])
                if isinstance(item, Mapping)
            ],
            commitments=[
                IdentityCommitment.from_dict(item)
                for item in data.get("commitments", [])
                if isinstance(item, Mapping)
            ],
            chapter_transition_evidence=[
                dict(item) for item in chapter_transition_evidence if isinstance(item, Mapping)
            ],
            contradiction_summary=dict(contradiction_summary),
            evidence_provenance=dict(evidence_provenance),
            last_updated_tick=int(data.get("last_updated_tick", 0)),
            version=int(data.get("version", 0)),
        )


@dataclass(slots=True)
class SelfInconsistencyEvent:
    tick: int
    action: str
    active_commitments: list[str] = field(default_factory=list)
    relevant_commitments: list[str] = field(default_factory=list)
    commitment_compatibility_score: float = 1.0
    self_inconsistency_error: float = 0.0
    conflict_type: str = "none"
    severity_level: str = "none"
    consistency_classification: str = "aligned"
    behavioral_classification: str = "aligned"
    repair_triggered: bool = False
    repair_policy: str = ""
    repair_result: dict[str, object] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "action": self.action,
            "active_commitments": list(self.active_commitments),
            "relevant_commitments": list(self.relevant_commitments),
            "commitment_compatibility_score": float(self.commitment_compatibility_score),
            "self_inconsistency_error": float(self.self_inconsistency_error),
            "conflict_type": self.conflict_type,
            "severity_level": self.severity_level,
            "consistency_classification": self.consistency_classification,
            "behavioral_classification": self.behavioral_classification,
            "repair_triggered": bool(self.repair_triggered),
            "repair_policy": self.repair_policy,
            "repair_result": dict(self.repair_result),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SelfInconsistencyEvent":
        if not payload:
            return cls(tick=0, action="")
        return cls(
            tick=int(payload.get("tick", 0)),
            action=str(payload.get("action", "")),
            active_commitments=[str(item) for item in payload.get("active_commitments", [])],
            relevant_commitments=[str(item) for item in payload.get("relevant_commitments", [])],
            commitment_compatibility_score=float(payload.get("commitment_compatibility_score", 1.0)),
            self_inconsistency_error=float(payload.get("self_inconsistency_error", 0.0)),
            conflict_type=str(payload.get("conflict_type", "none")),
            severity_level=str(payload.get("severity_level", "none")),
            consistency_classification=str(payload.get("consistency_classification", "aligned")),
            behavioral_classification=str(payload.get("behavioral_classification", "aligned")),
            repair_triggered=bool(payload.get("repair_triggered", False)),
            repair_policy=str(payload.get("repair_policy", "")),
            repair_result=dict(payload.get("repair_result", {})),
            evidence=dict(payload.get("evidence", {})),
        )


@dataclass(slots=True)
class RepairRecord:
    tick: int
    policy: str
    success: bool
    target_action: str = ""
    repaired_action: str = ""
    pre_alignment: float = 0.0
    post_alignment: float = 0.0
    recovery_ticks: int = 0
    bounded_update_applied: bool = False
    social_repair_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "policy": self.policy,
            "success": bool(self.success),
            "target_action": self.target_action,
            "repaired_action": self.repaired_action,
            "pre_alignment": float(self.pre_alignment),
            "post_alignment": float(self.post_alignment),
            "recovery_ticks": int(self.recovery_ticks),
            "bounded_update_applied": bool(self.bounded_update_applied),
            "social_repair_required": bool(self.social_repair_required),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "RepairRecord":
        if not payload:
            return cls(tick=0, policy="", success=False)
        return cls(
            tick=int(payload.get("tick", 0)),
            policy=str(payload.get("policy", "")),
            success=bool(payload.get("success", False)),
            target_action=str(payload.get("target_action", "")),
            repaired_action=str(payload.get("repaired_action", "")),
            pre_alignment=float(payload.get("pre_alignment", 0.0)),
            post_alignment=float(payload.get("post_alignment", 0.0)),
            recovery_ticks=int(payload.get("recovery_ticks", 0)),
            bounded_update_applied=bool(payload.get("bounded_update_applied", False)),
            social_repair_required=bool(payload.get("social_repair_required", False)),
        )


@dataclass(frozen=True, slots=True)
class PersonalitySignal:
    """Personality trait deltas extracted from a single narrative appraisal."""

    openness_delta: float = 0.0
    conscientiousness_delta: float = 0.0
    extraversion_delta: float = 0.0
    agreeableness_delta: float = 0.0
    neuroticism_delta: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "openness_delta": float(self.openness_delta),
            "conscientiousness_delta": float(self.conscientiousness_delta),
            "extraversion_delta": float(self.extraversion_delta),
            "agreeableness_delta": float(self.agreeableness_delta),
            "neuroticism_delta": float(self.neuroticism_delta),
        }


def _clamp_trait(value: float) -> float:
    return max(0.05, min(0.95, value))


@dataclass(slots=True)
class PersonalityProfile:
    """Big Five personality traits derived from accumulated narrative experience.

    Each trait is on [0, 1] with 0.5 as neutral/population mean.
    Traits drift slowly through narrative experience accumulation during sleep.
    """

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    update_count: int = 0
    last_updated_tick: int = 0

    # M2.7: CognitiveStyle extensions
    meaning_construction_tendency: float = 0.5  # VIA Spirituality / Transcendence
    emotional_regulation_style: float = 0.5     # VIA Humor / Reframing

    # Learning rate parameters
    _base_learning_rate: float = 0.27
    _learning_rate_decay: float = 0.02

    @property
    def learning_rate(self) -> float:
        return self._base_learning_rate / (1.0 + self._learning_rate_decay * self.update_count)

    def absorb_signal(self, signal: PersonalitySignal, tick: int) -> dict[str, float]:
        """Apply a personality signal, returning the trait deltas actually applied."""
        lr = self.learning_rate
        deltas: dict[str, float] = {}

        for trait_name, delta_name in (
            ("openness", "openness_delta"),
            ("conscientiousness", "conscientiousness_delta"),
            ("extraversion", "extraversion_delta"),
            ("agreeableness", "agreeableness_delta"),
            ("neuroticism", "neuroticism_delta"),
        ):
            old_value = getattr(self, trait_name)
            signal_value = getattr(signal, delta_name)
            # Target is 0.5 + signal_value (signal in [-0.5, 0.5] maps to [0, 1])
            target = max(0.0, min(1.0, 0.5 + signal_value))
            new_value = _clamp_trait(old_value * (1.0 - lr) + target * lr)
            deltas[trait_name] = new_value - old_value
            setattr(self, trait_name, new_value)

        self.update_count += 1
        self.last_updated_tick = tick
        return deltas

    def deviation_from_neutral(self) -> float:
        """How far this personality is from the default neutral profile."""
        return sum(
            abs(getattr(self, trait) - 0.5)
            for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
        )

    def drive_modulation(self) -> dict[str, float]:
        """Compute drive weight modulations from personality traits.

        Returns a dict of drive_name -> weight_delta.
        Positive values increase drive urgency, negative decrease it.
        """
        o = self.openness - 0.5
        c = self.conscientiousness - 0.5
        e = self.extraversion - 0.5
        a = self.agreeableness - 0.5
        n = self.neuroticism - 0.5

        return {
            "hunger": c * -0.10,  # conscientious agents manage resources better
            "safety": n * 0.25 + a * -0.10,  # neurotic = more safety-seeking
            "exploration": o * 0.20 + e * 0.10,  # open + extraverted = more exploring
            "comfort": c * 0.15 + o * -0.10,  # conscientious = comfort-seeking
            "thermal": n * 0.10,  # neurotic = more sensitive
            "social": e * 0.25 + a * 0.15,  # extraverted + agreeable = social
        }

    def strategic_modulation(self) -> dict[str, float]:
        """Compute strategic prior modulations from personality traits.

        Returns a dict of prior_name -> delta to apply to the strategic layer.
        """
        o = self.openness - 0.5
        c = self.conscientiousness - 0.5
        e = self.extraversion - 0.5
        n = self.neuroticism - 0.5

        return {
            "energy_floor": c * 0.10,  # conscientious = higher energy floor
            "danger_ceiling": n * 0.12 + (-o) * 0.05,  # neurotic = higher danger ceiling
            "novelty_floor": o * -0.10,  # open = lower novelty floor (seeks novelty)
            "shelter_floor": n * 0.08 + c * 0.05,  # neurotic/conscientious = needs shelter
            "temperature_ideal": 0.0,  # no personality effect on temperature ideal
            "social_floor": e * 0.12 + (-n) * 0.05,  # extraverted = higher social floor
        }

    def policy_bias(self, action: str, danger: float) -> float:
        """Compute personality-driven policy bias for an action.

        Returns a bounded bias term to add to identity_bias in policy evaluation.
        """
        o = self.openness - 0.5
        c = self.conscientiousness - 0.5
        e = self.extraversion - 0.5
        a = self.agreeableness - 0.5
        n = self.neuroticism - 0.5

        bias = 0.0
        if action == "scan":
            bias += o * 0.20 + e * 0.10
        elif action == "hide":
            bias += n * 0.20 + (-o) * 0.10 + (-e) * 0.15
        elif action == "rest":
            bias += c * 0.10 + (-e) * 0.10
        elif action == "exploit_shelter":
            bias += n * 0.15 + c * 0.10
        elif action == "seek_contact":
            bias += e * 0.25 + a * 0.20
        elif action == "forage":
            # Under danger, neurotic agents penalize forage more
            bias += (-n) * 0.15 * max(0.0, danger - 0.3)
            bias += (-c) * 0.10 * max(0.0, danger - 0.3)
        elif action == "thermoregulate":
            bias += n * 0.05

        return max(-0.30, min(0.30, bias))

    def to_dict(self) -> dict[str, object]:
        return {
            "openness": float(self.openness),
            "conscientiousness": float(self.conscientiousness),
            "extraversion": float(self.extraversion),
            "agreeableness": float(self.agreeableness),
            "neuroticism": float(self.neuroticism),
            "update_count": int(self.update_count),
            "last_updated_tick": int(self.last_updated_tick),
            "meaning_construction_tendency": float(self.meaning_construction_tendency),
            "emotional_regulation_style": float(self.emotional_regulation_style),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PersonalityProfile":
        if not payload:
            return cls()
        return cls(
            openness=float(payload.get("openness", 0.5)),
            conscientiousness=float(payload.get("conscientiousness", 0.5)),
            extraversion=float(payload.get("extraversion", 0.5)),
            agreeableness=float(payload.get("agreeableness", 0.5)),
            neuroticism=float(payload.get("neuroticism", 0.5)),
            update_count=int(payload.get("update_count", 0)),
            last_updated_tick=int(payload.get("last_updated_tick", 0)),
            meaning_construction_tendency=float(payload.get("meaning_construction_tendency", 0.5)),
            emotional_regulation_style=float(payload.get("emotional_regulation_style", 0.5)),
        )


@dataclass(slots=True)
class NarrativePriors:
    trust_prior: float = 0.0
    controllability_prior: float = 0.0
    trauma_bias: float = 0.0
    contamination_sensitivity: float = 0.0
    meaning_stability: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "trust_prior": float(self.trust_prior),
            "controllability_prior": float(self.controllability_prior),
            "trauma_bias": float(self.trauma_bias),
            "contamination_sensitivity": float(self.contamination_sensitivity),
            "meaning_stability": float(self.meaning_stability),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NarrativePriors":
        if not payload:
            return cls()
        return cls(
            trust_prior=float(payload.get("trust_prior", 0.0)),
            controllability_prior=float(payload.get("controllability_prior", 0.0)),
            trauma_bias=float(payload.get("trauma_bias", 0.0)),
            contamination_sensitivity=float(payload.get("contamination_sensitivity", 0.0)),
            meaning_stability=float(payload.get("meaning_stability", 0.0)),
        )


@dataclass(slots=True)
class DriftBudget:
    personality_window: float = 0.45
    narrative_window: float = 0.55
    policy_window: float = 0.45
    action_dominance_limit: float = 0.82
    restart_tolerance: float = 0.18

    def to_dict(self) -> dict[str, float]:
        return {
            "personality_window": self.personality_window,
            "narrative_window": self.narrative_window,
            "policy_window": self.policy_window,
            "action_dominance_limit": self.action_dominance_limit,
            "restart_tolerance": self.restart_tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "DriftBudget":
        if not payload:
            return cls()
        return cls(
            personality_window=float(payload.get("personality_window", 0.45)),
            narrative_window=float(payload.get("narrative_window", 0.55)),
            policy_window=float(payload.get("policy_window", 0.45)),
            action_dominance_limit=float(payload.get("action_dominance_limit", 0.82)),
            restart_tolerance=float(payload.get("restart_tolerance", 0.18)),
        )


@dataclass(slots=True)
class ContinuityAudit:
    continuity_score: float = 1.0
    dominant_action: str = ""
    dominant_action_ratio: float = 0.0
    action_distribution: dict[str, float] = field(default_factory=dict)
    personality_drift: float = 0.0
    narrative_drift: float = 0.0
    policy_drift: float = 0.0
    restart_divergence: float = 0.0
    chapter_shift_excused: bool = False
    protected_anchor_ids: list[str] = field(default_factory=list)
    rehearsal_queue: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    personality_snapshot: dict[str, float] = field(default_factory=dict)
    policy_snapshot: dict[str, float] = field(default_factory=dict)
    commitment_snapshot: list[str] = field(default_factory=list)
    updated_tick: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "continuity_score": self.continuity_score,
            "dominant_action": self.dominant_action,
            "dominant_action_ratio": self.dominant_action_ratio,
            "action_distribution": dict(self.action_distribution),
            "personality_drift": self.personality_drift,
            "narrative_drift": self.narrative_drift,
            "policy_drift": self.policy_drift,
            "restart_divergence": self.restart_divergence,
            "chapter_shift_excused": self.chapter_shift_excused,
            "protected_anchor_ids": list(self.protected_anchor_ids),
            "rehearsal_queue": list(self.rehearsal_queue),
            "interventions": list(self.interventions),
            "personality_snapshot": dict(self.personality_snapshot),
            "policy_snapshot": dict(self.policy_snapshot),
            "commitment_snapshot": list(self.commitment_snapshot),
            "updated_tick": self.updated_tick,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ContinuityAudit":
        if not payload:
            return cls()
        return cls(
            continuity_score=float(payload.get("continuity_score", 1.0)),
            dominant_action=str(payload.get("dominant_action", "")),
            dominant_action_ratio=float(payload.get("dominant_action_ratio", 0.0)),
            action_distribution={
                str(key): float(value)
                for key, value in dict(payload.get("action_distribution", {})).items()
            },
            personality_drift=float(payload.get("personality_drift", 0.0)),
            narrative_drift=float(payload.get("narrative_drift", 0.0)),
            policy_drift=float(payload.get("policy_drift", 0.0)),
            restart_divergence=float(payload.get("restart_divergence", 0.0)),
            chapter_shift_excused=bool(payload.get("chapter_shift_excused", False)),
            protected_anchor_ids=[
                str(item) for item in payload.get("protected_anchor_ids", [])
            ],
            rehearsal_queue=[str(item) for item in payload.get("rehearsal_queue", [])],
            interventions=[str(item) for item in payload.get("interventions", [])],
            personality_snapshot={
                str(key): float(value)
                for key, value in dict(payload.get("personality_snapshot", {})).items()
            },
            policy_snapshot={
                str(key): float(value)
                for key, value in dict(payload.get("policy_snapshot", {})).items()
            },
            commitment_snapshot=[str(item) for item in payload.get("commitment_snapshot", [])],
            updated_tick=int(payload.get("updated_tick", 0)),
        )

@dataclass(slots=True)
class SelfModel:
    """Self model for separating failures and persisting agent continuity."""

    body_schema: BodySchema
    capability_model: CapabilityModel
    resource_state: ResourceState
    threat_model: ThreatModel
    threat_profile: ThreatProfile = field(default_factory=ThreatProfile)
    error_classifier: ErrorClassifier = field(default_factory=ErrorClassifier)
    preferred_policies: PreferredPolicies | None = None
    identity_narrative: IdentityNarrative | None = None
    narrative_priors: NarrativePriors = field(default_factory=NarrativePriors)
    personality_profile: PersonalityProfile = field(default_factory=PersonalityProfile)
    drift_budget: DriftBudget = field(default_factory=DriftBudget)
    continuity_audit: ContinuityAudit = field(default_factory=ContinuityAudit)
    belief_calibration: dict[str, dict[str, object]] = field(default_factory=dict)
    commitments_enabled: bool = True
    repair_enabled: bool = False
    self_inconsistency_events: list[SelfInconsistencyEvent] = field(default_factory=list)
    repair_history: list[RepairRecord] = field(default_factory=list)
    log_sink: Callable[[str], None] | None = None
    last_result: ClassificationResult | None = field(init=False, default=None)

    def classify_event(self, event: object) -> str:
        return self.inspect_event(event).classification

    def inspect_event(self, event: object) -> ClassificationResult:
        event_name = _event_name(event)
        classification = self.error_classifier.classify(event)
        resource_state = (
            dict(event.resource_state)
            if isinstance(event, RuntimeFailureEvent) and event.resource_state is not None
            else self.resource_state.snapshot()
        )
        threat_resource_state = ResourceState(
            tokens_remaining=int(resource_state.get("tokens_remaining", self.resource_state.tokens_remaining)),
            cpu_budget=float(resource_state.get("cpu_budget", self.resource_state.cpu_budget)),
            memory_free=float(resource_state.get("memory_free", self.resource_state.memory_free)),
        )
        result = ClassificationResult(
            event=event_name,
            classification=classification,
            attribution=self.error_classifier.attribution(classification),
            resource_state=resource_state,
            surprise_source=self.error_classifier.surprise_source(classification),
            detected_threats=self.threat_model.detect(
                event_name,
                threat_resource_state,
                self.body_schema,
            ),
            evidence=self.error_classifier.evidence(
                event,
                resource_state=resource_state,
                body_schema=self.body_schema,
            ),
        )
        self.last_result = result
        if self.log_sink is not None:
            self.log_sink(result.to_log_string())
        return result

    def predict_resource_state(self) -> dict[str, bool]:
        return self.resource_state.predict(self.body_schema, self.threat_model)

    def detect_threats(self, event: object | None = None) -> tuple[str, ...]:
        return self.threat_model.detect(
            "Heartbeat" if event is None else event,
            self.resource_state,
            self.body_schema,
        )

    def assess_action_commitments(
        self,
        *,
        action: str,
        projected_state: Mapping[str, object],
        current_tick: int = 0,
    ) -> dict[str, object]:
        narrative = self.identity_narrative
        if narrative is None or not self.commitments_enabled:
            return {
                "bias": 0.0,
                "focus": [],
                "violations": [],
                "active_commitments": [],
                "relevant_commitments": [],
                "compatibility_score": 0.5,
                "self_inconsistency_error": 0.0,
            "conflict_type": "none",
            "severity_level": "none",
            "consistency_classification": "aligned",
            "behavioral_classification": "aligned",
            "repair_triggered": False,
                "repair_policy": "",
                "repair_result": {},
                "tension": 0.0,
            }

        danger = float(projected_state.get("danger", 0.0))
        novelty = float(projected_state.get("novelty", 0.0))
        shelter = float(projected_state.get("shelter", 0.0))
        stress = float(projected_state.get("stress", projected_state.get("predicted_stress", 0.0)))
        social = float(projected_state.get("social", 0.0))
        temptation_gain = float(projected_state.get("temptation_gain", projected_state.get("food", 0.0)))
        adaptation_pressure = float(projected_state.get("adaptation_pressure", novelty))
        active_commitments: list[str] = []
        relevant_commitments: list[str] = []
        focus: list[str] = []
        violations: list[str] = []
        bias = 0.0
        weighted_score = 0.0
        total_weight = 0.0

        for commitment in narrative.commitments:
            if not commitment.active:
                continue
            active_commitments.append(commitment.commitment_id)
            relevance = 0.15
            if action in commitment.target_actions or action in commitment.discouraged_actions:
                relevance = 1.0
            elif commitment.commitment_type == "value_guardrail":
                relevance = max(relevance, danger, 1.0 - shelter)
            elif commitment.commitment_type == "behavioral_style":
                relevance = max(relevance, novelty * max(0.2, 1.0 - danger))
            elif commitment.commitment_type == "capability":
                relevance = max(relevance, adaptation_pressure * 0.8)
            elif commitment.commitment_type == "social":
                relevance = max(relevance, social)
            if relevance < 0.25:
                continue
            relevant_commitments.append(commitment.commitment_id)
            weight = max(0.05, min(1.0, commitment.confidence)) * max(0.1, min(1.0, commitment.priority)) * relevance
            total_weight += weight
            if action in commitment.target_actions:
                local_score = 1.0
                focus.append(commitment.commitment_id)
                bias += 0.65 * weight
            elif action in commitment.discouraged_actions:
                local_score = 0.0
                violations.append(commitment.commitment_id)
                bias -= 0.85 * weight
            else:
                local_score = 0.6 if relevance <= 0.45 else 0.45
            weighted_score += local_score * weight

        compatibility_score = weighted_score / total_weight if total_weight > 0 else 0.5
        self_inconsistency_error = max(0.0, min(1.0, 1.0 - compatibility_score))
        conflict_type = "none"
        if violations:
            if social >= 0.6 and danger >= 0.55:
                conflict_type = "social_contradiction"
            elif stress >= 0.7:
                conflict_type = "stress_drift"
            elif adaptation_pressure >= 0.7 and novelty >= 0.55:
                conflict_type = "adaptation_vs_betrayal"
            elif temptation_gain >= 0.6:
                conflict_type = "temptation_conflict"
            else:
                conflict_type = "temporary_deviation"
        severity_value = self_inconsistency_error + (0.15 if stress >= 0.7 else 0.0) + (0.10 if danger >= 0.75 else 0.0)
        behavioral_classification = "aligned"
        if not violations:
            severity_level = "none"
            consistency_classification = "aligned" if relevant_commitments else "irrelevant"
            if adaptation_pressure >= 0.50 and novelty >= 0.55:
                behavioral_classification = "healthy_adaptation" if action == "scan" else "over_rigidity"
        elif adaptation_pressure >= 0.75 and novelty >= 0.55 and action == "scan" and compatibility_score >= 0.45:
            severity_level = "low"
            consistency_classification = "reasonable_adaptation"
            behavioral_classification = "healthy_adaptation"
        elif adaptation_pressure >= 0.75 and novelty >= 0.55 and action in {"rest", "hide", "exploit_shelter"}:
            behavioral_classification = "over_rigidity"
            if severity_value >= 0.72:
                severity_level = "high"
                consistency_classification = "self_conflict"
            else:
                severity_level = "medium"
                consistency_classification = "temporary_deviation"
        elif adaptation_pressure >= 0.75 and novelty >= 0.55:
            behavioral_classification = "narrative_rationalization"
            if severity_value >= 0.72:
                severity_level = "high"
                consistency_classification = "self_conflict"
            else:
                severity_level = "medium"
                consistency_classification = "temporary_deviation"
        elif severity_value >= 0.72:
            severity_level = "high"
            consistency_classification = "self_conflict"
            behavioral_classification = "self_conflict"
        elif severity_value >= 0.30:
            severity_level = "medium"
            consistency_classification = "temporary_deviation"
            behavioral_classification = "temporary_deviation"
        else:
            severity_level = "low"
            consistency_classification = "temporary_deviation"
            behavioral_classification = "temporary_deviation"
        repair_triggered = bool(
            self.repair_enabled
            and violations
            and relevant_commitments
            and severity_level in {"medium", "high"}
            and consistency_classification != "reasonable_adaptation"
        )
        repair_policy = ""
        if repair_triggered:
            if conflict_type == "stress_drift":
                repair_policy = "reflective_pause+policy_rebias"
            elif conflict_type == "social_contradiction":
                repair_policy = "social_repair+policy_rebias"
            elif conflict_type == "adaptation_vs_betrayal":
                repair_policy = "bounded_commitment_update+narrative_reconciliation"
            else:
                repair_policy = "metacognitive_review+policy_rebias"
        return {
            "bias": max(-0.9, min(0.9, bias)),
            "focus": focus,
            "violations": violations,
            "active_commitments": active_commitments,
            "relevant_commitments": relevant_commitments,
            "compatibility_score": max(0.0, min(1.0, compatibility_score)),
            "self_inconsistency_error": self_inconsistency_error,
            "conflict_type": conflict_type,
            "severity_level": severity_level,
            "consistency_classification": consistency_classification,
            "behavioral_classification": behavioral_classification,
            "repair_triggered": repair_triggered,
            "repair_policy": repair_policy,
            "repair_result": {},
            "tension": max(0.0, min(1.5, self_inconsistency_error + (0.25 if violations else 0.0))),
            "evidence": {
                "danger": danger,
                "stress": stress,
                "social": social,
                "temptation_gain": temptation_gain,
                "adaptation_pressure": adaptation_pressure,
            },
            "tick": current_tick,
        }

    def register_self_inconsistency(
        self,
        *,
        tick: int,
        action: str,
        assessment: Mapping[str, object],
    ) -> SelfInconsistencyEvent:
        event = SelfInconsistencyEvent(
            tick=tick,
            action=action,
            active_commitments=[str(item) for item in assessment.get("active_commitments", [])],
            relevant_commitments=[str(item) for item in assessment.get("relevant_commitments", [])],
            commitment_compatibility_score=float(assessment.get("compatibility_score", 1.0)),
            self_inconsistency_error=float(assessment.get("self_inconsistency_error", 0.0)),
            conflict_type=str(assessment.get("conflict_type", "none")),
            severity_level=str(assessment.get("severity_level", "none")),
            consistency_classification=str(assessment.get("consistency_classification", "aligned")),
            behavioral_classification=str(assessment.get("behavioral_classification", "aligned")),
            repair_triggered=bool(assessment.get("repair_triggered", False)),
            repair_policy=str(assessment.get("repair_policy", "")),
            repair_result=dict(assessment.get("repair_result", {})),
            evidence=dict(assessment.get("evidence", {})),
        )
        self.self_inconsistency_events.append(event)
        self.self_inconsistency_events = self.self_inconsistency_events[-256:]
        return event

    def record_repair_outcome(
        self,
        *,
        tick: int,
        policy: str,
        success: bool,
        target_action: str,
        repaired_action: str,
        pre_alignment: float,
        post_alignment: float,
        recovery_ticks: int = 1,
        bounded_update_applied: bool = False,
        social_repair_required: bool = False,
    ) -> RepairRecord:
        record = RepairRecord(
            tick=tick,
            policy=policy,
            success=success,
            target_action=target_action,
            repaired_action=repaired_action,
            pre_alignment=pre_alignment,
            post_alignment=post_alignment,
            recovery_ticks=recovery_ticks,
            bounded_update_applied=bounded_update_applied,
            social_repair_required=social_repair_required,
        )
        self.repair_history.append(record)
        self.repair_history = self.repair_history[-256:]
        return record

    def apply_repair_policy(
        self,
        *,
        tick: int,
        policy: str,
        assessment: Mapping[str, object],
    ) -> dict[str, object]:
        applied_updates: list[dict[str, object]] = []
        relevant_commitments = [str(item) for item in assessment.get("relevant_commitments", [])]
        violated_commitments = [str(item) for item in assessment.get("violations", [])]

        def _update(commitment_id: str, confidence_delta: float, priority_delta: float) -> None:
            if self.bounded_commitment_update(
                commitment_id=commitment_id,
                confidence_delta=confidence_delta,
                priority_delta=priority_delta,
                tick=tick,
            ):
                applied_updates.append(
                    {
                        "commitment_id": commitment_id,
                        "confidence_delta": max(-0.12, min(0.12, confidence_delta)),
                        "priority_delta": max(-0.08, min(0.08, priority_delta)),
                    }
                )

        if "bounded_commitment_update" in policy:
            _update("adaptive_exploration", 0.12, 0.08)
        if "social_repair" in policy:
            _update("core_social_repair", 0.12, 0.08)
        if "policy_rebias" in policy or "reflective_pause" in policy:
            targets = violated_commitments or relevant_commitments[:2]
            for commitment_id in targets[:2]:
                _update(commitment_id, 0.12, 0.08)

        return {
            "policy": policy,
            "applied_updates": applied_updates,
            "updated_commitments": [str(item["commitment_id"]) for item in applied_updates],
        }

    def apply_unresolved_conflict_drift(
        self,
        *,
        tick: int,
        assessment: Mapping[str, object],
    ) -> dict[str, object]:
        applied_updates: list[dict[str, object]] = []
        targets = [str(item) for item in assessment.get("violations", [])] or [
            str(item) for item in assessment.get("relevant_commitments", [])
        ][:2]
        for commitment_id in targets[:2]:
            changed = False
            for _ in range(2):
                changed = bool(
                    self.bounded_commitment_update(
                        commitment_id=commitment_id,
                        confidence_delta=-0.12,
                        priority_delta=-0.08,
                        tick=tick,
                    )
                ) or changed
            if changed:
                applied_updates.append(
                    {
                        "commitment_id": commitment_id,
                        "confidence_delta": -0.24,
                        "priority_delta": -0.16,
                    }
                )
        return {
            "applied_updates": applied_updates,
            "updated_commitments": [str(item["commitment_id"]) for item in applied_updates],
        }

    def bounded_commitment_update(
        self,
        *,
        commitment_id: str,
        confidence_delta: float,
        priority_delta: float = 0.0,
        tick: int,
    ) -> bool:
        narrative = self.identity_narrative
        if narrative is None:
            return False
        for commitment in narrative.commitments:
            if commitment.commitment_id != commitment_id:
                continue
            commitment.confidence = max(0.2, min(1.0, commitment.confidence + max(-0.12, min(0.12, confidence_delta))))
            commitment.priority = max(0.2, min(1.0, commitment.priority + max(-0.08, min(0.08, priority_delta))))
            commitment.last_reaffirmed_tick = tick
            return True
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "body_schema": self.body_schema.to_dict(),
            "capability_model": self.capability_model.to_dict(),
            "resource_state": self.resource_state.to_dict(),
            "threat_model": self.threat_model.to_dict(),
            "threat_profile": self.threat_profile.to_dict(),
            "error_classifier": self.error_classifier.to_dict(),
            "preferred_policies": (
                self.preferred_policies.to_dict() if self.preferred_policies else None
            ),
            "identity_narrative": (
                self.identity_narrative.to_dict() if self.identity_narrative else None
            ),
            "narrative_priors": self.narrative_priors.to_dict(),
            "personality_profile": self.personality_profile.to_dict(),
            "drift_budget": self.drift_budget.to_dict(),
            "continuity_audit": self.continuity_audit.to_dict(),
            "belief_calibration": {
                str(key): dict(value) for key, value in self.belief_calibration.items()
            },
            "commitments_enabled": self.commitments_enabled,
            "repair_enabled": self.repair_enabled,
            "self_inconsistency_events": [
                event.to_dict() for event in self.self_inconsistency_events[-128:]
            ],
            "repair_history": [record.to_dict() for record in self.repair_history[-128:]],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object] | None,
        *,
        log_sink: Callable[[str], None] | None = None,
    ) -> SelfModel:
        default = build_default_self_model(log_sink=log_sink)
        if not payload:
            return default
        return cls(
            body_schema=BodySchema.from_dict(payload.get("body_schema")),
            capability_model=CapabilityModel.from_dict(payload.get("capability_model")),
            resource_state=ResourceState.from_dict(payload.get("resource_state")),
            threat_model=ThreatModel.from_dict(payload.get("threat_model")),
            threat_profile=ThreatProfile.from_dict(payload.get("threat_profile")),
            error_classifier=ErrorClassifier.from_dict(payload.get("error_classifier")),
            preferred_policies=PreferredPolicies.from_dict(payload.get("preferred_policies")),
            identity_narrative=IdentityNarrative.from_dict(payload.get("identity_narrative")),
            narrative_priors=NarrativePriors.from_dict(payload.get("narrative_priors")),
            personality_profile=PersonalityProfile.from_dict(payload.get("personality_profile")),
            drift_budget=DriftBudget.from_dict(payload.get("drift_budget")),
            continuity_audit=ContinuityAudit.from_dict(payload.get("continuity_audit")),
            belief_calibration={
                str(key): dict(value)
                for key, value in dict(payload.get("belief_calibration", {})).items()
                if isinstance(value, Mapping)
            },
            commitments_enabled=bool(payload.get("commitments_enabled", True)),
            repair_enabled=bool(payload.get("repair_enabled", False)),
            self_inconsistency_events=[
                SelfInconsistencyEvent.from_dict(item)
                for item in payload.get("self_inconsistency_events", [])
                if isinstance(item, Mapping)
            ],
            repair_history=[
                RepairRecord.from_dict(item)
                for item in payload.get("repair_history", [])
                if isinstance(item, Mapping)
            ],
            log_sink=log_sink,
        )

    def update_continuity_audit(
        self,
        *,
        episodic_memory: list[dict[str, object]],
        archived_memory: list[dict[str, object]] | None = None,
        action_history: list[str],
        rehearsal_queue: list[str] | None = None,
        current_tick: int,
    ) -> ContinuityAudit:
        archived = archived_memory or []
        previous = self.continuity_audit
        current_personality = self._personality_anchor_snapshot()
        current_policy = (
            dict(self.preferred_policies.action_distribution)
            if self.preferred_policies is not None
            else {}
        )
        current_commitments = (
            [commitment.commitment_id for commitment in self.identity_narrative.commitments]
            if self.identity_narrative is not None
            else []
        )
        action_distribution = _distribution(action_history[-48:])
        dominant_action, dominant_ratio = _dominant_action(action_distribution)
        protected_anchor_ids = [
            str(payload.get("episode_id", ""))
            for payload in sorted(
                [
                    payload
                    for payload in [*episodic_memory, *archived]
                    if bool(payload.get("identity_critical", False))
                ],
                key=lambda payload: (
                    int(payload.get("last_seen_cycle", payload.get("cycle", 0))),
                    str(payload.get("episode_id", "")),
                ),
                reverse=True,
            )[:12]
            if payload.get("episode_id")
        ]

        personality_drift = _mean_abs_delta(
            previous.personality_snapshot,
            current_personality,
        )
        policy_drift = _distribution_delta(previous.policy_snapshot, current_policy)
        narrative_drift = _set_divergence(previous.commitment_snapshot, current_commitments)
        chapter_shift_excused = bool(
            self.identity_narrative is not None
            and self.identity_narrative.chapter_transition_evidence
            and narrative_drift > self.drift_budget.narrative_window
        )

        interventions: list[str] = []
        if dominant_ratio >= self.drift_budget.action_dominance_limit and dominant_action:
            interventions.append("action_collapse_guard")
            if self.preferred_policies is not None:
                avoidances = list(self.preferred_policies.learned_avoidances)
                if dominant_action not in avoidances:
                    avoidances.append(dominant_action)
                    self.preferred_policies.learned_avoidances = avoidances
        if (
            previous.personality_snapshot
            and personality_drift > self.drift_budget.personality_window
        ):
            interventions.append("personality_drift_guard")
            self._stabilize_personality(previous.personality_snapshot)
            current_personality = self._personality_anchor_snapshot()
            personality_drift = _mean_abs_delta(
                previous.personality_snapshot,
                current_personality,
            )
        if previous.policy_snapshot and policy_drift > self.drift_budget.policy_window:
            interventions.append("policy_prior_guard")
        if (
            previous.commitment_snapshot
            and narrative_drift > self.drift_budget.narrative_window
            and not chapter_shift_excused
        ):
            interventions.append("narrative_anchor_guard")

        continuity_penalty = (
            (dominant_ratio * 0.40)
            + (personality_drift * 0.20)
            + (policy_drift * 0.20)
            + (0.0 if chapter_shift_excused else narrative_drift * 0.20)
        )
        self.continuity_audit = ContinuityAudit(
            continuity_score=max(0.0, min(1.0, 1.0 - continuity_penalty)),
            dominant_action=dominant_action,
            dominant_action_ratio=dominant_ratio,
            action_distribution=action_distribution,
            personality_drift=personality_drift,
            narrative_drift=narrative_drift,
            policy_drift=policy_drift,
            restart_divergence=previous.restart_divergence,
            chapter_shift_excused=chapter_shift_excused,
            protected_anchor_ids=protected_anchor_ids,
            rehearsal_queue=list(rehearsal_queue or []),
            interventions=interventions,
            personality_snapshot=current_personality,
            policy_snapshot=current_policy,
            commitment_snapshot=current_commitments,
            updated_tick=current_tick,
        )
        return self.continuity_audit

    def record_restart_consistency(
        self,
        reference: Mapping[str, object] | None,
        *,
        current_tick: int,
    ) -> float:
        if not reference:
            self.continuity_audit.restart_divergence = 0.0
            self.continuity_audit.updated_tick = current_tick
            return 0.0
        reference_personality = {
            str(key): float(value)
            for key, value in dict(reference.get("personality_snapshot", {})).items()
        }
        reference_policy = {
            str(key): float(value)
            for key, value in dict(reference.get("policy_snapshot", {})).items()
        }
        reference_commitments = [
            str(item) for item in reference.get("commitment_snapshot", [])
        ]
        divergence = (
            _mean_abs_delta(reference_personality, self.continuity_audit.personality_snapshot)
            + _distribution_delta(reference_policy, self.continuity_audit.policy_snapshot)
            + _set_divergence(reference_commitments, self.continuity_audit.commitment_snapshot)
        ) / 3.0
        self.continuity_audit.restart_divergence = divergence
        self.continuity_audit.updated_tick = current_tick
        return divergence

    def build_restart_anchors(
        self,
        *,
        maintenance_agenda: Mapping[str, object] | None,
        memory_anchors: list[dict[str, object]],
        recent_actions: list[str],
    ) -> dict[str, object]:
        preferred = self.preferred_policies or PreferredPolicies()
        commitments = self.identity_narrative.commitments if self.identity_narrative is not None else []
        commitment_action_priors = [
            action
            for commitment in commitments
            for action in commitment.target_actions
            if action
        ]
        return {
            "preferred_policy_distribution": dict(preferred.action_distribution),
            "dominant_strategy": preferred.dominant_strategy,
            "learned_avoidances": list(preferred.learned_avoidances),
            "learned_preferences": list(preferred.learned_preferences),
            "risk_profile": preferred.risk_profile,
            "commitment_action_priors": list(dict.fromkeys(commitment_action_priors))[:8],
            "maintenance_agenda": dict(maintenance_agenda or {}),
            "continuity_audit": self.continuity_audit.to_dict(),
            "memory_anchors": [dict(item) for item in memory_anchors],
            "recent_actions": [str(action) for action in recent_actions[-32:]],
        }

    def apply_restart_anchors(self, anchors: Mapping[str, object] | None) -> None:
        if not anchors:
            return
        preferred = self.preferred_policies or PreferredPolicies()
        distribution = anchors.get("preferred_policy_distribution")
        if isinstance(distribution, Mapping) and distribution:
            preferred.action_distribution = {
                str(key): float(value) for key, value in distribution.items()
            }
        dominant_strategy = anchors.get("dominant_strategy")
        if dominant_strategy:
            preferred.dominant_strategy = str(dominant_strategy)
        risk_profile = anchors.get("risk_profile")
        if risk_profile:
            preferred.risk_profile = str(risk_profile)
        preferred.learned_avoidances = [
            str(item) for item in anchors.get("learned_avoidances", [])
        ][:8]
        preferred.learned_preferences = [
            str(item) for item in anchors.get("learned_preferences", [])
        ][:8]
        self.preferred_policies = preferred
        continuity_payload = anchors.get("continuity_audit")
        if isinstance(continuity_payload, Mapping):
            continuity = ContinuityAudit.from_dict(continuity_payload)
            continuity.restart_divergence = min(
                continuity.restart_divergence,
                self.continuity_audit.restart_divergence,
            )
            self.continuity_audit = continuity

    def _stabilize_personality(self, reference: Mapping[str, float]) -> None:
        if not reference:
            return
        for trait_name in (
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ):
            current = float(getattr(self.personality_profile, trait_name))
            anchor = float(reference.get(trait_name, current))
            setattr(
                self.personality_profile,
                trait_name,
                max(0.0, min(1.0, (current * 0.65) + (anchor * 0.35))),
            )

    def _personality_anchor_snapshot(self) -> dict[str, float]:
        profile = self.personality_profile
        return {
            "openness": float(profile.openness),
            "conscientiousness": float(profile.conscientiousness),
            "extraversion": float(profile.extraversion),
            "agreeableness": float(profile.agreeableness),
            "neuroticism": float(profile.neuroticism),
            "meaning_construction_tendency": float(profile.meaning_construction_tendency),
            "emotional_regulation_style": float(profile.emotional_regulation_style),
        }

    def update_preferred_policies(
        self,
        agent_history: list[dict[str, object]],
        *,
        counterfactual_insights: list[object] | None = None,
        drive_history: list[dict[str, float]] | None = None,
        current_tick: int,
    ) -> PreferredPolicies:
        if not agent_history:
            existing = self.preferred_policies or PreferredPolicies()
            existing.last_updated_tick = current_tick
            self.preferred_policies = existing
            return existing

        dominant_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        risks: list[float] = []
        for entry in agent_history:
            dominant = str(entry.get("dominant_component", "expected_free_energy"))
            action = str(entry.get("action", ""))
            dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
            try:
                risks.append(float(entry.get("risk", 0.0)))
            except (TypeError, ValueError):
                continue

        existing = self.preferred_policies
        if existing is not None and existing.dominant_strategy:
            dominant_counts[existing.dominant_strategy] = dominant_counts.get(
                existing.dominant_strategy,
                0,
            ) + max(2, len(agent_history) // 4)
            for action, frequency in existing.action_distribution.items():
                action_counts[action] = action_counts.get(action, 0) + max(
                    1,
                    int(round(frequency * max(1, len(agent_history)) * 0.25)),
                )

        total_actions = sum(action_counts.values()) or 1
        action_distribution = {
            action: count / total_actions for action, count in sorted(action_counts.items())
        }
        average_risk = mean(risks) if risks else 0.0
        if average_risk <= 1.0:
            risk_profile = "risk_averse"
        elif average_risk >= 2.5:
            risk_profile = "risk_seeking"
        else:
            risk_profile = "risk_neutral"

        learned_avoidances: list[str] = []
        learned_preferences: list[str] = []
        for insight in counterfactual_insights or []:
            absorbed = bool(getattr(insight, "absorbed", False))
            if not absorbed:
                continue
            original_action = str(getattr(insight, "original_action", ""))
            counterfactual_action = str(getattr(insight, "counterfactual_action", ""))
            if original_action and original_action not in learned_avoidances:
                learned_avoidances.append(original_action)
            if counterfactual_action and counterfactual_action not in learned_preferences:
                learned_preferences.append(counterfactual_action)

        if drive_history:
            averaged_drives: dict[str, float] = {}
            for snapshot in drive_history:
                for key, value in snapshot.items():
                    averaged_drives.setdefault(key, 0.0)
                    averaged_drives[key] += float(value)
            drive_count = len(drive_history)
            for key, value in sorted(averaged_drives.items()):
                if drive_count and (value / drive_count) >= 0.75:
                    learned_preferences.append(f"stabilize_{key}")

        learned_avoidances = learned_avoidances[:5]
        learned_preferences = list(dict.fromkeys(learned_preferences))[:5]
        dominant_strategy = max(
            dominant_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        if existing is not None and existing.dominant_strategy:
            existing_count = dominant_counts.get(existing.dominant_strategy, 0)
            new_count = dominant_counts.get(dominant_strategy, 0)
            if (
                current_tick - existing.last_updated_tick <= 150
                and dominant_strategy != existing.dominant_strategy
                and new_count < max(existing_count + 5, int(existing_count * 1.5))
            ):
                dominant_strategy = existing.dominant_strategy
        policies = PreferredPolicies(
            dominant_strategy=dominant_strategy,
            action_distribution=action_distribution,
            risk_profile=risk_profile,
            learned_avoidances=learned_avoidances,
            learned_preferences=learned_preferences,
            last_updated_tick=current_tick,
        )
        self.preferred_policies = policies
        return policies

    def update_identity_narrative(
        self,
        *,
        episodic_memory: list[dict[str, object]],
        preference_labels: Mapping[str, float],
        current_tick: int,
        decision_history: list[dict[str, object]] | None = None,
        sleep_metrics: Mapping[str, object] | None = None,
        conflict_history: list[object] | None = None,
        weight_adjustments: list[object] | None = None,
        chapter_signal: str | None = None,
    ) -> IdentityNarrative:
        narrative = self.generate_identity_narrative(
            episodic_memory=episodic_memory,
            preference_labels=preference_labels,
            current_tick=current_tick,
            decision_history=decision_history or [],
            sleep_metrics=sleep_metrics or {},
            conflict_history=conflict_history or [],
            weight_adjustments=weight_adjustments or [],
            chapter_signal=chapter_signal,
        )
        self.identity_narrative = narrative
        return narrative

    def evaluate_narrative_contradictions(
        self,
        *,
        episodic_memory: list[dict[str, object]],
        decision_history: list[dict[str, object]],
        current_tick: int,
        sleep_metrics: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        narrative = self.identity_narrative or IdentityNarrative()
        policies = self.preferred_policies or PreferredPolicies(last_updated_tick=current_tick)
        source_sleep_session_id = None
        if sleep_metrics is not None:
            raw_sleep_session = sleep_metrics.get("sleep_cycle_id")
            if isinstance(raw_sleep_session, (int, float)):
                source_sleep_session_id = int(raw_sleep_session)
        claims = self._build_narrative_claims(
            narrative=narrative,
            episodes=episodic_memory,
            decisions=decision_history,
            policies=policies,
            current_tick=current_tick,
            source_sleep_session_id=source_sleep_session_id,
        )
        return {
            "claims": [claim.to_dict() for claim in claims],
            "summary": self._summarize_claim_consistency(claims),
        }
    def generate_identity_narrative(
        self,
        *,
        episodic_memory: list[dict[str, object]],
        preference_labels: Mapping[str, float],
        current_tick: int,
        decision_history: list[dict[str, object]],
        sleep_metrics: Mapping[str, object],
        conflict_history: list[object],
        weight_adjustments: list[object],
        chapter_signal: str | None,
    ) -> IdentityNarrative:
        policies = self.preferred_policies or PreferredPolicies(last_updated_tick=current_tick)
        previous = self.identity_narrative or IdentityNarrative()
        previous_version = previous.version + 1 if self.identity_narrative else 0
        recent_episodes = [
            payload
            for payload in episodic_memory
            if int(payload.get("timestamp", payload.get("cycle", 0))) > previous.last_updated_tick
        ]
        recent_decisions = [
            payload
            for payload in decision_history
            if int(payload.get("tick", 0)) > previous.last_updated_tick
        ]
        recent_adjustments = [
            item
            for item in weight_adjustments
            if int(getattr(item, "tick", 0)) > previous.last_updated_tick
        ]

        ticks = [
            int(payload.get("timestamp", payload.get("cycle", 0)))
            for payload in recent_episodes
        ] + [int(payload.get("tick", 0)) for payload in recent_decisions]
        chapter_start = min(ticks) if ticks else current_tick
        chapter_end = max(ticks) if ticks else current_tick
        recent_state_summary = self._summarize_chapter_state(
            decisions=recent_decisions,
            episodes=recent_episodes,
            policies=policies,
        )
        recent_key_events = self._extract_key_events(recent_episodes)
        recent_theme = self._infer_dominant_theme(
            state_summary=recent_state_summary,
            episodes=recent_episodes,
            chapter_signal=chapter_signal,
        )

        chapters = [NarrativeChapter.from_dict(chapter.to_dict()) for chapter in previous.chapters]
        current_chapter = (
            NarrativeChapter.from_dict(previous.current_chapter.to_dict())
            if previous.current_chapter is not None
            else None
        )
        next_chapter_id = max(
            [chapter.chapter_id for chapter in chapters]
            + ([current_chapter.chapter_id] if current_chapter is not None else [])
            + [0]
        ) + 1

        if current_chapter is None:
            current_chapter = NarrativeChapter(
                chapter_id=next_chapter_id,
                tick_range=(chapter_start, chapter_end),
                dominant_theme=recent_theme,
                key_events=recent_key_events,
                state_summary=recent_state_summary,
            )
        elif self.should_start_new_chapter(
            current_chapter=current_chapter,
            recent_episodes=recent_episodes,
            recent_state_summary=recent_state_summary,
            sleep_metrics=sleep_metrics,
            chapter_signal=chapter_signal,
            current_tick=current_tick,
        ):
            chapters.append(current_chapter)
            current_chapter = NarrativeChapter(
                chapter_id=next_chapter_id,
                tick_range=(chapter_start, chapter_end),
                dominant_theme=recent_theme,
                key_events=recent_key_events,
                behavioral_shift=self._describe_behavioral_shift(
                    chapters[-1],
                    recent_state_summary,
                    recent_adjustments,
                    chapter_signal,
                ),
                state_summary=recent_state_summary,
            )
            if self.log_sink is not None:
                self.log_sink(
                    "[NARRATIVE] "
                    f"chapter={current_chapter.chapter_id} theme={current_chapter.dominant_theme} "
                    f"ticks={current_chapter.tick_range[0]}-{current_chapter.tick_range[1]}"
                )
        else:
            current_chapter.tick_range = (
                min(current_chapter.tick_range[0], chapter_start),
                max(current_chapter.tick_range[1], chapter_end),
            )
            current_chapter.dominant_theme = recent_theme
            current_chapter.key_events = self._merge_key_events(
                current_chapter.key_events,
                recent_key_events,
            )
            current_chapter.state_summary = self._merge_state_summaries(
                current_chapter.state_summary,
                recent_state_summary,
            )

        while len(chapters) > MAX_CHAPTERS:
            chapters = [self._merge_chapters(chapters[0], chapters[1])] + chapters[2:]

        chapter_views = chapters + ([current_chapter] if current_chapter is not None else [])
        source_sleep_session_id = None
        raw_sleep_session = sleep_metrics.get("sleep_cycle_id")
        if isinstance(raw_sleep_session, (int, float)):
            source_sleep_session_id = int(raw_sleep_session)
        narrative = IdentityNarrative(
            chapters=chapters,
            current_chapter=current_chapter,
            core_identity=self._derive_core_identity(chapter_views),
            core_summary="",
            behavioral_patterns=self._derive_behavioral_patterns(chapter_views, policies),
            significant_events=self._derive_significant_events(chapter_views),
            values_statement=self._derive_values_statement(preference_labels, policies),
            last_updated_tick=current_tick,
            version=previous_version,
        )
        narrative.claims = self._build_narrative_claims(
            narrative=narrative,
            episodes=episodic_memory,
            decisions=decision_history,
            policies=policies,
            current_tick=current_tick,
            source_sleep_session_id=source_sleep_session_id,
        )
        narrative.contradiction_summary = self._summarize_claim_consistency(narrative.claims)
        narrative.evidence_provenance = {
            claim.claim_id: {
                "claim_text": claim.text,
                "supported_by": list(claim.supported_by),
                "contradicted_by": list(claim.contradicted_by),
                "support_score": claim.support_score,
                "contradiction_score": claim.contradiction_score,
                "last_validated_at": claim.last_validated_at,
                "source_sleep_session_id": claim.source_sleep_session_id,
                "confidence": claim.confidence,
            }
            for claim in narrative.claims
        }
        narrative.trait_self_model = self._derive_trait_self_model(
            narrative=narrative,
            policies=policies,
        )
        narrative.chapter_transition_evidence = self._derive_chapter_transition_evidence(
            chapters=chapters,
            current_chapter=current_chapter,
        )
        narrative.commitments = self._derive_identity_commitments(
            narrative=narrative,
            policies=policies,
            current_tick=current_tick,
        )
        narrative.autobiographical_summary = self.generate_core_summary(narrative)
        narrative.core_summary = narrative.autobiographical_summary
        self._update_belief_calibration(
            current_tick=current_tick,
            policies=policies,
            claims=narrative.claims,
            episodes=episodic_memory,
        )
        return narrative

    def _build_narrative_claims(
        self,
        *,
        narrative: IdentityNarrative,
        episodes: list[dict[str, object]],
        decisions: list[dict[str, object]],
        policies: PreferredPolicies,
        current_tick: int,
        source_sleep_session_id: int | None,
    ) -> list[NarrativeClaim]:
        claim_specs: list[tuple[str, str, str, str]] = []
        identity_lower = narrative.core_identity.lower()
        if "risk-averse" in identity_lower or "cautious" in identity_lower or policies.risk_profile == "risk_averse":
            claim_specs.append(("trait", "cautious", "I am generally cautious under pressure.", "trait_cautious"))
        elif "risk-seeking" in identity_lower or "aggressive" in identity_lower or policies.risk_profile == "risk_seeking":
            claim_specs.append(("trait", "aggressive", "I am generally aggressive under pressure.", "trait_aggressive"))
        claim_specs.append((
            "value",
            "survival_priority",
            "Survival is prioritized over resource gain.",
            "value_survival_priority",
        ))
        dominant_action = ""
        if policies.action_distribution:
            dominant_action = max(
                policies.action_distribution.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
        if dominant_action:
            claim_specs.append((
                "capability",
                dominant_action,
                f"I can reliably execute {dominant_action} when conditions demand it.",
                f"capability_{dominant_action}",
            ))
        claims: list[NarrativeClaim] = []
        for index, (claim_type, claim_key, text_value, claim_name) in enumerate(claim_specs, start=1):
            claims.append(
                self._evaluate_narrative_claim(
                    claim_id=f"claim-{index:02d}-{claim_name}",
                    claim_type=claim_type,
                    claim_key=claim_key,
                    text_value=text_value,
                    episodes=episodes,
                    decisions=decisions,
                    current_tick=current_tick,
                    source_sleep_session_id=source_sleep_session_id,
                )
            )
        return claims

    def _evaluate_narrative_claim(
        self,
        *,
        claim_id: str,
        claim_type: str,
        claim_key: str,
        text_value: str,
        episodes: list[dict[str, object]],
        decisions: list[dict[str, object]],
        current_tick: int,
        source_sleep_session_id: int | None,
    ) -> NarrativeClaim:
        support_ids: list[str] = []
        contradiction_ids: list[str] = []
        support_score = 0.0
        contradiction_score = 0.0
        evidence_ticks: list[int] = []
        for payload in episodes:
            evidence_id = self._evidence_id(payload)
            if claim_type == "trait" and claim_key == "cautious":
                if self._episode_supports_caution(payload):
                    support_ids.append(evidence_id)
                    support_score += 1.0
                elif self._episode_contradicts_caution(payload):
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 1.0
            elif claim_type == "trait" and claim_key == "aggressive":
                if self._episode_contradicts_caution(payload):
                    support_ids.append(evidence_id)
                    support_score += 1.0
                elif self._episode_supports_caution(payload):
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 1.0
            elif claim_type == "value" and claim_key == "survival_priority":
                if self._episode_supports_survival_priority(payload):
                    support_ids.append(evidence_id)
                    support_score += 1.0
                elif self._episode_contradicts_survival_priority(payload):
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 1.0
            elif claim_type == "capability" and self._episode_matches_action(payload, claim_key):
                if self._episode_supports_capability(payload):
                    support_ids.append(evidence_id)
                    support_score += 1.0
                elif self._episode_contradicts_capability(payload):
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 1.0
            evidence_ticks.append(int(payload.get("timestamp", payload.get("cycle", 0))))
        for payload in decisions:
            evidence_id = self._evidence_id(payload)
            if claim_type == "trait" and claim_key == "cautious":
                if self._decision_supports_caution(payload):
                    support_ids.append(evidence_id)
                    support_score += 0.5
                elif self._decision_contradicts_caution(payload):
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 0.5
            elif claim_type == "value" and claim_key == "survival_priority":
                if self._decision_supports_survival_priority(payload):
                    support_ids.append(evidence_id)
                    support_score += 0.5
                elif self._decision_contradicts_survival_priority(payload):
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 0.5
            elif claim_type == "capability" and str(payload.get("action", "")) == claim_key:
                if float(payload.get("risk", 0.0)) <= 1.0:
                    support_ids.append(evidence_id)
                    support_score += 0.25
                else:
                    contradiction_ids.append(evidence_id)
                    contradiction_score += 0.25
            evidence_ticks.append(int(payload.get("tick", 0)))
        total = support_score + contradiction_score
        confidence = support_score / total if total > 0.0 else 0.0
        latest_evidence_tick = max(evidence_ticks, default=0)
        stale_since = None if current_tick - latest_evidence_tick <= MAX_CHAPTER_TICKS else latest_evidence_tick
        return NarrativeClaim(
            claim_id=claim_id,
            claim_type=claim_type,
            text=text_value,
            claim_key=claim_key,
            supported_by=support_ids,
            contradicted_by=contradiction_ids,
            support_score=round(support_score, 4),
            contradiction_score=round(contradiction_score, 4),
            support_count=len(support_ids),
            contradict_count=len(contradiction_ids),
            confidence=round(confidence, 4),
            stale_since=stale_since,
            last_validated_at=current_tick,
            source_sleep_session_id=source_sleep_session_id,
        )

    def _summarize_claim_consistency(self, claims: list[NarrativeClaim]) -> dict[str, object]:
        contradicted = [claim for claim in claims if claim.contradict_count > claim.support_count]
        mixed = [claim for claim in claims if claim.contradict_count > 0 and claim.support_count > 0]
        return {
            "total_claims": len(claims),
            "contradicted_claims": [claim.claim_id for claim in contradicted],
            "mixed_claims": [claim.claim_id for claim in mixed],
            "supporting_evidence_count": sum(claim.support_count for claim in claims),
            "contradicting_evidence_count": sum(claim.contradict_count for claim in claims),
        }

    def _derive_trait_self_model(
        self,
        *,
        narrative: IdentityNarrative,
        policies: PreferredPolicies,
    ) -> dict[str, object]:
        stable_traits = [
            claim.claim_key
            for claim in narrative.claims
            if claim.claim_type == "trait" and claim.confidence >= 0.5
        ]
        uncertain_traits = [
            claim.claim_key
            for claim in narrative.claims
            if claim.claim_type == "trait" and 0.0 < claim.confidence < 0.5
        ]
        return {
            "risk_profile": policies.risk_profile,
            "dominant_strategy": policies.dominant_strategy,
            "stable_traits": stable_traits,
            "uncertain_traits": uncertain_traits,
            "dominant_patterns": list(narrative.behavioral_patterns[:3]),
            "values_statement": narrative.values_statement,
        }

    def _derive_chapter_transition_evidence(
        self,
        *,
        chapters: list[NarrativeChapter],
        current_chapter: NarrativeChapter | None,
    ) -> list[dict[str, object]]:
        evidence: list[dict[str, object]] = []
        for chapter in chapters:
            if not chapter.behavioral_shift:
                continue
            evidence.append(
                {
                    "chapter_id": chapter.chapter_id,
                    "tick_range": [int(chapter.tick_range[0]), int(chapter.tick_range[1])],
                    "trigger": chapter.behavioral_shift,
                    "dominant_theme": chapter.dominant_theme,
                    "key_events": list(chapter.key_events[:3]),
                }
            )
        if current_chapter is not None and current_chapter.behavioral_shift:
            evidence.append(
                {
                    "chapter_id": current_chapter.chapter_id,
                    "tick_range": [int(current_chapter.tick_range[0]), int(current_chapter.tick_range[1])],
                    "trigger": current_chapter.behavioral_shift,
                    "dominant_theme": current_chapter.dominant_theme,
                    "key_events": list(current_chapter.key_events[:3]),
                }
            )
        return evidence[-5:]

    def _derive_identity_commitments(
        self,
        *,
        narrative: IdentityNarrative,
        policies: PreferredPolicies,
        current_tick: int,
    ) -> list[IdentityCommitment]:
        claim_by_key = {claim.claim_key: claim for claim in narrative.claims}
        commitments: list[IdentityCommitment] = []
        chapter_ids = [
            int(chapter.chapter_id)
            for chapter in narrative.chapters[-2:]
        ]
        if narrative.current_chapter is not None:
            chapter_ids.append(int(narrative.current_chapter.chapter_id))
        chapter_ids = list(dict.fromkeys(chapter_ids))

        survival_claim = claim_by_key.get("survival_priority")
        if survival_claim is not None and survival_claim.confidence >= 0.35:
            commitments.append(
                IdentityCommitment(
                    commitment_id="commitment-survival-priority",
                    commitment_type="value_guardrail",
                    statement="Protect survival and integrity before opportunistic gain.",
                    target_actions=["hide", "rest", "exploit_shelter", "thermoregulate"],
                    discouraged_actions=["forage"],
                    confidence=round(survival_claim.confidence, 4),
                    priority=0.95,
                    source_claim_ids=[survival_claim.claim_id],
                    source_chapter_ids=chapter_ids,
                    evidence_ids=list(survival_claim.supported_by[:5]),
                    last_reaffirmed_tick=current_tick,
                )
            )

        if "exploratory" in narrative.core_identity.lower() or policies.risk_profile == "risk_seeking":
            exploratory_claim = claim_by_key.get("aggressive")
            exploratory_confidence = exploratory_claim.confidence if exploratory_claim is not None else 0.45
            commitments.append(
                IdentityCommitment(
                    commitment_id="commitment-exploration-drive",
                    commitment_type="behavioral_style",
                    statement="When conditions are stable, reduce uncertainty through active exploration.",
                    target_actions=["scan", "seek_contact"],
                    discouraged_actions=["rest"],
                    confidence=round(exploratory_confidence, 4),
                    priority=0.55,
                    source_claim_ids=(
                        [exploratory_claim.claim_id] if exploratory_claim is not None else []
                    ),
                    source_chapter_ids=chapter_ids,
                    evidence_ids=(
                        list(exploratory_claim.supported_by[:5])
                        if exploratory_claim is not None
                        else []
                    ),
                    last_reaffirmed_tick=current_tick,
                )
            )

        dominant_action = ""
        if policies.action_distribution:
            dominant_action = max(
                policies.action_distribution.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
        capability_claim = claim_by_key.get(dominant_action) if dominant_action else None
        if capability_claim is not None and capability_claim.confidence >= 0.40:
            commitments.append(
                IdentityCommitment(
                    commitment_id=f"commitment-capability-{dominant_action}",
                    commitment_type="capability",
                    statement=f"Maintain competence for {dominant_action} when its context recurs.",
                    target_actions=[dominant_action],
                    discouraged_actions=[],
                    confidence=round(capability_claim.confidence, 4),
                    priority=0.45,
                    source_claim_ids=[capability_claim.claim_id],
                    source_chapter_ids=chapter_ids,
                    evidence_ids=list(capability_claim.supported_by[:5]),
                    last_reaffirmed_tick=current_tick,
                )
            )

        return commitments

    def _update_belief_calibration(
        self,
        *,
        current_tick: int,
        policies: PreferredPolicies,
        claims: list[NarrativeClaim],
        episodes: list[dict[str, object]],
    ) -> None:
        total_claim_support = sum(claim.support_count for claim in claims)
        mean_confidence = mean([claim.confidence for claim in claims]) if claims else 0.0
        protected_events = len([episode for episode in episodes if bool(episode.get("identity_critical", False))])
        self.belief_calibration = {
            "preferred_policies": {
                "confidence": round(mean_confidence, 4),
                "evidence_count": total_claim_support,
                "last_verified_at": current_tick,
                "risk_profile": policies.risk_profile,
            },
            "capability_profile": {
                "confidence": round(mean_confidence, 4),
                "evidence_count": len(self.capability_model.available_actions),
                "last_verified_at": current_tick,
            },
            "threat_profile": {
                "confidence": round(min(1.0, protected_events / max(1, len(episodes))), 4),
                "evidence_count": len(episodes),
                "last_verified_at": current_tick,
            },
        }

    def _evidence_id(self, payload: Mapping[str, object]) -> str:
        raw = payload.get("episode_id") or payload.get("claim_id")
        if raw:
            return str(raw)
        tick = int(payload.get("timestamp", payload.get("cycle", payload.get("tick", 0))))
        action = str(payload.get("action_taken", payload.get("action", "evidence")))
        return f"ev-{tick}-{action}"

    def _episode_matches_action(self, payload: Mapping[str, object], action: str) -> bool:
        return str(payload.get("action_taken", payload.get("action", ""))) == action

    def _episode_supports_caution(self, payload: Mapping[str, object]) -> bool:
        action = str(payload.get("action_taken", payload.get("action", "")))
        risk = float(payload.get("risk", 0.0))
        outcome = str(payload.get("predicted_outcome", "neutral"))
        return action in {"hide", "rest", "exploit_shelter"} and (risk >= 0.8 or outcome != "resource_gain")

    def _episode_contradicts_caution(self, payload: Mapping[str, object]) -> bool:
        action = str(payload.get("action_taken", payload.get("action", "")))
        risk = float(payload.get("risk", 0.0))
        outcome = str(payload.get("predicted_outcome", "neutral"))
        return action in {"forage", "scan", "seek_contact"} and (risk >= 1.0 or outcome in {"survival_threat", "integrity_loss"})

    def _episode_supports_survival_priority(self, payload: Mapping[str, object]) -> bool:
        return self._episode_supports_caution(payload) or bool(payload.get("identity_critical", False))

    def _episode_contradicts_survival_priority(self, payload: Mapping[str, object]) -> bool:
        action = str(payload.get("action_taken", payload.get("action", "")))
        outcome = str(payload.get("predicted_outcome", "neutral"))
        return action == "forage" and outcome in {"survival_threat", "integrity_loss", "resource_gain"}

    def _episode_supports_capability(self, payload: Mapping[str, object]) -> bool:
        outcome = str(payload.get("predicted_outcome", "neutral"))
        return outcome in {"neutral", "resource_gain"}

    def _episode_contradicts_capability(self, payload: Mapping[str, object]) -> bool:
        outcome = str(payload.get("predicted_outcome", "neutral"))
        return outcome in {"survival_threat", "integrity_loss"}

    def _decision_supports_caution(self, payload: Mapping[str, object]) -> bool:
        action = str(payload.get("action", ""))
        risk = float(payload.get("risk", 0.0))
        return action in {"hide", "rest", "exploit_shelter"} and risk <= 1.0

    def _decision_contradicts_caution(self, payload: Mapping[str, object]) -> bool:
        action = str(payload.get("action", ""))
        risk = float(payload.get("risk", 0.0))
        return action in {"forage", "scan", "seek_contact"} and risk >= 1.0

    def _decision_supports_survival_priority(self, payload: Mapping[str, object]) -> bool:
        return self._decision_supports_caution(payload)

    def _decision_contradicts_survival_priority(self, payload: Mapping[str, object]) -> bool:
        action = str(payload.get("action", ""))
        risk = float(payload.get("risk", 0.0))
        return action == "forage" and risk >= 1.0
    def should_start_new_chapter(
        self,
        *,
        current_chapter: NarrativeChapter,
        recent_episodes: list[dict[str, object]],
        recent_state_summary: Mapping[str, object],
        sleep_metrics: Mapping[str, object],
        chapter_signal: str | None,
        current_tick: int,
    ) -> bool:
        current_action = str(current_chapter.state_summary.get("dominant_action", ""))
        recent_action = str(recent_state_summary.get("dominant_action", ""))
        if current_action and recent_action and current_action != recent_action:
            return True
        if any(
            float(payload.get("total_surprise", payload.get("weighted_surprise", 0.0))) > HIGH_SURPRISE_THRESHOLD
            for payload in recent_episodes
        ) and (
            int(sleep_metrics.get("policy_bias_updates", 0)) > 0
            or int(sleep_metrics.get("threat_updates", 0)) > 0
        ):
            return True
        if current_tick - current_chapter.tick_range[0] > MAX_CHAPTER_TICKS:
            return True
        if chapter_signal:
            return True
        return False

    def generate_core_summary(self, narrative: IdentityNarrative) -> str:
        chapters = list(narrative.chapters)
        if narrative.current_chapter is not None:
            chapters.append(narrative.current_chapter)
        if not chapters:
            return "I am an agent still forming a coherent identity."
        parts = [narrative.core_identity or "I am an adaptive agent."]
        shifts = [chapter for chapter in chapters if chapter.behavioral_shift]
        if shifts:
            major_shift = max(shifts, key=self._shift_significance)
            parts.append(
                f"A significant shift occurred around tick {major_shift.tick_range[0]}: "
                f"{major_shift.behavioral_shift}."
            )
        latest = chapters[-1]
        parts.append(f"Currently in a {latest.dominant_theme} phase.")
        if latest.state_summary.get("dominant_strategy"):
            parts.append(f"My dominant strategy remains {latest.state_summary.get('dominant_strategy')}.")
        if latest.key_events and (
            "near-death" in latest.key_events[0]
            or latest.dominant_theme == "survival_crisis"
        ):
            parts.append(f"Recent memory remains anchored by {latest.key_events[0]}.")
        return " ".join(parts)

    def _extract_key_events(
        self,
        episodic_memory: list[dict[str, object]],
    ) -> list[str]:
        ranked_episodes = sorted(
            episodic_memory,
            key=lambda payload: (
                -float(payload.get("total_surprise", payload.get("weighted_surprise", 0.0))),
                -int(payload.get("timestamp", payload.get("cycle", 0))),
            ),
        )
        key_events: list[str] = []
        for payload in ranked_episodes[:5]:
            tick = int(payload.get("timestamp", payload.get("cycle", 0)))
            action = str(payload.get("action_taken", payload.get("action", "unknown")))
            surprise = float(payload.get("total_surprise", payload.get("weighted_surprise", 0.0)))
            outcome = str(payload.get("predicted_outcome", "neutral"))
            label = "near-death event" if outcome == "survival_threat" else f"{outcome} event"
            key_events.append(f"{label} at tick {tick} after {action} (surprise={surprise:.2f})")
        return key_events

    def _summarize_chapter_state(
        self,
        *,
        decisions: list[dict[str, object]],
        episodes: list[dict[str, object]],
        policies: PreferredPolicies,
    ) -> dict[str, object]:
        energy_values = [
            float(payload.get("body_state", {}).get("energy", 0.0))
            for payload in episodes
            if isinstance(payload.get("body_state"), dict)
        ]
        dominant_action_counts = Counter(
            str(payload.get("action", ""))
            for payload in decisions
            if payload.get("action")
        )
        if not dominant_action_counts:
            dominant_action_counts = Counter(
                str(payload.get("action_taken", payload.get("action", "")))
                for payload in episodes
                if payload.get("action_taken", payload.get("action"))
            )
        risks = [
            float(payload.get("risk", 0.0))
            for payload in (decisions + episodes)
            if isinstance(payload.get("risk", 0.0), (int, float))
        ]
        free_energy_from_episodes = [
            float(payload.get("outcome_state", payload.get("outcome", {})).get("free_energy_drop", 0.0))
            for payload in episodes
            if isinstance(payload.get("outcome_state", payload.get("outcome", {})), dict)
        ]
        if dominant_action_counts:
            dominant_action = dominant_action_counts.most_common(1)[0][0]
        elif policies.action_distribution:
            dominant_action = max(
                policies.action_distribution.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
        else:
            dominant_action = "rest"
        return {
            "energy_avg": mean(energy_values) if energy_values else self.body_schema.energy,
            "free_energy_avg": mean(free_energy_from_episodes) if free_energy_from_episodes else 0.0,
            "dominant_action": dominant_action,
            "risk_profile": policies.risk_profile,
            "risk_avg": mean(risks) if risks else 0.0,
            "dominant_strategy": policies.dominant_strategy,
        }

    def _infer_dominant_theme(
        self,
        *,
        state_summary: Mapping[str, object],
        episodes: list[dict[str, object]],
        chapter_signal: str | None,
    ) -> str:
        if chapter_signal:
            return "goal_realignment"
        dominant_action = str(state_summary.get("dominant_action", ""))
        energy_avg = float(state_summary.get("energy_avg", self.body_schema.energy))
        if any(
            str(payload.get("predicted_outcome", "neutral")) == "survival_threat"
            or float(payload.get("total_surprise", payload.get("weighted_surprise", 0.0))) > HIGH_SURPRISE_THRESHOLD
            for payload in episodes
        ):
            return "survival_crisis"
        if dominant_action in {"scan", "seek_contact"}:
            return "exploration_phase"
        if dominant_action in {"forage", "rest"} and energy_avg < 0.55:
            return "resource_recovery"
        return "consolidation"

    def _describe_behavioral_shift(
        self,
        previous_chapter: NarrativeChapter,
        current_state_summary: Mapping[str, object],
        weight_adjustments: list[object],
        chapter_signal: str | None,
    ) -> str | None:
        if chapter_signal:
            return chapter_signal
        previous_action = str(previous_chapter.state_summary.get("dominant_action", ""))
        current_action = str(current_state_summary.get("dominant_action", ""))
        previous_risk = str(previous_chapter.state_summary.get("risk_profile", ""))
        current_risk = str(current_state_summary.get("risk_profile", ""))
        if previous_action != current_action:
            return f"Behavior shifted from {previous_action} to {current_action}"
        if previous_risk != current_risk:
            return f"Risk posture shifted from {previous_risk} to {current_risk}"
        if weight_adjustments:
            return (
                "Goal weighting changed after conflicts at ticks "
                f"{[int(getattr(item, 'tick', 0)) for item in weight_adjustments[:3]]}"
            )
        return None

    def _derive_core_identity(
        self,
        chapters: list[NarrativeChapter],
    ) -> str:
        if not chapters:
            return "I am an adaptive agent still consolidating stable traits."
        chapter_features = [self._chapter_features(chapter) for chapter in chapters]
        recent_window = chapter_features[-3:] if len(chapter_features) >= 3 else chapter_features
        core_features = sorted(
            feature
            for feature in set().union(*chapter_features)
            if recent_window and all(feature in features for features in recent_window)
        )
        former_features = sorted(
            feature
            for feature in set().union(*chapter_features)
            if feature not in core_features and any(feature in features for features in chapter_features[:-len(recent_window)] or [])
        )
        if core_features:
            sentence = f"I am a {', '.join(core_features)} agent."
        else:
            latest = chapters[-1]
            sentence = (
                f"I am presently oriented around {latest.state_summary.get('dominant_action', 'rest')} "
                f"with a {latest.state_summary.get('risk_profile', 'risk_neutral')} stance."
            )
        if former_features:
            sentence += f" I used to be more {', '.join(former_features[:2])}."
        return sentence

    def _chapter_features(self, chapter: NarrativeChapter) -> set[str]:
        features: set[str] = set()
        action = str(chapter.state_summary.get("dominant_action", ""))
        risk_profile = str(chapter.state_summary.get("risk_profile", ""))
        if risk_profile == "risk_averse":
            features.add("risk-averse")
        elif risk_profile == "risk_seeking":
            features.add("risk-seeking")
        if action in {"hide", "rest", "exploit_shelter"}:
            features.add("resource-conservative")
        if action in {"scan", "seek_contact"}:
            features.add("exploratory")
        if chapter.dominant_theme == "survival_crisis":
            features.add("survival-focused")
        return features

    def _derive_behavioral_patterns(
        self,
        chapters: list[NarrativeChapter],
        policies: PreferredPolicies,
    ) -> list[str]:
        patterns: list[str] = []
        recent = chapters[-3:]
        action_counts = Counter(
            str(chapter.state_summary.get("dominant_action", ""))
            for chapter in recent
            if chapter.state_summary.get("dominant_action")
        )
        for action, _count in action_counts.most_common(3):
            patterns.append(f"I tend to {action} during {recent[-1].dominant_theme} phases")
        if not patterns:
            for action, frequency in sorted(
                policies.action_distribution.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]:
                patterns.append(f"I tend to {action} ({frequency:.0%} of decisions)")
        return patterns

    def _derive_significant_events(
        self,
        chapters: list[NarrativeChapter],
    ) -> list[str]:
        events: list[str] = []
        for chapter in reversed(chapters):
            for event in chapter.key_events:
                if event not in events:
                    events.append(event)
                if len(events) >= 5:
                    return events
        return events

    def _merge_key_events(self, existing: list[str], new_events: list[str]) -> list[str]:
        merged: list[str] = []
        for event in list(new_events) + list(existing):
            if event not in merged:
                merged.append(event)
            if len(merged) >= 5:
                break
        return merged

    def _merge_state_summaries(
        self,
        existing: Mapping[str, object],
        new_state: Mapping[str, object],
    ) -> dict[str, object]:
        merged = dict(existing)
        for numeric_key in ("energy_avg", "free_energy_avg", "risk_avg"):
            merged[numeric_key] = mean([
                float(merged.get(numeric_key, new_state.get(numeric_key, 0.0))),
                float(new_state.get(numeric_key, merged.get(numeric_key, 0.0))),
            ])
        for text_key in ("dominant_action", "risk_profile", "dominant_strategy"):
            if new_state.get(text_key):
                merged[text_key] = new_state[text_key]
        return merged

    def _merge_chapters(
        self,
        left: NarrativeChapter,
        right: NarrativeChapter,
    ) -> NarrativeChapter:
        return NarrativeChapter(
            chapter_id=left.chapter_id,
            tick_range=(left.tick_range[0], right.tick_range[1]),
            dominant_theme=(
                left.dominant_theme
                if left.dominant_theme == right.dominant_theme
                else "merged_history"
            ),
            key_events=self._merge_key_events(left.key_events, right.key_events),
            behavioral_shift=right.behavioral_shift or left.behavioral_shift,
            state_summary=self._merge_state_summaries(left.state_summary, right.state_summary),
        )

    def _shift_significance(self, chapter: NarrativeChapter) -> float:
        score = float(len(chapter.key_events))
        if chapter.behavioral_shift:
            score += 2.0
            if "Goal priority shifted" in chapter.behavioral_shift:
                score += 2.0
        if chapter.dominant_theme == "survival_crisis":
            score += 2.0
        return score

    def _derive_values_statement(
        self,
        preference_labels: Mapping[str, float],
        policies: PreferredPolicies,
    ) -> str:
        ranked_values = sorted(
            (
                (str(label), float(value))
                for label, value in preference_labels.items()
                if isinstance(value, (int, float))
            ),
            key=lambda item: item[1],
        )
        if ranked_values:
            most_avoided = ranked_values[0][0]
            most_sought = ranked_values[-1][0]
        else:
            most_avoided = "survival_threat"
            most_sought = "resource_gain"
        emphasis = f" I usually favor {policies.dominant_strategy}." if policies.dominant_strategy else ""
        return (
            f"I prioritize avoiding {most_avoided} while moving toward {most_sought}."
            f" My current risk profile is {policies.risk_profile}.{emphasis}"
        )


def build_default_self_model(
    *,
    log_sink: Callable[[str], None] | None = None,
) -> SelfModel:
    """Construct the hard-coded self prior used by the runtime."""

    return SelfModel(
        body_schema=BodySchema(
            energy=0.85,
            token_budget=256,
            memory_usage=128.0,
            compute_load=0.25,
        ),
        capability_model=CapabilityModel(
            action_schemas=tuple(ActionSchema(name=name) for name in CORE_ACTIONS),
            api_limits=CORE_API_LIMITS,
        ),
        resource_state=ResourceState(
            tokens_remaining=256,
            cpu_budget=1.0,
            memory_free=1024.0,
        ),
        threat_model=ThreatModel(
            token_exhaustion_threshold=32,
            memory_overflow_threshold=96.0,
            fatal_exceptions=("FatalException",),
        ),
        threat_profile=ThreatProfile(),
        preferred_policies=PreferredPolicies(),
        identity_narrative=IdentityNarrative(),
        narrative_priors=NarrativePriors(),
        log_sink=log_sink,
    )


def _mean_abs_delta(
    previous: Mapping[str, float] | None,
    current: Mapping[str, float] | None,
) -> float:
    previous = previous or {}
    current = current or {}
    keys = sorted(set(previous) | set(current))
    if not keys:
        return 0.0
    return mean(abs(float(current.get(key, 0.0)) - float(previous.get(key, 0.0))) for key in keys)


def _distribution(left: list[str]) -> dict[str, float]:
    if not left:
        return {}
    counts = Counter(item for item in left if item)
    total = sum(counts.values()) or 1
    return {str(key): value / total for key, value in sorted(counts.items())}


def _dominant_action(distribution: Mapping[str, float]) -> tuple[str, float]:
    if not distribution:
        return "", 0.0
    action, ratio = max(
        distribution.items(),
        key=lambda item: (float(item[1]), str(item[0])),
    )
    return str(action), float(ratio)


def _distribution_delta(
    previous: Mapping[str, float] | None,
    current: Mapping[str, float] | None,
) -> float:
    previous = previous or {}
    current = current or {}
    keys = sorted(set(previous) | set(current))
    if not keys:
        return 0.0
    return mean(abs(float(current.get(key, 0.0)) - float(previous.get(key, 0.0))) for key in keys)


def _set_divergence(previous: list[str] | tuple[str, ...], current: list[str] | tuple[str, ...]) -> float:
    previous_set = {str(item) for item in previous}
    current_set = {str(item) for item in current}
    union = previous_set | current_set
    if not union:
        return 0.0
    return 1.0 - (len(previous_set & current_set) / len(union))
