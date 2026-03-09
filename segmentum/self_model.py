from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping


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
}


def _event_name(event: object) -> str:
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
    }
    key = _event_name(event).strip().casefold()
    return aliases.get(key, key)


@dataclass(frozen=True, slots=True)
class BodySchema:
    """Immutable structural priors describing the agent's internal body."""

    energy: float
    token_budget: int
    memory_usage: float
    compute_load: float


@dataclass(frozen=True, slots=True)
class CapabilityModel:
    """Immutable prior over what the agent can do and where it is constrained."""

    available_actions: tuple[str, ...] = ()
    api_limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_actions", tuple(self.available_actions))
        object.__setattr__(
            self,
            "api_limits",
            MappingProxyType(dict(self.api_limits)),
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
    )
    world_error_events: tuple[str, ...] = (
        "http_timeout",
        "network_failure",
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

    def classify(self, event: object) -> str:
        normalized = _normalize_event_name(event)
        if normalized in self.existential_events:
            return EXISTENTIAL_THREAT
        if normalized in self.self_error_events:
            return SELF_ERROR
        if normalized in self.world_error_events:
            return WORLD_ERROR
        return WORLD_ERROR

    @staticmethod
    def surprise_source(classification: str) -> str:
        if classification == SELF_ERROR:
            return "interoceptive"
        if classification == WORLD_ERROR:
            return "exteroceptive"
        return "existential"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    event: str
    classification: str
    resource_state: Mapping[str, float | int]
    surprise_source: str
    detected_threats: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_state",
            MappingProxyType(dict(self.resource_state)),
        )

    def to_log_string(self) -> str:
        return "\n".join(
            [
                "[SelfModel]",
                f"event={self.event}",
                f"classification={self.classification}",
                f"resource_state={dict(self.resource_state)}",
                f"surprise_source={self.surprise_source}",
                f"detected_threats={list(self.detected_threats)}",
            ]
        )


@dataclass(slots=True)
class SelfModel:
    """Minimal self model for separating self, world, and survival failures."""

    body_schema: BodySchema
    capability_model: CapabilityModel
    resource_state: ResourceState
    threat_model: ThreatModel
    error_classifier: ErrorClassifier = field(default_factory=ErrorClassifier)
    log_sink: Callable[[str], None] | None = None
    last_result: ClassificationResult | None = field(init=False, default=None)

    def classify_event(self, event: object) -> str:
        return self.inspect_event(event).classification

    def inspect_event(self, event: object) -> ClassificationResult:
        event_name = _event_name(event)
        classification = self.error_classifier.classify(event_name)
        result = ClassificationResult(
            event=event_name,
            classification=classification,
            resource_state=self.resource_state.snapshot(),
            surprise_source=self.error_classifier.surprise_source(classification),
            detected_threats=self.threat_model.detect(
                event_name,
                self.resource_state,
                self.body_schema,
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
            available_actions=CORE_ACTIONS,
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
        log_sink=log_sink,
    )
