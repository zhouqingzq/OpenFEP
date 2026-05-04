from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC = timezone.utc
from typing import Mapping


COGNITIVE_EVENT_TYPES: tuple[str, ...] = (
    "ObservationEvent",
    "MemoryActivationEvent",
    "DecisionEvent",
    "CandidatePathEvent",
    "PathSelectionEvent",
    "PromptAssemblyEvent",
    "GenerationEvent",
    "OutcomeEvent",
    # M8.9: Memory write intent types
    "DialogueFactExtractionEvent",
    "MemoryWriteResultEvent",
)


COGNITIVE_EVENT_CONSUMERS: dict[str, tuple[str, ...]] = {
    "ObservationEvent": ("state_update", "trace"),
    "MemoryActivationEvent": ("state_update", "trace"),
    "DecisionEvent": ("trace", "evaluation"),
    "CandidatePathEvent": ("state_update", "trace"),
    "PathSelectionEvent": ("prompt_assembly_audit", "trace"),
    "PromptAssemblyEvent": ("prompt_assembly_audit", "trace"),
    "GenerationEvent": ("trace", "evaluation"),
    "OutcomeEvent": ("state_update", "trace", "evaluation"),
    # M8.9: Memory write intent consumers
    "DialogueFactExtractionEvent": ("state_update", "trace"),
    "MemoryWriteResultEvent": ("trace", "evaluation"),
}


JsonSafe = None | bool | int | float | str | list["JsonSafe"] | dict[str, "JsonSafe"]


def _json_safe(value: object) -> JsonSafe:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return _json_safe(converted)
    return str(value)


def _clean_id_part(value: object) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned.lower() or "none"


def make_cognitive_event_id(
    *,
    session_id: str,
    turn_id: str,
    cycle: int,
    event_type: str,
    source: str,
    sequence_index: int,
    persona_id: str | None = None,
) -> str:
    """Build a deterministic event id without consulting wall-clock time."""
    persona_part = (
        f"{_clean_id_part(persona_id)}-"
        if persona_id is not None
        else ""
    )
    return (
        "cognitive-"
        f"{_clean_id_part(session_id)}-"
        f"{persona_part}"
        f"{_clean_id_part(turn_id)}-"
        f"{int(cycle):06d}-"
        f"{_clean_id_part(event_type)}-"
        f"{_clean_id_part(source)}-"
        f"{int(sequence_index):04d}"
    )


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CognitiveEvent:
    event_id: str
    event_type: str
    turn_id: str
    cycle: int
    session_id: str
    persona_id: str
    source: str
    timestamp: str
    salience: float
    priority: float
    ttl: int
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "turn_id": self.turn_id,
            "cycle": int(self.cycle),
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "salience": float(self.salience),
            "priority": float(self.priority),
            "ttl": int(self.ttl),
            "payload": _json_safe(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CognitiveEvent":
        raw_payload = payload.get("payload", {})
        return cls(
            event_id=str(payload.get("event_id", "")),
            event_type=str(payload.get("event_type", "")),
            turn_id=str(payload.get("turn_id", "")),
            cycle=int(payload.get("cycle", 0)),
            session_id=str(payload.get("session_id", "")),
            persona_id=str(payload.get("persona_id", "")),
            source=str(payload.get("source", "")),
            timestamp=str(payload.get("timestamp", "")),
            salience=float(payload.get("salience", 0.0)),
            priority=float(payload.get("priority", 0.0)),
            ttl=int(payload.get("ttl", 0)),
            payload=dict(raw_payload) if isinstance(raw_payload, Mapping) else {},
        )


class CognitiveEventBus:
    def __init__(self, events: tuple[CognitiveEvent, ...] = ()) -> None:
        self._events: list[CognitiveEvent] = list(events)
        self._consumed_event_ids: list[str] = []

    def publish(self, event: CognitiveEvent) -> CognitiveEvent:
        self._events.append(event)
        return event

    def events(self) -> tuple[CognitiveEvent, ...]:
        return tuple(self._events)

    def consumed_event_ids(self) -> tuple[str, ...]:
        return tuple(self._consumed_event_ids)

    def filter(
        self,
        *,
        event_type: str | None = None,
        source: str | None = None,
        min_salience: float | None = None,
        persona_id: str | None = None,
    ) -> tuple[CognitiveEvent, ...]:
        selected = self._events
        if event_type is not None:
            selected = [event for event in selected if event.event_type == event_type]
        if source is not None:
            selected = [event for event in selected if event.source == source]
        if min_salience is not None:
            threshold = float(min_salience)
            selected = [event for event in selected if event.salience >= threshold]
        if persona_id is not None:
            selected = [event for event in selected if event.persona_id == persona_id]
        return tuple(selected)

    def consume(
        self,
        *,
        turn_id: str | None = None,
        persona_id: str | None = None,
        event_type: str | None = None,
        min_salience: float | None = None,
        include_expired: bool = False,
    ) -> tuple[CognitiveEvent, ...]:
        selected = self.filter(
            event_type=event_type,
            min_salience=min_salience,
            persona_id=persona_id,
        )
        if turn_id is not None:
            selected = tuple(event for event in selected if event.turn_id == turn_id)
        if not include_expired:
            selected = tuple(event for event in selected if event.ttl > 0)
        seen = set(self._consumed_event_ids)
        for event in selected:
            if event.event_id not in seen:
                self._consumed_event_ids.append(event.event_id)
                seen.add(event.event_id)
        return selected

    def clear_expired(self) -> None:
        self._events = [event for event in self._events if event.ttl > 0]

    def to_dict(self) -> dict[str, object]:
        return {"events": [event.to_dict() for event in self._events]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "CognitiveEventBus":
        if not payload:
            return cls()
        raw_events = payload.get("events", [])
        if not isinstance(raw_events, list):
            return cls()
        return cls(
            tuple(
                CognitiveEvent.from_dict(item)
                for item in raw_events
                if isinstance(item, Mapping)
            )
        )


def make_cognitive_event(
    *,
    event_type: str,
    turn_id: str,
    cycle: int,
    session_id: str,
    persona_id: str,
    source: str,
    sequence_index: int,
    payload: Mapping[str, object] | None = None,
    salience: float = 0.5,
    priority: float = 0.5,
    ttl: int = 1,
    timestamp: str | None = None,
) -> CognitiveEvent:
    event_payload: dict[str, object] = dict(payload or {})
    event_payload["persona_id"] = persona_id
    event_payload.setdefault(
        "planned_consumers",
        list(COGNITIVE_EVENT_CONSUMERS.get(event_type, ())),
    )
    return CognitiveEvent(
        event_id=make_cognitive_event_id(
            session_id=session_id,
            turn_id=turn_id,
            cycle=cycle,
            event_type=event_type,
            source=source,
            sequence_index=sequence_index,
            persona_id=persona_id,
        ),
        event_type=event_type,
        turn_id=turn_id,
        cycle=cycle,
        session_id=session_id,
        persona_id=persona_id,
        source=source,
        timestamp=timestamp or utc_timestamp(),
        salience=float(salience),
        priority=float(priority),
        ttl=int(ttl),
        payload=event_payload,
    )
