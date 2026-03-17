from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Mapping

from .action_schema import ActionSchema, action_name, ensure_action_schema
from .environment import Observation
from .interoception import InteroceptionReading
from .narrative_types import NarrativeEpisode


def _coerce_float_mapping(payload: object) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


@dataclass(frozen=True)
class BusSignal:
    channel: str
    raw_value: float
    normalized_value: float
    confidence: float
    source_id: str
    source_type: str
    role: str
    units: str = "normalized"
    provenance: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "role": self.role,
            "units": self.units,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BusSignal":
        return cls(
            channel=str(payload.get("channel", "")),
            raw_value=float(payload.get("raw_value", 0.0)),
            normalized_value=float(payload.get("normalized_value", 0.0)),
            confidence=float(payload.get("confidence", 1.0)),
            source_id=str(payload.get("source_id", "")),
            source_type=str(payload.get("source_type", "")),
            role=str(payload.get("role", "exteroceptive")),
            units=str(payload.get("units", "normalized")),
            provenance=(
                dict(payload.get("provenance"))
                if isinstance(payload.get("provenance"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class PerceptionPacket:
    packet_id: str
    cycle: int
    source_type: str
    source_id: str
    adapter_name: str
    confidence: float
    observation: dict[str, float]
    signals: tuple[BusSignal, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "packet_id": self.packet_id,
            "cycle": self.cycle,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "adapter_name": self.adapter_name,
            "confidence": self.confidence,
            "observation": dict(self.observation),
            "signals": [signal.to_dict() for signal in self.signals],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PerceptionPacket":
        raw_signals = payload.get("signals", [])
        return cls(
            packet_id=str(payload.get("packet_id", "")),
            cycle=int(payload.get("cycle", 0)),
            source_type=str(payload.get("source_type", "")),
            source_id=str(payload.get("source_id", "")),
            adapter_name=str(payload.get("adapter_name", "")),
            confidence=float(payload.get("confidence", 1.0)),
            observation=_coerce_float_mapping(payload.get("observation")),
            signals=tuple(
                BusSignal.from_dict(item)
                for item in raw_signals
                if isinstance(item, Mapping)
            ),
            metadata=(
                dict(payload.get("metadata"))
                if isinstance(payload.get("metadata"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class ActionEffectAck:
    ack_id: str
    cycle: int
    source_type: str
    source_id: str
    action: dict[str, object]
    action_name: str
    success: bool
    confidence: float
    feedback: dict[str, float]
    acknowledged_channels: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "ack_id": self.ack_id,
            "cycle": self.cycle,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "action": dict(self.action),
            "action_name": self.action_name,
            "success": self.success,
            "confidence": self.confidence,
            "feedback": dict(self.feedback),
            "acknowledged_channels": list(self.acknowledged_channels),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionEffectAck":
        return cls(
            ack_id=str(payload.get("ack_id", "")),
            cycle=int(payload.get("cycle", 0)),
            source_type=str(payload.get("source_type", "")),
            source_id=str(payload.get("source_id", "")),
            action=(
                dict(payload.get("action"))
                if isinstance(payload.get("action"), Mapping)
                else {}
            ),
            action_name=str(payload.get("action_name", "")),
            success=bool(payload.get("success", True)),
            confidence=float(payload.get("confidence", 1.0)),
            feedback=_coerce_float_mapping(payload.get("feedback")),
            acknowledged_channels=tuple(
                str(item) for item in payload.get("acknowledged_channels", [])
            ),
            metadata=(
                dict(payload.get("metadata"))
                if isinstance(payload.get("metadata"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class ActionDispatchRecord:
    dispatch_id: str
    cycle: int
    source_type: str
    source_id: str
    adapter_name: str
    action: dict[str, object]
    action_name: str
    feedback: dict[str, float]
    acknowledgment: ActionEffectAck

    def to_dict(self) -> dict[str, object]:
        return {
            "dispatch_id": self.dispatch_id,
            "cycle": self.cycle,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "adapter_name": self.adapter_name,
            "action": dict(self.action),
            "action_name": self.action_name,
            "feedback": dict(self.feedback),
            "acknowledgment": self.acknowledgment.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionDispatchRecord":
        return cls(
            dispatch_id=str(payload.get("dispatch_id", "")),
            cycle=int(payload.get("cycle", 0)),
            source_type=str(payload.get("source_type", "")),
            source_id=str(payload.get("source_id", "")),
            adapter_name=str(payload.get("adapter_name", "")),
            action=(
                dict(payload.get("action"))
                if isinstance(payload.get("action"), Mapping)
                else {}
            ),
            action_name=str(payload.get("action_name", "")),
            feedback=_coerce_float_mapping(payload.get("feedback")),
            acknowledgment=ActionEffectAck.from_dict(
                dict(payload.get("acknowledgment", {}))
            ),
        )


@dataclass
class PerceptionBus:
    packets_seen: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    last_packet_id: str = ""

    def _remember(self, packet: PerceptionPacket) -> PerceptionPacket:
        self.packets_seen += 1
        self.last_packet_id = packet.packet_id
        self.source_counts[packet.source_type] = self.source_counts.get(packet.source_type, 0) + 1
        return packet

    def _next_packet_id(self) -> str:
        return f"perception-{self.packets_seen + 1:06d}"

    def capture_simulated_world(
        self,
        observation: Observation,
        *,
        cycle: int,
        source_id: str = "simulated_world",
    ) -> PerceptionPacket:
        values = {
            "food": float(observation.food),
            "danger": float(observation.danger),
            "novelty": float(observation.novelty),
            "shelter": float(observation.shelter),
            "temperature": float(observation.temperature),
            "social": float(observation.social),
        }
        packet = PerceptionPacket(
            packet_id=self._next_packet_id(),
            cycle=cycle,
            source_type="simulated_world",
            source_id=source_id,
            adapter_name="SimulatedWorldAdapter",
            confidence=1.0,
            observation=values,
            signals=tuple(
                BusSignal(
                    channel=channel,
                    raw_value=value,
                    normalized_value=value,
                    confidence=1.0,
                    source_id=source_id,
                    source_type="simulated_world",
                    role="exteroceptive",
                )
                for channel, value in values.items()
            ),
            metadata={"channel_count": len(values)},
        )
        return self._remember(packet)

    def capture_interoception(
        self,
        reading: InteroceptionReading,
        *,
        cycle: int,
        source_id: str = "host_process",
    ) -> PerceptionPacket:
        values = {
            "cpu_prediction_error": float(reading.cpu_prediction_error),
            "memory_prediction_error": float(reading.memory_prediction_error),
            "resource_pressure": float(reading.resource_pressure),
            "surprise_signal": float(reading.surprise_signal),
            "boredom_signal": float(reading.boredom_signal),
            "energy_drain": float(reading.energy_drain),
        }
        packet = PerceptionPacket(
            packet_id=self._next_packet_id(),
            cycle=cycle,
            source_type="host_telemetry",
            source_id=source_id,
            adapter_name="ProcessInteroceptorAdapter",
            confidence=1.0,
            observation=values,
            signals=tuple(
                BusSignal(
                    channel=channel,
                    raw_value=value,
                    normalized_value=value,
                    confidence=1.0,
                    source_id=source_id,
                    source_type="host_telemetry",
                    role="interoceptive",
                )
                for channel, value in values.items()
            ),
            metadata={
                "cpu_percent": float(reading.cpu_percent),
                "memory_mb": float(reading.memory_mb),
            },
        )
        return self._remember(packet)

    def capture_narrative_episode(
        self,
        episode: NarrativeEpisode,
        *,
        cycle: int,
    ) -> PerceptionPacket:
        tags = list(episode.tags)
        packet = PerceptionPacket(
            packet_id=self._next_packet_id(),
            cycle=cycle,
            source_type="narrative_event",
            source_id=str(episode.episode_id),
            adapter_name="NarrativeEpisodeAdapter",
            confidence=1.0 if tags else 0.75,
            observation={},
            signals=(),
            metadata={
                "timestamp": int(episode.timestamp),
                "source": str(episode.source),
                "tags": tags,
                "raw_text": str(episode.raw_text),
            },
        )
        return self._remember(packet)

    def to_dict(self) -> dict[str, object]:
        return {
            "packets_seen": self.packets_seen,
            "source_counts": dict(self.source_counts),
            "last_packet_id": self.last_packet_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PerceptionBus":
        if not payload:
            return cls()
        source_counts = payload.get("source_counts", {})
        return cls(
            packets_seen=int(payload.get("packets_seen", 0)),
            source_counts=(
                {str(key): int(value) for key, value in source_counts.items()}
                if isinstance(source_counts, Mapping)
                else {}
            ),
            last_packet_id=str(payload.get("last_packet_id", "")),
        )


@dataclass
class ActionBus:
    dispatch_count: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    acknowledged_effects: int = 0
    last_dispatch_id: str = ""
    last_ack_id: str = ""

    def dispatch_to_simulated_world(
        self,
        world,
        action: ActionSchema | str | Mapping[str, object],
        *,
        cycle: int,
        source_id: str = "simulated_world",
    ) -> ActionDispatchRecord:
        action_schema = ensure_action_schema(action)
        feedback = _coerce_float_mapping(world.apply_action(action_schema))
        acknowledged_channels = tuple(sorted(feedback))
        next_dispatch_index = self.dispatch_count + 1
        ack = ActionEffectAck(
            ack_id=f"ack-{next_dispatch_index:06d}",
            cycle=cycle,
            source_type="simulated_world",
            source_id=source_id,
            action=action_schema.to_dict(),
            action_name=action_name(action_schema),
            success=True,
            confidence=1.0,
            feedback=feedback,
            acknowledged_channels=acknowledged_channels,
            metadata={
                "non_zero_feedback_count": sum(
                    1 for value in feedback.values() if value != 0.0
                )
            },
        )
        record = ActionDispatchRecord(
            dispatch_id=f"dispatch-{next_dispatch_index:06d}",
            cycle=cycle,
            source_type="simulated_world",
            source_id=source_id,
            adapter_name="SimulatedWorldActionAdapter",
            action=action_schema.to_dict(),
            action_name=action_schema.name,
            feedback=feedback,
            acknowledgment=ack,
        )
        self.dispatch_count += 1
        self.acknowledged_effects += len(acknowledged_channels)
        self.source_counts["simulated_world"] = self.source_counts.get("simulated_world", 0) + 1
        self.last_dispatch_id = record.dispatch_id
        self.last_ack_id = ack.ack_id
        return record

    def dispatch_to_external_adapter(
        self,
        adapter,
        action: ActionSchema | str | Mapping[str, object],
        *,
        cycle: int,
        source_type: str,
        source_id: str,
    ) -> ActionDispatchRecord:
        action_schema = ensure_action_schema(action)
        outcome = adapter.execute(action_schema, cycle=cycle)
        if not isinstance(outcome, Mapping):
            raise TypeError("external adapter outcome must be a mapping")
        success = bool(outcome.get("success", True))
        feedback = _coerce_float_mapping(outcome.get("feedback"))
        acknowledged_channels = tuple(sorted(feedback))
        next_dispatch_index = self.dispatch_count + 1
        ack = ActionEffectAck(
            ack_id=f"ack-{next_dispatch_index:06d}",
            cycle=cycle,
            source_type=source_type,
            source_id=source_id,
            action=action_schema.to_dict(),
            action_name=action_name(action_schema),
            success=success,
            confidence=1.0 if success else 0.35,
            feedback=feedback,
            acknowledged_channels=acknowledged_channels,
            metadata=(
                dict(outcome.get("metadata"))
                if isinstance(outcome.get("metadata"), Mapping)
                else {}
            ),
        )
        record = ActionDispatchRecord(
            dispatch_id=f"dispatch-{next_dispatch_index:06d}",
            cycle=cycle,
            source_type=source_type,
            source_id=source_id,
            adapter_name=str(getattr(adapter, "adapter_name", type(adapter).__name__)),
            action=action_schema.to_dict(),
            action_name=action_schema.name,
            feedback=feedback,
            acknowledgment=ack,
        )
        self.dispatch_count += 1
        self.acknowledged_effects += len(acknowledged_channels)
        self.source_counts[source_type] = self.source_counts.get(source_type, 0) + 1
        self.last_dispatch_id = record.dispatch_id
        self.last_ack_id = ack.ack_id
        return record

    def to_dict(self) -> dict[str, object]:
        return {
            "dispatch_count": self.dispatch_count,
            "source_counts": dict(self.source_counts),
            "acknowledged_effects": self.acknowledged_effects,
            "last_dispatch_id": self.last_dispatch_id,
            "last_ack_id": self.last_ack_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ActionBus":
        if not payload:
            return cls()
        source_counts = payload.get("source_counts", {})
        return cls(
            dispatch_count=int(payload.get("dispatch_count", 0)),
            source_counts=(
                {str(key): int(value) for key, value in source_counts.items()}
                if isinstance(source_counts, Mapping)
                else {}
            ),
            acknowledged_effects=int(payload.get("acknowledged_effects", 0)),
            last_dispatch_id=str(payload.get("last_dispatch_id", "")),
            last_ack_id=str(payload.get("last_ack_id", "")),
        )
