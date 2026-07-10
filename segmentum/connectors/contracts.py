"""Platform-neutral connector contracts.

Connectors translate platform events into bounded Segmentum inputs and translate
Segmentum replies back into platform deliveries. Platform-specific SDK objects
must not cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


CONNECTOR_CONTRACT_VERSION = "0.1"


def connector_session_id(
    *,
    platform: str,
    account_scope: str,
    surface_kind: str,
    surface_id: str,
) -> str:
    return f"{platform}:{account_scope}:{surface_kind}:{surface_id}"


def connector_participant_id(
    *,
    platform: str,
    account_scope: str,
    participant_kind: str,
    participant_id: str,
) -> str:
    return f"{platform}:{account_scope}:{participant_kind}:{participant_id}"


@dataclass(frozen=True)
class ConnectorCapabilities:
    """Features exposed by a platform adapter."""

    direct_messages: bool = True
    group_messages: bool = True
    threaded_messages: bool = False
    explicit_mentions: bool = False
    reply_links: bool = False
    message_edits: bool = False
    proactive_delivery: bool = False

    def to_payload(self) -> dict[str, bool]:
        return {
            "direct_messages": self.direct_messages,
            "group_messages": self.group_messages,
            "threaded_messages": self.threaded_messages,
            "explicit_mentions": self.explicit_mentions,
            "reply_links": self.reply_links,
            "message_edits": self.message_edits,
            "proactive_delivery": self.proactive_delivery,
        }


@runtime_checkable
class ConnectorDeliveryTarget(Protocol):
    """A serializable platform delivery address."""

    def to_payload(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class NormalizedConnectorInput:
    """Platform-neutral input accepted by the connector runtime."""

    platform: str
    persona_id: str
    session_id: str
    correlation_id: str
    text: str
    speaker_name: str
    group_turn_envelope: dict[str, Any]
    delivery_target: ConnectorDeliveryTarget
    ingress_evidence_band: str
    platform_event_id: str = ""
    platform_message_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONNECTOR_CONTRACT_VERSION


@dataclass(frozen=True)
class ConnectorDeliveryReceipt:
    """Bounded result returned after a platform delivery succeeds."""

    platform: str
    platform_message_id: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "platform": str(self.platform or ""),
            "message_id": str(self.platform_message_id or ""),
        }
        payload.update(dict(self.target))
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@runtime_checkable
class ConnectorAdapter(Protocol):
    """Minimum implementation required for a new external platform."""

    platform: str
    persona_id: str
    account_scope: str
    target_store_file: str
    capabilities: ConnectorCapabilities

    def normalize_event(self, event: Mapping[str, Any]) -> NormalizedConnectorInput | None:
        ...

    def target_from_payload(self, payload: Mapping[str, Any]) -> ConnectorDeliveryTarget | None:
        ...

    def deliver(self, *, target: ConnectorDeliveryTarget, text: str) -> ConnectorDeliveryReceipt:
        ...
