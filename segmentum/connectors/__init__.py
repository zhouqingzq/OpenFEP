"""External platform connector contracts and shared runtime."""

from .contracts import (
    CONNECTOR_CONTRACT_VERSION,
    ConnectorAdapter,
    ConnectorCapabilities,
    ConnectorDeliveryReceipt,
    ConnectorDeliveryTarget,
    NormalizedConnectorInput,
    connector_participant_id,
    connector_session_id,
)
from .runtime import ConnectorDeliveryTargetStore, ConnectorRuntime

__all__ = [
    "ConnectorAdapter",
    "CONNECTOR_CONTRACT_VERSION",
    "ConnectorCapabilities",
    "ConnectorDeliveryReceipt",
    "ConnectorDeliveryTarget",
    "ConnectorDeliveryTargetStore",
    "ConnectorRuntime",
    "NormalizedConnectorInput",
    "connector_participant_id",
    "connector_session_id",
]
