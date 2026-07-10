"""Canonical Telegram adapter import path.

The implementation remains at its historical module path for compatibility.
New integrations should import Telegram connector types from this module.
"""

from segmentum.dialogue.runtime.telegram_connector import (
    TelegramAdapter,
    TelegramApiError,
    TelegramBotApiClient,
    TelegramBotIdentity,
    TelegramConnector,
    TelegramDeliveryTarget,
    TelegramDeliveryTargetStore,
    TelegramNormalizedInput,
    normalize_telegram_update,
    telegram_delivery_target_from_payload,
    telegram_delivery_target_from_session_id,
)

__all__ = [
    "TelegramAdapter",
    "TelegramApiError",
    "TelegramBotApiClient",
    "TelegramBotIdentity",
    "TelegramConnector",
    "TelegramDeliveryTarget",
    "TelegramDeliveryTargetStore",
    "TelegramNormalizedInput",
    "normalize_telegram_update",
    "telegram_delivery_target_from_payload",
    "telegram_delivery_target_from_session_id",
]
