"""Telegram polling connector for Path B / M16 sessions.

This module keeps platform-specific parsing and delivery outside the cognition
runtime. Telegram updates are normalized into M16 client inputs, routed into one
session per conversation surface, and replies are sent back through the Bot API.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from segmentum.dialogue.runtime.m16_api import M16Gateway, M16SessionHandle


TELEGRAM_PLATFORM = "telegram"
DEFAULT_ACCOUNT_SCOPE = "bot_01"
DEFAULT_ALLOWED_UPDATES = ("message", "edited_message")
TARGET_STORE_FILE = "telegram_delivery_targets.jsonl"


def _now(clock: Any | None = None) -> int:
    if clock is None:
        return int(time.time())
    value = clock() if callable(clock) else clock
    return int(value)


def _mapping(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _telegram_slice(text: str, offset: int, length: int) -> str:
    if not text:
        return ""
    try:
        encoded = text.encode("utf-16-le")
        start = max(0, int(offset)) * 2
        end = max(int(offset), int(offset) + int(length)) * 2
        return encoded[start:end].decode("utf-16-le", errors="ignore")
    except Exception:
        return ""


def _display_name(user: Mapping[str, Any]) -> str:
    first = str(user.get("first_name", "") or "").strip()
    last = str(user.get("last_name", "") or "").strip()
    full = " ".join(part for part in (first, last) if part).strip()
    if full:
        return full[:64]
    username = str(user.get("username", "") or "").strip()
    if username:
        return username[:64]
    return f"user_{str(user.get('id', '') or '').strip()[:32]}"


def telegram_assistant_participant_id(account_scope: str, bot_user_id: str) -> str:
    return f"{TELEGRAM_PLATFORM}:{account_scope}:assistant:{bot_user_id}"


def telegram_user_participant_id(account_scope: str, platform_user_id: str) -> str:
    return f"{TELEGRAM_PLATFORM}:{account_scope}:user:{platform_user_id}"


def telegram_surface_kind(chat_type: str, message_thread_id: int | None = None) -> str:
    kind = str(chat_type or "").strip().lower()
    if kind == "private":
        return "dm"
    if message_thread_id is not None:
        return "topic"
    if kind in {"group", "supergroup"}:
        return "group"
    if kind == "channel":
        return "channel"
    return "group"


def telegram_session_id(
    *,
    account_scope: str,
    chat_id: str,
    chat_type: str,
    platform_user_id: str,
    message_thread_id: int | None = None,
) -> str:
    surface_kind = telegram_surface_kind(chat_type, message_thread_id)
    if surface_kind == "dm":
        return f"{TELEGRAM_PLATFORM}:{account_scope}:dm:user_{platform_user_id}"
    if surface_kind == "topic":
        return f"{TELEGRAM_PLATFORM}:{account_scope}:topic:chat_{chat_id}:thread_{int(message_thread_id or 0)}"
    return f"{TELEGRAM_PLATFORM}:{account_scope}:{surface_kind}:chat_{chat_id}"


def telegram_turn_id(
    *,
    account_scope: str,
    chat_id: str,
    chat_type: str,
    platform_user_id: str,
    message_id: str,
    message_thread_id: int | None = None,
) -> str:
    session = telegram_session_id(
        account_scope=account_scope,
        chat_id=chat_id,
        chat_type=chat_type,
        platform_user_id=platform_user_id,
        message_thread_id=message_thread_id,
    )
    return f"{session}:msg:{message_id}"


def telegram_delivery_target_from_session_id(session_id: str) -> TelegramDeliveryTarget | None:
    value = str(session_id or "").strip()
    if not value.startswith(f"{TELEGRAM_PLATFORM}:"):
        return None
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if len(parts) < 4:
        return None
    account_scope = parts[1]
    surface_kind = parts[2]
    if surface_kind == "dm" and len(parts) >= 4 and parts[3].startswith("user_"):
        return TelegramDeliveryTarget(
            chat_id=parts[3][5:],
            surface_kind=surface_kind,
            surface_id=parts[3],
            account_scope=account_scope,
        )
    if surface_kind == "group" and len(parts) >= 4 and parts[3].startswith("chat_"):
        return TelegramDeliveryTarget(
            chat_id=parts[3][5:],
            surface_kind=surface_kind,
            surface_id=parts[3],
            account_scope=account_scope,
        )
    if surface_kind == "topic" and len(parts) >= 5 and parts[3].startswith("chat_") and parts[4].startswith("thread_"):
        try:
            thread_id = int(parts[4][7:])
        except ValueError:
            return None
        return TelegramDeliveryTarget(
            chat_id=parts[3][5:],
            surface_kind=surface_kind,
            surface_id=f"{parts[3]}:{parts[4]}",
            account_scope=account_scope,
            message_thread_id=thread_id,
        )
    return None


@dataclass(frozen=True)
class TelegramBotIdentity:
    bot_user_id: str
    username: str
    account_scope: str = DEFAULT_ACCOUNT_SCOPE

    @property
    def assistant_participant_id(self) -> str:
        return telegram_assistant_participant_id(self.account_scope, self.bot_user_id)


@dataclass(frozen=True)
class TelegramDeliveryTarget:
    chat_id: str
    surface_kind: str
    surface_id: str
    account_scope: str
    reply_to_message_id: int | None = None
    message_thread_id: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "platform": TELEGRAM_PLATFORM,
            "account_scope": self.account_scope,
            "surface_kind": self.surface_kind,
            "surface_id": self.surface_id,
            "chat_id": self.chat_id,
        }
        if self.reply_to_message_id is not None:
            payload["reply_to_message_id"] = int(self.reply_to_message_id)
        if self.message_thread_id is not None:
            payload["message_thread_id"] = int(self.message_thread_id)
        return payload


@dataclass(frozen=True)
class TelegramNormalizedInput:
    persona_id: str
    session_id: str
    correlation_id: str
    text: str
    speaker_name: str
    group_turn_envelope: dict[str, Any]
    delivery_target: TelegramDeliveryTarget
    update_id: int
    platform_message_id: str
    ingress_evidence_band: str


class TelegramApiError(RuntimeError):
    pass


class TelegramBotApiClient:
    def __init__(self, *, token: str, base_url: str = "https://api.telegram.org", timeout_seconds: int = 45) -> None:
        self.token = str(token or "").strip()
        self.base_url = str(base_url or "https://api.telegram.org").rstrip("/")
        self.timeout_seconds = max(5, int(timeout_seconds))
        if not self.token:
            raise ValueError("telegram bot token is required")

    def _call(self, method: str, payload: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/bot{self.token}/{method}"
        body = json.dumps(dict(payload or {}), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise TelegramApiError(f"{method} http_error {exc.code}: {detail[:240]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise TelegramApiError(f"{method} url_error: {exc.reason}") from exc
        if not isinstance(data, Mapping) or not bool(data.get("ok")):
            description = ""
            if isinstance(data, Mapping):
                description = str(data.get("description", "") or "")
            raise TelegramApiError(f"{method} failed: {description or 'unknown_error'}")
        return data.get("result")

    def get_me(self) -> Mapping[str, Any]:
        raw = self._call("getMe")
        return _mapping(raw)

    def get_updates(
        self,
        *,
        offset: int | None = None,
        timeout_seconds: int = 30,
        allowed_updates: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": max(0, int(timeout_seconds))}
        if offset is not None:
            payload["offset"] = int(offset)
        if allowed_updates:
            payload["allowed_updates"] = [str(item) for item in allowed_updates]
        raw = self._call("getUpdates", payload)
        if not isinstance(raw, list):
            return []
        return [dict(item) for item in raw if isinstance(item, Mapping)]

    def send_message(
        self,
        *,
        chat_id: str,
        text: str,
        message_thread_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": str(text or ""),
        }
        if message_thread_id is not None:
            payload["message_thread_id"] = int(message_thread_id)
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": int(reply_to_message_id)}
        raw = self._call("sendMessage", payload)
        return _mapping(raw)


@dataclass
class TelegramDeliveryTargetStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.root / TARGET_STORE_FILE

    def record(
        self,
        *,
        event_id: str,
        correlation_id: str,
        target: TelegramDeliveryTarget,
        now: int | None = None,
    ) -> None:
        _append_jsonl(
            self.path,
            {
                "record_type": "target",
                "event_id": str(event_id),
                "correlation_id": str(correlation_id or ""),
                "status": "pending",
                "at": int(now if now is not None else time.time()),
                "target": target.to_payload(),
            },
        )

    def load(self, event_id: str) -> TelegramDeliveryTarget | None:
        needle = str(event_id or "").strip()
        if not needle:
            return None
        latest_target: dict[str, Any] | None = None
        delivered = False
        for row in reversed(_read_jsonl(self.path)):
            if str(row.get("event_id", "") or "") != needle:
                continue
            if str(row.get("record_type", "") or "") == "delivered":
                delivered = True
                break
            if str(row.get("record_type", "") or "") == "target" and latest_target is None:
                latest_target = _mapping(row.get("target"))
        if delivered or latest_target is None:
            return None
        return TelegramDeliveryTarget(
            chat_id=str(latest_target.get("chat_id", "") or ""),
            surface_kind=str(latest_target.get("surface_kind", "") or ""),
            surface_id=str(latest_target.get("surface_id", "") or ""),
            account_scope=str(latest_target.get("account_scope", "") or DEFAULT_ACCOUNT_SCOPE),
            reply_to_message_id=(
                int(latest_target["reply_to_message_id"])
                if latest_target.get("reply_to_message_id") is not None
                else None
            ),
            message_thread_id=(
                int(latest_target["message_thread_id"])
                if latest_target.get("message_thread_id") is not None
                else None
            ),
        )

    def mark_delivered(
        self,
        *,
        event_id: str,
        telegram_message_id: str = "",
        now: int | None = None,
    ) -> None:
        _append_jsonl(
            self.path,
            {
                "record_type": "delivered",
                "event_id": str(event_id),
                "telegram_message_id": str(telegram_message_id or ""),
                "at": int(now if now is not None else time.time()),
            },
        )


def _bot_identity_from_me(raw: Mapping[str, Any], *, account_scope: str) -> TelegramBotIdentity:
    return TelegramBotIdentity(
        bot_user_id=str(raw.get("id", "") or "").strip(),
        username=str(raw.get("username", "") or "").lstrip("@").strip(),
        account_scope=account_scope,
    )


def _extract_message_text(message: Mapping[str, Any]) -> str:
    text = str(message.get("text", "") or "")
    if text:
        return text[:8000]
    caption = str(message.get("caption", "") or "")
    return caption[:8000]


def _collect_entity_mentions(
    *,
    message: Mapping[str, Any],
    text: str,
    bot: TelegramBotIdentity,
    account_scope: str,
) -> tuple[list[str], list[str], bool]:
    explicit_mentions: list[str] = []
    mentioned_participant_ids: list[str] = []
    seen_mentions: set[str] = set()
    seen_participants: set[str] = set()
    addressed_assistant = False
    entities = list(message.get("entities") or []) + list(message.get("caption_entities") or [])
    for raw_entity in entities:
        entity = _mapping(raw_entity)
        entity_type = str(entity.get("type", "") or "").strip().lower()
        if entity_type not in {"mention", "text_mention"}:
            continue
        if entity_type == "mention":
            mention_text = _telegram_slice(
                text,
                int(entity.get("offset", 0) or 0),
                int(entity.get("length", 0) or 0),
            ).strip()
            if mention_text and mention_text not in seen_mentions:
                seen_mentions.add(mention_text)
                explicit_mentions.append(mention_text[:64])
            if mention_text.lstrip("@").casefold() == bot.username.casefold():
                addressed_assistant = True
            continue
        user = _mapping(entity.get("user"))
        user_id = str(user.get("id", "") or "").strip()
        if not user_id:
            continue
        participant_id = telegram_user_participant_id(account_scope, user_id)
        if participant_id not in seen_participants:
            seen_participants.add(participant_id)
            mentioned_participant_ids.append(participant_id)
        mention_name = _display_name(user)
        if mention_name and mention_name not in seen_mentions:
            seen_mentions.add(mention_name)
            explicit_mentions.append(mention_name[:64])
        if user_id == bot.bot_user_id:
            addressed_assistant = True
    return explicit_mentions[:8], mentioned_participant_ids[:8], addressed_assistant


def normalize_telegram_update(
    update: Mapping[str, Any],
    *,
    persona_id: str,
    bot: TelegramBotIdentity,
) -> TelegramNormalizedInput | None:
    message = _mapping(update.get("message") or update.get("edited_message"))
    if not message:
        return None
    text = _extract_message_text(message).strip()
    if not text:
        return None
    chat = _mapping(message.get("chat"))
    user = _mapping(message.get("from"))
    chat_id = str(chat.get("id", "") or "").strip()
    platform_user_id = str(user.get("id", "") or "").strip()
    if not chat_id or not platform_user_id:
        return None
    message_id = str(message.get("message_id", "") or "").strip()
    if not message_id:
        return None
    message_thread_id = message.get("message_thread_id")
    thread_id = int(message_thread_id) if message_thread_id is not None else None
    chat_type = str(chat.get("type", "") or "").strip().lower()
    surface_kind = telegram_surface_kind(chat_type, thread_id)
    surface_id = f"chat_{chat_id}"
    speaker_name = _display_name(user)
    speaker_participant_id = telegram_user_participant_id(bot.account_scope, platform_user_id)
    session_id = telegram_session_id(
        account_scope=bot.account_scope,
        chat_id=chat_id,
        chat_type=chat_type,
        platform_user_id=platform_user_id,
        message_thread_id=thread_id,
    )
    explicit_mentions, mentioned_participant_ids, addressed_by_mention = _collect_entity_mentions(
        message=message,
        text=text,
        bot=bot,
        account_scope=bot.account_scope,
    )
    visible_participant_ids: list[str] = [speaker_participant_id, bot.assistant_participant_id]
    seen_visible = set(visible_participant_ids)
    reply_to_turn_id = ""
    addressed_participant_ids: list[str] = []
    reply_to_message = _mapping(message.get("reply_to_message"))
    if reply_to_message:
        reply_from = _mapping(reply_to_message.get("from"))
        reply_from_id = str(reply_from.get("id", "") or "").strip()
        if reply_from_id:
            target_id = (
                bot.assistant_participant_id
                if reply_from_id == bot.bot_user_id
                else telegram_user_participant_id(bot.account_scope, reply_from_id)
            )
            if target_id not in seen_visible:
                seen_visible.add(target_id)
                visible_participant_ids.append(target_id)
        reply_to_message_id = str(reply_to_message.get("message_id", "") or "").strip()
        if reply_to_message_id:
            reply_to_turn_id = telegram_turn_id(
                account_scope=bot.account_scope,
                chat_id=chat_id,
                chat_type=chat_type,
                platform_user_id=platform_user_id,
                message_id=reply_to_message_id,
                message_thread_id=thread_id,
            )
        if reply_from_id == bot.bot_user_id:
            addressed_participant_ids = [bot.assistant_participant_id]
    if not addressed_participant_ids and (chat_type == "private" or addressed_by_mention):
        addressed_participant_ids = [bot.assistant_participant_id]
    for participant_id in mentioned_participant_ids:
        if participant_id not in seen_visible:
            seen_visible.add(participant_id)
            visible_participant_ids.append(participant_id)
    visible_participant_ids = visible_participant_ids[:8]
    envelope: dict[str, Any] = {
        "speaker_participant_id": speaker_participant_id,
        "visible_participant_ids": visible_participant_ids,
    }
    if addressed_participant_ids:
        envelope["addressed_participant_ids"] = addressed_participant_ids[:8]
    if mentioned_participant_ids:
        envelope["mentioned_participant_ids"] = mentioned_participant_ids[:8]
    if reply_to_turn_id:
        envelope["reply_to_turn_id"] = reply_to_turn_id
        envelope["quoted_turn_ids"] = [reply_to_turn_id]
    if explicit_mentions:
        envelope["explicit_mentions"] = explicit_mentions[:8]
    if chat_type == "private":
        ingress_evidence_band = "structured_full"
    elif reply_to_turn_id and addressed_participant_ids:
        ingress_evidence_band = "structured_partial"
    elif reply_to_turn_id:
        ingress_evidence_band = "reply_chain_only"
    elif addressed_participant_ids or explicit_mentions or mentioned_participant_ids:
        ingress_evidence_band = "structured_partial"
    else:
        ingress_evidence_band = "speaker_name_only"
    delivery_target = TelegramDeliveryTarget(
        chat_id=chat_id,
        surface_kind=surface_kind,
        surface_id=surface_id,
        account_scope=bot.account_scope,
        reply_to_message_id=int(message_id),
        message_thread_id=thread_id,
    )
    update_id = int(update.get("update_id", 0) or 0)
    correlation_id = f"tg:{bot.account_scope}:{update_id}:{message_id}"
    return TelegramNormalizedInput(
        persona_id=persona_id,
        session_id=session_id,
        correlation_id=correlation_id[:120],
        text=text,
        speaker_name=speaker_name,
        group_turn_envelope=envelope,
        delivery_target=delivery_target,
        update_id=update_id,
        platform_message_id=message_id,
        ingress_evidence_band=ingress_evidence_band,
    )


class TelegramConnector:
    def __init__(
        self,
        *,
        persona_id: str,
        account_scope: str = DEFAULT_ACCOUNT_SCOPE,
        gateway: M16Gateway | None = None,
        api_client: TelegramBotApiClient | Any,
        clock: Any | None = None,
    ) -> None:
        self.persona_id = str(persona_id or "").strip() or "default"
        self.account_scope = str(account_scope or DEFAULT_ACCOUNT_SCOPE).strip() or DEFAULT_ACCOUNT_SCOPE
        self.gateway = gateway or M16Gateway(clock=clock)
        self.api = api_client
        self.clock = clock
        self._bot_identity: TelegramBotIdentity | None = None

    def bot_identity(self) -> TelegramBotIdentity:
        if self._bot_identity is None:
            self._bot_identity = _bot_identity_from_me(self.api.get_me(), account_scope=self.account_scope)
        return self._bot_identity

    def normalize_update(self, update: Mapping[str, Any]) -> TelegramNormalizedInput | None:
        return normalize_telegram_update(update, persona_id=self.persona_id, bot=self.bot_identity())

    def ingest_update(self, update: Mapping[str, Any], *, max_cycles: int = 4) -> dict[str, Any]:
        normalized = self.normalize_update(update)
        if normalized is None:
            return {"accepted": False, "ignored": "unsupported_update"}
        handle = self.gateway.get_or_create_session(normalized.persona_id, normalized.session_id)
        target_store = TelegramDeliveryTargetStore(handle.session_root)
        event_id = handle.bridge.append_client_input(
            text=normalized.text,
            correlation_id=normalized.correlation_id,
            source="telegram_connector",
            speaker_name=normalized.speaker_name,
            group_turn_envelope=normalized.group_turn_envelope,
            ingress_evidence_band=normalized.ingress_evidence_band,
        )
        target_store.record(
            event_id=event_id,
            correlation_id=normalized.correlation_id,
            target=normalized.delivery_target,
            now=_now(self.clock),
        )
        runner = self.gateway.ensure_runner(handle)
        processed_rows: list[dict[str, Any]] = []
        sent_messages: list[dict[str, Any]] = []
        for _ in range(max(1, int(max_cycles))):
            step = runner.run_once(now=_now(self.clock), max_steps=1)
            processed_rows.extend(dict(row) for row in step.processed if isinstance(row, Mapping))
            sent_messages.extend(self._deliver_processed_rows(handle.session_root, step.processed))
            if handle.bridge.is_event_processed(event_id):
                break
        return {
            "accepted": True,
            "event_id": event_id,
            "persona_id": normalized.persona_id,
            "session_id": normalized.session_id,
            "correlation_id": normalized.correlation_id,
            "ingress_evidence_band": normalized.ingress_evidence_band,
            "processed": processed_rows,
            "sent_messages": sent_messages,
        }

    def _deliver_processed_rows(self, session_root: Path, processed_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        target_store = TelegramDeliveryTargetStore(session_root)
        sent: list[dict[str, Any]] = []
        for row in processed_rows:
            event_id = str(row.get("event_id", "") or "").strip()
            reply = str(row.get("reply", "") or "").strip()
            if not event_id or not reply:
                continue
            target = target_store.load(event_id)
            if target is None:
                continue
            message = self.api.send_message(
                chat_id=target.chat_id,
                text=reply,
                message_thread_id=target.message_thread_id,
                reply_to_message_id=target.reply_to_message_id,
            )
            target_store.mark_delivered(
                event_id=event_id,
                telegram_message_id=str(message.get("message_id", "") or ""),
                now=_now(self.clock),
            )
            sent.append({"event_id": event_id, "chat_id": target.chat_id, "message_id": message.get("message_id", "")})
        return sent

    def _telegram_handles(self) -> list[M16SessionHandle]:
        prefix = f"{TELEGRAM_PLATFORM}:{self.account_scope}:"
        return [
            handle
            for handle in self.gateway.sessions.values()
            if handle.persona_id == self.persona_id and handle.session_id.startswith(prefix)
        ]

    def drain_proactive_once(self, *, max_sessions: int = 8) -> dict[str, Any]:
        return {
            "sessions_considered": min(len(self._telegram_handles()), max(0, int(max_sessions))),
            "sent_messages": [],
            "results": [],
        }

    def poll_once(
        self,
        *,
        offset: int | None = None,
        timeout_seconds: int = 30,
        allowed_updates: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        updates = self.api.get_updates(
            offset=offset,
            timeout_seconds=timeout_seconds,
            allowed_updates=allowed_updates or list(DEFAULT_ALLOWED_UPDATES),
        )
        next_offset = offset
        results: list[dict[str, Any]] = []
        for update in updates:
            update_id = int(update.get("update_id", 0) or 0)
            next_offset = max(int(next_offset or 0), update_id + 1)
            results.append(self.ingest_update(update))
        return {"updates": len(updates), "next_offset": next_offset, "results": results}

    def serve_forever(
        self,
        *,
        offset_path: Path | None = None,
        timeout_seconds: int = 30,
        allowed_updates: list[str] | tuple[str, ...] | None = None,
        idle_sleep_seconds: float = 1.0,
    ) -> None:
        offset_file = Path(offset_path).resolve() if offset_path else None
        next_offset = _read_offset(offset_file) if offset_file else None
        while True:
            batch = self.poll_once(
                offset=next_offset,
                timeout_seconds=timeout_seconds,
                allowed_updates=allowed_updates,
            )
            next_offset = batch.get("next_offset")
            if offset_file is not None and next_offset is not None:
                _write_offset(offset_file, int(next_offset))
            if int(batch.get("updates", 0) or 0) <= 0:
                time.sleep(max(0.1, float(idle_sleep_seconds)))


def _read_offset(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, Mapping):
        value = raw.get("next_offset")
    else:
        value = raw
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_offset(path: Path, next_offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"next_offset": int(next_offset)}, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telegram_connector", description="Telegram polling connector for M16 sessions")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Long-poll Telegram and route into M16 sessions")
    serve.add_argument("--token", required=True)
    serve.add_argument("--persona", required=True)
    serve.add_argument("--account-scope", default=DEFAULT_ACCOUNT_SCOPE)
    serve.add_argument("--offset-file", default="")
    serve.add_argument("--poll-timeout", type=int, default=25)
    serve.add_argument("--idle-sleep", type=float, default=0.5)
    serve.set_defaults(action="serve")

    poll = sub.add_parser("poll-once", help="Run one Telegram polling batch")
    poll.add_argument("--token", required=True)
    poll.add_argument("--persona", required=True)
    poll.add_argument("--account-scope", default=DEFAULT_ACCOUNT_SCOPE)
    poll.add_argument("--offset", type=int, default=0)
    poll.add_argument("--poll-timeout", type=int, default=0)
    poll.set_defaults(action="poll_once")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    client = TelegramBotApiClient(token=args.token)
    connector = TelegramConnector(
        persona_id=args.persona,
        account_scope=args.account_scope,
        api_client=client,
    )
    if args.action == "poll_once":
        result = connector.poll_once(
            offset=(None if int(args.offset) <= 0 else int(args.offset)),
            timeout_seconds=max(0, int(args.poll_timeout)),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    connector.serve_forever(
        offset_path=(Path(args.offset_file) if str(args.offset_file or "").strip() else None),
        timeout_seconds=max(1, int(args.poll_timeout)),
        idle_sleep_seconds=max(0.1, float(args.idle_sleep)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
