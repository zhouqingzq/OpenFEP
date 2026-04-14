from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .observer import DialogueObserver
from .types import TranscriptUtterance


def _parse_ts(value: object) -> datetime:
    text = str(value or "")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.min


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(slots=True)
class DialogueTurn:
    session_id: str
    partner_uid: int
    speaker_uid: int
    body: str
    timestamp: datetime
    turn_index: int
    session_turn_count: int
    #: Last user (self) utterance in this session before this partner message; not six-channel encoded.
    prior_self_body: str = ""


class DialogueWorld:
    """Replay one user's chat history as dialogue observations."""

    def __init__(
        self,
        user_dataset: Mapping[str, object],
        observer: DialogueObserver,
        *,
        seed: int = 42,
        world_id: str = "dialogue",
    ) -> None:
        self._decision_master_seed = int(seed)
        self.world_id = world_id
        self.user_uid = _parse_int(user_dataset.get("uid", 0), 0)
        self.observer = observer
        self._session_contexts: dict[str, dict[str, object]] = {}
        self._turns = self._build_turns(user_dataset)
        self._cursor = 0
        self._session_boundary = False
        self._last_partner_uid: int | None = None
        self._last_turn: DialogueTurn | None = None
        self._history_by_partner: dict[int, list[TranscriptUtterance]] = {}

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._turns)

    @property
    def session_boundary(self) -> bool:
        return self._session_boundary

    @property
    def current_turn(self) -> dict[str, object]:
        turn = self._active_turn()
        if turn is None:
            return {}
        return {
            "session_id": turn.session_id,
            "partner_uid": turn.partner_uid,
            "speaker_uid": turn.speaker_uid,
            "body": turn.body,
            "prior_self_body": turn.prior_self_body,
            "timestamp": turn.timestamp.isoformat(),
            "turn_index": turn.turn_index,
            "session_turn_count": turn.session_turn_count,
        }

    def observe(self) -> dict[str, float]:
        turn = self._active_turn()
        if turn is None:
            return {}
        history = list(self._history_by_partner.get(turn.partner_uid, []))
        session_context = dict(self._session_contexts.get(turn.session_id, {}))
        obs = self.observer.observe(
            current_turn=turn.body,
            conversation_history=history,
            partner_uid=turn.partner_uid,
            session_context=session_context,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            speaker_uid=turn.speaker_uid,
            timestamp=turn.timestamp,
        )
        return dict(obs.channels)

    def advance(self) -> bool:
        turn = self._active_turn()
        if turn is None:
            self._session_boundary = False
            return False
        history = self._history_by_partner.setdefault(turn.partner_uid, [])
        history.append(TranscriptUtterance(role="interlocutor", text=turn.body))
        history[:] = history[-80:]
        self._last_turn = turn
        self._cursor += 1
        next_turn = self._active_turn()
        self._session_boundary = bool(next_turn and next_turn.session_id != turn.session_id)
        self._last_partner_uid = turn.partner_uid
        return not self.exhausted

    def partner_switched(self) -> bool:
        turn = self._active_turn()
        if turn is None or self._last_partner_uid is None:
            return False
        return turn.partner_uid != self._last_partner_uid

    def _active_turn(self) -> DialogueTurn | None:
        if self.exhausted:
            return None
        return self._turns[self._cursor]

    def _build_turns(self, payload: Mapping[str, object]) -> list[DialogueTurn]:
        sessions = payload.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        rows: list[DialogueTurn] = []
        for session in sessions:
            if not isinstance(session, Mapping):
                continue
            session_id = str(session.get("session_id", ""))
            uid_a = _parse_int(session.get("uid_a", self.user_uid), self.user_uid)
            uid_b = _parse_int(session.get("uid_b", self.user_uid), self.user_uid)
            partner_uid = uid_b if uid_a == self.user_uid else uid_a
            turns = session.get("turns", [])
            if not isinstance(turns, list):
                continue
            indexed_turns = [(idx, item) for idx, item in enumerate(turns) if isinstance(item, Mapping)]
            indexed_turns.sort(key=lambda pair: (_parse_ts(pair[1].get("timestamp")), pair[0]))
            last_self_body = ""
            inbound_index = 0
            for _, turn in indexed_turns:
                sender_uid = _parse_int(turn.get("sender_uid", partner_uid), partner_uid)
                body = str(turn.get("body", ""))
                if sender_uid == self.user_uid:
                    last_self_body = body
                    continue
                rows.append(
                    DialogueTurn(
                        session_id=session_id,
                        partner_uid=sender_uid,
                        speaker_uid=sender_uid,
                        body=body,
                        timestamp=_parse_ts(turn.get("timestamp")),
                        turn_index=inbound_index,
                        session_turn_count=len(turns),
                        prior_self_body=last_self_body,
                    )
                )
                inbound_index += 1
            self._session_contexts[session_id] = {
                "session_id": session_id,
                "partner_uid": partner_uid,
                "turn_count": int(session.get("metadata", {}).get("turn_count", len(turns)))
                if isinstance(session.get("metadata"), Mapping)
                else len(turns),
            }
        rows.sort(key=lambda item: (item.timestamp, item.session_id, item.turn_index))
        return rows
