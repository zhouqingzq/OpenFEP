from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .parser import ChatMessage


@dataclass
class ConversationSession:
    session_id: str
    uid_a: int
    uid_b: int
    start_time: datetime
    end_time: datetime
    turns: list[ChatMessage]
    metadata: dict[str, object]


def _build_session(
    uid_a: int,
    uid_b: int,
    turns: list[ChatMessage],
    sequence_id: int,
) -> ConversationSession:
    start_time = turns[0].timestamp
    end_time = turns[-1].timestamp
    turn_count = len(turns)
    duration_seconds = int((end_time - start_time).total_seconds())
    count_a = sum(1 for t in turns if t.sender_uid == uid_a)
    count_b = sum(1 for t in turns if t.sender_uid == uid_b)
    ratio = float(count_a) / float(count_b) if count_b else float(count_a)
    session_id = f"{uid_a}_{uid_b}_{start_time.strftime('%Y%m%d%H%M%S')}_{sequence_id:04d}"
    metadata: dict[str, object] = {
        "turn_count": turn_count,
        "duration_seconds": duration_seconds,
        "message_count_uid_a": count_a,
        "message_count_uid_b": count_b,
        "message_ratio_uid_a_to_uid_b": ratio,
    }
    return ConversationSession(
        session_id=session_id,
        uid_a=uid_a,
        uid_b=uid_b,
        start_time=start_time,
        end_time=end_time,
        turns=turns,
        metadata=metadata,
    )


def build_sessions(
    messages: Iterable[ChatMessage],
    *,
    gap_threshold_minutes: int = 30,
) -> dict[tuple[int, int], list[ConversationSession]]:
    grouped: dict[tuple[int, int], list[tuple[int, ChatMessage]]] = defaultdict(list)
    for index, msg in enumerate(messages):
        uid_a = min(msg.sender_uid, msg.receiver_uid)
        uid_b = max(msg.sender_uid, msg.receiver_uid)
        grouped[(uid_a, uid_b)].append((index, msg))

    threshold = timedelta(minutes=gap_threshold_minutes)
    sessions: dict[tuple[int, int], list[ConversationSession]] = {}
    for pair in sorted(grouped):
        ordered_messages = [m for _, m in sorted(grouped[pair], key=lambda x: (x[1].timestamp, x[0]))]
        pair_sessions: list[ConversationSession] = []
        current_turns: list[ChatMessage] = []
        sequence_id = 0
        for msg in ordered_messages:
            if not current_turns:
                current_turns = [msg]
                continue
            if msg.timestamp - current_turns[-1].timestamp > threshold:
                pair_sessions.append(_build_session(pair[0], pair[1], current_turns, sequence_id))
                sequence_id += 1
                current_turns = [msg]
            else:
                current_turns.append(msg)
        if current_turns:
            pair_sessions.append(_build_session(pair[0], pair[1], current_turns, sequence_id))
        sessions[pair] = pair_sessions
    return sessions
