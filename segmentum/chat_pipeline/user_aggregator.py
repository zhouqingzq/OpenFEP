from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import re

from .session_builder import ConversationSession

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class UserProfile:
    uid: int
    total_messages: int
    active_days: int
    unique_partners: int
    avg_message_length: float
    vocabulary_richness: float
    temporal_pattern: dict[str, float]
    sessions_by_partner: dict[int, list[str]]
    qualifies: bool


def aggregate_users(
    sessions: dict[tuple[int, int], list[ConversationSession]],
    *,
    min_messages: int = 200,
    min_partners: int = 3,
) -> dict[int, UserProfile]:
    messages_by_user: dict[int, list[str]] = defaultdict(list)
    active_days: dict[int, set[date]] = defaultdict(set)
    partner_sets: dict[int, set[int]] = defaultdict(set)
    sessions_by_partner: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    hour_counts: dict[int, Counter[int]] = defaultdict(Counter)

    for (uid_a, uid_b), pair_sessions in sessions.items():
        for session in pair_sessions:
            if session.metadata.get("dropped"):
                continue
            sessions_by_partner[uid_a][uid_b].append(session.session_id)
            sessions_by_partner[uid_b][uid_a].append(session.session_id)
            partner_sets[uid_a].add(uid_b)
            partner_sets[uid_b].add(uid_a)
            for turn in session.turns:
                messages_by_user[turn.sender_uid].append(turn.body)
                active_days[turn.sender_uid].add(turn.timestamp.date())
                hour_counts[turn.sender_uid].update([turn.timestamp.hour])

    profiles: dict[int, UserProfile] = {}
    for uid in sorted(messages_by_user):
        messages = messages_by_user[uid]
        total_messages = len(messages)
        unique_partners = len(partner_sets[uid])
        total_length = sum(len(m) for m in messages)
        avg_length = float(total_length) / float(total_messages) if total_messages else 0.0

        all_tokens: list[str] = []
        for text in messages:
            all_tokens.extend(TOKEN_RE.findall(text.lower()))
        token_count = len(all_tokens)
        vocab_richness = float(len(set(all_tokens))) / float(token_count) if token_count else 0.0

        user_hours = hour_counts[uid]
        hour_total = sum(user_hours.values()) or 1
        temporal_pattern = {f"{hour:02d}": round(user_hours.get(hour, 0) / hour_total, 6) for hour in range(24)}

        qualifies = total_messages >= min_messages and unique_partners >= min_partners
        profiles[uid] = UserProfile(
            uid=uid,
            total_messages=total_messages,
            active_days=len(active_days[uid]),
            unique_partners=unique_partners,
            avg_message_length=round(avg_length, 6),
            vocabulary_richness=round(vocab_richness, 6),
            temporal_pattern=temporal_pattern,
            sessions_by_partner={k: v for k, v in sorted(sessions_by_partner[uid].items())},
            qualifies=qualifies,
        )
    return profiles
