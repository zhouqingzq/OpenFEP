from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .session_builder import ConversationSession
from .user_aggregator import UserProfile


def _serialize_turn(turn: Any) -> dict[str, Any]:
    return {
        "timestamp": turn.timestamp.isoformat(),
        "msg_type": turn.msg_type,
        "sender_uid": turn.sender_uid,
        "receiver_uid": turn.receiver_uid,
        "body": turn.body,
    }


def _serialize_session(session: ConversationSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "uid_a": session.uid_a,
        "uid_b": session.uid_b,
        "start_time": session.start_time.isoformat(),
        "end_time": session.end_time.isoformat(),
        "metadata": session.metadata,
        "turns": [_serialize_turn(turn) for turn in session.turns],
    }


def export_user_dataset(
    uid: int,
    profile: UserProfile,
    sessions: list[ConversationSession],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uid}.json"
    ordered_sessions = sorted(sessions, key=lambda s: (s.start_time, s.session_id))
    payload = {
        "uid": uid,
        "profile": asdict(profile),
        "sessions": [_serialize_session(session) for session in ordered_sessions],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
