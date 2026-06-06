"""Shared dialogue typing for Path B transcript and bounded group-chat metadata."""

from __future__ import annotations

from typing import Literal, TypedDict

try:
    from typing import NotRequired  # Python >=3.11
except ImportError:
    from typing_extensions import NotRequired  # Python <3.11


class GroupTurnEnvelope(TypedDict, total=False):
    """Bounded multi-party input metadata for one visible user turn."""

    speaker_participant_id: str
    visible_participant_ids: list[str]
    addressed_participant_ids: list[str]
    mentioned_participant_ids: list[str]
    reply_to_turn_id: str
    quoted_turn_ids: list[str]
    explicit_mentions: list[str]
    surface_intent: str
    platform_command: str
    assistant_surface_label: str


class TranscriptUtterance(TypedDict):
    """One utterance in a chronological transcript (role + text + ownership)."""

    role: Literal["agent", "interlocutor"]
    text: str
    turn_index: NotRequired[int]
    speaker_name: NotRequired[str]
    speaker_participant_id: NotRequired[str]
    participant_ids: NotRequired[list[str]]
    addressed_participant_ids: NotRequired[list[str]]
    mentioned_participant_ids: NotRequired[list[str]]
    reply_to_turn_id: NotRequired[str]
    quoted_turn_ids: NotRequired[list[str]]
    explicit_mentions: NotRequired[list[str]]
    surface_intent: NotRequired[str]
    platform_command: NotRequired[str]
    assistant_surface_label: NotRequired[str]


# Alias for spec / readability (distinct from world.DialogueTurn replay row dataclass)
DialogueTurn = TranscriptUtterance
