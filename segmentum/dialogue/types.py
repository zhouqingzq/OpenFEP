"""Shared dialogue typing for M5.3 transcript and APIs."""

from __future__ import annotations

from typing import Literal, TypedDict

try:
    from typing import NotRequired  # Python >=3.11
except ImportError:
    from typing_extensions import NotRequired  # Python <3.11


class TranscriptUtterance(TypedDict):
    """One utterance in a chronological transcript (role + text)."""

    role: Literal["agent", "interlocutor"]
    text: str
    turn_index: NotRequired[int]


# Alias for spec / readability (distinct from world.DialogueTurn replay row dataclass)
DialogueTurn = TranscriptUtterance
