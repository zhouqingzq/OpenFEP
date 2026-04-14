"""M5.3 dialogue policy surface: scoring runs inside SegmentAgent (EFE + identity_bias)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .actions import DIALOGUE_ACTION_NAMES, DIALOGUE_ACTION_STRATEGY_MAP, is_dialogue_action

if TYPE_CHECKING:
    from ..agent import SegmentAgent


class DialoguePolicyEvaluator:
    """Adapter for the prompt API: do not duplicate PolicyEvaluator; use ``decision_cycle_from_dict``.

    Action utilities and strategy labels live here for callers that want a dialogue-local handle
    without reaching into the full decision stack.
    """

    def __init__(self, agent: "SegmentAgent") -> None:
        self._agent = agent

    def registered_dialogue_actions(self) -> tuple[str, ...]:
        """Dialogue actions present in the agent registry, in canonical M5.3 order."""
        reg = {a.name for a in self._agent.action_registry.get_all()}
        return tuple(n for n in DIALOGUE_ACTION_NAMES if n in reg)

    def strategy_for(self, action: str) -> str:
        return DIALOGUE_ACTION_STRATEGY_MAP.get(action, "explore")
