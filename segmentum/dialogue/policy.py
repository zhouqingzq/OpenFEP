"""M5.3 dialogue policy surface: scoring runs inside SegmentAgent (EFE + identity_bias)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

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

    def evaluate_actions(
        self,
        observation: Mapping[str, float],
        dialogue_context: Mapping[str, object] | None = None,
    ) -> dict[str, float]:
        """Return dialogue action -> expected free energy (lower is better)."""
        ctx = dict(dialogue_context or {})
        ctx.setdefault("event_type", "dialogue_turn")
        # Use an isolated shadow agent so facade evaluation stays read-only.
        shadow = type(self._agent).from_dict(self._agent.to_dict())
        result = shadow.decision_cycle_from_dict(dict(observation), context=ctx)
        diagnostics = result.get("diagnostics")
        if diagnostics is None:
            return {}
        scores: dict[str, float] = {}
        for option in getattr(diagnostics, "ranked_options", []):
            action = str(getattr(option, "choice", ""))
            if not is_dialogue_action(action):
                continue
            try:
                scores[action] = float(getattr(option, "expected_free_energy"))
            except (TypeError, ValueError):
                continue
        if not scores:
            return {}
        return {
            name: scores[name]
            for name in DIALOGUE_ACTION_NAMES
            if name in scores
        }

    def select_action(
        self,
        observation: Mapping[str, float],
        dialogue_context: Mapping[str, object] | None = None,
    ) -> str:
        """Pick the minimum-EFE dialogue action from ``evaluate_actions``."""
        scores = self.evaluate_actions(observation, dialogue_context)
        if not scores:
            return "ask_question"
        return min(
            scores,
            key=lambda action: (
                scores[action],
                DIALOGUE_ACTION_NAMES.index(action)
                if action in DIALOGUE_ACTION_NAMES
                else len(DIALOGUE_ACTION_NAMES),
            ),
        )
