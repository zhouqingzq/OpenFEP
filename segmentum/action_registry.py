"""Dynamic action registry used by agent planning and counterfactual search."""

from __future__ import annotations

from .action_schema import ActionSchema
from .constants import (
    ACTION_CONSTRAINTS,
    ACTION_COSTS,
    ACTION_FAILURE_MODES,
    ACTION_PARAM_SCHEMAS,
    ACTION_RESOURCE_COSTS,
)


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, ActionSchema] = {}
        self._cost_estimates: dict[str, float] = {}

    def register(self, action: ActionSchema, cost: float = 0.0) -> None:
        cost_estimate = float(cost if cost else action.cost_estimate)
        stored = ActionSchema(
            name=action.name,
            params=dict(action.params),
            params_schema=dict(action.params_schema),
            cost_estimate=cost_estimate,
            resource_cost=dict(action.resource_cost),
            failure_modes=tuple(action.failure_modes),
            constraints=dict(action.constraints),
            reversible=action.reversible,
        )
        self._actions[stored.name] = stored
        self._cost_estimates[stored.name] = cost_estimate

    def unregister(self, name: str) -> None:
        self._actions.pop(name, None)
        self._cost_estimates.pop(name, None)

    def get(self, name: str) -> ActionSchema | None:
        action = self._actions.get(name)
        if action is None:
            return None
        return ActionSchema(
            name=action.name,
            params=dict(action.params),
            params_schema=dict(action.params_schema),
            cost_estimate=self.get_cost(name),
            resource_cost=dict(action.resource_cost),
            failure_modes=tuple(action.failure_modes),
            constraints=dict(action.constraints),
            reversible=action.reversible,
        )

    def get_all(self) -> list[ActionSchema]:
        actions: list[ActionSchema] = []
        for name in self._actions:
            action = self.get(name)
            if action is not None:
                actions.append(action)
        return actions

    def get_alternatives(self, exclude: ActionSchema) -> list[ActionSchema]:
        return [
            action
            for action in self.get_all()
            if action.name != exclude.name
        ]

    def get_cost(self, name: str) -> float:
        return float(self._cost_estimates.get(name, 0.0))

    def contains(self, name: str) -> bool:
        return name in self._actions

    def names(self) -> list[str]:
        return list(self._actions)

    def to_dict(self) -> dict[str, object]:
        return {
            name: {
                "action": action.to_dict(),
                "cost": self._cost_estimates.get(name, 0.0),
            }
            for name, action in self._actions.items()
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ActionRegistry":
        registry = cls()
        for name, entry in dict(payload or {}).items():
            if not isinstance(entry, dict):
                continue
            action_payload = entry.get("action", {"name": str(name)})
            action = ActionSchema.from_dict(action_payload)
            registry.register(action, float(entry.get("cost", action.cost_estimate)))
        return registry


def build_default_action_registry() -> ActionRegistry:
    registry = ActionRegistry()
    for name, cost in ACTION_COSTS.items():
        registry.register(
            ActionSchema(
                name=name,
                params_schema=dict(ACTION_PARAM_SCHEMAS.get(name, {})),
                cost_estimate=float(cost),
                resource_cost=dict(ACTION_RESOURCE_COSTS.get(name, {})),
                failure_modes=tuple(ACTION_FAILURE_MODES.get(name, ())),
                constraints=dict(ACTION_CONSTRAINTS.get(name, {})),
            ),
            float(cost),
        )
    return registry
