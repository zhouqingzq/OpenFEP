from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(key), _freeze_value(val)) for key, val in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


@dataclass(frozen=True)
class ActionSchema:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    params_schema: dict[str, Any] = field(default_factory=dict)
    cost_estimate: float = 0.0
    resource_cost: dict[str, float] = field(default_factory=dict)
    failure_modes: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "params_schema": dict(self.params_schema),
            "cost_estimate": self.cost_estimate,
            "resource_cost": {str(key): float(value) for key, value in self.resource_cost.items()},
            "failure_modes": list(self.failure_modes),
            "constraints": dict(self.constraints),
            "reversible": self.reversible,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ActionSchema":
        if isinstance(payload, ActionSchema):
            return payload
        if isinstance(payload, str):
            return cls(name=payload)
        if not isinstance(payload, dict):
            return cls(name=str(payload or ""))
        params = payload.get("params", {})
        params_schema = payload.get("params_schema", {})
        resource_cost = payload.get("resource_cost", {})
        constraints = payload.get("constraints", {})
        return cls(
            name=str(payload.get("name", "")),
            params=dict(params) if isinstance(params, dict) else {},
            params_schema=(
                dict(params_schema) if isinstance(params_schema, dict) else {}
            ),
            cost_estimate=float(payload.get("cost_estimate", 0.0)),
            resource_cost=(
                {
                    str(key): float(value)
                    for key, value in resource_cost.items()
                    if isinstance(value, (int, float))
                }
                if isinstance(resource_cost, dict)
                else {}
            ),
            failure_modes=tuple(
                str(item) for item in payload.get("failure_modes", [])
            ),
            constraints=dict(constraints) if isinstance(constraints, dict) else {},
            reversible=bool(payload.get("reversible", True)),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.name == other
        if not isinstance(other, ActionSchema):
            return False
        return (
            self.name == other.name
            and self.params == other.params
            and self.params_schema == other.params_schema
            and self.cost_estimate == other.cost_estimate
            and self.resource_cost == other.resource_cost
            and self.failure_modes == other.failure_modes
            and self.constraints == other.constraints
            and self.reversible == other.reversible
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                _freeze_value(self.params),
                _freeze_value(self.params_schema),
                round(float(self.cost_estimate), 12),
                _freeze_value(self.resource_cost),
                self.failure_modes,
                _freeze_value(self.constraints),
                self.reversible,
            )
        )

    def __str__(self) -> str:
        if not self.params:
            return self.name
        return f"{self.name}({self.params})"


def ensure_action_schema(action: Any) -> ActionSchema:
    return ActionSchema.from_dict(action)


def action_name(action: Any) -> str:
    if isinstance(action, ActionSchema):
        return action.name
    if isinstance(action, dict):
        if "name" in action:
            return str(action.get("name", ""))
        if "action" in action:
            return action_name(action.get("action"))
    return str(action or "")
