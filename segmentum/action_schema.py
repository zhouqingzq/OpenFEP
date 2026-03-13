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
    cost_estimate: float = 0.0
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "cost_estimate": self.cost_estimate,
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
        return cls(
            name=str(payload.get("name", "")),
            params=dict(params) if isinstance(params, dict) else {},
            cost_estimate=float(payload.get("cost_estimate", 0.0)),
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
            and self.cost_estimate == other.cost_estimate
            and self.reversible == other.reversible
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                _freeze_value(self.params),
                round(float(self.cost_estimate), 12),
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
