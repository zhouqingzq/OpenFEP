from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .action_schema import ActionSchema, action_name, ensure_action_schema


def _clamp_int(value: int) -> int:
    return max(0, int(value))


@dataclass(frozen=True)
class CapabilityDescriptor:
    action_name: str
    adapter_name: str
    authorization: str
    budget_cost: dict[str, int] = field(default_factory=dict)
    risk_level: str = "low"
    reversible: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "action_name": self.action_name,
            "adapter_name": self.adapter_name,
            "authorization": self.authorization,
            "budget_cost": dict(self.budget_cost),
            "risk_level": self.risk_level,
            "reversible": self.reversible,
            "description": self.description,
        }


@dataclass(frozen=True)
class AuthorizationDecision:
    action_name: str
    status: str
    reason: str
    capability: dict[str, object]
    predicted_effects: dict[str, float]
    budget_before: dict[str, int]
    budget_after: dict[str, int]
    review_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "action_name": self.action_name,
            "status": self.status,
            "reason": self.reason,
            "capability": dict(self.capability),
            "predicted_effects": dict(self.predicted_effects),
            "budget_before": dict(self.budget_before),
            "budget_after": dict(self.budget_after),
            "review_required": self.review_required,
        }


@dataclass
class GovernanceState:
    context_budget_remaining: int = 64
    file_budget_remaining: int = 8
    network_budget_remaining: int = 2
    failure_budget_remaining: int = 3
    decision_history: list[dict[str, object]] = field(default_factory=list)
    last_failure: dict[str, object] | None = None

    def snapshot(self) -> dict[str, int]:
        return {
            "context_budget_remaining": int(self.context_budget_remaining),
            "file_budget_remaining": int(self.file_budget_remaining),
            "network_budget_remaining": int(self.network_budget_remaining),
            "failure_budget_remaining": int(self.failure_budget_remaining),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.snapshot(),
            "decision_history": list(self.decision_history),
            "last_failure": dict(self.last_failure) if self.last_failure is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "GovernanceState":
        if not payload:
            return cls()
        history = payload.get("decision_history", [])
        last_failure = payload.get("last_failure")
        return cls(
            context_budget_remaining=int(payload.get("context_budget_remaining", 64)),
            file_budget_remaining=int(payload.get("file_budget_remaining", 8)),
            network_budget_remaining=int(payload.get("network_budget_remaining", 2)),
            failure_budget_remaining=int(payload.get("failure_budget_remaining", 3)),
            decision_history=[dict(item) for item in history if isinstance(item, Mapping)],
            last_failure=(dict(last_failure) if isinstance(last_failure, Mapping) else None),
        )


class WorkspaceNoteAdapter:
    adapter_name = "WorkspaceNoteAdapter"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        action: ActionSchema,
        *,
        cycle: int,
    ) -> dict[str, object]:
        action_name_key = action_name(action)
        note_path = self.root / str(action.params.get("path", "governed_note.txt"))
        note_path.parent.mkdir(parents=True, exist_ok=True)
        text = str(action.params.get("text", ""))
        if action_name_key == "write_workspace_note":
            note_path.write_text(text, encoding="utf-8")
        elif action_name_key == "append_workspace_note":
            with note_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        elif action_name_key == "unstable_workspace_note":
            raise RuntimeError("simulated_external_failure")
        else:
            raise ValueError(f"unsupported governed action: {action_name_key}")
        return {
            "success": True,
            "feedback": {
                "file_write": 1.0,
                "context_delta": -0.02,
            },
            "metadata": {
                "path": str(note_path),
                "bytes_written": len(text.encode("utf-8")),
                "cycle": cycle,
            },
        }


class GovernanceController:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        state: GovernanceState | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.state = state or GovernanceState()
        note_root = self.workspace_root / "data" / "governed_actions"
        note_adapter = WorkspaceNoteAdapter(note_root)
        self.adapters = {
            "write_workspace_note": note_adapter,
            "append_workspace_note": note_adapter,
            "unstable_workspace_note": note_adapter,
        }
        self.capabilities = {
            "write_workspace_note": CapabilityDescriptor(
                action_name="write_workspace_note",
                adapter_name=note_adapter.adapter_name,
                authorization="allowed",
                budget_cost={"file_budget_remaining": 1, "context_budget_remaining": 4},
                description="Write a governed note inside the workspace sandbox.",
            ),
            "append_workspace_note": CapabilityDescriptor(
                action_name="append_workspace_note",
                adapter_name=note_adapter.adapter_name,
                authorization="allowed",
                budget_cost={"file_budget_remaining": 1, "context_budget_remaining": 2},
                description="Append to a governed workspace note.",
            ),
            "unstable_workspace_note": CapabilityDescriptor(
                action_name="unstable_workspace_note",
                adapter_name=note_adapter.adapter_name,
                authorization="allowed",
                budget_cost={"file_budget_remaining": 1, "context_budget_remaining": 2},
                description="Exercise failure and repair handling for external actions.",
            ),
            "fetch_remote_status": CapabilityDescriptor(
                action_name="fetch_remote_status",
                adapter_name="NetworkProbeAdapter",
                authorization="review-required",
                budget_cost={"network_budget_remaining": 1, "context_budget_remaining": 6},
                risk_level="medium",
                reversible=False,
                description="Contact a remote endpoint; held for review in M2.17.",
            ),
            "delete_workspace_note": CapabilityDescriptor(
                action_name="delete_workspace_note",
                adapter_name="WorkspaceDeleteAdapter",
                authorization="denied",
                budget_cost={"file_budget_remaining": 1},
                risk_level="high",
                reversible=False,
                description="Delete a workspace file; denied by M2.17 policy.",
            ),
        }

    def capability_descriptor(self, action_name_key: str) -> CapabilityDescriptor | None:
        return self.capabilities.get(action_name_key)

    def authorize(
        self,
        action: ActionSchema | str | Mapping[str, object],
        *,
        predicted_effects: Mapping[str, float] | None = None,
    ) -> AuthorizationDecision:
        schema = ensure_action_schema(action)
        action_name_key = schema.name
        capability = self.capability_descriptor(action_name_key)
        budget_before = self.state.snapshot()
        predicted = {
            str(key): float(value)
            for key, value in dict(predicted_effects or {}).items()
            if isinstance(value, (int, float))
        }
        if capability is None:
            decision = AuthorizationDecision(
                action_name=action_name_key,
                status="denied",
                reason="unknown_capability",
                capability={},
                predicted_effects=predicted,
                budget_before=budget_before,
                budget_after=budget_before,
            )
            self._remember_decision(decision)
            return decision
        if capability.authorization == "denied":
            decision = AuthorizationDecision(
                action_name=action_name_key,
                status="denied",
                reason="policy_denied",
                capability=capability.to_dict(),
                predicted_effects=predicted,
                budget_before=budget_before,
                budget_after=budget_before,
            )
            self._remember_decision(decision)
            return decision
        if capability.authorization == "review-required":
            decision = AuthorizationDecision(
                action_name=action_name_key,
                status="review-required",
                reason="review_gate",
                capability=capability.to_dict(),
                predicted_effects=predicted,
                budget_before=budget_before,
                budget_after=budget_before,
                review_required=True,
            )
            self._remember_decision(decision)
            return decision
        if self.state.failure_budget_remaining <= 0 and action_name_key != "append_workspace_note":
            decision = AuthorizationDecision(
                action_name=action_name_key,
                status="denied",
                reason="failure_budget_exhausted",
                capability=capability.to_dict(),
                predicted_effects=predicted,
                budget_before=budget_before,
                budget_after=budget_before,
            )
            self._remember_decision(decision)
            return decision

        budget_after = dict(budget_before)
        for field_name, cost in capability.budget_cost.items():
            if budget_after.get(field_name, 0) < int(cost):
                decision = AuthorizationDecision(
                    action_name=action_name_key,
                    status="denied",
                    reason=f"budget_exhausted:{field_name}",
                    capability=capability.to_dict(),
                    predicted_effects=predicted,
                    budget_before=budget_before,
                    budget_after=budget_before,
                )
                self._remember_decision(decision)
                return decision
            budget_after[field_name] = budget_after.get(field_name, 0) - int(cost)

        decision = AuthorizationDecision(
            action_name=action_name_key,
            status="allowed",
            reason="policy_allowed",
            capability=capability.to_dict(),
            predicted_effects=predicted,
            budget_before=budget_before,
            budget_after=budget_after,
        )
        self._remember_decision(decision)
        return decision

    def commit(self, decision: AuthorizationDecision, *, success: bool) -> None:
        if decision.status != "allowed":
            return
        for field_name, value in decision.budget_after.items():
            setattr(self.state, field_name, _clamp_int(value))
        if success:
            self.state.last_failure = None

    def record_failure(self, *, action_name: str, reason: str, cycle: int) -> None:
        self.state.failure_budget_remaining = _clamp_int(self.state.failure_budget_remaining - 1)
        self.state.last_failure = {
            "action_name": action_name,
            "reason": reason,
            "cycle": int(cycle),
        }

    def repair_action(self, *, failed_action_name: str, cycle: int) -> ActionSchema:
        return ActionSchema(
            name="append_workspace_note",
            params={
                "path": "repair.log",
                "text": (
                    f"[cycle {cycle}] repair after {failed_action_name}: "
                    "external action failure bounded and logged.\n"
                ),
            },
        )

    def _remember_decision(self, decision: AuthorizationDecision) -> None:
        self.state.decision_history.append(decision.to_dict())
        self.state.decision_history = self.state.decision_history[-128:]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "capabilities": {
                key: descriptor.to_dict()
                for key, descriptor in sorted(self.capabilities.items())
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object] | None,
        *,
        workspace_root: str | Path,
    ) -> "GovernanceController":
        state = GovernanceState.from_dict(
            payload.get("state") if isinstance(payload, Mapping) else None
        )
        return cls(workspace_root=workspace_root, state=state)
