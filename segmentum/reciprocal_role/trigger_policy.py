"""M12.2 durable-consolidation trigger policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, M122Hyperparams

TriggerKind = Literal[
    "explicit_user_request",
    "high_second_order_uncertainty",
    "relationship_turning_point",
    "contradiction_or_misread",
    "periodic_refresh",
    "developer_requested_audit",
]


@dataclass(frozen=True)
class TriggerPolicyInput:
    user_id: str
    current_turn_index: int
    current_hour_bucket: int
    previous_run_turn_indices: tuple[int, ...] = ()
    run_hour_buckets: tuple[int, ...] = ()
    has_existing_model: bool = False
    explicit_user_request: bool = False
    high_second_order_uncertainty: bool = False
    relationship_turning_point: bool = False
    contradiction_or_misread: bool = False
    periodic_refresh_due: bool = False
    developer_requested_audit: bool = False
    evidence_sparse: bool = False
    contradiction_cooldown: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "current_turn_index": self.current_turn_index,
            "current_hour_bucket": self.current_hour_bucket,
            "previous_run_turn_indices": list(self.previous_run_turn_indices),
            "run_hour_buckets": list(self.run_hour_buckets),
            "has_existing_model": self.has_existing_model,
            "explicit_user_request": self.explicit_user_request,
            "high_second_order_uncertainty": self.high_second_order_uncertainty,
            "relationship_turning_point": self.relationship_turning_point,
            "contradiction_or_misread": self.contradiction_or_misread,
            "periodic_refresh_due": self.periodic_refresh_due,
            "developer_requested_audit": self.developer_requested_audit,
            "evidence_sparse": self.evidence_sparse,
            "contradiction_cooldown": self.contradiction_cooldown,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TriggerPolicyInput":
        return cls(
            user_id=str(payload.get("user_id", "")),
            current_turn_index=int(payload.get("current_turn_index", 0) or 0),
            current_hour_bucket=int(payload.get("current_hour_bucket", 0) or 0),
            previous_run_turn_indices=_ints(payload.get("previous_run_turn_indices")),
            run_hour_buckets=_ints(payload.get("run_hour_buckets")),
            has_existing_model=bool(payload.get("has_existing_model", False)),
            explicit_user_request=bool(payload.get("explicit_user_request", False)),
            high_second_order_uncertainty=bool(payload.get("high_second_order_uncertainty", False)),
            relationship_turning_point=bool(payload.get("relationship_turning_point", False)),
            contradiction_or_misread=bool(payload.get("contradiction_or_misread", False)),
            periodic_refresh_due=bool(payload.get("periodic_refresh_due", False)),
            developer_requested_audit=bool(payload.get("developer_requested_audit", False)),
            evidence_sparse=bool(payload.get("evidence_sparse", False)),
            contradiction_cooldown=int(payload.get("contradiction_cooldown", 0) or 0),
        )


@dataclass(frozen=True)
class TriggerDecision:
    should_run: bool
    kind: str = "no_run"
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"should_run": self.should_run, "kind": self.kind, "reason": self.reason}


def decide_trigger(
    inputs: TriggerPolicyInput,
    *,
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
) -> TriggerDecision:
    if inputs.contradiction_cooldown > 0:
        return TriggerDecision(False, "no_run", "contradiction_cooldown")
    if _over_hour_cap(inputs, hyperparams=hyperparams):
        return TriggerDecision(False, "no_run", "per_hour_cap")
    if _inside_turn_window(inputs, hyperparams=hyperparams):
        return TriggerDecision(False, "no_run", "turn_window_cap")
    if not inputs.has_existing_model and not inputs.explicit_user_request and not inputs.developer_requested_audit:
        return TriggerDecision(False, "no_run", "bootstrap_first_turn_skips_durable")
    if inputs.evidence_sparse and not (inputs.explicit_user_request or inputs.developer_requested_audit):
        return TriggerDecision(False, "no_run", "sparse_evidence")
    checks: tuple[tuple[bool, str, str], ...] = (
        (inputs.developer_requested_audit, "developer_requested_audit", "audit_requested"),
        (inputs.explicit_user_request, "explicit_user_request", "explicit_user_request"),
        (inputs.contradiction_or_misread, "contradiction_or_misread", "contradiction_or_misread"),
        (inputs.relationship_turning_point, "relationship_turning_point", "relationship_turning_point"),
        (inputs.high_second_order_uncertainty, "high_second_order_uncertainty", "high_second_order_uncertainty"),
        (inputs.periodic_refresh_due, "periodic_refresh", "periodic_refresh_due"),
    )
    for active, kind, reason in checks:
        if active:
            return TriggerDecision(True, kind, reason)
    return TriggerDecision(False, "no_run", "no_trigger")


def _over_hour_cap(inputs: TriggerPolicyInput, *, hyperparams: M122Hyperparams) -> bool:
    return sum(1 for hour in inputs.run_hour_buckets if hour == inputs.current_hour_bucket) >= hyperparams.max_consolidations_per_hour


def _inside_turn_window(inputs: TriggerPolicyInput, *, hyperparams: M122Hyperparams) -> bool:
    if not inputs.previous_run_turn_indices:
        return False
    latest = max(inputs.previous_run_turn_indices)
    return inputs.current_turn_index - latest < hyperparams.min_turn_window_between_consolidations


def _ints(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)
