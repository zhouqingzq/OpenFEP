"""Pure M12.1 trigger policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, M121Hyperparams

TriggerKind = Literal[
    "explicit_request",
    "turn_count_cadence",
    "new_evidence_threshold",
    "identity_state_change",
    "strangeness_followup",
    "session_boundary",
]


@dataclass(frozen=True)
class TriggerPolicyInput:
    user_id: str
    current_turn_index: int
    current_hour_bucket: int
    last_successful_report_turn_index: int = -1
    run_hour_buckets: tuple[int, ...] = ()
    new_evidence_count: int = 0
    identity_state_changed: bool = False
    high_strangeness_turns: int = 0
    session_boundary: bool = False
    explicit_request: bool = False
    consecutive_step1_insufficient: int = 0
    new_evidence_since_step1_insufficient: bool = False
    weekday: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "current_turn_index": self.current_turn_index,
            "current_hour_bucket": self.current_hour_bucket,
            "last_successful_report_turn_index": self.last_successful_report_turn_index,
            "run_hour_buckets": list(self.run_hour_buckets),
            "new_evidence_count": self.new_evidence_count,
            "identity_state_changed": self.identity_state_changed,
            "high_strangeness_turns": self.high_strangeness_turns,
            "session_boundary": self.session_boundary,
            "explicit_request": self.explicit_request,
            "consecutive_step1_insufficient": self.consecutive_step1_insufficient,
            "new_evidence_since_step1_insufficient": self.new_evidence_since_step1_insufficient,
            "weekday": self.weekday,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TriggerPolicyInput":
        raw_runs = payload.get("run_hour_buckets", ())
        runs = tuple(int(item) for item in raw_runs) if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, (str, bytes)) else ()
        weekday_value = payload.get("weekday")
        return cls(
            user_id=str(payload.get("user_id", "")),
            current_turn_index=int(payload.get("current_turn_index", 0)),
            current_hour_bucket=int(payload.get("current_hour_bucket", 0)),
            last_successful_report_turn_index=int(payload.get("last_successful_report_turn_index", -1)),
            run_hour_buckets=runs,
            new_evidence_count=int(payload.get("new_evidence_count", 0)),
            identity_state_changed=bool(payload.get("identity_state_changed", False)),
            high_strangeness_turns=int(payload.get("high_strangeness_turns", 0)),
            session_boundary=bool(payload.get("session_boundary", False)),
            explicit_request=bool(payload.get("explicit_request", False)),
            consecutive_step1_insufficient=int(payload.get("consecutive_step1_insufficient", 0)),
            new_evidence_since_step1_insufficient=bool(payload.get("new_evidence_since_step1_insufficient", False)),
            weekday=int(weekday_value) if weekday_value is not None else None,
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
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> TriggerDecision:
    if not _calendar_allowed(inputs, hyperparams=hyperparams):
        return TriggerDecision(False, "no_run", "calendar_gate")
    if _over_hour_cap(inputs, hyperparams=hyperparams):
        return TriggerDecision(False, "no_run", "per_hour_cap")
    if inputs.explicit_request:
        return TriggerDecision(True, "explicit_request", "operator_request")
    if inputs.session_boundary:
        return TriggerDecision(True, "session_boundary", "session_boundary")
    if inputs.identity_state_changed:
        return TriggerDecision(True, "identity_state_change", "identity_state_changed")
    if inputs.high_strangeness_turns >= hyperparams.strangeness_high_turn_threshold:
        return TriggerDecision(True, "strangeness_followup", "sustained_high_strangeness")
    if inputs.new_evidence_count >= hyperparams.new_evidence_threshold:
        return TriggerDecision(True, "new_evidence_threshold", "new_evidence_threshold")
    if _cadence_suspended(inputs, hyperparams=hyperparams):
        return TriggerDecision(False, "no_run", "cadence_suspended_after_sparse_step1")
    if inputs.last_successful_report_turn_index < 0:
        return TriggerDecision(False, "no_run", "initial_profile_waits_for_trigger")
    turns_elapsed = inputs.current_turn_index - inputs.last_successful_report_turn_index
    if turns_elapsed >= hyperparams.cadence_turn_interval:
        return TriggerDecision(True, "turn_count_cadence", "cadence_elapsed")
    return TriggerDecision(False, "no_run", "under_cadence")


def _over_hour_cap(inputs: TriggerPolicyInput, *, hyperparams: M121Hyperparams) -> bool:
    count = sum(1 for hour in inputs.run_hour_buckets if int(hour) == inputs.current_hour_bucket)
    return count >= hyperparams.max_personality_runs_per_hour


def _cadence_suspended(inputs: TriggerPolicyInput, *, hyperparams: M121Hyperparams) -> bool:
    return (
        inputs.consecutive_step1_insufficient >= hyperparams.step1_insufficient_suspend_count
        and not inputs.new_evidence_since_step1_insufficient
    )


def _calendar_allowed(inputs: TriggerPolicyInput, *, hyperparams: M121Hyperparams) -> bool:
    window = hyperparams.trigger_calendar_window
    if window is None or inputs.weekday is None:
        return True
    start, end = window
    return start <= inputs.weekday <= end
