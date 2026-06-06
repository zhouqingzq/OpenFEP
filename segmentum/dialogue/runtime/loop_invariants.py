"""M20.3 §4 fast_chat minimum loop invariant.

M20.3 closes M20.0–M20.2 gap E: `fast_chat` previously skipped
admission, settlement, and dispatch. The runtime invariant module
emits `MinimumLoopCoverageMissed` audit events when the rule matrix
is breached, but NEVER blocks the turn. The invariant is **audit-
only**: it tells the operator the rule was missed, but it does not
refuse to run.

The rule matrix (M20.3 §4.2):

| Rule | Trigger | Required |
|---|---|---|
| A | every external turn (non-idle) | `ActiveCommitment` with `source_kind = "policy"` >= 1 |
| B | `surface_intent ∈ {chat, bot}` | `ActiveCommitment` with `owner_id = "runtime_mode_state"` >= 1 |

Both rules are hard. A miss on either rule emits the audit event.

The `fast_chat` path MUST call PolicyProducer even when
`fast_chat_skip_admit` is on for the conscious loop. M20.3 §4
verifies that via the rule matrix; a `MinimumLoopCoverageMissed`
event on a `fast_chat` turn is a regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from segmentum.dialogue.runtime.active_commitment import ActiveCommitment


# Frozen engineering proxy label. M20.3 §4 audit envelope.
_ENGINEERING_PROXY_LABEL = "mvp_local_minimum_loop"

# Rule A: every external turn produces a policy commitment.
RULE_POLICY_SOURCE_REQUIRED = "policy_source_required"

# Rule B: chat / bot surface produces a runtime_mode_state commitment.
RULE_RUNTIME_MODE_STATE_REQUIRED = "runtime_mode_state_required_for_chat_or_bot"

# Surface intents that trigger rule B (per M18.x group_turn_binding).
_RULE_B_SURFACE_INTENTS: frozenset[str] = frozenset({"chat", "bot"})


@dataclass(frozen=True)
class LoopCoverageVerdict:
    """Frozen verdict returned by `enforce_minimum_loop_coverage`.

    `missed` is a list of dicts (one per breached rule). Empty list
    means the matrix passed. The caller emits one
    `MinimumLoopCoverageMissed` event when `missed` is non-empty;
    the verdict itself is the structured return.
    """

    turn_index: int
    missed: tuple[dict[str, Any], ...]
    engineering_proxy_label: str

    @property
    def passed(self) -> bool:
        return not self.missed


def build_minimum_loop_coverage_missed_event(verdict: LoopCoverageVerdict) -> dict[str, Any]:
    """Build the `MinimumLoopCoverageMissed` audit envelope (M20.3 §4)."""
    return {
        "type": "MinimumLoopCoverageMissed",
        "turn_index": verdict.turn_index,
        "missing": [dict(row) for row in verdict.missed],
        "engineering_proxy_label": verdict.engineering_proxy_label,
        "at": "",  # caller fills the timestamp when recording the event
    }


# === LoopInvariants =====================================================


class LoopInvariants:
    """M20.3 §4 runtime invariant module.

    Audit-only. The module never blocks the turn. The caller runs
    `enforce_minimum_loop_coverage` after both admission sources
    (PolicyProducer + M20.0 conscious-loop) have produced their rows
    for the current turn.
    """

    ENGINEERING_PROXY_LABEL: str = _ENGINEERING_PROXY_LABEL

    def __init__(self) -> None:
        # Per-rule counters that increment on miss and decrement on
        # fix. The counter is reset to 0 when a turn passes the
        # rule; the counter is incremented by 1 when a turn misses
        # the rule. M20.3 §6.3 fixture asserts that
        # `MinimumLoopCoverageMissed` is NOT emitted on a `+1
        # fast_chat` turn, which means the counter must stay at 0
        # on that turn.
        self._miss_counters: dict[str, int] = {
            RULE_POLICY_SOURCE_REQUIRED: 0,
            RULE_RUNTIME_MODE_STATE_REQUIRED: 0,
        }
        self._last_miss_turn: dict[str, int | None] = {
            RULE_POLICY_SOURCE_REQUIRED: None,
            RULE_RUNTIME_MODE_STATE_REQUIRED: None,
        }

    # -- public counters (read-only) --------------------------------------

    @property
    def miss_counters(self) -> Mapping[str, int]:
        return dict(self._miss_counters)

    @property
    def last_miss_turn(self) -> Mapping[str, int | None]:
        return dict(self._last_miss_turn)

    # -- the runtime invariant call --------------------------------------

    def enforce_minimum_loop_coverage(
        self,
        *,
        turn_index: int,
        proposed_commitments: Iterable[ActiveCommitment],
        surface_intent: str = "",
        is_external_turn: bool = True,
    ) -> LoopCoverageVerdict:
        """Apply the rule matrix and return a verdict.

        `proposed_commitments` is the union of PolicyProducer + M20.0
        conscious-loop admissions for the current turn. The
        invariant does NOT retry the admission; if a rule is
        violated, the verdict lists the rule and the turn proceeds.

        The verdict is also recorded in the per-rule counters so
        later turns can read `miss_counters` / `last_miss_turn` for
        the diagnose surface.
        """
        commitments = [c for c in proposed_commitments if isinstance(c, ActiveCommitment)]
        missed: list[dict[str, Any]] = []

        if is_external_turn:
            policy_count = sum(1 for c in commitments if c.source_kind == "policy")
            if policy_count < 1:
                missed.append(
                    {
                        "rule": RULE_POLICY_SOURCE_REQUIRED,
                        "actual_count": policy_count,
                    }
                )

            surface = (surface_intent or "").strip().lower()
            if surface in _RULE_B_SURFACE_INTENTS:
                rms_count = sum(1 for c in commitments if c.owner_id == "runtime_mode_state")
                if rms_count < 1:
                    missed.append(
                        {
                            "rule": RULE_RUNTIME_MODE_STATE_REQUIRED,
                            "actual_count": rms_count,
                            "surface_intent": surface,
                        }
                    )

        # Update counters.
        seen_rules: set[str] = set()
        for row in missed:
            rule = str(row.get("rule", "") or "")
            if not rule:
                continue
            seen_rules.add(rule)
            self._miss_counters[rule] = self._miss_counters.get(rule, 0) + 1
            self._last_miss_turn[rule] = int(turn_index)

        # Decrement counters for rules that DID pass (this turn).
        for rule in (
            RULE_POLICY_SOURCE_REQUIRED,
            RULE_RUNTIME_MODE_STATE_REQUIRED,
        ):
            if rule in seen_rules:
                continue
            # Only decrement on external turns (idles don't tick the
            # rule matrix, so their counters stay where they were).
            if is_external_turn:
                if self._miss_counters.get(rule, 0) > 0:
                    self._miss_counters[rule] -= 1

        return LoopCoverageVerdict(
            turn_index=int(turn_index),
            missed=tuple(missed),
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )


__all__ = [
    "LoopInvariants",
    "LoopCoverageVerdict",
    "RULE_POLICY_SOURCE_REQUIRED",
    "RULE_RUNTIME_MODE_STATE_REQUIRED",
    "build_minimum_loop_coverage_missed_event",
]
