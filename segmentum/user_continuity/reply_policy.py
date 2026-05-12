"""M12.0 deterministic reply-policy selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .identity_conflict_detector import ConflictRecord
from .identity_profile import IdentityProfile
from .strangeness_signal import IdentityStrangenessSignal


@dataclass(frozen=True)
class ReplyPolicyDecision:
    permitted_response: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "permitted_response": self.permitted_response,
            "reasons": list(self.reasons),
        }


def select_reply_policy(
    *,
    profile: IdentityProfile,
    active_conflicts: Sequence[ConflictRecord],
    strangeness_signal: IdentityStrangenessSignal | None,
    identity_anchored_action: bool,
) -> ReplyPolicyDecision:
    major_conflict = any(record.severity_band == "major" for record in active_conflicts)
    if profile.identity_state == "unverified" and identity_anchored_action:
        return ReplyPolicyDecision("ask", ("unverified_alias",))
    if major_conflict and identity_anchored_action:
        return ReplyPolicyDecision("refuse", ("major_identity_conflict_sensitive_request",))
    if major_conflict:
        return ReplyPolicyDecision("ask", ("major_identity_conflict",))
    if strangeness_signal is not None and strangeness_signal.strangeness_band in {"med", "high"}:
        return ReplyPolicyDecision("probe", ("active_identity_strangeness",))
    if profile.identity_state == "corroborated":
        return ReplyPolicyDecision("accept", ("corroborated_binding_no_conflict",))
    if profile.identity_state == "retracted":
        return ReplyPolicyDecision("observe", ("recent_retraction",))
    return ReplyPolicyDecision("hedge", ("insufficient_continuity_evidence",))
