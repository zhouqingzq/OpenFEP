"""M8.9 State Ownership Contract.

Encodes the ownership table for every self-related state surface as runtime
data so tests can verify:

- Who owns each state surface.
- What update path is canonical.
- How it may (or may not) appear in prompts.

This module does NOT participate in the cognitive loop. It is a static
contract that other modules can reference and tests can audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptEligibility = Literal[
    "compressed_only",
    "summary_only",
    "compact_self_prior_only",
    "evidence_filtered_only",
    "bounded_guidance_only",
    "prompt_safe_fields_only",
]


@dataclass(frozen=True)
class StateOwnershipEntry:
    """A single entry in the state ownership table.

    Attributes:
        state_surface: The self-related state or data type.
        owner: Named module, reducer, or component that owns writes.
        update_path: Canonical path through which this surface is updated.
        prompt_eligibility: How this surface may appear in prompts.
    """

    state_surface: str
    owner: str
    update_path: str
    prompt_eligibility: PromptEligibility


# ── Canonical ownership table ────────────────────────────────────────────
#
# Every self-related state surface must appear here.  The table is
# referenced by tests that verify prompt-facing code respects these
# boundaries.

STATE_OWNERSHIP_TABLE: tuple[StateOwnershipEntry, ...] = (
    StateOwnershipEntry(
        state_surface="CognitiveStateMVP",
        owner="cognitive state reducer",
        update_path="CognitiveLoop.consume_and_update",
        prompt_eligibility="compressed_only",
    ),
    StateOwnershipEntry(
        state_surface="SubjectState",
        owner="subject state owner",
        update_path="explicit derivation or later patch commit",
        prompt_eligibility="summary_only",
    ),
    StateOwnershipEntry(
        state_surface="SelfModel",
        owner="slow persona model owner",
        update_path="slow consolidation / audited update",
        prompt_eligibility="compact_self_prior_only",
    ),
    StateOwnershipEntry(
        state_surface="MemoryStore / anchored memory",
        owner="memory owner",
        update_path="write intent or explicit commit",
        prompt_eligibility="evidence_filtered_only",
    ),
    StateOwnershipEntry(
        state_surface="MetaControlSignal",
        owner="meta-control owner",
        update_path="deterministic derivation from state",
        prompt_eligibility="bounded_guidance_only",
    ),
    StateOwnershipEntry(
        state_surface="FEPPromptCapsule",
        owner="prompt adapter",
        update_path="compressed summary builder",
        prompt_eligibility="prompt_safe_fields_only",
    ),
)


def get_ownership_for(surface_name: str) -> StateOwnershipEntry | None:
    """Look up the ownership entry for a named state surface."""
    for entry in STATE_OWNERSHIP_TABLE:
        if entry.state_surface == surface_name:
            return entry
    return None
