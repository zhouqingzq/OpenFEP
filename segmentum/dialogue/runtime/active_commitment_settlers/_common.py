"""Shared helpers for the M20.1 reference settlers."""

from __future__ import annotations

from typing import Any, Mapping


def bounded_evidence_refs(
    commitment_evidence_refs: list[str],
    observation_evidence_refs: list[str] | None,
    *,
    limit: int = 32,
) -> tuple[str, ...]:
    """Intersect and clamp `evidence_refs` for a `SettledValue` (M20.1 §1).

    Rules:
    - A settler MUST NOT invent new evidence ids.
    - It may only pick from the commitment's `evidence_refs` plus
      bounded turn-local observation handles (`turn_<n>_<slot>`).
    - The result MUST be non-empty; otherwise `evidence_ref_filtered`
      is surfaced.

    The simple v1 policy: prefer commitment_evidence_refs that are
    also present in observation_evidence_refs (when the latter is
    supplied), and fall back to the commitment's refs otherwise.
    """
    commit_set = {ref for ref in commitment_evidence_refs if isinstance(ref, str) and ref}
    if not commit_set:
        return ()
    if observation_evidence_refs:
        obs_set = {ref for ref in observation_evidence_refs if isinstance(ref, str) and ref}
        if obs_set:
            intersect = sorted(commit_set & obs_set)
            if intersect:
                return tuple(intersect[:limit])
    return tuple(sorted(commit_set)[:limit])


def get_observation_row(
    observation_context: Mapping[str, Any],
    *,
    key: str,
    match_field: str,
    match_value: str,
) -> dict[str, Any] | None:
    """Return the first row in `observation_context[key]` whose
    `match_field == match_value`, or None.
    """
    rows = observation_context.get(key)
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get(match_field) == match_value:
            return dict(row)
    return None


def clamp_to_unit_interval(value: float) -> float:
    if value != value:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
