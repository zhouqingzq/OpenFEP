"""M12.2 reciprocal-role deterministic limits and bands."""

from __future__ import annotations

from dataclasses import dataclass


BANDS: tuple[str, ...] = ("low", "medium", "high")
BAND_ORDER: dict[str, int] = {"insufficient_evidence": -1, "low": 0, "medium": 1, "high": 2}
BAND_BY_ORDER: dict[int, str] = {-1: "insufficient_evidence", 0: "low", 1: "medium", 2: "high"}


@dataclass(frozen=True)
class M122Hyperparams:
    hyperparams_version: str = "m12.2.v1"
    default_persona_label: str = "hutao"
    max_claims_per_axis: int = 24
    max_groups_per_axis: int = 16
    max_uncertainty_points: int = 16
    max_candidates: int = 16
    max_boundaries: int = 16
    max_summary_chars: int = 260
    max_action_chars: int = 220
    second_order_high_recent_probe_turns: int = 5
    max_consolidations_per_hour: int = 2
    min_turn_window_between_consolidations: int = 2
    cooldown_window_turns: int = 5
    cooldown_threshold: int = 2
    cooldown_turns: int = 3
    max_transcript_quote_refs: int = 12
    max_readonly_summary_items: int = 8


DEFAULT_HYPERPARAMS = M122Hyperparams()


def normalize_band(value: object, *, default: str = "low") -> str:
    text = str(value or default)
    return text if text in BAND_ORDER else default


def lower_band(value: str) -> str:
    level = max(-1, BAND_ORDER.get(value, 0) - 1)
    return BAND_BY_ORDER[level]


def raise_band(value: str) -> str:
    level = min(2, BAND_ORDER.get(value, 0) + 1)
    return BAND_BY_ORDER[level]
