"""Shared utilities for dialogue subpackage (clamping, normalisation, etc.)."""

from __future__ import annotations


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clip *value* to [*lo*, *hi*] — single canonical definition."""
    return max(lo, min(hi, float(value)))
