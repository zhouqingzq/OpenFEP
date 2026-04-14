"""Deterministic sub-seeds for M5.3 policy / surface layers (no wall-clock entropy)."""

from __future__ import annotations

import hashlib
from typing import Any


def derive_subseed(master_seed: int, domain: str, *parts: Any) -> int:
    """Stable 63-bit int from master seed + domain + parts (same across processes)."""
    payload = "|".join(str(p) for p in (master_seed, domain) + parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) >> 1


def pick_index(master_seed: int, domain: str, *parts: Any, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return int(derive_subseed(master_seed, domain, *parts) % modulo)
