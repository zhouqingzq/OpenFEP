from __future__ import annotations

from .agent import SegmentAgent


class MemoryAwareSegmentAgent(SegmentAgent):
    """Thin compatibility wrapper now that SegmentAgent is memory-aware by default."""

    pass
