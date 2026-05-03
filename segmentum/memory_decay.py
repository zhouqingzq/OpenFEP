from __future__ import annotations

from dataclasses import dataclass, field
from math import exp

from .memory_model import MemoryClass, MemoryEntry, StoreLevel


SHORT_CLEANUP_TRACE_THRESHOLD = 0.05
LONG_DORMANT_TRACE_THRESHOLD = 0.02
LONG_DORMANT_ACCESS_THRESHOLD = 0.01

TRACE_DECAY_RATES: dict[StoreLevel, float] = {
    StoreLevel.SHORT: 0.010,
    StoreLevel.MID: 0.002,
    StoreLevel.LONG: 0.0002,
}
ACCESS_DECAY_RATES: dict[StoreLevel, float] = {
    StoreLevel.SHORT: 0.050,
    StoreLevel.MID: 0.020,
    StoreLevel.LONG: 0.005,
}


@dataclass
class DecayReport:
    current_cycle: int
    deleted_short_residue: list[str] = field(default_factory=list)
    dormant_marked: list[str] = field(default_factory=list)
    abstracted_entries: list[str] = field(default_factory=list)
    source_confidence_drifted: list[str] = field(default_factory=list)
    reality_confidence_drifted: list[str] = field(default_factory=list)
    confidence_drifted: list[str] = field(default_factory=list)
    processed_entries: int = 0
    anchored_pruned: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "current_cycle": self.current_cycle,
            "deleted_short_residue": list(self.deleted_short_residue),
            "dormant_marked": list(self.dormant_marked),
            "abstracted_entries": list(self.abstracted_entries),
            "source_confidence_drifted": list(self.source_confidence_drifted),
            "reality_confidence_drifted": list(self.reality_confidence_drifted),
            "confidence_drifted": list(self.confidence_drifted),
            "processed_entries": self.processed_entries,
            "anchored_pruned": self.anchored_pruned,
        }


def decay_trace_strength(entry: MemoryEntry, elapsed_cycles: int) -> float:
    if elapsed_cycles <= 0:
        return entry.trace_strength
    return decay_trace_strength_for_level(
        entry.trace_strength,
        entry.store_level,
        elapsed_cycles,
        memory_class=entry.memory_class,
    )


def decay_accessibility(entry: MemoryEntry, elapsed_cycles: int) -> float:
    if elapsed_cycles <= 0:
        return entry.accessibility
    return decay_accessibility_for_level(entry.accessibility, entry.store_level, elapsed_cycles)


def trace_decay_rate(store_level: StoreLevel, *, memory_class: MemoryClass = MemoryClass.EPISODIC) -> float:
    decay_rate = TRACE_DECAY_RATES[store_level]
    if store_level is StoreLevel.LONG and memory_class is MemoryClass.PROCEDURAL:
        decay_rate *= 0.1
    return decay_rate


def access_decay_rate(store_level: StoreLevel) -> float:
    return ACCESS_DECAY_RATES[store_level]


def decay_trace_strength_for_level(
    base_trace_strength: float,
    store_level: StoreLevel,
    elapsed_cycles: int,
    *,
    memory_class: MemoryClass = MemoryClass.EPISODIC,
) -> float:
    if elapsed_cycles <= 0:
        return base_trace_strength
    decay_rate = trace_decay_rate(store_level, memory_class=memory_class)
    return max(0.0, base_trace_strength * exp(-(decay_rate * elapsed_cycles)))


def decay_accessibility_for_level(
    base_accessibility: float,
    store_level: StoreLevel,
    elapsed_cycles: int,
) -> float:
    if elapsed_cycles <= 0:
        return base_accessibility
    decay_rate = access_decay_rate(store_level)
    return max(0.0, base_accessibility * exp(-(decay_rate * elapsed_cycles)))
