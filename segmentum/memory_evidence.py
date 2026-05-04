"""M9.0 Unified Memory Evidence Surface.

A single compact evidence structure that represents anchored facts,
retrieved episodic memory, semantic memory, and inferred hypotheses
without losing source or confidence.  Consumed by cognitive state and
generation through the response evidence contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

MemoryClass = Literal[
    "anchored_fact",
    "retrieved_episodic",
    "semantic",
    "inferred_hypothesis",
    "user_assertion",
    "agent_self_utterance",
]

PermittedUse = Literal[
    "explicit_fact",
    "cautious_hypothesis",
    "strategy_only",
    "forbidden",
]

ConflictStatus = Literal[
    "none",
    "resolved",
    "unresolved",
    "overdominant",
]

DecayState = Literal[
    "fresh",
    "active",
    "fading",
    "dormant",
    "pruned",
]


@dataclass(frozen=True)
class MemoryEvidence:
    """Unified memory evidence for a single memory item.

    Represents anchored facts, retrieved episodic memory, semantic memory,
    and inferred hypotheses in one compact structure.  Generation can
    distinguish an asserted user statement from a verified fact by checking
    ``memory_class`` and ``confidence`` together.
    """

    memory_id: str = ""
    memory_class: MemoryClass = "anchored_fact"
    source_turn_id: str = ""
    source_utterance_id: str = ""
    speaker: str = ""
    status: str = ""  # asserted | corroborated | retracted | hypothesis
    confidence: float = 0.0
    retrieval_score: float = 0.0
    cue_match: str = ""  # the specific cue that triggered recall
    salience: float = 0.0
    value_score: float = 0.0
    decay_state: DecayState = "fresh"
    conflict_status: ConflictStatus = "none"
    permitted_use: PermittedUse = "explicit_fact"
    content_summary: str = ""

    # ── Classification helpers ──────────────────────────────────────────

    @property
    def is_user_statement(self) -> bool:
        return self.memory_class == "user_assertion"

    @property
    def is_verified_fact(self) -> bool:
        return (
            self.memory_class == "anchored_fact"
            and self.status == "corroborated"
            and self.confidence >= 0.8
        )

    @property
    def is_hypothesis(self) -> bool:
        return self.memory_class in ("inferred_hypothesis",)

    @property
    def is_agent_utterance(self) -> bool:
        return self.memory_class == "agent_self_utterance"

    @property
    def can_be_asserted(self) -> bool:
        """True when this evidence is safe to state as a fact in generation."""
        return (
            self.permitted_use == "explicit_fact"
            and self.confidence >= 0.6
            and self.status not in ("retracted",)
            and self.conflict_status != "unresolved"
            and self.decay_state not in ("dormant", "pruned")
        )

    @property
    def should_be_cautious(self) -> bool:
        """True when the evidence exists but must be hedged."""
        return (
            self.permitted_use == "cautious_hypothesis"
            or self.status == "hypothesis"
            or self.confidence < 0.6
            or self.conflict_status == "unresolved"
            or self.decay_state == "fading"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "memory_class": self.memory_class,
            "source_turn_id": self.source_turn_id,
            "source_utterance_id": self.source_utterance_id,
            "speaker": self.speaker,
            "status": self.status,
            "confidence": self.confidence,
            "retrieval_score": self.retrieval_score,
            "cue_match": self.cue_match,
            "salience": self.salience,
            "value_score": self.value_score,
            "decay_state": self.decay_state,
            "conflict_status": self.conflict_status,
            "permitted_use": self.permitted_use,
            "content_summary": self.content_summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MemoryEvidence":
        return cls(
            memory_id=str(payload.get("memory_id", "")),
            memory_class=str(payload.get("memory_class", "anchored_fact")),
            source_turn_id=str(payload.get("source_turn_id", "")),
            source_utterance_id=str(payload.get("source_utterance_id", "")),
            speaker=str(payload.get("speaker", "")),
            status=str(payload.get("status", "")),
            confidence=float(payload.get("confidence", 0.0)),
            retrieval_score=float(payload.get("retrieval_score", 0.0)),
            cue_match=str(payload.get("cue_match", "")),
            salience=float(payload.get("salience", 0.0)),
            value_score=float(payload.get("value_score", 0.0)),
            decay_state=str(payload.get("decay_state", "fresh")),
            conflict_status=str(payload.get("conflict_status", "none")),
            permitted_use=str(payload.get("permitted_use", "explicit_fact")),
            content_summary=str(payload.get("content_summary", "")),
        )


# ── Converters from existing types ──────────────────────────────────────

def evidence_from_anchored_item(
    item: object,
    *,
    retrieval_score: float = 0.0,
    cue_match: str = "",
    value_score: float = 0.5,
    decay_state: DecayState = "fresh",
    conflict_status: ConflictStatus = "none",
) -> MemoryEvidence:
    """Convert an AnchoredMemoryItem into a MemoryEvidence record."""
    if not hasattr(item, "to_dict"):
        return MemoryEvidence(memory_id="error:not_an_anchored_item")
    d = getattr(item, "to_dict", lambda: {})()
    status = str(d.get("status", "asserted"))
    visibility = str(d.get("visibility", "explicit"))
    memory_type = str(d.get("memory_type", "user_fact"))

    if memory_type == "agent_self_utterance":
        memory_class: MemoryClass = "agent_self_utterance"
        permitted_use: PermittedUse = "strategy_only"
    elif memory_type in ("hypothesis", "system_inferred_hypothesis"):
        memory_class = "inferred_hypothesis"
        permitted_use = "cautious_hypothesis"
    elif status == "asserted":
        memory_class = "user_assertion" if str(d.get("speaker", "")) == "user" else "anchored_fact"
        permitted_use = "explicit_fact" if visibility == "explicit" else "strategy_only"
    elif status == "corroborated":
        memory_class = "anchored_fact"
        permitted_use = "explicit_fact"
    elif status == "retracted":
        memory_class = "anchored_fact"
        permitted_use = "forbidden"
    else:
        memory_class = "anchored_fact"
        permitted_use = "cautious_hypothesis"

    return MemoryEvidence(
        memory_id=str(d.get("memory_id", "")),
        memory_class=memory_class,
        source_turn_id=str(d.get("turn_id", "")),
        source_utterance_id=str(d.get("utterance_id", "")),
        speaker=str(d.get("speaker", "")),
        status=status,
        confidence=float(d.get("confidence", 0.0)),
        retrieval_score=retrieval_score,
        cue_match=cue_match,
        salience=float(d.get("salience", 0.5)),
        value_score=value_score,
        decay_state=decay_state,
        conflict_status=conflict_status,
        permitted_use=permitted_use,
        content_summary=str(d.get("proposition", "")),
    )


def evidence_from_retrieval(
    retrieval_entry: Mapping[str, object],
    *,
    cue_match: str = "",
) -> MemoryEvidence:
    """Convert a retrieval result entry into MemoryEvidence."""
    memory_id = str(retrieval_entry.get("episode_id", retrieval_entry.get("memory_id", "")))
    status = str(retrieval_entry.get("status", "asserted"))
    confidence = float(retrieval_entry.get("confidence", retrieval_entry.get("retrieval_score", 0.5)))

    return MemoryEvidence(
        memory_id=memory_id,
        memory_class="retrieved_episodic",
        source_turn_id=str(retrieval_entry.get("turn_id", retrieval_entry.get("source_turn_id", ""))),
        source_utterance_id=str(retrieval_entry.get("utterance_id", "")),
        speaker=str(retrieval_entry.get("speaker", "")),
        status=status,
        confidence=confidence,
        retrieval_score=float(retrieval_entry.get("retrieval_score", confidence)),
        cue_match=cue_match,
        salience=float(retrieval_entry.get("salience", 0.5)),
        value_score=float(retrieval_entry.get("value_score", 0.5)),
        decay_state=str(retrieval_entry.get("decay_state", "active")),
        conflict_status=str(retrieval_entry.get("conflict_status", "none")),
        permitted_use="cautious_hypothesis" if confidence < 0.6 else "explicit_fact",
        content_summary=str(retrieval_entry.get("proposition",
            retrieval_entry.get("summary", retrieval_entry.get("content", "")))),
    )


def evidence_from_hypothesis(
    hypothesis_text: str,
    *,
    memory_id: str = "",
    confidence: float = 0.4,
    source_turn_id: str = "",
    cue_match: str = "",
) -> MemoryEvidence:
    """Create a MemoryEvidence entry for an inferred hypothesis."""
    return MemoryEvidence(
        memory_id=memory_id or f"hyp:{hash(hypothesis_text) & 0x7FFFFFFF:08x}",
        memory_class="inferred_hypothesis",
        source_turn_id=source_turn_id,
        source_utterance_id="",
        speaker="system",
        status="hypothesis",
        confidence=confidence,
        retrieval_score=0.0,
        cue_match=cue_match,
        salience=0.3,
        value_score=0.3,
        decay_state="fading",
        conflict_status="none",
        permitted_use="cautious_hypothesis",
        content_summary=hypothesis_text,
    )


# ── Collection helpers ──────────────────────────────────────────────────

def unify_evidence(
    *,
    anchored_items: Sequence[object] | None = None,
    retrieval_results: Sequence[Mapping[str, object]] | None = None,
    hypotheses: Sequence[str] | None = None,
    current_cue: str = "",
) -> list[MemoryEvidence]:
    """Collect all evidence sources into a unified MemoryEvidence list.

    Items are sorted by: permitted_use priority, then confidence descending,
    then value_score descending.
    """
    combined: list[MemoryEvidence] = []

    for item in (anchored_items or []):
        combined.append(evidence_from_anchored_item(item, cue_match=current_cue))

    for entry in (retrieval_results or []):
        combined.append(evidence_from_retrieval(entry, cue_match=current_cue))

    for hypothesis in (hypotheses or []):
        combined.append(evidence_from_hypothesis(hypothesis, cue_match=current_cue))

    _permitted_order: dict[str, int] = {
        "explicit_fact": 0,
        "cautious_hypothesis": 1,
        "strategy_only": 2,
        "forbidden": 3,
    }

    combined.sort(key=lambda e: (
        _permitted_order.get(e.permitted_use, 9),
        -e.confidence,
        -e.value_score,
        e.memory_id,
    ))
    return combined


def evidence_summary_for_prompt(
    evidence_list: Sequence[MemoryEvidence],
    *,
    max_items: int = 8,
) -> str:
    """Produce a prompt-safe compact summary of unified evidence."""
    visible = [e for e in evidence_list if e.permitted_use != "forbidden"]
    lines: list[str] = ["[Memory Evidence]"]

    for e in visible[:max_items]:
        tag = (
            "FACT" if e.can_be_asserted
            else "HYP" if e.should_be_cautious
            else "INFO"
        )
        source = f"@{e.speaker}" if e.speaker else ""
        decay = "" if e.decay_state in ("fresh", "active") else f"[{e.decay_state}]"
        conflict = " [CONFLICT]" if e.conflict_status == "unresolved" else ""
        lines.append(
            f"  [{tag}]{source} {e.content_summary}{decay}{conflict} "
            f"(conf={e.confidence:.2f})"
        )

    if len(visible) > max_items:
        lines.append(f"  ... ({len(visible) - max_items} more items)")

    if not any(e.can_be_asserted for e in visible):
        lines.append("  [No high-confidence evidence available]")

    return "\n".join(lines)
