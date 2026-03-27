from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping

from .narrative_types import SemanticGrounding


def _coerce_str_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(value) for value in values if str(value))


@dataclass(frozen=True)
class SchemaSupport:
    episode_id: str
    source_type: str
    confidence: float
    motifs: tuple[str, ...] = ()
    contexts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "source_type": self.source_type,
            "confidence": round(float(self.confidence), 6),
            "motifs": list(self.motifs),
            "contexts": list(self.contexts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SchemaSupport":
        if not payload:
            return cls(episode_id="", source_type="", confidence=0.0)
        return cls(
            episode_id=str(payload.get("episode_id", "")),
            source_type=str(payload.get("source_type", "")),
            confidence=float(payload.get("confidence", 0.0)),
            motifs=_coerce_str_tuple(payload.get("motifs", [])),
            contexts=_coerce_str_tuple(payload.get("contexts", [])),
        )


@dataclass(frozen=True)
class SchemaConflict:
    schema_id: str
    conflicting_episode_id: str
    reason: str
    severity: float
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "conflicting_episode_id": self.conflicting_episode_id,
            "reason": self.reason,
            "severity": round(float(self.severity), 6),
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SchemaConflict":
        if not payload:
            return cls(schema_id="", conflicting_episode_id="", reason="", severity=0.0, outcome="")
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            conflicting_episode_id=str(payload.get("conflicting_episode_id", "")),
            reason=str(payload.get("reason", "")),
            severity=float(payload.get("severity", 0.0)),
            outcome=str(payload.get("outcome", "")),
        )


@dataclass(frozen=True)
class SemanticSchema:
    schema_id: str
    label: str
    motif_signature: tuple[str, ...]
    dominant_direction: str
    confidence: float
    support_count: int
    applicable_contexts: tuple[str, ...] = ()
    protected_anchors: tuple[str, ...] = ()
    support: tuple[SchemaSupport, ...] = ()
    conflict_history: tuple[SchemaConflict, ...] = ()
    split_from_schema_id: str = ""
    active: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "label": self.label,
            "motif_signature": list(self.motif_signature),
            "dominant_direction": self.dominant_direction,
            "confidence": round(float(self.confidence), 6),
            "support_count": int(self.support_count),
            "applicable_contexts": list(self.applicable_contexts),
            "protected_anchors": list(self.protected_anchors),
            "support": [item.to_dict() for item in self.support],
            "conflict_history": [item.to_dict() for item in self.conflict_history],
            "split_from_schema_id": self.split_from_schema_id,
            "active": bool(self.active),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SemanticSchema":
        if not payload:
            return cls(
                schema_id="",
                label="",
                motif_signature=(),
                dominant_direction="",
                confidence=0.0,
                support_count=0,
            )
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            label=str(payload.get("label", "")),
            motif_signature=_coerce_str_tuple(payload.get("motif_signature", [])),
            dominant_direction=str(payload.get("dominant_direction", "")),
            confidence=float(payload.get("confidence", 0.0)),
            support_count=int(payload.get("support_count", 0)),
            applicable_contexts=_coerce_str_tuple(payload.get("applicable_contexts", [])),
            protected_anchors=_coerce_str_tuple(payload.get("protected_anchors", [])),
            support=tuple(
                SchemaSupport.from_dict(item) for item in payload.get("support", []) if isinstance(item, dict)
            ),
            conflict_history=tuple(
                SchemaConflict.from_dict(item)
                for item in payload.get("conflict_history", [])
                if isinstance(item, dict)
            ),
            split_from_schema_id=str(payload.get("split_from_schema_id", "")),
            active=bool(payload.get("active", True)),
        )


@dataclass(frozen=True)
class SchemaUpdateResult:
    created_schema_ids: tuple[str, ...] = ()
    strengthened_schema_ids: tuple[str, ...] = ()
    weakened_schema_ids: tuple[str, ...] = ()
    split_schema_ids: tuple[str, ...] = ()
    archived_schema_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "created_schema_ids": list(self.created_schema_ids),
            "strengthened_schema_ids": list(self.strengthened_schema_ids),
            "weakened_schema_ids": list(self.weakened_schema_ids),
            "split_schema_ids": list(self.split_schema_ids),
            "archived_schema_ids": list(self.archived_schema_ids),
        }


class SemanticSchemaStore:
    """Derive reusable schemas from narrative-grounded memory episodes."""

    def build_from_groundings(
        self,
        episode_payloads: list[dict[str, object]],
    ) -> tuple[list[SemanticSchema], SchemaUpdateResult]:
        by_signature: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
        for payload in episode_payloads:
            grounding_payload = payload.get("semantic_grounding")
            if not isinstance(grounding_payload, dict):
                provenance = payload.get("narrative_provenance", {})
                if isinstance(provenance, dict):
                    grounding_payload = provenance.get("semantic_grounding")
            if not isinstance(grounding_payload, dict):
                continue
            grounding = SemanticGrounding.from_dict(grounding_payload)
            signature = tuple(sorted(set(grounding.motifs)))
            if signature:
                by_signature[signature].append(payload)

        schemas: list[SemanticSchema] = []
        created: list[str] = []
        strengthened: list[str] = []
        weakened: list[str] = []
        split: list[str] = []
        for signature, payloads in sorted(by_signature.items()):
            if not signature:
                continue
            support: list[SchemaSupport] = []
            contexts: set[str] = set()
            anchors: set[str] = set()
            directions: dict[str, float] = defaultdict(float)
            confidence_values: list[float] = []
            outcome_labels = {str(payload.get("predicted_outcome", "neutral")) for payload in payloads}
            for payload in payloads:
                grounding_payload = payload.get("semantic_grounding")
                if not isinstance(grounding_payload, dict):
                    provenance = payload.get("narrative_provenance", {})
                    grounding_payload = provenance.get("semantic_grounding") if isinstance(provenance, dict) else {}
                grounding = SemanticGrounding.from_dict(grounding_payload if isinstance(grounding_payload, dict) else {})
                confidence_values.append(max(0.1, float(payload.get("compiler_confidence", 0.0))))
                for direction, score in grounding.semantic_direction_scores.items():
                    directions[direction] += float(score)
                contexts.update(str(tag) for tag in payload.get("narrative_tags", []) if str(tag))
                contexts.update(str(tag) for tag in payload.get("continuity_tags", []) if str(tag))
                if bool(payload.get("identity_critical", False)):
                    anchors.add("identity_critical")
                if bool(payload.get("restart_protected", False)):
                    anchors.add("restart_protected")
                support.append(
                    SchemaSupport(
                        episode_id=str(payload.get("episode_id", payload.get("source_episode_id", ""))),
                        source_type=str(payload.get("source_type", payload.get("source", "memory"))),
                        confidence=max(0.1, float(payload.get("compiler_confidence", 0.0))),
                        motifs=tuple(signature),
                        contexts=tuple(sorted(contexts))[:6],
                    )
                )
            dominant_direction = sorted(directions.items(), key=lambda item: (-item[1], item[0]))[0][0]
            conflict_history: list[SchemaConflict] = []
            if len(outcome_labels) > 1:
                conflict_history.append(
                    SchemaConflict(
                        schema_id=f"schema:{'-'.join(signature)}",
                        conflicting_episode_id=support[-1].episode_id if support else "",
                        reason="divergent_outcomes",
                        severity=min(1.0, 0.25 + len(outcome_labels) * 0.15),
                        outcome="split" if len(signature) > 1 else "downgrade",
                    )
                )
            schema_id = f"schema:{'-'.join(signature)}"
            confidence = min(0.98, mean(confidence_values) + min(0.25, len(payloads) * 0.04))
            active = True
            split_from = ""
            if conflict_history and conflict_history[0].outcome == "split":
                split.append(schema_id)
                split_from = schema_id
            elif conflict_history:
                weakened.append(schema_id)
                confidence = max(0.2, confidence - 0.18)
            else:
                strengthened.append(schema_id)
            created.append(schema_id)
            schemas.append(
                SemanticSchema(
                    schema_id=schema_id,
                    label=signature[0],
                    motif_signature=signature,
                    dominant_direction=dominant_direction,
                    confidence=confidence,
                    support_count=len(payloads),
                    applicable_contexts=tuple(sorted(contexts))[:8],
                    protected_anchors=tuple(sorted(anchors)),
                    support=tuple(support[-16:]),
                    conflict_history=tuple(conflict_history),
                    split_from_schema_id=split_from,
                    active=active,
                )
            )
        update = SchemaUpdateResult(
            created_schema_ids=tuple(created),
            strengthened_schema_ids=tuple(strengthened),
            weakened_schema_ids=tuple(weakened),
            split_schema_ids=tuple(split),
            archived_schema_ids=(),
        )
        return schemas, update
