from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _social_tag_set(tags: list[str]) -> set[str]:
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


@dataclass(slots=True)
class OtherModel:
    other_id: str
    display_name: str = ""
    trust: float = 0.5
    threat: float = 0.0
    reciprocity: float = 0.5
    predictability: float = 0.5
    attachment: float = 0.0
    interaction_count: int = 0
    last_seen_tick: int = 0
    rupture_count: int = 0
    repair_count: int = 0
    last_event_type: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "other_id": self.other_id,
            "display_name": self.display_name,
            "trust": self.trust,
            "threat": self.threat,
            "reciprocity": self.reciprocity,
            "predictability": self.predictability,
            "attachment": self.attachment,
            "interaction_count": self.interaction_count,
            "last_seen_tick": self.last_seen_tick,
            "rupture_count": self.rupture_count,
            "repair_count": self.repair_count,
            "last_event_type": self.last_event_type,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "OtherModel":
        if not payload:
            return cls(other_id="")
        return cls(
            other_id=str(payload.get("other_id", "")),
            display_name=str(payload.get("display_name", "")),
            trust=float(payload.get("trust", 0.5)),
            threat=float(payload.get("threat", 0.0)),
            reciprocity=float(payload.get("reciprocity", 0.5)),
            predictability=float(payload.get("predictability", 0.5)),
            attachment=float(payload.get("attachment", 0.0)),
            interaction_count=int(payload.get("interaction_count", 0)),
            last_seen_tick=int(payload.get("last_seen_tick", 0)),
            rupture_count=int(payload.get("rupture_count", 0)),
            repair_count=int(payload.get("repair_count", 0)),
            last_event_type=str(payload.get("last_event_type", "")),
            tags=[str(item) for item in payload.get("tags", [])],
        )


@dataclass(slots=True)
class SocialMemory:
    others: dict[str, OtherModel] = field(default_factory=dict)
    interaction_history: list[dict[str, object]] = field(default_factory=list)

    def observe_counterpart(
        self,
        *,
        other_id: str,
        tick: int,
        appraisal: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        tags: list[str] | None = None,
        event_type: str = "",
    ) -> OtherModel:
        if not other_id:
            return OtherModel(other_id="")
        payload = self.others.get(other_id, OtherModel(other_id=other_id))
        appraisal = appraisal or {}
        metadata = metadata or {}
        tags = tags or []
        tag_set = _social_tag_set(list(tags))

        display_name = str(metadata.get("counterpart_name", payload.display_name or other_id))
        trust_impact = float(appraisal.get("trust_impact", metadata.get("trust_impact", 0.0)))
        attachment_signal = float(
            appraisal.get("attachment_signal", metadata.get("attachment_signal", 0.0))
        )
        social_threat = max(
            0.0,
            float(appraisal.get("social_threat", metadata.get("social_threat", 0.0))),
        )
        uncertainty = max(
            0.0,
            float(appraisal.get("uncertainty", metadata.get("uncertainty", 0.0))),
        )
        reciprocity_signal = float(
            metadata.get("reciprocity_signal", trust_impact + attachment_signal * 0.25)
        )
        rupture = bool(metadata.get("rupture", False)) or bool(
            {"betrayal", "rejection", "threat", "rupture"} & tag_set
        )
        repair = bool(metadata.get("repair", False)) or bool(
            {"repair", "apology", "reconciliation", "help", "cooperation"} & tag_set
        )

        payload.display_name = display_name
        payload.interaction_count += 1
        payload.last_seen_tick = int(tick)
        payload.last_event_type = event_type
        payload.tags = sorted(set(payload.tags) | tag_set)

        payload.trust = _clamp(payload.trust * 0.78 + (0.5 + trust_impact * 0.5) * 0.22)
        payload.threat = _clamp(payload.threat * 0.70 + social_threat * 0.30)
        payload.reciprocity = _clamp(
            payload.reciprocity * 0.75 + (0.5 + reciprocity_signal * 0.5) * 0.25
        )
        payload.predictability = _clamp(payload.predictability * 0.70 + (1.0 - uncertainty) * 0.30)
        payload.attachment = _clamp(
            payload.attachment * 0.75 + (0.5 + attachment_signal * 0.5) * 0.25
        )

        if rupture:
            payload.rupture_count += 1
            payload.trust = _clamp(payload.trust - 0.25)
            payload.threat = _clamp(payload.threat + 0.20)
            payload.reciprocity = _clamp(payload.reciprocity - 0.12)
        if repair:
            payload.repair_count += 1
            payload.trust = _clamp(payload.trust + max(0.0, 0.18 - payload.threat * 0.10))
            payload.threat = _clamp(payload.threat - 0.10)
            payload.reciprocity = _clamp(payload.reciprocity + 0.08)

        self.others[other_id] = payload
        self.interaction_history.append(
            {
                "tick": int(tick),
                "other_id": other_id,
                "display_name": payload.display_name,
                "event_type": event_type,
                "rupture": rupture,
                "repair": repair,
                "trust": round(payload.trust, 6),
                "threat": round(payload.threat, 6),
                "reciprocity": round(payload.reciprocity, 6),
                "predictability": round(payload.predictability, 6),
                "attachment": round(payload.attachment, 6),
            }
        )
        self.interaction_history = self.interaction_history[-128:]
        return payload

    def policy_assessment(
        self,
        *,
        action: str,
        observation: Mapping[str, float],
    ) -> dict[str, object]:
        social_signal = float(observation.get("social", 0.0))
        if not self.others:
            return {"bias": 0.0, "focus": [], "alerts": [], "snapshot": self.snapshot()}

        ordered = sorted(
            self.others.values(),
            key=lambda model: (
                model.last_seen_tick,
                model.interaction_count,
                model.other_id,
            ),
            reverse=True,
        )
        trusted = [model for model in ordered if model.trust >= 0.58 and model.threat <= 0.48]
        threatening = [model for model in ordered if model.threat >= 0.45]
        focus = [model.other_id for model in trusted[:1] + threatening[:1]]
        alerts = [
            f"{model.other_id}:rupture-risk"
            for model in threatening[:1]
        ]
        bias = 0.0
        if action == "seek_contact":
            if trusted:
                best = trusted[0]
                bias += max(0.0, 0.55 - social_signal) * (
                    0.35 + best.trust * 0.35 + best.attachment * 0.15
                )
            if threatening:
                bias -= threatening[0].threat * 0.55
        elif action == "hide":
            if threatening:
                bias += threatening[0].threat * 0.35
        elif action == "scan":
            if threatening:
                bias += (1.0 - threatening[0].predictability) * 0.25
            elif trusted:
                bias += (1.0 - trusted[0].predictability) * 0.10

        return {
            "bias": max(-0.8, min(0.8, bias)),
            "focus": focus,
            "alerts": alerts,
            "snapshot": self.snapshot(),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "others": {
                other_id: model.to_dict()
                for other_id, model in sorted(self.others.items())
            },
            "history_size": len(self.interaction_history),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "others": {
                other_id: model.to_dict()
                for other_id, model in sorted(self.others.items())
            },
            "interaction_history": list(self.interaction_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SocialMemory":
        if not payload:
            return cls()
        others_raw = payload.get("others", {})
        history_raw = payload.get("interaction_history", [])
        memory = cls()
        if isinstance(others_raw, Mapping):
            memory.others = {
                str(other_id): OtherModel.from_dict(model)
                for other_id, model in others_raw.items()
                if isinstance(model, Mapping)
            }
        if isinstance(history_raw, list):
            memory.interaction_history = [
                dict(item) for item in history_raw if isinstance(item, Mapping)
            ]
        return memory
