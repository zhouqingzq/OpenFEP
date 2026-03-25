from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from .homeostasis import MaintenanceAgenda

if TYPE_CHECKING:
    from .agent import SegmentAgent
    from .types import DecisionDiagnostics


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class DominantNeed:
    name: str
    intensity: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "intensity": round(self.intensity, 6), "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "DominantNeed":
        if not payload:
            return cls(name="", intensity=0.0, reason="")
        return cls(
            name=str(payload.get("name", "")),
            intensity=float(payload.get("intensity", 0.0)),
            reason=str(payload.get("reason", "")),
        )


@dataclass(frozen=True)
class ActiveTension:
    label: str
    tension_type: str
    intensity: float
    repair_target: str = ""
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "tension_type": self.tension_type,
            "intensity": round(self.intensity, 6),
            "repair_target": self.repair_target,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ActiveTension":
        if not payload:
            return cls(label="", tension_type="none", intensity=0.0)
        return cls(
            label=str(payload.get("label", "")),
            tension_type=str(payload.get("tension_type", "none")),
            intensity=float(payload.get("intensity", 0.0)),
            repair_target=str(payload.get("repair_target", "")),
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass(frozen=True)
class SubjectBinding:
    binding_id: str
    binding_type: str
    salience: float
    summary: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "binding_type": self.binding_type,
            "salience": round(self.salience, 6),
            "summary": self.summary,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SubjectBinding":
        if not payload:
            return cls(binding_id="", binding_type="", salience=0.0, summary="")
        return cls(
            binding_id=str(payload.get("binding_id", "")),
            binding_type=str(payload.get("binding_type", "")),
            salience=float(payload.get("salience", 0.0)),
            summary=str(payload.get("summary", "")),
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass(frozen=True)
class SubjectPriority:
    label: str
    weight: float
    priority_type: str
    preferred_actions: tuple[str, ...] = ()
    avoid_actions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "weight": round(self.weight, 6),
            "priority_type": self.priority_type,
            "preferred_actions": list(self.preferred_actions),
            "avoid_actions": list(self.avoid_actions),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SubjectPriority":
        if not payload:
            return cls(label="", weight=0.0, priority_type="")
        return cls(
            label=str(payload.get("label", "")),
            weight=float(payload.get("weight", 0.0)),
            priority_type=str(payload.get("priority_type", "")),
            preferred_actions=tuple(str(item) for item in payload.get("preferred_actions", [])),
            avoid_actions=tuple(str(item) for item in payload.get("avoid_actions", [])),
            evidence=tuple(str(item) for item in payload.get("evidence", [])),
        )


@dataclass(frozen=True)
class NarrativeUncertaintyFocus:
    unknown_id: str
    label: str
    unknown_type: str
    relevance: float
    linked_entities: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "unknown_id": self.unknown_id,
            "label": self.label,
            "unknown_type": self.unknown_type,
            "relevance": round(self.relevance, 6),
            "linked_entities": list(self.linked_entities),
            "hypotheses": list(self.hypotheses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NarrativeUncertaintyFocus":
        if not payload:
            return cls(unknown_id="", label="", unknown_type="", relevance=0.0)
        return cls(
            unknown_id=str(payload.get("unknown_id", "")),
            label=str(payload.get("label", "")),
            unknown_type=str(payload.get("unknown_type", "")),
            relevance=float(payload.get("relevance", 0.0)),
            linked_entities=tuple(str(item) for item in payload.get("linked_entities", [])),
            hypotheses=tuple(str(item) for item in payload.get("hypotheses", [])),
        )


@dataclass(frozen=True)
class InquiryFocus:
    plan_id: str
    action: str
    status: str
    target_unknown_id: str
    salience: float
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "action": self.action,
            "status": self.status,
            "target_unknown_id": self.target_unknown_id,
            "salience": round(self.salience, 6),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "InquiryFocus":
        if not payload:
            return cls(plan_id="", action="", status="", target_unknown_id="", salience=0.0, summary="")
        return cls(
            plan_id=str(payload.get("plan_id", "")),
            action=str(payload.get("action", "")),
            status=str(payload.get("status", "")),
            target_unknown_id=str(payload.get("target_unknown_id", "")),
            salience=float(payload.get("salience", 0.0)),
            summary=str(payload.get("summary", "")),
        )


@dataclass(frozen=True)
class SubjectState:
    tick: int = 0
    core_identity_summary: str = ""
    current_phase: str = "forming"
    current_self_narrative_phase: str = "forming"
    active_commitments: tuple[str, ...] = ()
    dominant_preferences: tuple[str, ...] = ()
    dominant_goal: str = ""
    dominant_needs: tuple[DominantNeed, ...] = ()
    dominant_workspace_contents: tuple[str, ...] = ()
    active_social_focus: tuple[str, ...] = ()
    socially_salient_bindings: tuple[SubjectBinding, ...] = ()
    identity_tension_level: float = 0.0
    self_inconsistency_level: float = 0.0
    maintenance_pressure: float = 0.0
    continuity_score: float = 1.0
    continuity_anchors: tuple[str, ...] = ()
    unresolved_tensions: tuple[ActiveTension, ...] = ()
    narrative_uncertainties: tuple[NarrativeUncertaintyFocus, ...] = ()
    active_inquiries: tuple[InquiryFocus, ...] = ()
    deferred_inquiries: tuple[InquiryFocus, ...] = ()
    ambiguity_profile: dict[str, float] = field(default_factory=dict)
    subject_priority_stack: tuple[SubjectPriority, ...] = ()
    slow_biases: dict[str, float] = field(default_factory=dict)
    status_flags: dict[str, bool] = field(default_factory=dict)
    protected_targets: tuple[str, ...] = ()
    repair_targets: tuple[str, ...] = ()
    same_subject_basis: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "core_identity_summary": self.core_identity_summary,
            "current_phase": self.current_phase,
            "current_self_narrative_phase": self.current_self_narrative_phase,
            "active_commitments": list(self.active_commitments),
            "dominant_preferences": list(self.dominant_preferences),
            "dominant_goal": self.dominant_goal,
            "dominant_needs": [item.to_dict() for item in self.dominant_needs],
            "dominant_workspace_contents": list(self.dominant_workspace_contents),
            "active_social_focus": list(self.active_social_focus),
            "socially_salient_bindings": [item.to_dict() for item in self.socially_salient_bindings],
            "identity_tension_level": round(self.identity_tension_level, 6),
            "self_inconsistency_level": round(self.self_inconsistency_level, 6),
            "maintenance_pressure": round(self.maintenance_pressure, 6),
            "continuity_score": round(self.continuity_score, 6),
            "continuity_anchors": list(self.continuity_anchors),
            "unresolved_tensions": [item.to_dict() for item in self.unresolved_tensions],
            "narrative_uncertainties": [item.to_dict() for item in self.narrative_uncertainties],
            "active_inquiries": [item.to_dict() for item in self.active_inquiries],
            "deferred_inquiries": [item.to_dict() for item in self.deferred_inquiries],
            "ambiguity_profile": {
                str(key): float(value) for key, value in self.ambiguity_profile.items()
            },
            "subject_priority_stack": [item.to_dict() for item in self.subject_priority_stack],
            "slow_biases": {str(key): float(value) for key, value in self.slow_biases.items()},
            "status_flags": {str(key): bool(value) for key, value in self.status_flags.items()},
            "protected_targets": list(self.protected_targets),
            "repair_targets": list(self.repair_targets),
            "same_subject_basis": self.same_subject_basis,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SubjectState":
        if not payload:
            return cls()
        status_flags = payload.get("status_flags", {})
        return cls(
            tick=int(payload.get("tick", 0)),
            core_identity_summary=str(payload.get("core_identity_summary", "")),
            current_phase=str(payload.get("current_phase", "forming")),
            current_self_narrative_phase=str(
                payload.get("current_self_narrative_phase", payload.get("current_phase", "forming"))
            ),
            active_commitments=tuple(str(item) for item in payload.get("active_commitments", [])),
            dominant_preferences=tuple(str(item) for item in payload.get("dominant_preferences", [])),
            dominant_goal=str(payload.get("dominant_goal", "")),
            dominant_needs=tuple(
                DominantNeed.from_dict(item)
                for item in payload.get("dominant_needs", [])
                if isinstance(item, Mapping)
            ),
            dominant_workspace_contents=tuple(
                str(item) for item in payload.get("dominant_workspace_contents", [])
            ),
            active_social_focus=tuple(str(item) for item in payload.get("active_social_focus", [])),
            socially_salient_bindings=tuple(
                SubjectBinding.from_dict(item)
                for item in payload.get("socially_salient_bindings", [])
                if isinstance(item, Mapping)
            ),
            identity_tension_level=float(payload.get("identity_tension_level", 0.0)),
            self_inconsistency_level=float(payload.get("self_inconsistency_level", 0.0)),
            maintenance_pressure=float(payload.get("maintenance_pressure", 0.0)),
            continuity_score=float(payload.get("continuity_score", 1.0)),
            continuity_anchors=tuple(str(item) for item in payload.get("continuity_anchors", [])),
            unresolved_tensions=tuple(
                ActiveTension.from_dict(item)
                for item in payload.get("unresolved_tensions", [])
                if isinstance(item, Mapping)
            ),
            narrative_uncertainties=tuple(
                NarrativeUncertaintyFocus.from_dict(item)
                for item in payload.get("narrative_uncertainties", [])
                if isinstance(item, Mapping)
            ),
            active_inquiries=tuple(
                InquiryFocus.from_dict(item)
                for item in payload.get("active_inquiries", [])
                if isinstance(item, Mapping)
            ),
            deferred_inquiries=tuple(
                InquiryFocus.from_dict(item)
                for item in payload.get("deferred_inquiries", [])
                if isinstance(item, Mapping)
            ),
            ambiguity_profile={
                str(key): float(value)
                for key, value in dict(payload.get("ambiguity_profile", {})).items()
                if isinstance(value, (int, float))
            },
            subject_priority_stack=tuple(
                SubjectPriority.from_dict(item)
                for item in payload.get("subject_priority_stack", [])
                if isinstance(item, Mapping)
            ),
            slow_biases={
                str(key): float(value)
                for key, value in dict(payload.get("slow_biases", {})).items()
                if isinstance(value, (int, float))
            },
            status_flags=(
                {str(key): bool(value) for key, value in status_flags.items()}
                if isinstance(status_flags, Mapping)
                else {}
            ),
            protected_targets=tuple(str(item) for item in payload.get("protected_targets", [])),
            repair_targets=tuple(str(item) for item in payload.get("repair_targets", [])),
            same_subject_basis=str(payload.get("same_subject_basis", "")),
        )

    def summary_text(self) -> str:
        labels = [priority.label for priority in self.subject_priority_stack[:2] if priority.label]
        flags = [name.replace("_", "-") for name, active in self.status_flags.items() if active]
        tensions = [tension.label for tension in self.unresolved_tensions[:2] if tension.label]
        uncertainties = [item.label for item in self.narrative_uncertainties[:2] if item.label]
        inquiries = [item.action for item in self.active_inquiries[:2] if item.action]
        parts = [f"My current subject state is {self.current_phase or 'forming'}-dominant."]
        if self.dominant_goal:
            parts.append(f"Primary goal: {self.dominant_goal}.")
        if flags:
            parts.append("Flags: " + ", ".join(flags) + ".")
        if labels:
            parts.append("Priorities: " + ", ".join(labels) + ".")
        if tensions:
            parts.append("Unresolved tensions: " + ", ".join(tensions) + ".")
        if uncertainties:
            parts.append("Narrative uncertainty: " + ", ".join(uncertainties) + ".")
        if inquiries:
            parts.append("Active inquiry: " + ", ".join(inquiries) + ".")
        if self.same_subject_basis:
            parts.append(self.same_subject_basis)
        return " ".join(parts)

    def explanation_payload(self) -> dict[str, object]:
        return {
            "summary": self.summary_text(),
            "dominant_goal": self.dominant_goal,
            "current_phase": self.current_phase,
            "active_commitments": list(self.active_commitments),
            "dominant_workspace_contents": list(self.dominant_workspace_contents),
            "active_social_focus": list(self.active_social_focus),
            "status_flags": {str(key): bool(value) for key, value in self.status_flags.items()},
            "priority_stack": [item.to_dict() for item in self.subject_priority_stack[:4]],
            "slow_biases": {str(key): float(value) for key, value in self.slow_biases.items()},
            "unresolved_tensions": [item.to_dict() for item in self.unresolved_tensions[:4]],
            "narrative_uncertainties": [
                item.to_dict() for item in self.narrative_uncertainties[:4]
            ],
            "active_inquiries": [item.to_dict() for item in self.active_inquiries[:4]],
            "deferred_inquiries": [item.to_dict() for item in self.deferred_inquiries[:4]],
            "ambiguity_profile": {
                str(key): float(value) for key, value in self.ambiguity_profile.items()
            },
            "continuity_score": round(self.continuity_score, 6),
            "continuity_anchors": list(self.continuity_anchors),
        }


GOAL_ACTIONS = {
    "SURVIVAL": (("hide", "exploit_shelter", "rest"), ("forage",)),
    "INTEGRITY": (("rest", "thermoregulate", "exploit_shelter"), ("forage",)),
    "CONTROL": (("scan", "hide"), ()),
    "RESOURCES": (("forage", "rest"), ("hide",)),
    "SOCIAL": (("seek_contact", "scan"), ("hide",)),
}

NEED_ACTIONS = {
    "energy": (("rest", "forage"), ()),
    "stress": (("hide", "rest"), ("seek_contact",)),
    "fatigue": (("rest", "exploit_shelter"), ("forage",)),
    "temperature": (("thermoregulate", "exploit_shelter"), ()),
    "danger": (("hide", "exploit_shelter", "scan"), ("forage", "seek_contact")),
    "continuity": (("rest", "scan", "hide"), ("forage",)),
    "social": (("seek_contact", "scan"), ("hide",)),
    "uncertainty": (("scan", "hide"), ("forage",)),
}


def _top_commitments(agent: "SegmentAgent") -> tuple[list[str], list[str], list[str]]:
    narrative = agent.self_model.identity_narrative
    if narrative is None:
        return [], [], []
    active = [commitment for commitment in narrative.commitments if commitment.active]
    active.sort(
        key=lambda commitment: (
            float(commitment.priority),
            float(commitment.confidence),
            commitment.commitment_id,
        ),
        reverse=True,
    )
    summaries = [
        commitment.statement or commitment.commitment_id
        for commitment in active[:4]
        if commitment.statement or commitment.commitment_id
    ]
    target_actions = [
        action
        for commitment in active[:4]
        for action in commitment.target_actions[:3]
        if action
    ]
    commitment_ids = [commitment.commitment_id for commitment in active[:4] if commitment.commitment_id]
    return summaries, target_actions, commitment_ids


def _inquiry_focus(agent: "SegmentAgent") -> tuple[tuple[InquiryFocus, ...], tuple[InquiryFocus, ...]]:
    experiment = getattr(agent, "latest_narrative_experiment", None)
    if experiment is None:
        return (), ()
    active: list[InquiryFocus] = []
    deferred: list[InquiryFocus] = []
    for plan in getattr(experiment, "plans", ()):
        focus = InquiryFocus(
            plan_id=str(getattr(plan, "plan_id", "")),
            action=str(getattr(plan, "selected_action", "")),
            status=str(getattr(plan, "status", "")),
            target_unknown_id=str(getattr(plan, "target_unknown_id", "")),
            salience=float(getattr(plan, "informative_value", 0.0)),
            summary=str(getattr(plan, "selected_reason", "")),
        )
        if focus.status == "active_experiment":
            active.append(focus)
        elif (
            focus.status.startswith("deferred")
            or focus.status.startswith("rejected")
            or focus.status == "blocked_by_governance"
        ):
            deferred.append(focus)
    active.sort(key=lambda item: (-item.salience, item.plan_id))
    deferred.sort(key=lambda item: (-item.salience, item.plan_id))
    return tuple(active[:4]), tuple(deferred[:4])


def _dominant_needs(agent: "SegmentAgent", maintenance_agenda: MaintenanceAgenda | None) -> list[DominantNeed]:
    observation = {}
    if agent.last_decision_diagnostics is not None:
        observation = dict(agent.last_decision_diagnostics.prediction_after_memory)
    needs = [
        DominantNeed("energy", max(0.0, 0.55 - agent.energy), "low energy reserve"),
        DominantNeed("stress", max(0.0, agent.stress - 0.45), "elevated stress"),
        DominantNeed("fatigue", max(0.0, agent.fatigue - 0.40), "fatigue load"),
        DominantNeed("temperature", max(0.0, abs(agent.temperature - 0.5) - 0.08), "thermal drift"),
        DominantNeed("danger", max(0.0, float(observation.get("danger", 0.0)) - 0.35), "hazard exposure"),
        DominantNeed("social", max(0.0, 0.35 - float(observation.get("social", 0.35))), "social deprivation"),
    ]
    if maintenance_agenda is not None and maintenance_agenda.policy_shift_strength > 0.25:
        needs.append(
            DominantNeed(
                "maintenance",
                max(maintenance_agenda.policy_shift_strength, maintenance_agenda.chronic_debt_pressure),
                "active homeostatic protection",
            )
        )
    needs.sort(key=lambda item: (-item.intensity, item.name))
    return [item for item in needs if item.intensity > 0.02][:3]


def _social_bindings(agent: "SegmentAgent") -> list[SubjectBinding]:
    bindings: list[SubjectBinding] = []
    current_tick = max(1, agent.cycle)
    for other_id, model in sorted(agent.social_memory.others.items()):
        recency = 1.0 / (1.0 + max(0, current_tick - int(model.last_seen_tick)))
        salience = max(
            float(model.threat),
            float(model.trust) * 0.85,
            float(model.attachment) * 0.90,
        ) * (0.65 + 0.35 * recency)
        if salience <= 0.10:
            continue
        if model.threat >= 0.45:
            binding_type = "social_threat"
            summary = f"{other_id} is socially destabilizing"
        elif model.trust >= 0.58:
            binding_type = "trusted_other"
            summary = f"{other_id} is a trusted counterpart"
        else:
            binding_type = "social_monitor"
            summary = f"{other_id} remains socially salient"
        bindings.append(
            SubjectBinding(
                binding_id=other_id,
                binding_type=binding_type,
                salience=round(salience, 6),
                summary=summary,
                evidence=(
                    f"trust={model.trust:.2f}",
                    f"threat={model.threat:.2f}",
                    f"attachment={model.attachment:.2f}",
                ),
            )
        )
    bindings.sort(key=lambda item: (-item.salience, item.binding_id))
    return bindings[:3]


def _narrative_uncertainty_focus(agent: "SegmentAgent") -> list[NarrativeUncertaintyFocus]:
    payload = getattr(agent, "latest_narrative_uncertainty", None)
    if payload is None:
        return []
    unknowns = getattr(payload, "unknowns", ())
    hypotheses = getattr(payload, "competing_hypotheses", ())
    focus: list[NarrativeUncertaintyFocus] = []
    for unknown in unknowns[:4]:
        if not getattr(unknown, "action_relevant", False):
            continue
        linked_hypotheses = [
            hypothesis.statement
            for hypothesis in hypotheses
            if hypothesis.parent_unknown_id == unknown.unknown_id
        ][:3]
        label = unknown.unknown_type.replace("_", " ")
        if unknown.linked_entities:
            label += f" for {unknown.linked_entities[0]}"
        focus.append(
            NarrativeUncertaintyFocus(
                unknown_id=unknown.unknown_id,
                label=label,
                unknown_type=unknown.unknown_type,
                relevance=float(unknown.decision_relevance.total_score),
                linked_entities=tuple(unknown.linked_entities[:2]),
                hypotheses=tuple(linked_hypotheses),
            )
        )
    focus.sort(key=lambda item: (-item.relevance, item.unknown_id))
    return focus[:3]


def _tensions(
    agent: "SegmentAgent",
    diagnostics: "DecisionDiagnostics | None",
    maintenance_agenda: MaintenanceAgenda | None,
    continuity_score: float,
) -> list[ActiveTension]:
    tensions: list[ActiveTension] = []
    prediction_ledger = getattr(agent, "prediction_ledger", None)
    narrative_uncertainties = _narrative_uncertainty_focus(agent)
    if diagnostics is not None and diagnostics.identity_tension > 0.05:
        tensions.append(
            ActiveTension(
                label="identity tension",
                tension_type="identity",
                intensity=diagnostics.identity_tension,
                repair_target=diagnostics.identity_repair_policy,
                evidence=tuple(diagnostics.violated_commitments[:3]),
            )
        )
    if diagnostics is not None and diagnostics.self_inconsistency_error > 0.05:
        tensions.append(
            ActiveTension(
                label="self inconsistency",
                tension_type="continuity",
                intensity=diagnostics.self_inconsistency_error,
                repair_target=diagnostics.repair_policy,
                evidence=(diagnostics.conflict_type, diagnostics.severity_level),
            )
        )
    if maintenance_agenda is not None and maintenance_agenda.policy_shift_strength > 0.25:
        tensions.append(
            ActiveTension(
                label="maintenance pressure",
                tension_type="maintenance",
                intensity=max(
                    maintenance_agenda.policy_shift_strength,
                    maintenance_agenda.chronic_debt_pressure,
                ),
                repair_target=maintenance_agenda.recommended_action,
                evidence=tuple(maintenance_agenda.active_tasks[:3]),
            )
        )
    if continuity_score < 0.78:
        tensions.append(
            ActiveTension(
                label="continuity fragility",
                tension_type="continuity",
                intensity=1.0 - continuity_score,
                repair_target="protect continuity anchors",
                evidence=("continuity_score_drop",),
            )
        )
    if diagnostics is not None:
        for alert in diagnostics.social_alerts[:2]:
            tensions.append(
                ActiveTension(
                    label="social destabilization",
                    tension_type="social",
                    intensity=0.45,
                    repair_target="stabilize social bindings",
                    evidence=(alert,),
                )
            )
    for uncertainty in narrative_uncertainties[:2]:
        tensions.append(
            ActiveTension(
                label=f"narrative uncertainty: {uncertainty.label}",
                tension_type="uncertainty",
                intensity=min(1.0, 0.18 + uncertainty.relevance * 0.75),
                repair_target="verify latent cause",
                evidence=tuple(uncertainty.hypotheses[:2] or (uncertainty.unknown_type,)),
            )
        )
    if prediction_ledger is not None:
        for discrepancy in prediction_ledger.top_discrepancies(limit=3):
            tensions.append(
                ActiveTension(
                    label=discrepancy.label,
                    tension_type=discrepancy.discrepancy_type,
                    intensity=discrepancy.severity + (0.08 if discrepancy.chronic else 0.0),
                    repair_target=discrepancy.linked_goal or discrepancy.priority,
                    evidence=tuple(discrepancy.supporting_evidence[:2]),
                )
            )
    reconciliation_engine = getattr(agent, "reconciliation_engine", None)
    if reconciliation_engine is not None:
        for thread in reconciliation_engine.active_unresolved_threads()[:3]:
            tensions.append(
                ActiveTension(
                    label=thread.title or "long-horizon conflict",
                    tension_type="reconciliation",
                    intensity=min(1.0, 0.18 + 0.12 * len(set(thread.linked_chapter_ids)) + 0.08 * min(thread.recurrence_count, 4)),
                    repair_target=thread.status,
                    evidence=(thread.thread_id, thread.current_outcome or thread.source_category),
                )
            )
    tensions.sort(key=lambda item: (-item.intensity, item.label))
    return tensions[:4]


def _priority_stack(
    dominant_goal: str,
    needs: list[DominantNeed],
    tensions: list[ActiveTension],
    commitment_targets: list[str],
    active_inquiries: tuple[InquiryFocus, ...],
    status_flags: Mapping[str, bool],
) -> list[SubjectPriority]:
    priorities: list[SubjectPriority] = []
    goal_pref, goal_avoid = GOAL_ACTIONS.get(dominant_goal, ((), ()))
    if dominant_goal:
        priorities.append(
            SubjectPriority(
                label=f"goal:{dominant_goal.lower()}",
                weight=0.70,
                priority_type="goal",
                preferred_actions=tuple(goal_pref),
                avoid_actions=tuple(goal_avoid),
                evidence=(dominant_goal,),
            )
        )
    for need in needs:
        pref, avoid = NEED_ACTIONS.get(need.name, ((), ()))
        priorities.append(
            SubjectPriority(
                label=f"need:{need.name}",
                weight=_clamp(need.intensity, 0.0, 1.0),
                priority_type="need",
                preferred_actions=tuple(pref),
                avoid_actions=tuple(avoid),
                evidence=(need.reason,),
            )
        )
    if commitment_targets:
        priorities.append(
            SubjectPriority(
                label="commitment continuity",
                weight=0.62 if status_flags.get("continuity_fragile", False) else 0.48,
                priority_type="commitment",
                preferred_actions=tuple(dict.fromkeys(commitment_targets))[:4],
                avoid_actions=(),
                evidence=("active commitments",),
            )
        )
    if active_inquiries:
        top_inquiry = active_inquiries[0]
        priorities.append(
            SubjectPriority(
                label=f"inquiry:{top_inquiry.target_unknown_id}",
                weight=min(1.0, 0.44 + top_inquiry.salience * 0.50),
                priority_type="inquiry",
                preferred_actions=(top_inquiry.action,) if top_inquiry.action else (),
                avoid_actions=(),
                evidence=(top_inquiry.summary,),
            )
        )
    for tension in tensions:
        pref, avoid = NEED_ACTIONS.get(tension.tension_type, ((), ()))
        priorities.append(
            SubjectPriority(
                label=tension.label,
                weight=_clamp(tension.intensity, 0.0, 1.0),
                priority_type=tension.tension_type,
                preferred_actions=tuple(pref),
                avoid_actions=tuple(avoid),
                evidence=tension.evidence,
            )
        )
    priorities.sort(key=lambda item: (-item.weight, item.label))
    return priorities[:6]


def derive_subject_state(
    agent: "SegmentAgent",
    *,
    diagnostics: "DecisionDiagnostics | None" = None,
    continuity_report: Mapping[str, object] | None = None,
    maintenance_agenda: MaintenanceAgenda | None = None,
    previous_state: SubjectState | None = None,
    restart_anchors: Mapping[str, object] | None = None,
) -> SubjectState:
    slow_biases = agent.slow_variable_learner.state.bias_payload()
    narrative = agent.self_model.identity_narrative
    current_chapter = narrative.current_chapter if narrative is not None else None
    core_identity_summary = (
        (narrative.core_summary if narrative is not None and narrative.core_summary else "")
        or (narrative.core_identity if narrative is not None and narrative.core_identity else "")
        or "I am an adaptive subject maintaining continuity under uncertainty."
    )
    current_phase = (
        current_chapter.dominant_theme
        if current_chapter is not None and current_chapter.dominant_theme
        else "forming"
    )
    continuity_payload = (
        dict(continuity_report)
        if isinstance(continuity_report, Mapping)
        else agent.self_model.continuity_audit.to_dict()
    )
    continuity_score = _clamp(float(continuity_payload.get("continuity_score", 1.0)))
    workspace_contents = (
        tuple(agent.global_workspace.report_focus(agent.last_workspace_state))
        if agent.last_workspace_state is not None
        else ()
    )
    active_goal = (
        diagnostics.active_goal
        if diagnostics is not None and diagnostics.active_goal
        else agent.goal_stack.active_goal.name
    )
    preferences: list[str] = []
    preferred_policies = agent.self_model.preferred_policies
    if preferred_policies is not None:
        if preferred_policies.dominant_strategy:
            preferences.append(f"strategy:{preferred_policies.dominant_strategy}")
        preferences.extend(
            f"prefer:{action}"
            for action in sorted(preferred_policies.learned_preferences)[:3]
        )
    commitment_summaries, commitment_targets, commitment_ids = _top_commitments(agent)
    bindings = _social_bindings(agent)
    narrative_uncertainties = _narrative_uncertainty_focus(agent)
    active_inquiries, deferred_inquiries = _inquiry_focus(agent)
    active_social_focus = list(
        diagnostics.social_focus if diagnostics is not None else [item.binding_id for item in bindings[:2]]
    )
    identity_tension_level = (
        float(diagnostics.identity_tension)
        if diagnostics is not None
        else float(agent.identity_tension_history[-1].get("identity_tension", 0.0))
        if agent.identity_tension_history
        else 0.0
    )
    self_inconsistency_level = (
        float(diagnostics.self_inconsistency_error)
        if diagnostics is not None
        else float(agent.self_model.self_inconsistency_events[-1].self_inconsistency_error)
        if agent.self_model.self_inconsistency_events
        else 0.0
    )
    dominant_needs = _dominant_needs(agent, maintenance_agenda)
    maintenance_pressure = max(
        max((need.intensity for need in dominant_needs), default=0.0),
        float(maintenance_agenda.policy_shift_strength) if maintenance_agenda is not None else 0.0,
        float(maintenance_agenda.chronic_debt_pressure) if maintenance_agenda is not None else 0.0,
    )
    continuity_anchors = list(
        dict.fromkeys(
            [
                *[str(item) for item in continuity_payload.get("protected_anchor_ids", []) if str(item)],
                *commitment_ids,
                *[
                    str(item)
                    for item in getattr(agent.long_term_memory, "restart_continuity_anchor_ids", [])
                    if str(item)
                ],
                *[str(item) for item in (restart_anchors or {}).get("commitment_snapshot", []) if str(item)],
            ]
        )
    )[:12]
    continuity_fragile = continuity_score < 0.78 and (
        bool(continuity_anchors)
        or bool(commitment_summaries)
        or bool(previous_state is not None and previous_state.continuity_anchors)
    )
    active_reconciliation_threads = getattr(agent, "reconciliation_engine", None)
    unresolved_reconciliation = (
        len(active_reconciliation_threads.active_unresolved_threads())
        if active_reconciliation_threads is not None
        else 0
    )
    status_flags = {
        "threatened": (
            agent.energy < 0.25
            or any(need.name == "danger" and need.intensity > 0.15 for need in dominant_needs)
            or (maintenance_agenda.protected_mode if maintenance_agenda is not None else False)
            or float(slow_biases.get("threat_sensitivity", 0.5)) >= 0.68
        ),
        "repairing": bool(diagnostics and (diagnostics.repair_triggered or diagnostics.repair_policy)),
        "overloaded": (
            agent.stress > 0.72
            or agent.fatigue > 0.70
            or (
                agent.last_workspace_state is not None
                and agent.last_workspace_state.replacement_pressure > 0.45
            )
        ),
        "socially_destabilized": bool(
            (diagnostics and diagnostics.social_alerts)
            or any(binding.binding_type == "social_threat" for binding in bindings)
            or float(slow_biases.get("trust_stance", 0.5)) <= 0.34
        ),
        "narrative_ambiguity_active": bool(narrative_uncertainties),
        "active_inquiry": bool(active_inquiries),
        "inquiry_deferred": bool(deferred_inquiries),
        "continuity_fragile": continuity_fragile,
        "long_horizon_conflict": unresolved_reconciliation > 0,
    }
    tensions = _tensions(agent, diagnostics, maintenance_agenda, continuity_score)
    priorities = _priority_stack(
        active_goal,
        dominant_needs,
        tensions,
        commitment_targets,
        active_inquiries,
        status_flags,
    )
    protected_targets = tuple(
        item.label for item in priorities[:3] if item.priority_type in {"goal", "need", "commitment"}
    )
    repair_targets = tuple(item.repair_target for item in tensions if item.repair_target)[:4]
    same_subject_bits = []
    if commitment_summaries:
        same_subject_bits.append("I remain the same subject through active commitments.")
    if continuity_anchors:
        same_subject_bits.append(f"Anchors held: {', '.join(continuity_anchors[:3])}.")
    elif previous_state is not None and previous_state.continuity_anchors:
        same_subject_bits.append("Continuity still depends on prior anchors awaiting rebind.")
    if current_phase:
        same_subject_bits.append(f"Current narrative phase: {current_phase}.")
    if unresolved_reconciliation:
        same_subject_bits.append(
            f"{unresolved_reconciliation} long-horizon conflict thread(s) still shape continuity."
        )
    if narrative_uncertainties:
        same_subject_bits.append(
            f"{len(narrative_uncertainties)} retained narrative uncertainty item(s) still shape policy."
        )
    ambiguity_profile = getattr(getattr(agent, "latest_narrative_uncertainty", None), "profile", None)
    return SubjectState(
        tick=int(agent.cycle),
        core_identity_summary=core_identity_summary,
        current_phase=current_phase,
        current_self_narrative_phase=current_phase,
        active_commitments=tuple(commitment_summaries),
        dominant_preferences=tuple(dict.fromkeys(preferences))[:4],
        dominant_goal=active_goal,
        dominant_needs=tuple(dominant_needs),
        dominant_workspace_contents=tuple(workspace_contents[:4]),
        active_social_focus=tuple(active_social_focus[:3]),
        socially_salient_bindings=tuple(bindings),
        identity_tension_level=_clamp(identity_tension_level),
        self_inconsistency_level=_clamp(self_inconsistency_level),
        maintenance_pressure=_clamp(maintenance_pressure),
        continuity_score=continuity_score,
        continuity_anchors=tuple(continuity_anchors),
        unresolved_tensions=tuple(tensions),
        narrative_uncertainties=tuple(narrative_uncertainties),
        active_inquiries=active_inquiries,
        deferred_inquiries=deferred_inquiries,
        ambiguity_profile=(
            ambiguity_profile.to_dict() if ambiguity_profile is not None else {}
        ),
        subject_priority_stack=tuple(priorities),
        slow_biases=slow_biases,
        status_flags=status_flags,
        protected_targets=protected_targets,
        repair_targets=repair_targets,
        same_subject_basis=" ".join(same_subject_bits).strip(),
    )


def subject_action_bias(subject_state: SubjectState, action: str) -> float:
    bias = 0.0
    for priority in subject_state.subject_priority_stack[:4]:
        gain = 0.0 if priority.priority_type == "goal" else 0.14
        penalty = 0.0 if priority.priority_type == "goal" else 0.12
        if action in priority.preferred_actions:
            bias += gain * priority.weight
        if action in priority.avoid_actions:
            bias -= penalty * priority.weight
    if subject_state.status_flags.get("threatened", False):
        if action in {"hide", "exploit_shelter", "rest", "thermoregulate"}:
            bias += 0.12
        elif action in {"forage", "seek_contact"}:
            bias -= 0.10
    if subject_state.status_flags.get("repairing", False) and action in {"rest", "scan", "seek_contact"}:
        bias += 0.08
    if subject_state.status_flags.get("socially_destabilized", False):
        if action == "hide":
            bias += 0.06
        elif action == "seek_contact":
            bias -= 0.06
    if subject_state.status_flags.get("continuity_fragile", False):
        if action in {"rest", "scan", "hide"}:
            bias += 0.05
        if subject_state.active_commitments and action == "forage":
            bias -= 0.04
    if subject_state.status_flags.get("long_horizon_conflict", False):
        if action in {"rest", "scan"}:
            bias += 0.05
        elif action == "forage":
            bias -= 0.03
    if subject_state.status_flags.get("narrative_ambiguity_active", False):
        if action in {"scan", "hide"}:
            bias += 0.04
        elif action == "forage":
            bias -= 0.03
    if subject_state.status_flags.get("active_inquiry", False):
        if any(item.action == action for item in subject_state.active_inquiries[:2]):
            bias += 0.10
    if subject_state.status_flags.get("inquiry_deferred", False):
        if any(item.action == action for item in subject_state.deferred_inquiries[:2]):
            bias -= 0.06
    caution_bias = float(subject_state.slow_biases.get("caution_bias", 0.5))
    threat_sensitivity = float(subject_state.slow_biases.get("threat_sensitivity", 0.5))
    trust_stance = float(subject_state.slow_biases.get("trust_stance", 0.5))
    exploration_posture = float(subject_state.slow_biases.get("exploration_posture", 0.5))
    continuity_resilience = float(subject_state.slow_biases.get("continuity_resilience", 0.5))
    if action in {"hide", "rest", "exploit_shelter", "thermoregulate"}:
        bias += max(0.0, caution_bias - 0.5) * 0.14
        bias += max(0.0, threat_sensitivity - 0.5) * 0.10
        bias += max(0.0, 0.55 - continuity_resilience) * 0.06
    if action in {"scan", "seek_contact"}:
        bias += (exploration_posture - 0.5) * 0.12
        if action == "seek_contact":
            bias += (trust_stance - 0.5) * 0.10
    if action == "forage":
        bias -= max(0.0, caution_bias - 0.5) * 0.10
    return max(-0.45, min(0.45, round(bias, 6)))


def subject_memory_threshold_delta(subject_state: SubjectState) -> float:
    delta = 0.0
    delta -= min(0.12, subject_state.maintenance_pressure * 0.12)
    delta -= min(0.10, subject_state.identity_tension_level * 0.10)
    delta -= min(0.10, subject_state.self_inconsistency_level * 0.10)
    if subject_state.status_flags.get("continuity_fragile", False):
        delta -= 0.08
    if subject_state.status_flags.get("repairing", False):
        delta -= 0.06
    if subject_state.status_flags.get("long_horizon_conflict", False):
        delta -= 0.05
    if subject_state.status_flags.get("threatened", False):
        delta -= 0.05
    if subject_state.status_flags.get("narrative_ambiguity_active", False):
        delta -= 0.04
    delta -= max(0.0, float(subject_state.slow_biases.get("threat_sensitivity", 0.5)) - 0.5) * 0.07
    delta -= max(0.0, float(subject_state.slow_biases.get("commitment_stability", 0.5)) - 0.5) * 0.05
    delta -= max(0.0, 0.55 - float(subject_state.slow_biases.get("continuity_resilience", 0.5))) * 0.06
    return max(-0.30, min(0.05, round(delta, 6)))


def apply_subject_state_to_maintenance_agenda(
    subject_state: SubjectState,
    agenda: MaintenanceAgenda,
) -> tuple[MaintenanceAgenda, dict[str, object]]:
    priority_gain = min(
        0.35,
        0.14 * subject_state.maintenance_pressure
        + (0.10 if subject_state.status_flags.get("continuity_fragile", False) else 0.0)
        + (0.08 if subject_state.status_flags.get("repairing", False) else 0.0),
    )
    priority_gain = min(
        0.4,
        priority_gain
        + max(0.0, float(subject_state.slow_biases.get("maintenance_weight", 0.5)) - 0.5) * 0.16
        + max(0.0, float(subject_state.slow_biases.get("caution_bias", 0.5)) - 0.5) * 0.08,
    )
    active_tasks = list(agenda.active_tasks)
    recommended_action = agenda.recommended_action
    interrupt_action = agenda.interrupt_action
    interrupt_reason = agenda.interrupt_reason
    if subject_state.status_flags.get("continuity_fragile", False) and "continuity_guard" not in active_tasks:
        active_tasks.append("continuity_guard")
    if subject_state.status_flags.get("repairing", False) and "repair_stabilization" not in active_tasks:
        active_tasks.append("repair_stabilization")
    if subject_state.status_flags.get("long_horizon_conflict", False) and "reconciliation_review" not in active_tasks:
        active_tasks.append("reconciliation_review")
    if subject_state.status_flags.get("threatened", False):
        recommended_action = "hide"
        if interrupt_action is None and agenda.policy_shift_strength + priority_gain > 0.30:
            interrupt_action = "hide"
            interrupt_reason = "subject-state threat mitigation"
    elif subject_state.status_flags.get("continuity_fragile", False) and recommended_action not in {
        "rest",
        "hide",
        "exploit_shelter",
        "scan",
    }:
        recommended_action = "scan"
    updated = MaintenanceAgenda(
        cycle=agenda.cycle,
        active_tasks=tuple(dict.fromkeys(active_tasks)),
        recommended_action=recommended_action,
        interrupt_action=interrupt_action,
        interrupt_reason=interrupt_reason,
        sleep_recommended=agenda.sleep_recommended,
        memory_compaction_recommended=agenda.memory_compaction_recommended,
        telemetry_backoff_recommended=agenda.telemetry_backoff_recommended,
        protected_mode=agenda.protected_mode or subject_state.status_flags.get("threatened", False),
        protected_mode_ticks_remaining=agenda.protected_mode_ticks_remaining,
        recovery_rebound_active=agenda.recovery_rebound_active,
        recovery_rebound_ticks_remaining=agenda.recovery_rebound_ticks_remaining,
        policy_shift_strength=round(min(1.0, agenda.policy_shift_strength + priority_gain), 6),
        suppressed_actions=agenda.suppressed_actions,
        recovery_focus=agenda.recovery_focus,
        chronic_debt_pressure=agenda.chronic_debt_pressure,
        state=agenda.state,
    )
    details = {
        "priority_gain": round(priority_gain, 6),
        "recommended_action": recommended_action,
        "interrupt_action": interrupt_action,
        "active_tasks": list(updated.active_tasks),
        "status_flags": {str(key): bool(value) for key, value in subject_state.status_flags.items()},
    }
    return updated, details
