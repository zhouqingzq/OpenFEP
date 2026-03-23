from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _rounded(value: float) -> float:
    return round(float(value), 6)


@dataclass(slots=True)
class SlowTraitState:
    caution_bias: float = 0.5
    threat_sensitivity: float = 0.5
    trust_stance: float = 0.5
    exploration_posture: float = 0.5
    social_approach: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "caution_bias": _rounded(self.caution_bias),
            "threat_sensitivity": _rounded(self.threat_sensitivity),
            "trust_stance": _rounded(self.trust_stance),
            "exploration_posture": _rounded(self.exploration_posture),
            "social_approach": _rounded(self.social_approach),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SlowTraitState":
        if not payload:
            return cls()
        return cls(
            caution_bias=float(payload.get("caution_bias", 0.5)),
            threat_sensitivity=float(payload.get("threat_sensitivity", 0.5)),
            trust_stance=float(payload.get("trust_stance", 0.5)),
            exploration_posture=float(payload.get("exploration_posture", 0.5)),
            social_approach=float(payload.get("social_approach", 0.5)),
        )


@dataclass(slots=True)
class ValueStabilityState:
    survival_weight: float = 0.72
    social_weight: float = 0.5
    exploration_weight: float = 0.46
    maintenance_weight: float = 0.62
    hierarchy_stability: float = 0.68

    def to_dict(self) -> dict[str, float]:
        return {
            "survival_weight": _rounded(self.survival_weight),
            "social_weight": _rounded(self.social_weight),
            "exploration_weight": _rounded(self.exploration_weight),
            "maintenance_weight": _rounded(self.maintenance_weight),
            "hierarchy_stability": _rounded(self.hierarchy_stability),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ValueStabilityState":
        if not payload:
            return cls()
        return cls(
            survival_weight=float(payload.get("survival_weight", 0.72)),
            social_weight=float(payload.get("social_weight", 0.5)),
            exploration_weight=float(payload.get("exploration_weight", 0.46)),
            maintenance_weight=float(payload.get("maintenance_weight", 0.62)),
            hierarchy_stability=float(payload.get("hierarchy_stability", 0.68)),
        )


@dataclass(slots=True)
class IdentityStabilityState:
    commitment_stability: float = 0.7
    identity_rigidity: float = 0.58
    plasticity: float = 0.42
    continuity_resilience: float = 0.66

    def to_dict(self) -> dict[str, float]:
        return {
            "commitment_stability": _rounded(self.commitment_stability),
            "identity_rigidity": _rounded(self.identity_rigidity),
            "plasticity": _rounded(self.plasticity),
            "continuity_resilience": _rounded(self.continuity_resilience),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "IdentityStabilityState":
        if not payload:
            return cls()
        return cls(
            commitment_stability=float(payload.get("commitment_stability", 0.7)),
            identity_rigidity=float(payload.get("identity_rigidity", 0.58)),
            plasticity=float(payload.get("plasticity", 0.42)),
            continuity_resilience=float(payload.get("continuity_resilience", 0.66)),
        )


@dataclass(slots=True)
class DriftBudget:
    max_total_delta_per_cycle: float = 0.18
    per_variable_budget: dict[str, float] = field(
        default_factory=lambda: {
            "traits.caution_bias": 0.05,
            "traits.threat_sensitivity": 0.05,
            "traits.trust_stance": 0.045,
            "traits.exploration_posture": 0.05,
            "traits.social_approach": 0.045,
            "values.survival_weight": 0.04,
            "values.social_weight": 0.04,
            "values.exploration_weight": 0.04,
            "values.maintenance_weight": 0.04,
            "values.hierarchy_stability": 0.035,
            "identity.commitment_stability": 0.035,
            "identity.identity_rigidity": 0.03,
            "identity.plasticity": 0.03,
            "identity.continuity_resilience": 0.04,
        }
    )
    max_divergent_updates: int = 7

    def to_dict(self) -> dict[str, object]:
        return {
            "max_total_delta_per_cycle": _rounded(self.max_total_delta_per_cycle),
            "per_variable_budget": {
                str(key): _rounded(value) for key, value in self.per_variable_budget.items()
            },
            "max_divergent_updates": int(self.max_divergent_updates),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "DriftBudget":
        if not payload:
            return cls()
        budgets = payload.get("per_variable_budget", {})
        return cls(
            max_total_delta_per_cycle=float(payload.get("max_total_delta_per_cycle", 0.18)),
            per_variable_budget=(
                {str(key): float(value) for key, value in budgets.items()}
                if isinstance(budgets, Mapping)
                else cls().per_variable_budget
            ),
            max_divergent_updates=max(1, int(payload.get("max_divergent_updates", 7))),
        )


@dataclass(slots=True)
class PlasticityWindow:
    variable_path: str
    learning_rate: float
    evidence_threshold: int
    resistance: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "variable_path": self.variable_path,
            "learning_rate": _rounded(self.learning_rate),
            "evidence_threshold": int(self.evidence_threshold),
            "resistance": _rounded(self.resistance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PlasticityWindow":
        if not payload:
            return cls(variable_path="", learning_rate=0.25, evidence_threshold=2)
        return cls(
            variable_path=str(payload.get("variable_path", "")),
            learning_rate=float(payload.get("learning_rate", 0.25)),
            evidence_threshold=max(1, int(payload.get("evidence_threshold", 2))),
            resistance=float(payload.get("resistance", 0.0)),
        )


@dataclass(slots=True)
class ProtectedAnchor:
    variable_path: str
    label: str
    min_value: float
    max_value: float
    required_evidence: int
    anchor_strength: float = 0.8

    def to_dict(self) -> dict[str, object]:
        return {
            "variable_path": self.variable_path,
            "label": self.label,
            "min_value": _rounded(self.min_value),
            "max_value": _rounded(self.max_value),
            "required_evidence": int(self.required_evidence),
            "anchor_strength": _rounded(self.anchor_strength),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ProtectedAnchor":
        if not payload:
            return cls(
                variable_path="",
                label="",
                min_value=0.0,
                max_value=1.0,
                required_evidence=3,
            )
        return cls(
            variable_path=str(payload.get("variable_path", "")),
            label=str(payload.get("label", "")),
            min_value=float(payload.get("min_value", 0.0)),
            max_value=float(payload.get("max_value", 1.0)),
            required_evidence=max(1, int(payload.get("required_evidence", 3))),
            anchor_strength=float(payload.get("anchor_strength", 0.8)),
        )


@dataclass(slots=True)
class LearningPressure:
    variable_path: str
    signed_pressure: float
    evidence_count: int
    support: float
    sources: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "variable_path": self.variable_path,
            "signed_pressure": _rounded(self.signed_pressure),
            "evidence_count": int(self.evidence_count),
            "support": _rounded(self.support),
            "sources": list(self.sources),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "LearningPressure":
        if not payload:
            return cls(variable_path="", signed_pressure=0.0, evidence_count=0, support=0.0)
        return cls(
            variable_path=str(payload.get("variable_path", "")),
            signed_pressure=float(payload.get("signed_pressure", 0.0)),
            evidence_count=max(0, int(payload.get("evidence_count", 0))),
            support=float(payload.get("support", 0.0)),
            sources=[str(item) for item in payload.get("sources", [])],
            rationale=str(payload.get("rationale", "")),
        )


@dataclass(slots=True)
class ConsolidationUpdate:
    variable_path: str
    previous_value: float
    attempted_value: float
    new_value: float
    delta: float
    status: str
    source_pressures: list[str] = field(default_factory=list)
    rationale: str = ""
    protected_anchor: str = ""
    clipped_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "variable_path": self.variable_path,
            "previous_value": _rounded(self.previous_value),
            "attempted_value": _rounded(self.attempted_value),
            "new_value": _rounded(self.new_value),
            "delta": _rounded(self.delta),
            "status": self.status,
            "source_pressures": list(self.source_pressures),
            "rationale": self.rationale,
            "protected_anchor": self.protected_anchor,
            "clipped_reason": self.clipped_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "ConsolidationUpdate":
        if not payload:
            return cls(
                variable_path="",
                previous_value=0.0,
                attempted_value=0.0,
                new_value=0.0,
                delta=0.0,
                status="rejected",
            )
        return cls(
            variable_path=str(payload.get("variable_path", "")),
            previous_value=float(payload.get("previous_value", 0.0)),
            attempted_value=float(payload.get("attempted_value", 0.0)),
            new_value=float(payload.get("new_value", 0.0)),
            delta=float(payload.get("delta", 0.0)),
            status=str(payload.get("status", "rejected")),
            source_pressures=[str(item) for item in payload.get("source_pressures", [])],
            rationale=str(payload.get("rationale", "")),
            protected_anchor=str(payload.get("protected_anchor", "")),
            clipped_reason=str(payload.get("clipped_reason", "")),
        )


@dataclass(slots=True)
class SlowUpdateAudit:
    audit_id: str
    tick: int
    sleep_cycle_id: int
    pressures: list[LearningPressure] = field(default_factory=list)
    updates: list[ConsolidationUpdate] = field(default_factory=list)
    anti_collapse_triggered: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "tick": int(self.tick),
            "sleep_cycle_id": int(self.sleep_cycle_id),
            "pressures": [item.to_dict() for item in self.pressures],
            "updates": [item.to_dict() for item in self.updates],
            "anti_collapse_triggered": bool(self.anti_collapse_triggered),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SlowUpdateAudit":
        if not payload:
            return cls(audit_id="", tick=0, sleep_cycle_id=0)
        return cls(
            audit_id=str(payload.get("audit_id", "")),
            tick=int(payload.get("tick", 0)),
            sleep_cycle_id=int(payload.get("sleep_cycle_id", 0)),
            pressures=[
                LearningPressure.from_dict(item)
                for item in payload.get("pressures", [])
                if isinstance(item, Mapping)
            ],
            updates=[
                ConsolidationUpdate.from_dict(item)
                for item in payload.get("updates", [])
                if isinstance(item, Mapping)
            ],
            anti_collapse_triggered=bool(payload.get("anti_collapse_triggered", False)),
            summary=str(payload.get("summary", "")),
        )


@dataclass(slots=True)
class SlowLearningState:
    traits: SlowTraitState = field(default_factory=SlowTraitState)
    values: ValueStabilityState = field(default_factory=ValueStabilityState)
    identity: IdentityStabilityState = field(default_factory=IdentityStabilityState)
    last_processed_tick: int = 0
    sleep_cycles: int = 0
    last_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "traits": self.traits.to_dict(),
            "values": self.values.to_dict(),
            "identity": self.identity.to_dict(),
            "last_processed_tick": int(self.last_processed_tick),
            "sleep_cycles": int(self.sleep_cycles),
            "last_summary": self.last_summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SlowLearningState":
        if not payload:
            return cls()
        return cls(
            traits=SlowTraitState.from_dict(
                payload.get("traits") if isinstance(payload.get("traits"), Mapping) else None
            ),
            values=ValueStabilityState.from_dict(
                payload.get("values") if isinstance(payload.get("values"), Mapping) else None
            ),
            identity=IdentityStabilityState.from_dict(
                payload.get("identity") if isinstance(payload.get("identity"), Mapping) else None
            ),
            last_processed_tick=int(payload.get("last_processed_tick", 0)),
            sleep_cycles=int(payload.get("sleep_cycles", 0)),
            last_summary=str(payload.get("last_summary", "")),
        )

    def flattened(self) -> dict[str, float]:
        return {
            "traits.caution_bias": self.traits.caution_bias,
            "traits.threat_sensitivity": self.traits.threat_sensitivity,
            "traits.trust_stance": self.traits.trust_stance,
            "traits.exploration_posture": self.traits.exploration_posture,
            "traits.social_approach": self.traits.social_approach,
            "values.survival_weight": self.values.survival_weight,
            "values.social_weight": self.values.social_weight,
            "values.exploration_weight": self.values.exploration_weight,
            "values.maintenance_weight": self.values.maintenance_weight,
            "values.hierarchy_stability": self.values.hierarchy_stability,
            "identity.commitment_stability": self.identity.commitment_stability,
            "identity.identity_rigidity": self.identity.identity_rigidity,
            "identity.plasticity": self.identity.plasticity,
            "identity.continuity_resilience": self.identity.continuity_resilience,
        }

    def set_value(self, variable_path: str, value: float) -> None:
        value = _clamp(value)
        root, _, name = variable_path.partition(".")
        target = getattr(self, root, None)
        if target is None or not name:
            return
        if hasattr(target, name):
            setattr(target, name, value)

    def bias_payload(self) -> dict[str, float]:
        return {
            **self.traits.to_dict(),
            **self.values.to_dict(),
            **self.identity.to_dict(),
        }


def _window_map() -> dict[str, PlasticityWindow]:
    items = [
        PlasticityWindow("traits.caution_bias", learning_rate=0.7, evidence_threshold=2, resistance=0.15),
        PlasticityWindow("traits.threat_sensitivity", learning_rate=0.65, evidence_threshold=2, resistance=0.1),
        PlasticityWindow("traits.trust_stance", learning_rate=0.5, evidence_threshold=3, resistance=0.2),
        PlasticityWindow("traits.exploration_posture", learning_rate=0.6, evidence_threshold=2, resistance=0.15),
        PlasticityWindow("traits.social_approach", learning_rate=0.45, evidence_threshold=3, resistance=0.2),
        PlasticityWindow("values.survival_weight", learning_rate=0.4, evidence_threshold=3, resistance=0.35),
        PlasticityWindow("values.social_weight", learning_rate=0.35, evidence_threshold=3, resistance=0.25),
        PlasticityWindow("values.exploration_weight", learning_rate=0.4, evidence_threshold=2, resistance=0.25),
        PlasticityWindow("values.maintenance_weight", learning_rate=0.45, evidence_threshold=2, resistance=0.2),
        PlasticityWindow("values.hierarchy_stability", learning_rate=0.3, evidence_threshold=3, resistance=0.35),
        PlasticityWindow("identity.commitment_stability", learning_rate=0.3, evidence_threshold=4, resistance=0.45),
        PlasticityWindow("identity.identity_rigidity", learning_rate=0.22, evidence_threshold=4, resistance=0.5),
        PlasticityWindow("identity.plasticity", learning_rate=0.24, evidence_threshold=3, resistance=0.35),
        PlasticityWindow("identity.continuity_resilience", learning_rate=0.32, evidence_threshold=3, resistance=0.4),
    ]
    return {item.variable_path: item for item in items}


def _anchor_list() -> list[ProtectedAnchor]:
    return [
        ProtectedAnchor(
            variable_path="values.survival_weight",
            label="survival-priority",
            min_value=0.58,
            max_value=1.0,
            required_evidence=4,
            anchor_strength=0.9,
        ),
        ProtectedAnchor(
            variable_path="identity.commitment_stability",
            label="commitment-continuity",
            min_value=0.48,
            max_value=1.0,
            required_evidence=4,
            anchor_strength=0.85,
        ),
        ProtectedAnchor(
            variable_path="identity.continuity_resilience",
            label="continuity-resilience",
            min_value=0.45,
            max_value=1.0,
            required_evidence=3,
            anchor_strength=0.82,
        ),
        ProtectedAnchor(
            variable_path="traits.caution_bias",
            label="survival-caution-floor",
            min_value=0.25,
            max_value=0.95,
            required_evidence=2,
            anchor_strength=0.65,
        ),
    ]


def _human_label(variable_path: str) -> str:
    labels = {
        "traits.caution_bias": "Caution bias",
        "traits.threat_sensitivity": "Threat sensitivity",
        "traits.trust_stance": "Trust stance",
        "traits.exploration_posture": "Exploration posture",
        "traits.social_approach": "Social approach",
        "values.survival_weight": "Survival value weighting",
        "values.social_weight": "Social value weighting",
        "values.exploration_weight": "Exploration value weighting",
        "values.maintenance_weight": "Maintenance priority",
        "values.hierarchy_stability": "Value hierarchy stability",
        "identity.commitment_stability": "Commitment stability",
        "identity.identity_rigidity": "Identity rigidity",
        "identity.plasticity": "Plasticity",
        "identity.continuity_resilience": "Continuity resilience",
    }
    return labels.get(variable_path, variable_path.replace(".", " "))


@dataclass(slots=True)
class SlowVariableLearner:
    state: SlowLearningState = field(default_factory=SlowLearningState)
    drift_budget: DriftBudget = field(default_factory=DriftBudget)
    plasticity_windows: dict[str, PlasticityWindow] = field(default_factory=_window_map)
    protected_anchors: list[ProtectedAnchor] = field(default_factory=_anchor_list)
    audit_history: list[SlowUpdateAudit] = field(default_factory=list)
    max_audit_history: int = 64

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "drift_budget": self.drift_budget.to_dict(),
            "plasticity_windows": {
                key: value.to_dict() for key, value in sorted(self.plasticity_windows.items())
            },
            "protected_anchors": [item.to_dict() for item in self.protected_anchors],
            "audit_history": [item.to_dict() for item in self.audit_history[-self.max_audit_history :]],
            "max_audit_history": int(self.max_audit_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SlowVariableLearner":
        if not payload:
            return cls()
        windows_payload = payload.get("plasticity_windows", {})
        if isinstance(windows_payload, Mapping):
            windows = {
                str(key): PlasticityWindow.from_dict(value)
                for key, value in windows_payload.items()
                if isinstance(value, Mapping)
            }
        else:
            windows = _window_map()
        anchors = [
            ProtectedAnchor.from_dict(item)
            for item in payload.get("protected_anchors", [])
            if isinstance(item, Mapping)
        ]
        return cls(
            state=SlowLearningState.from_dict(
                payload.get("state") if isinstance(payload.get("state"), Mapping) else None
            ),
            drift_budget=DriftBudget.from_dict(
                payload.get("drift_budget")
                if isinstance(payload.get("drift_budget"), Mapping)
                else None
            ),
            plasticity_windows=windows or _window_map(),
            protected_anchors=anchors or _anchor_list(),
            audit_history=[
                SlowUpdateAudit.from_dict(item)
                for item in payload.get("audit_history", [])
                if isinstance(item, Mapping)
            ],
            max_audit_history=max(8, int(payload.get("max_audit_history", 64))),
        )

    def latest_audit(self) -> SlowUpdateAudit | None:
        return self.audit_history[-1] if self.audit_history else None

    def recent_explanations(self) -> list[str]:
        audit = self.latest_audit()
        if audit is None:
            return []
        explanations: list[str] = []
        for update in audit.updates:
            label = _human_label(update.variable_path)
            if update.status == "accepted":
                direction = "increased" if update.delta > 0 else "decreased"
                explanations.append(f"{label} {direction} because {update.rationale}.")
            elif update.status in {"clipped", "rejected"} and update.clipped_reason:
                explanations.append(f"{label} remained stable because {update.clipped_reason}.")
        return explanations[:6]

    def explanation_payload(self) -> dict[str, object]:
        audit = self.latest_audit()
        return {
            "state": self.state.to_dict(),
            "latest_audit": audit.to_dict() if audit is not None else None,
            "recent_explanations": self.recent_explanations(),
        }

    def action_bias(self, action: str) -> float:
        traits = self.state.traits
        values = self.state.values
        identity = self.state.identity
        bias = 0.0
        if action in {"hide", "rest", "exploit_shelter", "thermoregulate"}:
            bias += (traits.caution_bias - 0.5) * 0.35
            bias += (traits.threat_sensitivity - 0.5) * 0.25
            bias += (values.maintenance_weight - 0.5) * 0.20
            bias += (identity.continuity_resilience - 0.5) * 0.12
        if action in {"scan", "seek_contact"}:
            bias += (traits.exploration_posture - 0.5) * 0.32
            bias += (values.exploration_weight - 0.5) * 0.24
            bias += (traits.trust_stance - 0.5) * (0.16 if action == "seek_contact" else 0.08)
            bias -= (traits.caution_bias - 0.5) * 0.14
        if action == "seek_contact":
            bias += (traits.social_approach - 0.5) * 0.22
            bias += (values.social_weight - 0.5) * 0.18
        if action == "forage":
            bias -= (traits.caution_bias - 0.5) * 0.20
            bias -= (values.survival_weight - 0.5) * 0.18
            bias -= (identity.commitment_stability - 0.5) * 0.10
        return max(-0.32, min(0.32, round(bias, 6)))

    def memory_threshold_delta(self) -> float:
        traits = self.state.traits
        identity = self.state.identity
        values = self.state.values
        delta = 0.0
        delta -= max(0.0, traits.threat_sensitivity - 0.5) * 0.08
        delta -= max(0.0, traits.caution_bias - 0.5) * 0.08
        delta -= max(0.0, values.maintenance_weight - 0.5) * 0.06
        delta -= max(0.0, identity.commitment_stability - 0.5) * 0.05
        delta -= max(0.0, 0.55 - identity.continuity_resilience) * 0.08
        return round(max(-0.24, min(0.04, delta)), 6)

    def continuity_modifier(self) -> float:
        identity = self.state.identity
        values = self.state.values
        modifier = (
            (identity.continuity_resilience - 0.5) * 0.18
            + (identity.commitment_stability - 0.5) * 0.10
            + (values.hierarchy_stability - 0.5) * 0.10
            - max(0.0, identity.plasticity - 0.62) * 0.08
        )
        return round(max(-0.12, min(0.12, modifier)), 6)

    def verification_priority_delta(
        self,
        *,
        target_channels: tuple[str, ...],
        prediction_type: str,
    ) -> float:
        traits = self.state.traits
        identity = self.state.identity
        values = self.state.values
        delta = 0.0
        if "danger" in target_channels or "maintenance" in target_channels:
            delta += max(0.0, traits.threat_sensitivity - 0.5) * 0.18
            delta += max(0.0, values.survival_weight - 0.5) * 0.12
        if "social" in target_channels or "social" in prediction_type:
            delta += max(0.0, 0.5 - traits.trust_stance) * 0.10
            delta += max(0.0, traits.social_approach - 0.5) * 0.06
        if "continuity" in prediction_type:
            delta += max(0.0, identity.commitment_stability - 0.5) * 0.10
            delta += max(0.0, 0.55 - identity.continuity_resilience) * 0.12
        return round(max(-0.04, min(0.22, delta)), 6)

    def _pressure(
        self,
        variable_path: str,
        signed_pressure: float,
        evidence_count: int,
        *sources: str,
        rationale: str,
    ) -> LearningPressure | None:
        if evidence_count <= 0 or abs(signed_pressure) < 0.01:
            return None
        return LearningPressure(
            variable_path=variable_path,
            signed_pressure=signed_pressure,
            evidence_count=evidence_count,
            support=min(1.0, evidence_count / 5.0),
            sources=[source for source in sources if source],
            rationale=rationale,
        )

    def aggregate_pressures(
        self,
        *,
        tick: int,
        replay_batch: list[dict[str, object]],
        decision_history: list[dict[str, object]],
        prediction_ledger,
        verification_loop,
        social_memory,
        identity_tension_history: list[dict[str, object]],
        self_model,
        body_state: Mapping[str, float],
    ) -> list[LearningPressure]:
        cutoff = self.state.last_processed_tick
        new_episodes = [
            payload
            for payload in replay_batch
            if int(payload.get("timestamp", payload.get("cycle", 0))) > cutoff
        ]
        new_decisions = [
            payload for payload in decision_history if int(payload.get("tick", 0)) > cutoff
        ]
        verification_targets = [
            item for item in getattr(verification_loop, "archived_targets", [])
            if int(getattr(item, "outcome_tick", 0)) > cutoff
        ]
        falsifications = [
            item for item in getattr(verification_loop, "falsification_history", [])
            if int(getattr(item, "tick", 0)) > cutoff
        ]
        discrepancies = [
            item for item in (
                list(getattr(prediction_ledger, "discrepancies", []))
                + list(getattr(prediction_ledger, "archived_discrepancies", []))
            )
            if int(getattr(item, "last_seen_tick", 0)) > cutoff
        ]
        social_events = [
            payload
            for payload in getattr(social_memory, "interaction_history", [])
            if int(payload.get("tick", 0)) > cutoff
        ]
        tensions = [
            payload for payload in identity_tension_history if int(payload.get("tick", 0)) > cutoff
        ]
        inconsistency_events = [
            item for item in getattr(self_model, "self_inconsistency_events", [])
            if int(getattr(item, "tick", 0)) > cutoff
        ]
        repair_history = [
            item for item in getattr(self_model, "repair_history", [])
            if int(getattr(item, "tick", 0)) > cutoff and bool(getattr(item, "success", False))
        ]

        threat_events = sum(
            1
            for payload in new_episodes
            if str(payload.get("predicted_outcome", "neutral")) in {"survival_threat", "integrity_loss"}
            or float(payload.get("observation", {}).get("danger", payload.get("danger", 0.0))) >= 0.72
        )
        safe_repairs = sum(
            1
            for payload in new_episodes
            if str(payload.get("predicted_outcome", "neutral")) in {"neutral", "resource_gain"}
            and str(payload.get("action_taken", payload.get("action", ""))) in {"hide", "rest", "exploit_shelter", "thermoregulate"}
        ) + len(repair_history)
        exploratory_successes = sum(
            1
            for payload in new_episodes
            if str(payload.get("action_taken", payload.get("action", ""))) in {"scan", "seek_contact"}
            and str(payload.get("predicted_outcome", "neutral")) in {"neutral", "resource_gain"}
        )
        exploratory_failures = sum(
            1
            for payload in new_episodes
            if str(payload.get("action_taken", payload.get("action", ""))) in {"scan", "seek_contact", "forage"}
            and str(payload.get("predicted_outcome", "neutral")) in {"survival_threat", "integrity_loss"}
        )
        confirmations = sum(
            1
            for item in verification_targets
            if str(getattr(item, "outcome", "")) in {"confirmed", "partially_supported"}
        )
        falsification_count = len(falsifications) + sum(
            1
            for item in verification_targets
            if str(getattr(item, "outcome", "")) in {
                "falsified",
                "contradicted_by_new_evidence",
                "expired_unverified",
            }
        )
        rupture_count = sum(1 for payload in social_events if bool(payload.get("rupture", False)))
        repair_count = sum(1 for payload in social_events if bool(payload.get("repair", False)))
        overload_count = sum(
            1
            for payload in new_episodes
            if float(payload.get("body_state", {}).get("stress", 0.0)) >= 0.7
            or float(payload.get("body_state", {}).get("fatigue", 0.0)) >= 0.7
        )
        if float(body_state.get("stress", 0.0)) >= 0.72 or float(body_state.get("fatigue", 0.0)) >= 0.72:
            overload_count += 1
        continuity_strain = len(tensions) + len(inconsistency_events)
        danger_discrepancies = sum(
            1
            for item in discrepancies
            if "danger" in tuple(getattr(item, "target_channels", ()))
            or "continuity" in str(getattr(item, "discrepancy_type", ""))
        )
        active_commitments = (
            len(
                [
                    commitment
                    for commitment in getattr(
                        getattr(self_model, "identity_narrative", None),
                        "commitments",
                        [],
                    )
                    if bool(getattr(commitment, "active", False))
                ]
            )
            if getattr(self_model, "identity_narrative", None) is not None
            else 0
        )
        exploratory_decisions = sum(
            1 for payload in new_decisions if str(payload.get("action", "")) in {"scan", "seek_contact"}
        )

        pressures = [
            self._pressure("traits.caution_bias", (threat_events * 0.10) + (falsification_count * 0.04) + (overload_count * 0.03) - (safe_repairs * 0.04), threat_events + falsification_count + overload_count + safe_repairs, "repeated_survival_threat", "verification_failures", rationale="repeated unresolved threat accumulated across recent sleep evidence"),
            self._pressure("traits.threat_sensitivity", (threat_events * 0.08) + (danger_discrepancies * 0.05) - (safe_repairs * 0.03), threat_events + danger_discrepancies + safe_repairs, "survival_threat", "danger_discrepancy", rationale="threat-signalling episodes recurred across prediction and replay surfaces"),
            self._pressure("traits.trust_stance", (repair_count * 0.07) - (rupture_count * 0.10) - (falsification_count * 0.02), repair_count + rupture_count + falsification_count, "social_repair", "social_rupture", rationale="recent social repair and rupture patterns changed how much trust feels supportable"),
            self._pressure("traits.exploration_posture", (exploratory_successes * 0.08) + (exploratory_decisions * 0.02) - (exploratory_failures * 0.10) - (threat_events * 0.05), exploratory_successes + exploratory_failures + threat_events + exploratory_decisions, "exploratory_recurrence", "exploratory_outcomes", rationale="exploration was updated by repeated evidence rather than one-step novelty"),
            self._pressure("traits.social_approach", (repair_count * 0.05) - (rupture_count * 0.07), repair_count + rupture_count, "social_memory", rationale="repeated social outcomes shifted approach tendency"),
            self._pressure("values.survival_weight", (threat_events * 0.07) + (overload_count * 0.04) - (exploratory_successes * 0.02), threat_events + overload_count + exploratory_successes, "survival_threat", "maintenance_overload", rationale="survival-protective experiences repeatedly outweighed exploratory reward"),
            self._pressure("values.social_weight", (repair_count * 0.05) - (rupture_count * 0.04), repair_count + rupture_count, "social_memory", rationale="social value weighting moved with repeated rupture and repair"),
            self._pressure("values.exploration_weight", (exploratory_successes * 0.06) - (exploratory_failures * 0.07) - (threat_events * 0.03), exploratory_successes + exploratory_failures + threat_events, "exploration_outcomes", rationale="exploration value weighting followed repeated confirmation and failure signals"),
            self._pressure("values.maintenance_weight", (overload_count * 0.08) + (safe_repairs * 0.03), overload_count + safe_repairs, "maintenance_overload", rationale="maintenance demands kept recurring strongly enough to change long-horizon priorities"),
            self._pressure("values.hierarchy_stability", (confirmations * 0.04) - (falsification_count * 0.05) - (continuity_strain * 0.03), confirmations + falsification_count + continuity_strain, "verification_loop", rationale="the value hierarchy stabilized only where repeated confirmation exceeded contradiction"),
            self._pressure("identity.commitment_stability", (confirmations * 0.05) + (len(repair_history) * 0.03) - (continuity_strain * 0.08), confirmations + len(repair_history) + continuity_strain, "commitment_evidence", rationale="commitment stability moved with repeated reaffirmation versus inconsistency"),
            self._pressure("identity.identity_rigidity", (active_commitments * 0.02) + (confirmations * 0.02) - (falsification_count * 0.04) - (exploratory_successes * 0.02), active_commitments + confirmations + falsification_count + exploratory_successes, "identity_anchor_pressure", rationale="identity rigidity responds slowly to repeated support and contradiction"),
            self._pressure("identity.plasticity", (exploratory_successes * 0.03) + (repair_count * 0.02) - (threat_events * 0.04) - (active_commitments * 0.01), exploratory_successes + repair_count + threat_events + active_commitments, "plasticity_window", rationale="plasticity widens only under repeated adaptive evidence and narrows under repeated threat"),
            self._pressure("identity.continuity_resilience", (safe_repairs * 0.05) + (confirmations * 0.04) - (continuity_strain * 0.06) - (overload_count * 0.03), safe_repairs + confirmations + continuity_strain + overload_count, "continuity_repair", rationale="continuity resilience was shaped by repeated repair versus chronic strain"),
        ]
        del tick
        return [item for item in pressures if item is not None]

    def apply_sleep_cycle(
        self,
        *,
        sleep_cycle_id: int,
        tick: int,
        replay_batch: list[dict[str, object]],
        decision_history: list[dict[str, object]],
        prediction_ledger,
        verification_loop,
        social_memory,
        identity_tension_history: list[dict[str, object]],
        self_model,
        body_state: Mapping[str, float],
    ) -> SlowUpdateAudit:
        pressures = self.aggregate_pressures(
            tick=tick,
            replay_batch=replay_batch,
            decision_history=decision_history,
            prediction_ledger=prediction_ledger,
            verification_loop=verification_loop,
            social_memory=social_memory,
            identity_tension_history=identity_tension_history,
            self_model=self_model,
            body_state=body_state,
        )
        flat = self.state.flattened()
        updates: list[ConsolidationUpdate] = []
        remaining_budget = self.drift_budget.max_total_delta_per_cycle
        anti_collapse_triggered = len(pressures) > self.drift_budget.max_divergent_updates
        for pressure in sorted(pressures, key=lambda item: (-abs(item.signed_pressure), -item.evidence_count, item.variable_path)):
            previous_value = flat.get(pressure.variable_path, 0.5)
            window = self.plasticity_windows.get(pressure.variable_path, PlasticityWindow(pressure.variable_path, learning_rate=0.25, evidence_threshold=2))
            anchor = next((item for item in self.protected_anchors if item.variable_path == pressure.variable_path), None)
            if pressure.evidence_count < window.evidence_threshold:
                updates.append(ConsolidationUpdate(variable_path=pressure.variable_path, previous_value=previous_value, attempted_value=previous_value, new_value=previous_value, delta=0.0, status="rejected", source_pressures=list(pressure.sources), rationale=pressure.rationale, protected_anchor=anchor.label if anchor is not None else "", clipped_reason="insufficient repeated evidence"))
                continue
            scale = window.learning_rate * (0.4 + pressure.support * 0.6)
            if pressure.variable_path.startswith("identity."):
                scale *= 0.75 + (self.state.identity.plasticity * 0.5)
            attempted_delta = pressure.signed_pressure * scale * (1.0 - window.resistance * 0.5)
            budget_cap = self.drift_budget.per_variable_budget.get(pressure.variable_path, 0.03)
            status = "accepted"
            clipped_reason = ""
            if anti_collapse_triggered:
                budget_cap *= 0.75
                clipped_reason = "anti-collapse safeguard clipped multi-variable divergence"
                status = "clipped"
            if remaining_budget <= 0.0:
                updates.append(ConsolidationUpdate(variable_path=pressure.variable_path, previous_value=previous_value, attempted_value=_clamp(previous_value + attempted_delta), new_value=previous_value, delta=0.0, status="rejected", source_pressures=list(pressure.sources), rationale=pressure.rationale, protected_anchor=anchor.label if anchor is not None else "", clipped_reason="sleep-cycle drift budget exhausted"))
                continue
            clipped_delta = max(-budget_cap, min(budget_cap, attempted_delta))
            clipped_delta = max(-remaining_budget, min(remaining_budget, clipped_delta))
            attempted_value = _clamp(previous_value + attempted_delta)
            new_value = _clamp(previous_value + clipped_delta)
            if abs(clipped_delta - attempted_delta) > 1e-9:
                status = "clipped"
                if not clipped_reason:
                    clipped_reason = "per-cycle drift budget clipped the attempted update"
            if anchor is not None:
                protected = new_value < anchor.min_value or new_value > anchor.max_value
                if protected and pressure.evidence_count < anchor.required_evidence:
                    updates.append(ConsolidationUpdate(variable_path=pressure.variable_path, previous_value=previous_value, attempted_value=attempted_value, new_value=previous_value, delta=0.0, status="rejected", source_pressures=list(pressure.sources), rationale=pressure.rationale, protected_anchor=anchor.label, clipped_reason="protected anchor blocked abrupt drift"))
                    continue
                if protected:
                    new_value = _clamp(max(anchor.min_value, min(anchor.max_value, new_value)))
                    status = "clipped"
                    clipped_reason = "protected anchor bounded the accepted update"
            delta = new_value - previous_value
            self.state.set_value(pressure.variable_path, new_value)
            flat[pressure.variable_path] = new_value
            remaining_budget = max(0.0, remaining_budget - abs(delta))
            updates.append(ConsolidationUpdate(variable_path=pressure.variable_path, previous_value=previous_value, attempted_value=attempted_value, new_value=new_value, delta=delta, status=status if abs(delta) > 1e-9 else "rejected", source_pressures=list(pressure.sources), rationale=pressure.rationale, protected_anchor=anchor.label if anchor is not None else "", clipped_reason=clipped_reason if abs(delta) > 1e-9 or clipped_reason else "no effective update"))
        accepted = [item for item in updates if item.status in {"accepted", "clipped"} and abs(item.delta) > 1e-9]
        rejected = [item for item in updates if item.status == "rejected" or abs(item.delta) <= 1e-9]
        summary_bits: list[str] = []
        if accepted:
            strongest = max(accepted, key=lambda item: (abs(item.delta), item.variable_path))
            if strongest.variable_path == "traits.caution_bias" and strongest.delta > 0:
                summary_bits.append("Repeated unresolved threat has slowly increased my caution bias.")
            elif strongest.variable_path == "traits.trust_stance" and strongest.delta < 0:
                summary_bits.append("Trust became more fragile after repeated social rupture.")
            elif strongest.variable_path == "values.maintenance_weight":
                summary_bits.append("Chronic maintenance pressure changed my long-horizon priorities.")
            else:
                summary_bits.append(f"{_human_label(strongest.variable_path)} changed through repeated experience.")
        blocked = next((item for item in rejected if "drift budget" in item.clipped_reason or "protected anchor" in item.clipped_reason), None)
        if blocked is not None:
            summary_bits.append(f"{_human_label(blocked.variable_path)} remained stable because {blocked.clipped_reason}.")
        if not summary_bits:
            summary_bits.append("Recent experience was too weak or too inconsistent to move slow variables.")
        self.state.last_processed_tick = max(self.state.last_processed_tick, int(tick))
        self.state.sleep_cycles += 1
        self.state.last_summary = " ".join(summary_bits)
        audit = SlowUpdateAudit(audit_id=f"slow:{sleep_cycle_id}:{tick}", tick=tick, sleep_cycle_id=sleep_cycle_id, pressures=pressures, updates=updates, anti_collapse_triggered=anti_collapse_triggered, summary=self.state.last_summary)
        self.audit_history.append(audit)
        self.audit_history = self.audit_history[-self.max_audit_history :]
        return audit
