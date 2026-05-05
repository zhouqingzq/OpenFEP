from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


REQUIRED_BOUNDARY_FIELDS = (
    "trigger_conditions",
    "action_steps",
    "expected_benefits",
    "applicability_bounds",
    "disable_conditions",
    "evidence_refs",
)

VALUE_MEMORY_WEIGHTS = {
    "future_reuse_gain": 0.22,
    "cost_reduction": 0.18,
    "error_avoidance_gain": 0.18,
    "transferability": 0.14,
    "activation_clarity": 0.10,
    "evidence_strength": 0.10,
    "recurrence_potential": 0.08,
    "overgeneralization_risk": -0.15,
    "maintenance_cost": -0.10,
}

QUARANTINE_KIND = "quarantined_candidate"
REJECTED_KIND = "rejected_candidate"


def _clamp(value: object, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(low, min(high, numeric))


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _lower_tokens(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in normalized.split() if token}


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(needle in lowered for needle in needles)


@dataclass(frozen=True)
class ValueMemoryCandidate:
    summary: str
    trigger_conditions: list[str]
    action_steps: list[str]
    expected_benefits: list[str]
    applicability_bounds: list[str]
    disable_conditions: list[str]
    evidence_refs: list[str]
    source_material: str = ""
    candidate_id: str = ""
    proposed_kind: str = "experience"
    extractor_source: str = "deterministic"
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], *, source_material: str = "") -> "ValueMemoryCandidate":
        return cls(
            summary=str(payload.get("summary", "")).strip(),
            trigger_conditions=_string_list(payload.get("trigger_conditions")),
            action_steps=_string_list(payload.get("action_steps")),
            expected_benefits=_string_list(payload.get("expected_benefits")),
            applicability_bounds=_string_list(payload.get("applicability_bounds")),
            disable_conditions=_string_list(payload.get("disable_conditions")),
            evidence_refs=_string_list(payload.get("evidence_refs")),
            source_material=str(payload.get("source_material", source_material or "")).strip(),
            candidate_id=str(payload.get("candidate_id", "")).strip(),
            proposed_kind=str(payload.get("proposed_kind", "experience")).strip() or "experience",
            extractor_source=str(payload.get("extractor_source", "llm")).strip() or "llm",
            metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def missing_boundary_fields(self) -> list[str]:
        missing: list[str] = []
        for field_name in REQUIRED_BOUNDARY_FIELDS:
            value = getattr(self, field_name)
            if not value:
                missing.append(field_name)
        return missing

    def has_action_structure(self) -> bool:
        return bool(
            self.trigger_conditions
            and self.action_steps
            and self.expected_benefits
            and self.applicability_bounds
            and self.disable_conditions
            and self.evidence_refs
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "summary": self.summary,
            "trigger_conditions": list(self.trigger_conditions),
            "action_steps": list(self.action_steps),
            "expected_benefits": list(self.expected_benefits),
            "applicability_bounds": list(self.applicability_bounds),
            "disable_conditions": list(self.disable_conditions),
            "evidence_refs": list(self.evidence_refs),
            "source_material": self.source_material,
            "proposed_kind": self.proposed_kind,
            "extractor_source": self.extractor_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValueMemoryScoreBreakdown:
    future_reuse_gain: float = 0.0
    cost_reduction: float = 0.0
    error_avoidance_gain: float = 0.0
    transferability: float = 0.0
    activation_clarity: float = 0.0
    evidence_strength: float = 0.0
    recurrence_potential: float = 0.0
    overgeneralization_risk: float = 0.0
    maintenance_cost: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ValueMemoryScoreBreakdown":
        return cls(
            future_reuse_gain=_clamp(payload.get("future_reuse_gain")),
            cost_reduction=_clamp(payload.get("cost_reduction")),
            error_avoidance_gain=_clamp(payload.get("error_avoidance_gain")),
            transferability=_clamp(payload.get("transferability")),
            activation_clarity=_clamp(payload.get("activation_clarity")),
            evidence_strength=_clamp(payload.get("evidence_strength")),
            recurrence_potential=_clamp(payload.get("recurrence_potential")),
            overgeneralization_risk=_clamp(payload.get("overgeneralization_risk")),
            maintenance_cost=_clamp(payload.get("maintenance_cost")),
        )

    def weighted_score(self) -> float:
        values = self.to_dict()
        score = sum(float(values[key]) * weight for key, weight in VALUE_MEMORY_WEIGHTS.items())
        return max(-1.0, min(1.0, score))

    def to_dict(self) -> dict[str, float]:
        return {
            "future_reuse_gain": round(self.future_reuse_gain, 6),
            "cost_reduction": round(self.cost_reduction, 6),
            "error_avoidance_gain": round(self.error_avoidance_gain, 6),
            "transferability": round(self.transferability, 6),
            "activation_clarity": round(self.activation_clarity, 6),
            "evidence_strength": round(self.evidence_strength, 6),
            "recurrence_potential": round(self.recurrence_potential, 6),
            "overgeneralization_risk": round(self.overgeneralization_risk, 6),
            "maintenance_cost": round(self.maintenance_cost, 6),
        }


@dataclass(frozen=True)
class ValueMemoryEvaluation:
    candidate: ValueMemoryCandidate
    score_breakdown: ValueMemoryScoreBreakdown
    future_path_utility: float
    value_memory_score: float
    candidate_kind: str
    quarantine_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    audit_flags: tuple[str, ...] = ()

    @property
    def is_quarantined(self) -> bool:
        return self.candidate_kind == QUARANTINE_KIND

    @property
    def is_rejected(self) -> bool:
        return self.candidate_kind == REJECTED_KIND

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "score_breakdown": self.score_breakdown.to_dict(),
            "future_path_utility": round(self.future_path_utility, 6),
            "value_memory_score": round(self.value_memory_score, 6),
            "candidate_kind": self.candidate_kind,
            "quarantine_reasons": list(self.quarantine_reasons),
            "rejection_reasons": list(self.rejection_reasons),
            "audit_flags": list(self.audit_flags),
            "weights": dict(VALUE_MEMORY_WEIGHTS),
        }


class ValueMemoryEvaluator:
    def evaluate(self, candidate: ValueMemoryCandidate) -> ValueMemoryEvaluation:
        rejection_reasons: list[str] = []
        quarantine_reasons: list[str] = []
        audit_flags: list[str] = []
        missing = candidate.missing_boundary_fields()
        if missing:
            rejection_reasons.extend(f"missing_boundary:{field}" for field in missing)
        if not candidate.summary:
            rejection_reasons.append("missing_summary")
        if candidate.summary and not candidate.has_action_structure():
            quarantine_reasons.append("summary_without_complete_action_structure")

        text = " ".join(
            [
                candidate.summary,
                *candidate.trigger_conditions,
                *candidate.action_steps,
                *candidate.expected_benefits,
                *candidate.applicability_bounds,
                *candidate.disable_conditions,
                candidate.source_material,
            ]
        )
        tokens = _lower_tokens(text)
        evidence_strength = self._evidence_strength(candidate)
        activation_clarity = self._activation_clarity(candidate)
        overgeneralization_risk = self._overgeneralization_risk(candidate)
        maintenance_cost = self._maintenance_cost(candidate)
        breakdown = ValueMemoryScoreBreakdown(
            future_reuse_gain=self._future_reuse_gain(candidate, tokens),
            cost_reduction=self._cost_reduction(candidate, tokens),
            error_avoidance_gain=self._error_avoidance_gain(candidate, tokens),
            transferability=self._transferability(candidate, tokens),
            activation_clarity=activation_clarity,
            evidence_strength=evidence_strength,
            recurrence_potential=self._recurrence_potential(candidate, tokens),
            overgeneralization_risk=overgeneralization_risk,
            maintenance_cost=maintenance_cost,
        )
        score = breakdown.weighted_score()
        if evidence_strength < 0.35:
            quarantine_reasons.append("evidence_strength_below_threshold")
        if activation_clarity < 0.35:
            quarantine_reasons.append("activation_clarity_below_threshold")
        if overgeneralization_risk >= 0.65:
            quarantine_reasons.append("overgeneralization_risk_high")
        if maintenance_cost >= 0.75 and score < 0.45:
            quarantine_reasons.append("maintenance_cost_high")
        if not candidate.evidence_refs:
            quarantine_reasons.append("missing_evidence_refs")
        if candidate.evidence_refs and candidate.source_material:
            unsupported = [
                ref for ref in candidate.evidence_refs
                if ref.casefold() not in candidate.source_material.casefold()
            ]
            if unsupported and len(unsupported) == len(candidate.evidence_refs):
                quarantine_reasons.append("evidence_not_supported_by_material")
        if _contains_any(text, ("always", "never", "all tasks", "every task", "任何情况", "所有任务")):
            audit_flags.append("absolute_language")

        if rejection_reasons:
            kind = REJECTED_KIND
        elif quarantine_reasons:
            kind = QUARANTINE_KIND
        else:
            kind = self._candidate_kind(candidate, breakdown, score)

        return ValueMemoryEvaluation(
            candidate=candidate,
            score_breakdown=breakdown,
            future_path_utility=score,
            value_memory_score=score,
            candidate_kind=kind,
            quarantine_reasons=tuple(dict.fromkeys(quarantine_reasons)),
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            audit_flags=tuple(dict.fromkeys(audit_flags)),
        )

    def _future_reuse_gain(self, candidate: ValueMemoryCandidate, tokens: set[str]) -> float:
        signal = 0.25 if candidate.action_steps else 0.0
        if len(candidate.action_steps) >= 2:
            signal += 0.25
        if tokens & {"reuse", "reusable", "similar", "repeat", "recurring", "pattern", "复用", "相似"}:
            signal += 0.30
        if candidate.applicability_bounds:
            signal += 0.15
        return _clamp(signal)

    def _cost_reduction(self, candidate: ValueMemoryCandidate, tokens: set[str]) -> float:
        signal = 0.15 if candidate.expected_benefits else 0.0
        if tokens & {"cost", "cheap", "minimal", "minimum", "fast", "verify", "沟通", "验证", "成本", "最小"}:
            signal += 0.45
        if any("cost" in item.casefold() or "成本" in item for item in candidate.expected_benefits):
            signal += 0.25
        return _clamp(signal)

    def _error_avoidance_gain(self, candidate: ValueMemoryCandidate, tokens: set[str]) -> float:
        signal = 0.0
        if tokens & {"error", "avoid", "prevent", "mistake", "risk", "failure", "返工", "错误", "避免"}:
            signal += 0.50
        if candidate.disable_conditions:
            signal += 0.20
        if any("avoid" in item.casefold() or "避免" in item for item in candidate.expected_benefits):
            signal += 0.20
        return _clamp(signal)

    def _transferability(self, candidate: ValueMemoryCandidate, tokens: set[str]) -> float:
        signal = 0.15 if candidate.applicability_bounds else 0.0
        if tokens & {"similar", "cross", "general", "transfer", "domain", "相似", "跨", "泛化"}:
            signal += 0.35
        if len(candidate.applicability_bounds) >= 2:
            signal += 0.20
        if len(candidate.disable_conditions) >= 1:
            signal += 0.10
        return _clamp(signal)

    def _activation_clarity(self, candidate: ValueMemoryCandidate) -> float:
        if not candidate.trigger_conditions:
            return 0.0
        scores = []
        for trigger in candidate.trigger_conditions:
            lowered = trigger.casefold()
            clarity = 0.35
            if any(token in lowered for token in ("when", "if", "under", "遇到", "如果", "当")):
                clarity += 0.25
            if len(trigger.split()) >= 4 or len(trigger) >= 8:
                clarity += 0.20
            if any(token in lowered for token in ("uncertain", "similar", "task", "需求", "不确定", "相似")):
                clarity += 0.15
            scores.append(_clamp(clarity))
        return _clamp(mean(scores))

    def _evidence_strength(self, candidate: ValueMemoryCandidate) -> float:
        if not candidate.evidence_refs:
            return 0.0
        support = min(1.0, len(candidate.evidence_refs) / 3.0)
        if candidate.source_material:
            matched = sum(
                1 for ref in candidate.evidence_refs
                if ref.casefold() in candidate.source_material.casefold()
            )
            support = (support * 0.45) + ((matched / max(1, len(candidate.evidence_refs))) * 0.55)
        return _clamp(support)

    def _recurrence_potential(self, candidate: ValueMemoryCandidate, tokens: set[str]) -> float:
        signal = 0.10 if candidate.trigger_conditions else 0.0
        if tokens & {"recurring", "repeat", "again", "often", "weekly", "similar", "反复", "经常", "相似"}:
            signal += 0.45
        if len(candidate.evidence_refs) >= 2:
            signal += 0.15
        return _clamp(signal)

    def _overgeneralization_risk(self, candidate: ValueMemoryCandidate) -> float:
        text = " ".join([
            candidate.summary,
            *candidate.applicability_bounds,
            *candidate.disable_conditions,
            *candidate.trigger_conditions,
        ]).casefold()
        risk = 0.15
        if len(candidate.evidence_refs) <= 1:
            risk += 0.20
        weak_disable = (
            not candidate.disable_conditions
            or all(item.strip().casefold() in {"none", "n/a", "no", "无", "无禁用边界"} for item in candidate.disable_conditions)
        )
        if weak_disable:
            risk += 0.35
        if _contains_any(text, ("always", "never", "all tasks", "every task", "任何情况", "所有任务")):
            risk += 0.35
        if any(len(item) < 4 for item in candidate.applicability_bounds):
            risk += 0.10
        if candidate.applicability_bounds and candidate.disable_conditions and not weak_disable:
            risk -= 0.15
        return _clamp(risk)

    def _maintenance_cost(self, candidate: ValueMemoryCandidate) -> float:
        size = (
            len(candidate.trigger_conditions)
            + len(candidate.action_steps)
            + len(candidate.applicability_bounds)
            + len(candidate.disable_conditions)
        )
        cost = 0.10 + max(0, size - 8) * 0.06
        if len(candidate.summary) > 240:
            cost += 0.10
        return _clamp(cost)

    def _candidate_kind(
        self,
        candidate: ValueMemoryCandidate,
        breakdown: ValueMemoryScoreBreakdown,
        score: float,
    ) -> str:
        proposed = candidate.proposed_kind
        if proposed in {"experience", "skill_candidate", "hook_candidate", "guardrail"}:
            return proposed
        if breakdown.error_avoidance_gain >= 0.55 and breakdown.future_reuse_gain >= 0.35:
            return "guardrail" if score >= 0.30 else "hook_candidate"
        if breakdown.future_reuse_gain >= 0.55 and breakdown.activation_clarity >= 0.55:
            return "skill_candidate"
        return "experience"


class ValueMemoryExtractor:
    def __init__(
        self,
        llm_extractor: Callable[..., Sequence[Mapping[str, object]]] | None = None,
        evaluator: ValueMemoryEvaluator | None = None,
    ) -> None:
        self.llm_extractor = llm_extractor
        self.evaluator = evaluator or ValueMemoryEvaluator()

    def extract(self, material: str, *, source_id: str = "material") -> list[ValueMemoryEvaluation]:
        raw_candidates: list[Mapping[str, object]] = []
        if self.llm_extractor is not None:
            try:
                raw = self.llm_extractor(material=material, source_id=source_id)
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                    raw_candidates = [item for item in raw if isinstance(item, Mapping)]
            except Exception:
                raw_candidates = []
        if not raw_candidates:
            raw_candidates = self._deterministic_candidates(material, source_id=source_id)

        evaluations: list[ValueMemoryEvaluation] = []
        for index, payload in enumerate(raw_candidates):
            enriched = dict(payload)
            enriched.setdefault("candidate_id", f"{source_id}:value:{index}")
            enriched.setdefault("source_material", material)
            candidate = ValueMemoryCandidate.from_mapping(enriched, source_material=material)
            evaluations.append(self.evaluator.evaluate(candidate))
        return evaluations

    def _deterministic_candidates(self, material: str, *, source_id: str) -> list[dict[str, object]]:
        text = " ".join(str(material or "").split())
        if not text:
            return []
        lowered = text.casefold()
        structural = (
            _contains_any(lowered, ("uncertain", "ambiguous", "不确定", "模糊"))
            and _contains_any(lowered, ("hypothesis", "assumption", "假设"))
            and _contains_any(lowered, ("verify", "test", "验证", "可验证"))
            and _contains_any(lowered, ("minimal", "minimum", "最小"))
        )
        avoid_error = _contains_any(lowered, ("avoid", "prevent", "error", "mistake", "risk", "避免", "错误"))
        reusable = _contains_any(lowered, ("similar", "reuse", "repeat", "pattern", "相似", "复用"))
        if structural:
            return [{
                "summary": "Under uncertain requirements, form a verifiable hypothesis before the smallest implementation.",
                "trigger_conditions": ["when requirements are uncertain or ambiguous"],
                "action_steps": [
                    "state the current assumption as a testable hypothesis",
                    "define the smallest implementation that can validate it",
                    "verify the result before broadening the solution",
                ],
                "expected_benefits": [
                    "reduces planning and implementation cost",
                    "avoids building the wrong thing",
                    "improves reuse in similar ambiguous tasks",
                ],
                "applicability_bounds": ["ambiguous engineering or planning tasks", "similar tasks with verifiable assumptions"],
                "disable_conditions": ["requirements are already explicit and verified", "the smallest test would be unsafe or irreversible"],
                "evidence_refs": [text[:160]],
                "proposed_kind": "skill_candidate",
                "extractor_source": "deterministic_structure",
                "metadata": {"source_id": source_id},
            }]
        if avoid_error and reusable:
            return [{
                "summary": "Use prior failure evidence as a guardrail before repeating a similar action.",
                "trigger_conditions": ["when a similar action previously produced an error or avoidable risk"],
                "action_steps": ["retrieve the prior failure condition", "check whether the current context matches it", "add a guardrail before acting"],
                "expected_benefits": ["avoids repeated errors", "reduces verification cost"],
                "applicability_bounds": ["similar tasks with matching failure conditions"],
                "disable_conditions": ["current context has material differences from the prior failure"],
                "evidence_refs": [text[:160]],
                "proposed_kind": "guardrail",
                "extractor_source": "deterministic_structure",
                "metadata": {"source_id": source_id},
            }]
        if _contains_any(lowered, ("success", "risk", "trust", "protect")):
            return [{
                "summary": text[:180],
                "trigger_conditions": [],
                "action_steps": [],
                "expected_benefits": ["surface positive or risk language may matter later"],
                "applicability_bounds": [],
                "disable_conditions": [],
                "evidence_refs": [text[:160]],
                "proposed_kind": "experience",
                "extractor_source": "deterministic_surface_guard",
                "metadata": {"source_id": source_id},
            }]
        return []


def value_memory_payload_from_evaluation(evaluation: ValueMemoryEvaluation) -> dict[str, object]:
    return evaluation.to_dict()


def value_memory_utility_from_metadata(metadata: Mapping[str, object] | None, default: float = 0.0) -> float:
    if not isinstance(metadata, Mapping):
        return float(default)
    payload = metadata.get("value_memory")
    if isinstance(payload, Mapping):
        if payload.get("candidate_kind") in {QUARANTINE_KIND, REJECTED_KIND}:
            return 0.0
        return _clamp(payload.get("future_path_utility"), -1.0, 1.0)
    return float(default)


@dataclass(frozen=True)
class QuarantineDecision:
    action: str
    reasons: tuple[str, ...]
    candidate_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "reasons": list(self.reasons),
            "candidate_kind": self.candidate_kind,
        }


class ValueMemoryQuarantinePolicy:
    def __init__(
        self,
        *,
        evidence_threshold: float = 0.50,
        activation_threshold: float = 0.50,
        overgeneralization_threshold: float = 0.45,
        stale_audit_threshold: int = 3,
    ) -> None:
        self.evidence_threshold = evidence_threshold
        self.activation_threshold = activation_threshold
        self.overgeneralization_threshold = overgeneralization_threshold
        self.stale_audit_threshold = stale_audit_threshold

    def decide(self, payload: Mapping[str, object], *, observed: Mapping[str, object] | None = None) -> QuarantineDecision:
        observed = dict(observed or {})
        kind = str(payload.get("candidate_kind", ""))
        reasons = _string_list(payload.get("quarantine_reasons"))
        rejection_reasons = _string_list(payload.get("rejection_reasons"))
        breakdown = payload.get("score_breakdown")
        breakdown = dict(breakdown or {}) if isinstance(breakdown, Mapping) else {}
        evidence = _clamp(breakdown.get("evidence_strength"))
        activation = _clamp(breakdown.get("activation_clarity"))
        overgeneralization = _clamp(breakdown.get("overgeneralization_risk"))
        maintenance = _clamp(breakdown.get("maintenance_cost"))
        observed_success = _clamp(observed.get("reuse_success_rate"))
        no_activation_count = int(observed.get("no_activation_audit_count", 0) or 0)
        contradicted = bool(observed.get("contradicted", False))
        duplicate_absorbed = bool(observed.get("duplicate_absorbed", False))

        if kind == REJECTED_KIND or rejection_reasons:
            return QuarantineDecision("reject", tuple(rejection_reasons or reasons or ["rejected_candidate"]), REJECTED_KIND)
        if contradicted:
            return QuarantineDecision("retire", ("contradicted_by_later_evidence",), QUARANTINE_KIND)
        if duplicate_absorbed:
            return QuarantineDecision("retire", ("duplicate_absorbed_no_independent_value",), QUARANTINE_KIND)
        if no_activation_count >= self.stale_audit_threshold:
            return QuarantineDecision("retire", ("stale_without_activation",), QUARANTINE_KIND)
        if maintenance > max(0.75, observed_success + 0.30):
            return QuarantineDecision("retire", ("maintenance_cost_exceeds_observed_benefit",), QUARANTINE_KIND)
        if (
            evidence >= self.evidence_threshold
            and activation >= self.activation_threshold
            and overgeneralization <= self.overgeneralization_threshold
        ):
            return QuarantineDecision("release", ("quarantine_release_thresholds_met",), "experience")
        return QuarantineDecision("keep_quarantined", tuple(reasons or ["quarantine_thresholds_not_met"]), QUARANTINE_KIND)


@dataclass(frozen=True)
class CalibrationAuditReport:
    sample_count: int
    mean_predicted_utility: float
    mean_observed_utility: float
    drift: float
    dimension_drifts: dict[str, float]
    recommended_weight_updates: dict[str, float]
    audit_record: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "mean_predicted_utility": round(self.mean_predicted_utility, 6),
            "mean_observed_utility": round(self.mean_observed_utility, 6),
            "drift": round(self.drift, 6),
            "dimension_drifts": {key: round(value, 6) for key, value in self.dimension_drifts.items()},
            "recommended_weight_updates": {key: round(value, 6) for key, value in self.recommended_weight_updates.items()},
            "audit_record": dict(self.audit_record),
        }


class ValueMemoryCalibrator:
    def audit(self, samples: Sequence[Mapping[str, object]]) -> CalibrationAuditReport:
        rows = [dict(sample) for sample in samples]
        if not rows:
            return CalibrationAuditReport(0, 0.0, 0.0, 0.0, {}, {}, {"event": "value_memory_calibration", "empty": True})
        predicted = [_clamp(row.get("predicted_utility"), -1.0, 1.0) for row in rows]
        observed = [_clamp(row.get("observed_utility"), -1.0, 1.0) for row in rows]
        drift = mean(observed) - mean(predicted)
        dimension_drifts: dict[str, float] = {}
        recommended: dict[str, float] = {}
        for key in VALUE_MEMORY_WEIGHTS:
            values = []
            for row, obs in zip(rows, observed):
                breakdown = row.get("score_breakdown")
                if not isinstance(breakdown, Mapping):
                    continue
                values.append(obs - _clamp(breakdown.get(key), -1.0, 1.0))
            if not values:
                continue
            dimension_drift = mean(values)
            dimension_drifts[key] = dimension_drift
            if abs(dimension_drift) >= 0.12:
                recommended[key] = VALUE_MEMORY_WEIGHTS[key] + (dimension_drift * 0.05)
        audit_record = {
            "event": "value_memory_calibration",
            "weight_update_policy": "explicit_report_only",
            "sample_count": len(rows),
            "requires_manual_application": True,
        }
        return CalibrationAuditReport(
            sample_count=len(rows),
            mean_predicted_utility=mean(predicted),
            mean_observed_utility=mean(observed),
            drift=drift,
            dimension_drifts=dimension_drifts,
            recommended_weight_updates=recommended,
            audit_record=audit_record,
        )


def offline_gold_set() -> list[dict[str, object]]:
    return [
        {
            "case_id": "structural_no_keyword",
            "material": "When requirements are uncertain, state a verifiable hypothesis and make the minimum implementation to test it.",
            "expected_kind": "skill_candidate",
            "expected_score_min": 0.35,
        },
        {
            "case_id": "keyword_false_positive",
            "material": "Success, risk, trust, and protect are listed as inspiring words on a poster.",
            "expected_kind": QUARANTINE_KIND,
            "expected_score_max": 0.25,
        },
        {
            "case_id": "overgeneralized_single_evidence",
            "material": "Always apply this one example to every task forever.",
            "expected_kind": QUARANTINE_KIND,
            "expected_risk_min": 0.65,
        },
    ]
