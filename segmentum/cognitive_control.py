"""Bounded causal control signals derived from compact cognitive state.

This module deliberately stays below any claim of subjective consciousness.  It
turns observable loop state into small, deterministic control nudges that can be
audited, ablated, and rolled back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .meta_control import MetaControlSignal, memory_overdominance_detected


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(lo, min(hi, numeric)), 6)


def _float_attr(obj: object | None, name: str, default: float = 0.0) -> float:
    if obj is None:
        return default
    try:
        return float(getattr(obj, name, default))
    except (TypeError, ValueError):
        return default


def _section(obj: object | None, name: str) -> object | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _value(obj: object | None, name: str, default: object = None) -> object:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _ranked_margins(diagnostics: object | None) -> tuple[float, float]:
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    if len(ranked) < 2:
        return 1.0, 1.0
    policy_margin = abs(
        _float_attr(ranked[0], "policy_score", 0.0)
        - _float_attr(ranked[1], "policy_score", 0.0)
    )
    efe_margin = abs(
        _float_attr(ranked[0], "expected_free_energy", 0.0)
        - _float_attr(ranked[1], "expected_free_energy", 0.0)
    )
    return _clamp(policy_margin), _clamp(efe_margin)


def _path_margin(path_summary: Mapping[str, object] | None, key: str, default: float) -> float:
    if not isinstance(path_summary, Mapping):
        return default
    try:
        return float(path_summary.get(key, default))
    except (TypeError, ValueError):
        return default


def _negative_outcome(value: str) -> bool:
    lowered = str(value or "").lower()
    return any(
        token in lowered
        for token in ("fail", "failed", "failure", "negative", "rupture", "worse", "miss")
    )


@dataclass(frozen=True)
class CognitiveControlSignal:
    memory_retrieval_gain: float = 1.0
    epistemic_bonus_gain: float = 1.0
    clarification_bias: float = 0.0
    repair_bias: float = 0.0
    effective_temperature_delta: float = 0.0
    candidate_budget_delta: int = 0
    assertion_strength: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MetaControlPolicy:
    """Derive a small causal control signal from bounded loop state."""

    def derive(
        self,
        state: object | None,
        diagnostics: object | None = None,
        path_summary: Mapping[str, object] | None = None,
        *,
        previous_outcome: str = "",
    ) -> CognitiveControlSignal:
        gaps = _section(state, "gaps")
        memory = _section(state, "memory")
        resource = _section(state, "resource")
        task = _section(state, "task")
        candidate_paths = _section(state, "candidate_paths")

        reasons: list[str] = []
        memory_gain = 1.0
        epistemic_gain = 1.0
        clarification_bias = 0.0
        repair_bias = 0.0
        temperature_delta = 0.0
        candidate_delta = 0
        assertion_strength = 1.0

        blocking = bool(_strings(_value(gaps, "blocking_gaps", [])))
        structured = list(_value(gaps, "structured_gaps", []) or [])
        high_contextual_or_criteria = False
        for gap in structured:
            kind = str(_value(gap, "kind", "")).lower()
            status = str(_value(gap, "status", "")).lower()
            severity = _clamp(_value(gap, "severity", 0.0))
            if kind in {"contextual", "criteria"} and (
                severity >= 0.7 or status in {"blocking", "high"}
            ):
                high_contextual_or_criteria = True
                break
        contextual = _strings(_value(gaps, "contextual_gaps", []))
        if blocking or high_contextual_or_criteria or (
            contextual and _clamp(_value(_section(state, "user"), "ambiguity", 0.0)) >= 0.7
        ):
            clarification_bias = max(clarification_bias, 0.28 if blocking else 0.22)
            epistemic_gain = max(epistemic_gain, 1.08)
            assertion_strength = min(assertion_strength, 0.68)
            reasons.append("blocking_or_high_severity_gap")

        policy_margin, efe_margin = _ranked_margins(diagnostics)
        policy_margin = min(policy_margin, abs(_path_margin(path_summary, "policy_margin", 1.0)))
        efe_margin = min(efe_margin, abs(_path_margin(path_summary, "efe_margin", 1.0)))
        posterior_margin = abs(_path_margin(path_summary, "posterior_margin", 1.0))
        selection_margin = _clamp(_value(candidate_paths, "selection_margin", 1.0))
        low_margin = (
            policy_margin < 0.12
            or efe_margin < 0.05
            or posterior_margin < 0.05
            or selection_margin < 0.08
        )
        if low_margin:
            assertion_strength = min(assertion_strength, 0.62)
            clarification_bias = max(clarification_bias, 0.18)
            temperature_delta = max(temperature_delta, 0.08)
            reasons.append("low_selection_margin")

        memory_conflict = bool(_strings(_value(memory, "memory_conflicts", [])))
        if memory_conflict or memory_overdominance_detected(diagnostics):
            memory_gain = min(memory_gain, 0.55)
            assertion_strength = min(assertion_strength, 0.72)
            reasons.append("memory_overdominance_or_conflict")

        overload = bool(_value(resource, "overload", False))
        cognitive_load = _clamp(_value(resource, "cognitive_load", 0.0))
        if overload or cognitive_load >= 0.82:
            temperature_delta = max(temperature_delta, 0.12)
            candidate_delta = min(candidate_delta, -2)
            assertion_strength = min(assertion_strength, 0.78)
            reasons.append("resource_overload")

        task_phase = str(_value(task, "task_phase", "")).lower()
        if _negative_outcome(previous_outcome) or task_phase == "repair" or blocking:
            repair_bias = max(repair_bias, 0.24)
            assertion_strength = min(assertion_strength, 0.7)
            reasons.append("repair_pressure")

        return CognitiveControlSignal(
            memory_retrieval_gain=_clamp(memory_gain, 0.1, 1.5),
            epistemic_bonus_gain=_clamp(epistemic_gain, 0.1, 1.5),
            clarification_bias=_clamp(clarification_bias),
            repair_bias=_clamp(repair_bias),
            effective_temperature_delta=round(min(0.35, temperature_delta), 6),
            candidate_budget_delta=int(candidate_delta),
            assertion_strength=_clamp(assertion_strength),
            reason=";".join(dict.fromkeys(reasons)),
        )


class CognitiveControlAdapter:
    """Compatibility adapter for injecting bounded control into existing surfaces."""

    @staticmethod
    def decision_context(
        context: Mapping[str, object] | None,
        signal: CognitiveControlSignal | None,
    ) -> dict[str, object]:
        payload = dict(context or {})
        if signal is not None:
            payload["cognitive_control"] = signal.to_dict()
        return payload

    @staticmethod
    def to_meta_control_signal(
        signal: CognitiveControlSignal | None,
    ) -> MetaControlSignal:
        if signal is None:
            return MetaControlSignal(signal_id="cognitive-control-none")
        retrieval_delta = -1 if signal.memory_retrieval_gain < 0.8 else 0
        candidate_limit = 3 if signal.candidate_budget_delta < 0 else None
        reasons = tuple(item for item in signal.reason.split(";") if item)
        return MetaControlSignal(
            signal_id="m7-cognitive-control",
            memory_retrieval_gain_multiplier=signal.memory_retrieval_gain,
            retrieval_k_delta=retrieval_delta,
            lambda_memory_multiplier=signal.memory_retrieval_gain,
            beta_efe_multiplier=signal.epistemic_bonus_gain,
            effective_temperature_delta=signal.effective_temperature_delta,
            candidate_limit=candidate_limit,
            reasons=reasons,
        )

    @staticmethod
    def path_scoring_config(
        meta_control: object | None,
        signal: CognitiveControlSignal | None,
    ) -> dict[str, object]:
        config = dict(meta_control) if isinstance(meta_control, Mapping) else {}
        if signal is not None:
            if "exploration_temperature" in config:
                config["exploration_temperature"] = _clamp(
                    float(config["exploration_temperature"])
                    + signal.effective_temperature_delta,
                    0.01,
                    1.0,
                )
            if signal.candidate_budget_delta < 0:
                current = int(config.get("candidate_limit", 5) or 5)
                config["candidate_limit"] = max(1, current + signal.candidate_budget_delta)
        return config

    @staticmethod
    def compact_prompt_guidance(
        signal: CognitiveControlSignal | None,
    ) -> dict[str, object]:
        if signal is None:
            return {}
        result: dict[str, object] = {
            "assertion_strength": signal.assertion_strength,
            "clarification_bias": signal.clarification_bias,
            "repair_bias": signal.repair_bias,
            "memory_retrieval_gain": signal.memory_retrieval_gain,
        }
        if signal.effective_temperature_delta:
            result["effective_temperature_delta"] = signal.effective_temperature_delta
        if signal.candidate_budget_delta:
            result["candidate_budget_delta"] = signal.candidate_budget_delta
        if signal.reason:
            result["reasons"] = signal.reason.split(";")[:6]
        return result
