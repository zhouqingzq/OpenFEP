from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return round(max(lo, min(hi, numeric)), 6)


def _bool_value(source: object, name: str) -> bool:
    if isinstance(source, Mapping):
        return bool(source.get(name, False))
    return bool(getattr(source, name, False))


def _state_section(state: object | None, name: str) -> object | None:
    if state is None:
        return None
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def _mapping_value(source: object | None, name: str, default: object = None) -> object:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


@dataclass(frozen=True)
class MetaControlSignal:
    signal_id: str
    memory_retrieval_gain_multiplier: float = 1.0
    retrieval_k_delta: int = 0
    lambda_energy_multiplier: float = 1.0
    lambda_attention_multiplier: float = 1.0
    lambda_memory_multiplier: float = 1.0
    lambda_control_multiplier: float = 1.0
    beta_efe_multiplier: float = 1.0
    effective_temperature_delta: float = 0.0
    candidate_limit: int | None = None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class MetaControlAdjustment:
    domain: str
    original: dict[str, object]
    adjusted: dict[str, object]
    signal: dict[str, object]
    rollback: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def memory_overdominance_detected(diagnostics: object | None) -> bool:
    if bool(getattr(diagnostics, "memory_overdominance", False)):
        return True
    interference = getattr(diagnostics, "memory_interference", None)
    if isinstance(interference, Mapping) and bool(interference.get("detected")):
        try:
            if float(interference.get("severity", 0.0)) >= 0.55:
                return True
        except (TypeError, ValueError):
            return True
    chosen = getattr(diagnostics, "chosen", None)
    if str(getattr(chosen, "dominant_component", "")) in {
        "memory_bias",
        "pattern_bias",
        "threat_memory_bias",
    }:
        return True
    ranked = list(getattr(diagnostics, "ranked_options", []) or [])
    for option in ranked[:3]:
        total_memory_pull = (
            abs(float(getattr(option, "memory_bias", 0.0) or 0.0))
            + abs(float(getattr(option, "pattern_bias", 0.0) or 0.0))
            + abs(float(getattr(option, "threat_memory_bias", 0.0) or 0.0))
        )
        if total_memory_pull >= 0.75:
            return True
    return False


def derive_meta_control_signal(
    *,
    state: object | None = None,
    guidance: object | None = None,
    diagnostics: object | None = None,
) -> MetaControlSignal:
    reasons: list[str] = []
    memory_multiplier = 1.0
    retrieval_k_delta = 0
    lambda_energy = 1.0
    lambda_attention = 1.0
    lambda_memory = 1.0
    lambda_control = 1.0
    beta_efe = 1.0
    temperature_delta = 0.0
    candidate_limit: int | None = None

    if _bool_value(guidance, "reduce_memory_reliance") or memory_overdominance_detected(diagnostics):
        memory_multiplier = 0.55
        lambda_memory = 0.75
        retrieval_k_delta = -1
        reasons.append("memory_overdominance_or_conflict")

    if _bool_value(guidance, "increase_control_gain"):
        lambda_control = 1.25
        reasons.append("increase_control_gain")

    if _bool_value(guidance, "increase_exploration_temperature"):
        temperature_delta += 0.18
        beta_efe = 1.10
        reasons.append("increase_exploration_temperature")

    resource = _state_section(state, "resource")
    resource_overload = bool(_mapping_value(resource, "overload", False))
    cognitive_load = _clamp(_mapping_value(resource, "cognitive_load", 0.0))
    if resource_overload or cognitive_load >= 0.82 or _bool_value(guidance, "compress_context"):
        lambda_energy = 1.15
        lambda_attention = 1.20
        temperature_delta += 0.12
        candidate_limit = 3
        reasons.append("resource_overload")

    return MetaControlSignal(
        signal_id="meta-control-stage4",
        memory_retrieval_gain_multiplier=memory_multiplier,
        retrieval_k_delta=retrieval_k_delta,
        lambda_energy_multiplier=lambda_energy,
        lambda_attention_multiplier=lambda_attention,
        lambda_memory_multiplier=lambda_memory,
        lambda_control_multiplier=lambda_control,
        beta_efe_multiplier=beta_efe,
        effective_temperature_delta=round(min(0.35, temperature_delta), 6),
        candidate_limit=candidate_limit,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def adjust_memory_retrieval(
    *,
    k: int,
    memory_retrieval_gain: float,
    signal: MetaControlSignal | None,
) -> MetaControlAdjustment:
    signal = signal or MetaControlSignal(signal_id="none")
    original = {
        "k": int(k),
        "memory_retrieval_gain": _clamp(memory_retrieval_gain),
    }
    adjusted_k = max(0, min(12, int(k) + int(signal.retrieval_k_delta)))
    adjusted_gain = _clamp(
        _clamp(memory_retrieval_gain) * signal.memory_retrieval_gain_multiplier
    )
    adjusted = {
        "k": adjusted_k,
        "memory_retrieval_gain": adjusted_gain,
    }
    return MetaControlAdjustment(
        domain="memory_retrieval",
        original=original,
        adjusted=adjusted,
        signal=signal.to_dict(),
        rollback=original,
    )


def adjust_path_scoring_meta_control(
    meta_control: object | None,
    signal: MetaControlSignal | None,
) -> MetaControlAdjustment:
    signal = signal or MetaControlSignal(signal_id="none")
    base = {
        "lambda_energy": _clamp(_mapping_value(meta_control, "lambda_energy", 0.25)),
        "lambda_attention": _clamp(_mapping_value(meta_control, "lambda_attention", 0.35)),
        "lambda_memory": _clamp(_mapping_value(meta_control, "lambda_memory", 0.25)),
        "lambda_control": _clamp(_mapping_value(meta_control, "lambda_control", 0.35)),
        "beta_efe": _clamp(_mapping_value(meta_control, "beta_efe", 0.5)),
        "exploration_temperature": _clamp(
            _mapping_value(meta_control, "exploration_temperature", 0.35),
            0.01,
            1.0,
        ),
    }
    adjusted = {
        "lambda_energy": _clamp(base["lambda_energy"] * signal.lambda_energy_multiplier),
        "lambda_attention": _clamp(
            base["lambda_attention"] * signal.lambda_attention_multiplier
        ),
        "lambda_memory": _clamp(base["lambda_memory"] * signal.lambda_memory_multiplier),
        "lambda_control": _clamp(
            base["lambda_control"] * signal.lambda_control_multiplier
        ),
        "beta_efe": _clamp(base["beta_efe"] * signal.beta_efe_multiplier),
        "exploration_temperature": _clamp(
            base["exploration_temperature"] + signal.effective_temperature_delta,
            0.01,
            1.0,
        ),
    }
    if signal.candidate_limit is not None:
        adjusted["candidate_limit"] = int(signal.candidate_limit)
    return MetaControlAdjustment(
        domain="path_scoring",
        original=base,
        adjusted=adjusted,
        signal=signal.to_dict(),
        rollback=base,
    )
