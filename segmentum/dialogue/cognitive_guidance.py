"""Compressed cognitive guidance for prompt-facing consumers.

This module intentionally renders a small, derived guidance surface. It does
not expose raw cognitive events, raw diagnostics, payload dumps, or private
prompt/source material.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_SENSITIVE_KEY_FRAGMENTS = (
    "api",
    "authorization",
    "body",
    "conscious",
    "content",
    "diagnostic",
    "event",
    "history",
    "key",
    "markdown",
    "message",
    "payload",
    "prompt",
    "raw",
    "secret",
    "self-consciousness",
    "system",
    "token",
    "user",
)

_SENSITIVE_TEXT_MARKERS = (
    "Self-consciousness.md",
    "Conscious.md",
    "FULL ",
    "RAW ",
    "```",
)


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _safe_key(key: object) -> bool:
    lower = str(key).lower()
    return not any(fragment in lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _safe_text(value: object, *, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    lower = text.lower()
    if any(marker.lower() in lower for marker in _SENSITIVE_TEXT_MARKERS):
        return ""
    return text[:limit]


def _safe_float(value: object) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _safe_list(value: object, *, limit: int = 4) -> list[str]:
    result: list[str] = []
    for item in list(_as_sequence(value))[:limit]:
        text = _safe_text(item, limit=96)
        if text and text not in result:
            result.append(text)
    return result


def _flatten_gaps(active_gaps: object) -> list[str]:
    gaps = _as_mapping(active_gaps)
    result: list[str] = []
    for key in (
        "blocking_gaps",
        "epistemic_gaps",
        "contextual_gaps",
        "instrumental_gaps",
        "resource_gaps",
        "social_gaps",
    ):
        for item in _safe_list(gaps.get(key), limit=3):
            label = key.replace("_gaps", "")
            result.append(f"{label}: {item}")
            if len(result) >= 6:
                return result
    return result


def _selected_path(capsule: Mapping[str, object]) -> dict[str, object]:
    summary = _as_mapping(capsule.get("selected_path_summary"))
    if not summary:
        paths = _as_sequence(capsule.get("cognitive_paths"))
        summary = _as_mapping(paths[0]) if paths else {}
    if not summary:
        return {}
    action = _safe_text(
        summary.get("proposed_action") or summary.get("action") or capsule.get("chosen_action"),
        limit=64,
    )
    path: dict[str, object] = {}
    if action:
        path["action"] = action
    outcome = _safe_text(summary.get("expected_outcome") or capsule.get("chosen_predicted_outcome"), limit=80)
    if outcome:
        path["expected_outcome"] = outcome
    for key in ("total_cost", "posterior_weight", "expected_free_energy"):
        value = _safe_float(summary.get(key))
        if value is not None:
            path[key] = value
    return path


def _meta_flags(meta: Mapping[str, object]) -> list[str]:
    flags: list[str] = []
    for key, value in sorted(meta.items()):
        if _safe_key(key) and isinstance(value, bool) and value:
            flags.append(str(key))
        if len(flags) >= 8:
            break
    return flags


def build_compressed_cognitive_guidance(capsule: Mapping[str, object]) -> dict[str, object]:
    """Build a prompt-safe guidance packet from an FEP capsule-like mapping."""
    selected_path = _selected_path(capsule)
    meta = _as_mapping(capsule.get("meta_control_guidance"))
    memory = _as_mapping(capsule.get("memory_use_guidance"))
    affective = _as_mapping(capsule.get("affective_guidance"))
    prior = _as_mapping(capsule.get("self_prior_summary"))
    path_competition = _as_mapping(capsule.get("path_competition_summary"))

    uncertainty_label = _safe_text(capsule.get("decision_uncertainty"), limit=32) or "low"
    policy_margin = _safe_float(capsule.get("policy_margin"))
    efe_margin = _safe_float(capsule.get("efe_margin"))
    selection_margin = _safe_float(path_competition.get("selection_margin"))
    lower_assertiveness = (
        uncertainty_label in {"high", "medium"}
        or _safe_bool(meta.get("lower_assertiveness"))
        or (policy_margin is not None and policy_margin < 0.12)
        or (efe_margin is not None and efe_margin < 0.05)
        or (selection_margin is not None and selection_margin < 0.08)
    )

    guidance: dict[str, object] = {
        "current_task": {
            "chosen_action": _safe_text(capsule.get("chosen_action"), limit=64),
            "previous_outcome": _safe_text(capsule.get("previous_outcome"), limit=64),
        },
        "current_goal": _safe_text(capsule.get("chosen_predicted_outcome"), limit=96),
        "selected_path": selected_path,
        "missing_gaps": _flatten_gaps(capsule.get("active_gaps")),
        "uncertainty": {
            "level": uncertainty_label,
            "policy_margin": policy_margin,
            "efe_margin": efe_margin,
            "selection_margin": selection_margin,
        },
        "assertiveness": "lower" if lower_assertiveness else "normal",
        "memory_constraints": {
            "reduce_memory_reliance": _safe_bool(memory.get("reduce_memory_reliance")),
            "memory_conflict_count": memory.get("memory_conflict_count", 0) or 0,
            "activated_memory_count": memory.get("activated_memory_count", 0) or 0,
        },
        "generation_style_constraints": {
            "affective_actions": _safe_list(affective.get("actions"), limit=4),
            "meta_flags": _meta_flags(meta),
        },
    }

    prior_items: list[str] = []
    for key in ("summary", "current_prior", "stable_patterns", "reusable_patterns"):
        value = prior.get(key)
        if isinstance(value, str):
            text = _safe_text(value, limit=120)
            if text:
                prior_items.append(text)
        else:
            prior_items.extend(_safe_list(value, limit=2))
    if prior_items:
        guidance["self_prior_summary"] = prior_items[:3]

    affective_summary = _safe_text(affective.get("summary"), limit=120)
    if affective_summary:
        guidance["affective_guidance_summary"] = affective_summary

    if capsule.get("hidden_intent_label") in {"clear_subtext", "possible_subtext"}:
        guidance["hidden_intent_constraint"] = "observable_low_confidence_only"

    if _as_sequence(capsule.get("omitted_signals")):
        guidance["omitted_internal_signals"] = True

    return guidance


def format_compressed_cognitive_guidance(guidance: Mapping[str, object]) -> list[str]:
    """Render compressed guidance as prompt-safe bullet text."""
    lines: list[str] = []
    task = _as_mapping(guidance.get("current_task"))
    action = _safe_text(task.get("chosen_action"), limit=64)
    previous = _safe_text(task.get("previous_outcome"), limit=64)
    if action:
        suffix = f"; previous outcome: {previous}" if previous else ""
        lines.append(f"Current task: follow chosen action {action}{suffix}.")

    goal = _safe_text(guidance.get("current_goal"), limit=96)
    if goal:
        lines.append(f"Current goal: aim for {goal}.")

    selected_path = _as_mapping(guidance.get("selected_path"))
    path_action = _safe_text(selected_path.get("action"), limit=64)
    if path_action:
        parts = [f"Selected path: {path_action}"]
        outcome = _safe_text(selected_path.get("expected_outcome"), limit=80)
        if outcome:
            parts.append(f"expected {outcome}")
        for label, key in (
            ("total cost", "total_cost"),
            ("posterior weight", "posterior_weight"),
        ):
            value = selected_path.get(key)
            if value not in (None, ""):
                parts.append(f"{label} {value}")
        lines.append("; ".join(parts) + ".")

    gaps = _safe_list(guidance.get("missing_gaps"), limit=6)
    if gaps:
        lines.append("Missing gaps: " + " | ".join(gaps) + ".")

    uncertainty = _as_mapping(guidance.get("uncertainty"))
    level = _safe_text(uncertainty.get("level"), limit=32)
    if level:
        margin_bits: list[str] = []
        for label, key in (
            ("policy", "policy_margin"),
            ("EFE", "efe_margin"),
            ("selection", "selection_margin"),
        ):
            value = uncertainty.get(key)
            if value not in (None, ""):
                margin_bits.append(f"{label} margin {value}")
        detail = f" ({', '.join(margin_bits)})" if margin_bits else ""
        lines.append(f"Uncertainty: {level}{detail}.")

    assertiveness = _safe_text(guidance.get("assertiveness"), limit=32) or "normal"
    if assertiveness == "lower":
        lines.append("Assertiveness: lower; use provisional wording and avoid overclaiming.")
    else:
        lines.append("Assertiveness: normal; stay natural and bounded.")

    style = _as_mapping(guidance.get("generation_style_constraints"))
    actions = _safe_list(style.get("affective_actions"), limit=4)
    flags = _safe_list(style.get("meta_flags"), limit=8)
    if actions or flags:
        bits = []
        if actions:
            bits.append("affective actions " + ", ".join(actions))
        if flags:
            bits.append("meta flags " + ", ".join(flags))
        lines.append("Generation style constraints: " + "; ".join(bits) + ".")

    prior_items = _safe_list(guidance.get("self_prior_summary"), limit=3)
    if prior_items:
        lines.append("Compact self-prior for stance only: " + " | ".join(prior_items))

    summary = _safe_text(guidance.get("affective_guidance_summary"), limit=120)
    if summary:
        lines.append("Affective guidance is about response stance, not claims about the user: " + summary)

    memory = _as_mapping(guidance.get("memory_constraints"))
    if memory.get("reduce_memory_reliance"):
        lines.append("Memory use: treat recalled context as tentative when current evidence conflicts.")
    conflicts = memory.get("memory_conflict_count")
    if conflicts not in (None, "", 0):
        lines.append(f"Memory use: {conflicts} compact conflict signal(s); avoid over-relying on memory.")

    if guidance.get("hidden_intent_constraint") == "observable_low_confidence_only":
        lines.append("Hidden-intent cues are low-confidence observable signals; avoid motive claims.")

    if guidance.get("omitted_internal_signals"):
        lines.append("Some internal signals were omitted for prompt budget; do not invent missing internal state.")

    return lines


def render_compressed_cognitive_guidance(capsule: Mapping[str, object]) -> str:
    """Convenience helper used by prompt builders."""
    guidance = build_compressed_cognitive_guidance(capsule)
    return "\n".join(format_compressed_cognitive_guidance(guidance))
