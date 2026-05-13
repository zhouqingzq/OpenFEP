"""Strict M12.1 step-extractor schema validators.

The functions here validate structured extractor outputs.  They do not call an
LLM; callers can inject an LLM-backed function or recorded fixture.
"""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

from .hyperparams import DEFAULT_HYPERPARAMS, M121Hyperparams

ALLOWED_BANDS = {"low", "med", "high"}
ALLOWED_DEFENSES = {
    "intellectualisation",
    "avoidance",
    "pleasing",
    "attack",
    "self_deprecation",
    "rationalisation",
    "control",
    "projection",
    "cold_detachment",
}
ALLOWED_CONFLICT_STYLES = {"confront", "yield", "cold_war", "argue", "flee"}
CORE_LOOP_STAGES = (
    "trigger_event",
    "interpretation",
    "emotion",
    "action",
    "outcome",
    "belief_reinforcement",
)

STEP_KEYS: dict[int, set[str]] = {
    1: {"summary", "evidence_quote_refs", "confidence_band"},
    2: {"evidence_items"},
    3: {"wants", "fears", "hypersensitive_to", "ignores", "default_interpretation", "evidence_quote_refs", "confidence_band"},
    4: {"about_self", "about_others", "about_world"},
    5: {"dominant_emotional_baseline", "threat_response", "defenses", "evidence_quote_refs", "confidence_band"},
    6: {"close_relationship_role", "recurring_loop_summary", "conflict_style", "drawn_to", "clashes_with", "evidence_quote_refs", "confidence_band"},
    7: {"trigger_event", "interpretation", "emotion", "action", "outcome", "belief_reinforcement", "evidence_quote_refs", "confidence_band"},
    8: {"stable_parts", "fragile_spots", "soft_spots", "communication_styles_likely_accepted", "communication_styles_that_trigger_defenses", "evidence_quote_refs", "confidence_band"},
}


class StepExtractorValidationError(ValueError):
    """Raised when a step extractor violates M12.1 boundaries."""

    def __init__(self, message: str, *, findings: tuple[object, ...] = ()) -> None:
        super().__init__(message)
        self.findings = findings


def build_step_extractor_prompt(step: int, snapshot: Mapping[str, object]) -> tuple[str, str]:
    """Build a bounded prompt for a single step extractor.

    The prompt exposes only the structured snapshot assembled by the
    orchestrator.  It asks for strict JSON; callers must still pass the result
    through ``validate_step_output`` before patching profile state.
    """
    if step not in STEP_KEYS:
        raise StepExtractorValidationError("unknown step")
    schema = _schema_for_step(step)
    allowed_refs = snapshot.get("allowed_evidence_quote_refs", [])
    system_prompt = (
        "You are an M12.1 bounded step extractor. Return only one JSON object. "
        "Do not invent turn ids, quote ids, hypothesis ids, or cue ids. "
        "Use only allowed evidence_quote_refs. Do not output floats. "
        "If evidence is thin, return {\"status\":\"insufficient_evidence\",\"reason\":\"...\"}. "
        "Return the best structured analysis supported by the snapshot."
    )
    user_prompt = {
        "step": step,
        "schema": schema,
        "allowed_evidence_quote_refs": list(allowed_refs) if isinstance(allowed_refs, Sequence) and not isinstance(allowed_refs, (str, bytes)) else [],
        "snapshot": snapshot,
    }
    import json

    return system_prompt, json.dumps(user_prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def insufficient_evidence(reason: str = "not_enough_material") -> dict[str, object]:
    return {"status": "insufficient_evidence", "reason": str(reason)[: DEFAULT_HYPERPARAMS.max_reason_chars]}


def noop_step_extractor(step: int) -> dict[str, object]:
    return insufficient_evidence(f"step_{step}_not_run")


def validate_step_output(
    step: int,
    payload: Mapping[str, object],
    *,
    snapshot: Mapping[str, object],
    hyperparams: M121Hyperparams = DEFAULT_HYPERPARAMS,
) -> dict[str, object]:
    if step not in STEP_KEYS:
        raise StepExtractorValidationError("unknown step")
    if not isinstance(payload, Mapping):
        raise StepExtractorValidationError("step output must be an object")
    _reject_float(payload)
    if str(payload.get("status", "")) == "insufficient_evidence":
        unknown = set(payload) - {"status", "reason", "evidence_quote_refs"}
        if unknown:
            raise StepExtractorValidationError(f"unknown insufficient-evidence fields: {sorted(unknown)}")
        return {
            "status": "insufficient_evidence",
            "reason": str(payload.get("reason", ""))[: hyperparams.max_reason_chars],
            "evidence_quote_refs": _quote_refs(payload.get("evidence_quote_refs", []), snapshot=snapshot),
        }
    unknown = set(payload) - STEP_KEYS[step]
    if unknown:
        raise StepExtractorValidationError(f"unknown step {step} fields: {sorted(unknown)}")
    result = copy.deepcopy(dict(payload))
    _validate_step_shape(step, result, snapshot=snapshot, hyperparams=hyperparams)
    return result


def _schema_for_step(step: int) -> dict[str, object]:
    if step == 1:
        return {"summary": "string", "evidence_quote_refs": ["allowed_ref"], "confidence_band": "low|med|high"}
    if step == 2:
        return {"evidence_items": [{"kind": "string", "content_summary": "string", "evidence_quote_refs": ["allowed_ref"], "confidence_band": "low|med|high"}]}
    if step == 3:
        return {
            "wants": "string",
            "fears": "string",
            "hypersensitive_to": ["string"],
            "ignores": ["string"],
            "default_interpretation": "string",
            "evidence_quote_refs": ["allowed_ref"],
            "confidence_band": "low|med|high",
        }
    if step == 4:
        belief = {"content_summary": "string", "evidence_quote_refs": ["allowed_ref"], "confidence_band": "low|med|high"}
        return {"about_self": belief, "about_others": belief, "about_world": belief}
    if step == 5:
        return {
            "dominant_emotional_baseline": "string",
            "threat_response": "string",
            "defenses": [
                {
                    "defense_kind": sorted(ALLOWED_DEFENSES),
                    "protects_what": "string",
                    "short_term_benefit": "string",
                    "long_term_cost": "string",
                    "evidence_quote_refs": ["allowed_ref"],
                    "confidence_band": "low|med|high",
                }
            ],
            "evidence_quote_refs": ["allowed_ref"],
            "confidence_band": "low|med|high",
        }
    if step == 6:
        target = {"kind": "string", "why": "string", "evidence_quote_refs": ["allowed_ref"], "confidence_band": "low|med|high"}
        return {
            "close_relationship_role": "string",
            "recurring_loop_summary": "string",
            "conflict_style": sorted(ALLOWED_CONFLICT_STYLES),
            "drawn_to": target,
            "clashes_with": target,
            "evidence_quote_refs": ["allowed_ref"],
            "confidence_band": "low|med|high",
        }
    if step == 7:
        return {**{stage: "string" for stage in CORE_LOOP_STAGES}, "evidence_quote_refs": ["allowed_ref"], "confidence_band": "low|med|high"}
    if step == 8:
        return {
            "stable_parts": ["string"],
            "fragile_spots": ["string"],
            "soft_spots": ["string"],
            "communication_styles_likely_accepted": ["string"],
            "communication_styles_that_trigger_defenses": ["string"],
            "evidence_quote_refs": ["allowed_ref"],
            "confidence_band": "low|med|high",
        }
    raise StepExtractorValidationError("unknown step")


def _validate_step_shape(
    step: int,
    payload: Mapping[str, object],
    *,
    snapshot: Mapping[str, object],
    hyperparams: M121Hyperparams,
) -> None:
    if step == 1:
        _bounded_string(payload.get("summary"), hyperparams.max_summary_chars, "summary")
        _band(payload.get("confidence_band"), "confidence_band")
        _quote_refs(payload.get("evidence_quote_refs"), snapshot=snapshot)
    elif step == 2:
        rows = _object_list(payload.get("evidence_items"), "evidence_items")[: hyperparams.max_evidence_items]
        for row in rows:
            _check_keys(row, {"kind", "content_summary", "evidence_quote_refs", "confidence_band"}, "evidence_item")
            _bounded_string(row.get("content_summary"), hyperparams.max_summary_chars, "evidence_item.content_summary")
            _band(row.get("confidence_band"), "evidence_item.confidence_band")
            _quote_refs(row.get("evidence_quote_refs"), snapshot=snapshot)
    elif step == 3:
        for key in ("wants", "fears", "default_interpretation"):
            _bounded_string(payload.get(key), hyperparams.max_summary_chars, key)
        _string_list(payload.get("hypersensitive_to"), "hypersensitive_to")
        _string_list(payload.get("ignores"), "ignores")
        _band(payload.get("confidence_band"), "confidence_band")
        _quote_refs(payload.get("evidence_quote_refs"), snapshot=snapshot)
    elif step == 4:
        for key in ("about_self", "about_others", "about_world"):
            row = _object(payload.get(key), key)
            _check_keys(row, {"content_summary", "evidence_quote_refs", "confidence_band"}, key)
            _bounded_string(row.get("content_summary"), hyperparams.max_summary_chars, f"{key}.content_summary")
            _band(row.get("confidence_band"), f"{key}.confidence_band")
            _quote_refs(row.get("evidence_quote_refs"), snapshot=snapshot)
    elif step == 5:
        for key in ("dominant_emotional_baseline", "threat_response"):
            _bounded_string(payload.get(key), hyperparams.max_summary_chars, key)
        _band(payload.get("confidence_band"), "confidence_band")
        _quote_refs(payload.get("evidence_quote_refs"), snapshot=snapshot)
        for row in _object_list(payload.get("defenses"), "defenses")[: hyperparams.max_defenses]:
            _check_keys(row, {"defense_kind", "protects_what", "short_term_benefit", "long_term_cost", "evidence_quote_refs", "confidence_band"}, "defense")
            _enum(row.get("defense_kind"), ALLOWED_DEFENSES, "defense_kind")
            for key in ("protects_what", "short_term_benefit", "long_term_cost"):
                _bounded_string(row.get(key), hyperparams.max_summary_chars, f"defense.{key}")
            _band(row.get("confidence_band"), "defense.confidence_band")
            _quote_refs(row.get("evidence_quote_refs"), snapshot=snapshot)
    elif step == 6:
        for key in ("close_relationship_role", "recurring_loop_summary"):
            _bounded_string(payload.get(key), hyperparams.max_summary_chars, key)
        _enum(payload.get("conflict_style"), ALLOWED_CONFLICT_STYLES, "conflict_style")
        _band(payload.get("confidence_band"), "confidence_band")
        _quote_refs(payload.get("evidence_quote_refs"), snapshot=snapshot)
        for key in ("drawn_to", "clashes_with"):
            row = _object(payload.get(key), key)
            _check_keys(row, {"kind", "why", "evidence_quote_refs", "confidence_band"}, key)
            _bounded_string(row.get("why"), hyperparams.max_summary_chars, f"{key}.why")
            _band(row.get("confidence_band"), f"{key}.confidence_band")
            _quote_refs(row.get("evidence_quote_refs"), snapshot=snapshot)
    elif step == 7:
        for key in CORE_LOOP_STAGES:
            _bounded_string(payload.get(key), hyperparams.max_summary_chars, key)
        _band(payload.get("confidence_band"), "confidence_band")
        refs = _quote_refs(payload.get("evidence_quote_refs"), snapshot=snapshot)
        if not refs:
            raise StepExtractorValidationError("step 7 requires evidence_quote_refs")
    elif step == 8:
        for key in (
            "stable_parts",
            "fragile_spots",
            "soft_spots",
            "communication_styles_likely_accepted",
            "communication_styles_that_trigger_defenses",
        ):
            _string_list(payload.get(key), key)
        _band(payload.get("confidence_band"), "confidence_band")
        _quote_refs(payload.get("evidence_quote_refs"), snapshot=snapshot)


def _surface_rows_for_lint(step: int, payload: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    section = f"step_{step}"
    if step == 1:
        rows.append({"section": section, "summary": payload.get("summary", "")})
    elif step == 2:
        for item in _object_list(payload.get("evidence_items"), "evidence_items"):
            rows.append({"section": section, "content_summary": item.get("content_summary", "")})
    elif step == 3:
        rows.append({key: payload.get(key, "") for key in ("wants", "fears", "default_interpretation")} | {"section": section})
    elif step == 4:
        for key in ("about_self", "about_others", "about_world"):
            row = _object(payload.get(key), key)
            rows.append({"section": section, "content_summary": row.get("content_summary", "")})
    elif step == 5:
        rows.append({"section": section, "content_summary": payload.get("dominant_emotional_baseline", ""), "reason": payload.get("threat_response", "")})
        for item in _object_list(payload.get("defenses"), "defenses"):
            rows.append({"section": section, **dict(item)})
    elif step == 6:
        rows.append({"section": section, "content_summary": payload.get("close_relationship_role", ""), "reason": payload.get("recurring_loop_summary", "")})
        for key in ("drawn_to", "clashes_with"):
            row = _object(payload.get(key), key)
            rows.append({"section": section, **dict(row)})
    elif step == 7:
        for key in CORE_LOOP_STAGES:
            rows.append({"section": section, "content_summary": payload.get(key, "")})
    elif step == 8:
        for key in ("stable_parts", "fragile_spots", "soft_spots", "communication_styles_likely_accepted", "communication_styles_that_trigger_defenses"):
            for text in _string_list(payload.get(key), key):
                rows.append({"section": section, "content_summary": text})
    return tuple(rows)


def _reject_float(value: object) -> None:
    if isinstance(value, float):
        raise StepExtractorValidationError("floats are forbidden in step extractor output")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_float(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_float(child)


def _quote_refs(value: object, *, snapshot: Mapping[str, object]) -> tuple[str, ...]:
    refs = _string_list(value, "evidence_quote_refs")
    allowed = _allowed_quote_refs(snapshot)
    unknown = [ref for ref in refs if ref not in allowed]
    if unknown:
        raise StepExtractorValidationError(f"unknown evidence quote refs: {unknown}")
    return refs


def _allowed_quote_refs(snapshot: Mapping[str, object]) -> set[str]:
    raw = snapshot.get("allowed_evidence_quote_refs", ())
    allowed = {str(item) for item in raw} if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else set()
    quotes = snapshot.get("current_turn_quotes", {})
    turn_id = str(snapshot.get("turn_id", ""))
    if isinstance(quotes, Mapping):
        for quote_id in quotes:
            allowed.add(str(quote_id))
            if turn_id:
                allowed.add(f"{turn_id}:{quote_id}")
    transcript_refs = snapshot.get("transcript_quote_refs", ())
    if isinstance(transcript_refs, Sequence) and not isinstance(transcript_refs, (str, bytes)):
        for ref in transcript_refs:
            if isinstance(ref, Mapping):
                qid = str(ref.get("quote_id", ""))
                tid = str(ref.get("turn_id", turn_id))
                allowed.add(qid)
                allowed.add(f"{tid}:{qid}")
    return allowed


def _object(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StepExtractorValidationError(f"{field} must be an object")
    return value


def _object_list(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StepExtractorValidationError(f"{field} must be a list")
    rows: list[Mapping[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise StepExtractorValidationError(f"{field} entries must be objects")
        rows.append(row)
    return rows


def _check_keys(payload: Mapping[str, object], allowed: set[str], field: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise StepExtractorValidationError(f"unknown {field} fields: {sorted(unknown)}")


def _bounded_string(value: object, limit: int, field: str) -> str:
    text = str(value or "")
    if not text:
        raise StepExtractorValidationError(f"{field} is required")
    if len(text) > limit:
        raise StepExtractorValidationError(f"{field} exceeds max length")
    return text


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StepExtractorValidationError(f"{field} must be a list")
    return tuple(str(item) for item in value if str(item))


def _band(value: object, field: str) -> None:
    _enum(value, ALLOWED_BANDS, field)


def _enum(value: object, allowed: set[str], field: str) -> None:
    if str(value) not in allowed:
        raise StepExtractorValidationError(f"{field} has invalid enum")
