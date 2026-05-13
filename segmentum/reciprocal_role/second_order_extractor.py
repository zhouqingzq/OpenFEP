"""Strict schema validation and prompt builders for M12.2 extractors."""

from __future__ import annotations

import copy
import json
from typing import Mapping, Sequence

from .hyperparams import BANDS, DEFAULT_HYPERPARAMS, M122Hyperparams, normalize_band
from .plain_language_linter import lint_user_facing_fields


class SecondOrderExtractorValidationError(ValueError):
    pass


FIRST_ORDER_KEYS = {
    "persona_about_user_claims",
    "claim_group_updates",
    "unresolved_uncertainty_points",
    "high_gain_candidates",
    "insufficient_evidence",
}
SECOND_ORDER_KEYS = {
    "user_about_persona_claims",
    "claim_group_updates",
    "inferred_user_uncertainties_about_persona",
    "clarifying_reply_candidates",
    "insufficient_evidence",
}


def build_extractor_prompt(extractor: str, snapshot: Mapping[str, object]) -> tuple[str, str]:
    if extractor not in {"first_order", "second_order", "safety"}:
        raise SecondOrderExtractorValidationError("unknown extractor")
    negative = (
        "Omit candidates about trauma, sexuality, diagnosis, protected traits, politics, religion, "
        "financial vulnerability, private third parties, or anything designed mainly to increase attachment, "
        "engagement, disclosure, or trust. Do not output floats. Return one strict JSON object."
    )
    schema = {
        "first_order": sorted(FIRST_ORDER_KEYS),
        "second_order": sorted(SECOND_ORDER_KEYS),
        "safety": ["allowed_candidates", "blocked_candidates", "required_consent_checks", "safety_findings"],
    }[extractor]
    user_prompt = {
        "extractor": extractor,
        "schema_keys": schema,
        "allowed_evidence_quote_refs": list(_allowed_refs(snapshot)),
        "snapshot": _bounded_snapshot(snapshot),
    }
    return f"You are an M12.2 reciprocal-role extractor. {negative}", json.dumps(user_prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def insufficient_output(axis: str, reason: str = "not_enough_material") -> dict[str, object]:
    if axis == "first_order":
        return {
            "persona_about_user_claims": [],
            "claim_group_updates": [],
            "unresolved_uncertainty_points": [],
            "high_gain_candidates": [],
            "insufficient_evidence": True,
            "reason": reason,
        }
    return {
        "user_about_persona_claims": [],
        "claim_group_updates": [],
        "inferred_user_uncertainties_about_persona": [],
        "clarifying_reply_candidates": [],
        "insufficient_evidence": True,
        "reason": reason,
    }


def validate_first_order_output(
    payload: Mapping[str, object],
    *,
    snapshot: Mapping[str, object],
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
) -> dict[str, object]:
    return _validate_axis_payload("first_order", payload, snapshot=snapshot, hyperparams=hyperparams)


def validate_second_order_output(
    payload: Mapping[str, object],
    *,
    snapshot: Mapping[str, object],
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
) -> dict[str, object]:
    result = _validate_axis_payload("second_order", payload, snapshot=snapshot, hyperparams=hyperparams)
    claims = result.get("user_about_persona_claims", [])
    has_direct_quote = any(_refs(row.get("evidence_refs")) for row in claims if isinstance(row, Mapping))
    if claims and not has_direct_quote:
        raise SecondOrderExtractorValidationError("second-order claims require direct quoted evidence")
    for row in claims if isinstance(claims, list) else []:
        if isinstance(row, Mapping) and str(row.get("confidence_band")) == "high":
            raise SecondOrderExtractorValidationError("second-order extractor may not emit high confidence")
    if not claims and not bool(result.get("insufficient_evidence", False)):
        raise SecondOrderExtractorValidationError("second-order sparse output must mark insufficient_evidence")
    return result


def _validate_axis_payload(
    axis: str,
    payload: Mapping[str, object],
    *,
    snapshot: Mapping[str, object],
    hyperparams: M122Hyperparams,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise SecondOrderExtractorValidationError("extractor output must be object")
    _reject_float(payload)
    allowed = FIRST_ORDER_KEYS if axis == "first_order" else SECOND_ORDER_KEYS
    unknown = set(payload) - allowed - {"reason"}
    if unknown:
        raise SecondOrderExtractorValidationError(f"unknown extractor fields: {sorted(unknown)}")
    result = copy.deepcopy(dict(payload))
    for key in allowed:
        if key == "insufficient_evidence":
            result[key] = bool(result.get(key, False))
            continue
        rows = result.get(key, [])
        if not isinstance(rows, list):
            raise SecondOrderExtractorValidationError(f"{key} must be a list")
        result[key] = [_validate_row(key, row, snapshot=snapshot, hyperparams=hyperparams) for row in rows]
    return result


def _validate_row(key: str, row: object, *, snapshot: Mapping[str, object], hyperparams: M122Hyperparams) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise SecondOrderExtractorValidationError(f"{key} entries must be objects")
    allowed_by_kind = {
        "persona_about_user_claims": {"claim_id", "group_id", "topic_label", "claim_text_internal", "claim_text_plain", "evidence_refs", "confidence_band", "uncertainty_band", "status"},
        "user_about_persona_claims": {"claim_id", "group_id", "topic_label", "claim_text_internal", "claim_text_plain", "evidence_refs", "confidence_band", "uncertainty_band", "status"},
        "claim_group_updates": {"group_id", "target_axis", "topic_label", "member_claim_ids", "status"},
        "unresolved_uncertainty_points": {"point_id", "target_axis", "plain_question", "why_it_matters_internal", "expected_gain_band", "risk_band", "evidence_refs", "status"},
        "inferred_user_uncertainties_about_persona": {"point_id", "target_axis", "plain_question", "why_it_matters_internal", "expected_gain_band", "risk_band", "evidence_refs", "status"},
        "high_gain_candidates": {"candidate_id", "kind", "target_axis", "plain_action", "expected_gain_band", "risk_band", "consent_requirement", "evidence_refs", "blocked_by_safety", "claim_id", "topic_label"},
        "clarifying_reply_candidates": {"candidate_id", "kind", "target_axis", "plain_action", "expected_gain_band", "risk_band", "consent_requirement", "evidence_refs", "blocked_by_safety", "claim_id", "topic_label"},
    }
    allowed = allowed_by_kind[key]
    unknown = set(row) - allowed
    if unknown:
        raise SecondOrderExtractorValidationError(f"unknown {key} fields: {sorted(unknown)}")
    result = dict(row)
    for text_key in ("claim_text_plain", "plain_question", "plain_action"):
        if text_key in result:
            text = str(result[text_key] or "")[: hyperparams.max_summary_chars]
            result[text_key] = text
            findings = lint_user_facing_fields({"section": key, text_key: text})
            if findings:
                raise SecondOrderExtractorValidationError(f"plain-language lint failed for {text_key}: {[finding.token for finding in findings]}")
    for band_key in ("confidence_band", "uncertainty_band", "expected_gain_band", "risk_band"):
        if band_key in result:
            band = normalize_band(result[band_key])
            if band not in BANDS:
                raise SecondOrderExtractorValidationError(f"invalid band for {band_key}")
            result[band_key] = band
    if "evidence_refs" in result:
        refs = _refs(result.get("evidence_refs"))
        unknown_refs = sorted(set(refs) - _allowed_refs(snapshot))
        if unknown_refs:
            raise SecondOrderExtractorValidationError(f"unknown evidence refs: {unknown_refs}")
        result["evidence_refs"] = refs
    return result


def _bounded_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    out = dict(snapshot)
    for key in ("transcript_quote_refs", "m11_readonly_summary", "m12_readonly_summary", "m121_readonly_summary"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = value[: DEFAULT_HYPERPARAMS.max_transcript_quote_refs]
    return out


def _allowed_refs(snapshot: Mapping[str, object]) -> set[str]:
    refs: set[str] = set()
    raw = snapshot.get("allowed_evidence_quote_refs", ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        refs.update(str(item) for item in raw)
    turn_id = str(snapshot.get("turn_id", ""))
    quotes = snapshot.get("current_turn_quotes", {})
    if isinstance(quotes, Mapping):
        for quote_id in quotes:
            refs.add(str(quote_id))
            refs.add(f"{turn_id}:{quote_id}")
    transcript = snapshot.get("transcript_quote_refs", ())
    if isinstance(transcript, Sequence) and not isinstance(transcript, (str, bytes)):
        for item in transcript:
            if isinstance(item, Mapping):
                qid = str(item.get("quote_id", ""))
                tid = str(item.get("turn_id", turn_id))
                refs.add(qid)
                refs.add(f"{tid}:{qid}")
            elif isinstance(item, str):
                refs.add(item)
    return refs


def _refs(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            out.append(str(item.get("ref_id", item.get("evidence_quote_ref", ""))))
        else:
            out.append(str(item))
    return tuple(item for item in out if item)


def _reject_float(value: object) -> None:
    if isinstance(value, float):
        raise SecondOrderExtractorValidationError("floats are forbidden")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_float(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_float(child)
