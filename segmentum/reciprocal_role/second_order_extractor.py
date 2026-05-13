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
        "snapshot": bound_extractor_snapshot(snapshot),
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
    for row in claims if isinstance(claims, list) else []:
        if not isinstance(row, Mapping):
            continue
        refs = _refs(row.get("evidence_refs"))
        if not refs:
            raise SecondOrderExtractorValidationError("each second-order claim requires direct quoted evidence")
        if str(row.get("confidence_band")) == "high":
            raise SecondOrderExtractorValidationError("second-order extractor may not emit high confidence")
        if str(row.get("confidence_band")) == "medium" and not refs:
            raise SecondOrderExtractorValidationError("medium second-order claims require quoted evidence")
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


def bound_extractor_snapshot(
    snapshot: Mapping[str, object],
    *,
    hyperparams: M122Hyperparams = DEFAULT_HYPERPARAMS,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in snapshot.items():
        if key == "current_turn_quotes" and isinstance(value, Mapping):
            out[key] = {
                str(quote_id): str(quote)[: hyperparams.max_summary_chars]
                for quote_id, quote in sorted(value.items())[: hyperparams.max_readonly_summary_items]
            }
        elif key == "transcript_quote_refs" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            out[key] = [_json_safe(item, hyperparams=hyperparams, depth=1) for item in value[: hyperparams.max_transcript_quote_refs]]
        elif key in {"m11_readonly_summary", "m12_readonly_summary", "m121_readonly_summary"}:
            out[key] = _bounded_readonly_summary(key, value, hyperparams=hyperparams)
        elif key == "model":
            out[key] = _bounded_model_summary(value, hyperparams=hyperparams)
        elif key == "allowed_evidence_quote_refs" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            out[key] = [str(item) for item in value[: hyperparams.max_transcript_quote_refs]]
        elif isinstance(value, str):
            out[key] = value[: hyperparams.max_summary_chars]
        else:
            out[key] = _json_safe(value, hyperparams=hyperparams, depth=1)
    return out


def _bounded_readonly_summary(key: str, value: object, *, hyperparams: M122Hyperparams) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    if key == "m11_readonly_summary":
        allowed = ("m11_evidence_cards", "prompt_safe_evidence_cards", "reply_policy_effects", "active_hypotheses")
    elif key == "m12_readonly_summary":
        allowed = ("entity_binding_context", "reply_policy", "prompt_safe_evidence_cards", "trigger_decision", "identity_state", "binding_confidence_band", "new_evidence_count")
    else:
        allowed = ("enabled", "trigger_decision", "prompt_safe_evidence_cards", "published_event_ids")
    out: dict[str, object] = {}
    for summary_key in allowed:
        if summary_key in value:
            out[summary_key] = _json_safe(value.get(summary_key), hyperparams=hyperparams, depth=2)
        if len(out) >= hyperparams.max_readonly_summary_items:
            break
    if key == "m121_readonly_summary":
        orchestrator = value.get("orchestrator_result")
        if isinstance(orchestrator, Mapping):
            report = orchestrator.get("report")
            if isinstance(report, Mapping):
                out["latest_report"] = {
                    "report_status": str(report.get("report_status", ""))[: hyperparams.max_summary_chars],
                    "report_id": str(report.get("report_id", ""))[: hyperparams.max_summary_chars],
                }
    return out


def _bounded_model_summary(value: object, *, hyperparams: M122Hyperparams) -> dict[str, object]:
    if not hasattr(value, "to_dict"):
        return {}
    raw = value.to_dict()
    if not isinstance(raw, Mapping):
        return {}
    return {
        "user_id": str(raw.get("user_id", ""))[: hyperparams.max_summary_chars],
        "persona_label": str(raw.get("persona_label", ""))[: hyperparams.max_summary_chars],
        "last_consolidated_turn_id": str(raw.get("last_consolidated_turn_id", ""))[: hyperparams.max_summary_chars],
        "contradiction_cooldown": int(raw.get("contradiction_cooldown", 0) or 0),
        "recent_probe_turn_ids": list(raw.get("recent_probe_turn_ids", []))[-hyperparams.second_order_high_recent_probe_turns:] if isinstance(raw.get("recent_probe_turn_ids", []), list) else [],
        "open_group_count": len(raw.get("reciprocal_claim_groups", [])) if isinstance(raw.get("reciprocal_claim_groups"), list) else 0,
        "claim_count": (
            len(raw.get("persona_about_user_claims", [])) if isinstance(raw.get("persona_about_user_claims"), list) else 0
        ) + (
            len(raw.get("user_about_persona_claims", [])) if isinstance(raw.get("user_about_persona_claims"), list) else 0
        ),
    }


def _json_safe(value: object, *, hyperparams: M122Hyperparams, depth: int) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return value[: hyperparams.max_action_chars]
    if isinstance(value, Mapping):
        if depth <= 0:
            return {"truncated": True, "keys": sorted(str(key) for key in value)[: hyperparams.max_readonly_summary_items]}
        out: dict[str, object] = {}
        for idx, key in enumerate(sorted(value, key=str)):
            if idx >= hyperparams.max_readonly_summary_items:
                break
            out[str(key)] = _json_safe(value.get(key), hyperparams=hyperparams, depth=depth - 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item, hyperparams=hyperparams, depth=depth - 1) for item in value[: hyperparams.max_readonly_summary_items]]
    return str(value)[: hyperparams.max_summary_chars]


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
