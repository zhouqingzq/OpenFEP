"""Strict M12 identity extractor schema and validator."""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

ALLOWED_MODALITIES = {"factual", "roleplay", "joke", "hypothetical", "request", "command"}
ALLOWED_CUE_KINDS = {"style", "knowledge", "relationship", "history", "preference", "timing"}
ALLOWED_SUPPORTS = {"binds", "weakens", "contradicts"}
ALLOWED_BANDS = {"low", "med", "high"}

TOP_LEVEL_KEYS = {"identity_claims", "continuity_cues", "strangeness_band", "surprise_explanation"}
CLAIM_KEYS = {"id", "claimant_user_id", "asserted_alias", "modality", "evidence_quote_ids", "confidence_band"}
CUE_KEYS = {"id", "cue_kind", "supports", "content_summary", "evidence_quote_ids", "confidence_band"}


class ExtractorValidationError(ValueError):
    """Raised when extractor payload violates the M12 boundary."""


def noop_extraction() -> dict[str, object]:
    return {
        "identity_claims": [],
        "continuity_cues": [],
        "strangeness_band": "low",
        "surprise_explanation": "",
    }


def validate_extractor_output(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ExtractorValidationError("extractor output must be an object")
    _reject_float(payload)
    unknown = set(payload) - TOP_LEVEL_KEYS
    if unknown:
        raise ExtractorValidationError(f"unknown top-level fields: {sorted(unknown)}")
    result = noop_extraction()
    result.update(copy.deepcopy(dict(payload)))
    claims = _list_of_objects(result.get("identity_claims"), "identity_claims")
    cues = _list_of_objects(result.get("continuity_cues"), "continuity_cues")
    for claim in claims:
        _check_keys(claim, CLAIM_KEYS, "claim")
        _enum(claim.get("modality"), ALLOWED_MODALITIES, "claim.modality")
        _enum(claim.get("confidence_band"), ALLOWED_BANDS, "claim.confidence_band")
        _max_len(claim.get("asserted_alias"), 80, "claim.asserted_alias")
        _string_list(claim.get("evidence_quote_ids"), "claim.evidence_quote_ids")
    for cue in cues:
        _check_keys(cue, CUE_KEYS, "cue")
        _enum(cue.get("cue_kind"), ALLOWED_CUE_KINDS, "cue.cue_kind")
        _enum(cue.get("supports"), ALLOWED_SUPPORTS, "cue.supports")
        _enum(cue.get("confidence_band"), ALLOWED_BANDS, "cue.confidence_band")
        _max_len(cue.get("content_summary"), 120, "cue.content_summary")
        _string_list(cue.get("evidence_quote_ids"), "cue.evidence_quote_ids")
    _enum(result.get("strangeness_band"), ALLOWED_BANDS, "strangeness_band")
    _max_len(result.get("surprise_explanation"), 200, "surprise_explanation")
    return result


def _reject_float(value: object) -> None:
    if isinstance(value, float):
        raise ExtractorValidationError("floats are forbidden in extractor output")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_float(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_float(child)


def _list_of_objects(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExtractorValidationError(f"{field} must be a list")
    rows: list[Mapping[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ExtractorValidationError(f"{field} entries must be objects")
        rows.append(row)
    return rows


def _check_keys(payload: Mapping[str, object], allowed: set[str], field: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ExtractorValidationError(f"unknown {field} fields: {sorted(unknown)}")


def _enum(value: object, allowed: set[str], field: str) -> None:
    if str(value) not in allowed:
        raise ExtractorValidationError(f"{field} has invalid enum")


def _max_len(value: object, limit: int, field: str) -> None:
    if len(str(value or "")) > limit:
        raise ExtractorValidationError(f"{field} exceeds max length")


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExtractorValidationError(f"{field} must be a list")
    return tuple(str(item) for item in value)
