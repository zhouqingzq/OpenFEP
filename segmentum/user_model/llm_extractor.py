"""Strict M11 extractor boundary.

The live LLM call is deliberately outside the deterministic layer. This module
defines the only accepted JSON shape and a validator that rejects unknown fields,
floats, and invented snapshot references.
"""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

ALLOWED_DOMAINS = {
    "self_reported_preferences",
    "self_reported_history",
    "task_requirements",
    "emotional_state",
    "technical_claims",
    "social_relationship_claims",
}
ALLOWED_MODALITIES = {"factual", "roleplay", "joke", "hypothetical", "request", "command"}
ALLOWED_BANDS = {"low", "med", "high"}
ALLOWED_PREDICTION_TYPES = {
    "intent_prediction",
    "preference_prediction",
    "reaction_prediction",
    "claim_reliability_prediction",
    "relationship_state_prediction",
    "needed_memory_prediction",
}
ALLOWED_EVIDENCE_BASIS = {
    "no_evidence",
    "current_user_question",
    "current_user_request",
    "current_user_statement",
    "recent_topic_continuity",
    "direct_user_statement",
    "repeated_evidence",
    "prior_confirmation",
    "system_invariant",
    "product_invariant",
}
ALLOWED_STATUSES = {"confirmed", "violated", "uncertain"}
ALLOWED_RELATIONS = {"activates", "contradicts", "irrelevant"}
ALLOWED_SEVERITY = {"minor", "major"}

EXTRACTOR_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims_made": {"type": "array"},
        "prediction_judgments": {"type": "array"},
        "prediction_proposals": {"type": "array"},
        "hypothesis_activations": {"type": "array"},
        "contradiction_detections": {"type": "array"},
        "calibration_need_band": {"type": "string", "enum": sorted(ALLOWED_BANDS)},
        "memory_value_band": {"type": "string", "enum": sorted(ALLOWED_BANDS)},
        "surprise_explanation": {"type": "string", "maxLength": 200},
    },
}

EXTRACTOR_PROMPT_TEMPLATE = """Extract bounded enum-only user-model events.

Use only the bounded snapshot:
- current turn quotes and quote ids
- last K turn summaries
- active hypotheses by id and summary
- open predictions by id and summary

Do not invent prediction ids or hypothesis ids from the snapshot. New proposal
ids are allowed, but each proposal source id must already appear in the bounded
snapshot. Floats remain forbidden except:
- prediction_proposals[].raw_confidence
- prediction_judgments[].settlement_confidence
Return only strict JSON matching the schema.
"""

TOP_LEVEL_KEYS = set(EXTRACTOR_OUTPUT_SCHEMA["properties"])  # type: ignore[index]
CLAIM_KEYS = {"id", "domain", "modality", "content_summary", "evidence_quote_ids", "confidence_band", "tags", "permitted_use", "hypothesis_id"}
JUDGMENT_KEYS = {
    "prediction_id",
    "status",
    "outcome",
    "evidence_quote_ids",
    "evidence_refs",
    "settlement_confidence",
    "evidence_span",
    "reason_codes",
}
PROPOSAL_KEYS = {
    "id",
    "prediction_type",
    "proposed_prediction_type",
    "predicted_value_summary",
    "confidence_band",
    "raw_confidence",
    "evidence_basis",
    "evidence_quote_ids",
    "source_hypothesis_ids",
    "source_judgment_ids",
    "expires_after_turns",
}
ACTIVATION_KEYS = {"hypothesis_id", "relation"}
CONTRADICTION_KEYS = {"claim_id", "conflicts_with_memory_id", "severity_band"}


class ExtractorValidationError(ValueError):
    """Raised when extractor output violates the M11 boundary."""


def validate_extractor_output(
    payload: Mapping[str, object],
    *,
    snapshot_prediction_ids: set[str] | None = None,
    snapshot_hypothesis_ids: set[str] | None = None,
    snapshot_judgment_ids: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ExtractorValidationError("extractor output must be an object")
    _reject_float(payload, path=())
    unknown = set(payload) - TOP_LEVEL_KEYS
    if unknown:
        raise ExtractorValidationError(f"unknown top-level fields: {sorted(unknown)}")

    prediction_ids = set(snapshot_prediction_ids or set())
    hypothesis_ids = set(snapshot_hypothesis_ids or set())
    judgment_ids = set(snapshot_judgment_ids or set())
    result = _default_output()
    result.update(copy.deepcopy(dict(payload)))

    for claim in _list(result["claims_made"], "claims_made"):
        _check_keys(claim, CLAIM_KEYS, "claim")
        _enum(claim.get("domain"), ALLOWED_DOMAINS, "claim.domain")
        _enum(claim.get("modality"), ALLOWED_MODALITIES, "claim.modality")
        _enum(claim.get("confidence_band"), ALLOWED_BANDS, "claim.confidence_band")
        _max_len(claim.get("content_summary"), 120, "claim.content_summary")
        _string_list(claim.get("evidence_quote_ids"), "claim.evidence_quote_ids")
        hyp_id = str(claim.get("hypothesis_id", ""))
        if hyp_id and hyp_id not in hypothesis_ids:
            raise ExtractorValidationError("unknown hypothesis_id")

    for judgment in _list(result["prediction_judgments"], "prediction_judgments"):
        _check_keys(judgment, JUDGMENT_KEYS, "judgment")
        pred_id = str(judgment.get("prediction_id", ""))
        if pred_id not in prediction_ids:
            raise ExtractorValidationError("unknown prediction_id")
        status_value = judgment.get("outcome", judgment.get("status"))
        _enum(status_value, ALLOWED_STATUSES, "judgment.status")
        _string_list(judgment.get("evidence_quote_ids"), "judgment.evidence_quote_ids")
        _string_list(judgment.get("evidence_refs", []), "judgment.evidence_refs")
        _string_list(judgment.get("reason_codes", []), "judgment.reason_codes")
        _max_len(judgment.get("evidence_span"), 200, "judgment.evidence_span")
        _confidence_float(judgment.get("settlement_confidence"), "judgment.settlement_confidence", low=0.0)

    for proposal in _list(result["prediction_proposals"], "prediction_proposals"):
        _check_keys(proposal, PROPOSAL_KEYS, "proposal")
        ptype = proposal.get("prediction_type", proposal.get("proposed_prediction_type"))
        _enum(ptype, ALLOWED_PREDICTION_TYPES, "proposal.prediction_type")
        _enum(proposal.get("confidence_band"), ALLOWED_BANDS, "proposal.confidence_band")
        _max_len(proposal.get("predicted_value_summary"), 200, "proposal.predicted_value_summary")
        _positive_int(proposal.get("expires_after_turns"), "proposal.expires_after_turns")
        _confidence_float(proposal.get("raw_confidence"), "proposal.raw_confidence")
        for basis in _string_list(proposal.get("evidence_basis", []), "proposal.evidence_basis"):
            _enum(basis, ALLOWED_EVIDENCE_BASIS, "proposal.evidence_basis")
        _string_list(proposal.get("evidence_quote_ids", []), "proposal.evidence_quote_ids")
        for hyp_id in _string_list(proposal.get("source_hypothesis_ids"), "proposal.source_hypothesis_ids"):
            if hyp_id not in hypothesis_ids:
                raise ExtractorValidationError("unknown source_hypothesis_id")
        for judgment_id in _string_list(proposal.get("source_judgment_ids"), "proposal.source_judgment_ids"):
            if judgment_id not in judgment_ids:
                raise ExtractorValidationError("unknown source_judgment_id")

    for activation in _list(result["hypothesis_activations"], "hypothesis_activations"):
        _check_keys(activation, ACTIVATION_KEYS, "activation")
        hyp_id = str(activation.get("hypothesis_id", ""))
        if hyp_id not in hypothesis_ids:
            raise ExtractorValidationError("unknown activation hypothesis_id")
        _enum(activation.get("relation"), ALLOWED_RELATIONS, "activation.relation")

    for contradiction in _list(result["contradiction_detections"], "contradiction_detections"):
        _check_keys(contradiction, CONTRADICTION_KEYS, "contradiction")
        memory_id = str(contradiction.get("conflicts_with_memory_id", ""))
        if memory_id and memory_id not in hypothesis_ids:
            raise ExtractorValidationError("unknown contradiction memory id")
        _enum(contradiction.get("severity_band"), ALLOWED_SEVERITY, "contradiction.severity_band")

    _enum(result.get("calibration_need_band"), ALLOWED_BANDS, "calibration_need_band")
    _enum(result.get("memory_value_band"), ALLOWED_BANDS, "memory_value_band")
    _max_len(result.get("surprise_explanation"), 200, "surprise_explanation")
    return result


def noop_extraction() -> dict[str, object]:
    return _default_output()


def _default_output() -> dict[str, object]:
    return {
        "claims_made": [],
        "prediction_judgments": [],
        "prediction_proposals": [],
        "hypothesis_activations": [],
        "contradiction_detections": [],
        "calibration_need_band": "low",
        "memory_value_band": "low",
        "surprise_explanation": "",
    }


def _reject_float(value: object, *, path: tuple[object, ...]) -> None:
    if isinstance(value, float):
        if not _float_allowed(path):
            raise ExtractorValidationError("floats are forbidden in extractor output")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_float(child, path=(*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_float(child, path=(*path, index))


def _list(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExtractorValidationError(f"{field} must be a list")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ExtractorValidationError(f"{field} entries must be objects")
        rows.append(item)
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
        raise ExtractorValidationError(f"{field} is too long")


def _positive_int(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExtractorValidationError(f"{field} must be a positive integer")


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExtractorValidationError(f"{field} must be a list")
    return tuple(str(item) for item in value)


def _confidence_float(value: object, field: str, *, low: float = 0.50, high: float = 0.90) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ExtractorValidationError(f"{field} must be a float")
    if parsed < low or parsed > high:
        raise ExtractorValidationError(f"{field} must be within bounds")
    return parsed


def _float_allowed(path: tuple[object, ...]) -> bool:
    if len(path) >= 3 and path[0] == "prediction_proposals" and isinstance(path[1], int) and path[2] == "raw_confidence":
        return True
    if len(path) >= 3 and path[0] == "prediction_judgments" and isinstance(path[1], int) and path[2] == "settlement_confidence":
        return True
    return False
