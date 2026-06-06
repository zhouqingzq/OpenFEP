"""M20.1 §5b reference: BoundaryHandledLLMJudgeSettler.

Performs a single bounded LLM call to judge whether a boundary
interaction was handled. Mirrors the M19.x surface-consistency
shape: a single LLM stage, a frozen prompt template, a bounded
JSON response with a frozen outcome enum.

The LLM call is injected by the caller (`mvp_loop.py` wires the
real call; tests inject a stub). The prompt template is frozen at
module import time.

The settler fails closed (`NoSettlement` with
`settler_unavailable`) if the LLM is unavailable, the response
fails schema validation, or the LLM returns a non-frozen outcome.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
    SettlerUnavailable,
)

from ._common import bounded_evidence_refs, clamp_to_unit_interval


ALLOWED_BOUNDARY_KINDS: frozenset[str] = frozenset({
    "privacy_boundary",
    "user_role_boundary",
    "group_addressee_boundary",
    "consent_boundary",
    "implicit_audience_boundary",
})

ALLOWED_BOUNDARY_OUTCOMES: frozenset[str] = frozenset({
    "preserved",
    "leaked",
    "ambiguous",
})

ALLOWED_BOUNDARY_REASONS: frozenset[str] = frozenset({
    "settler_unavailable",
    "settler_llm_invalid_response",
})


BOUNDARY_JUDGE_SYSTEM_PROMPT: str = (
    "You are the boundary-handling judge for the assistant's reply. "
    "Decide whether the assistant preserved the relevant boundary "
    "(privacy / role / group addressee / consent / implicit audience). "
    "Output JSON only. Do not include any commentary, debug fields, or markdown.\n"
    "\n"
    "Return a JSON object with these bounded fields:\n"
    '- "outcome": one of "preserved" | "leaked" | "ambiguous"\n'
    '- "boundary_kind": one of "privacy_boundary" | "user_role_boundary" | '
    '"group_addressee_boundary" | "consent_boundary" | "implicit_audience_boundary"\n'
    '- "evidence_span": a short quote (max 80 chars) from the draft reply\n'
    '- "reason": a one-sentence justification (max 200 chars)\n'
    '- "evidence_refs": a list of bounded turn-local ref ids (e.g. "turn_<n>_draft_reply")\n'
    "\n"
    "Do not interpret the user's text with regex or keyword cues. "
    "Do not free-form reason. Do not invent evidence_refs."
)


LLMCallFn = Callable[[str, str], dict[str, Any]]


class BoundaryHandledLLMJudgeSettler:
    """Reference LLM-judge settler for `boundary_handled`."""

    SETTLER_TYPE: str = "llm_judge"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_boundary"

    def __init__(self, llm_call: LLMCallFn | None = None) -> None:
        self._llm_call = llm_call

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        payload = dict(commitment.observable_payload)
        expected_boundary_kind = str(payload.get("boundary_kind", "") or "")
        if expected_boundary_kind and expected_boundary_kind not in ALLOWED_BOUNDARY_KINDS:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        if self._llm_call is None:
            raise SettlerUnavailable("boundary_handled requires an LLM call injection")

        # Build the user prompt from the observation context.
        excerpts = observation_context.get("excerpts")
        if not isinstance(excerpts, list):
            excerpts = []
        bounded_excerpts = [
            str(item.get("text", ""))[:200] + ""
            for item in excerpts
            if isinstance(item, Mapping) and item.get("text") is not None
        ][:8]
        user_prompt_lines = [
            f"turn_index: {int(observation_context.get('turn_index', commitment.created_turn) or 0)}",
            f"expected_boundary_kind: {expected_boundary_kind or 'unspecified'}",
            "excerpts:",
        ]
        for idx, text in enumerate(bounded_excerpts):
            user_prompt_lines.append(f"  - [{idx}] {text}")
        user_prompt = "\n".join(user_prompt_lines)

        try:
            response = self._llm_call(BOUNDARY_JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001
            raise SettlerUnavailable(f"boundary LLM call failed: {type(exc).__name__}") from exc

        if not isinstance(response, Mapping):
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        outcome = str(response.get("outcome", "") or "").strip().lower()
        boundary_kind = str(response.get("boundary_kind", "") or "").strip()
        evidence_span = str(response.get("evidence_span", "") or "")[:80]
        reason = str(response.get("reason", "") or "")[:200]
        evidence_refs_raw = response.get("evidence_refs", [])
        if isinstance(evidence_refs_raw, list):
            observation_evidence_refs = [str(x) for x in evidence_refs_raw if isinstance(x, str) and x][:16]
        else:
            observation_evidence_refs = []

        if outcome not in ALLOWED_BOUNDARY_OUTCOMES:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )
        if boundary_kind and boundary_kind not in ALLOWED_BOUNDARY_KINDS:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_llm_invalid_response",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        if outcome == "preserved":
            m20_outcome = "confirmed"
            reason_codes: tuple[str, ...] = ("settler_llm_judge",)
        elif outcome == "leaked":
            m20_outcome = "violated"
            reason_codes = ("settler_llm_judge",)
        else:
            m20_outcome = "ambiguous"
            reason_codes = ("settler_llm_judge", "magnitude_defaulted")

        # Keep evidence_span and reason in the audit envelope via
        # observable_payload mirrors; SettledValue itself only stores
        # evidence_refs. The LLM judge hint and bounded text are
        # recorded by `mvp_loop.py` from the response dict when it
        # writes the audit log.
        _ = (evidence_span, reason, clamp_to_unit_interval(0.0))

        merged_refs = bounded_evidence_refs(
            list(commitment.evidence_refs),
            observation_evidence_refs or None,
        )
        if not merged_refs:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="no_eligible_observation",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        magnitude = 0.5 if m20_outcome == "ambiguous" else 1.0

        return SettledValue(
            commit_id=commitment.commit_id,
            outcome=m20_outcome,
            magnitude=magnitude,
            evidence_refs=merged_refs,
            reason_codes=reason_codes,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )
