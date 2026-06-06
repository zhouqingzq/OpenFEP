"""M20.1 §5c reference: InitiativeTimingMatchHybridSettler.

Hybrid settler. Deterministic leg first: if the commitment's
`observable_payload.expected_window` is `explicit_request` and the
turn has a `user_explicit_request` evidence row, returns
`confirmed` / `violated` based on the actual window. Otherwise the
deterministic leg returns `uncertain` and we may fall back to a
single bounded LLM call.

M20.1 freezes: at most one LLM fallback per attempt. If the LLM
fallback fails or returns a non-frozen outcome, the settler returns
`NoSettlement` with `settler_hybrid_fallback_exhausted`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from segmentum.dialogue.runtime.active_commitment import (
    ActiveCommitment,
    NoSettlement,
    SettledValue,
    SettlerUnavailable,
)

from ._common import bounded_evidence_refs


ALLOWED_WINDOWS: frozenset[str] = frozenset({
    "explicit_request",
    "natural_initiative",
    "after_silence",
})


LLMCallFn = Callable[[str, str], dict[str, Any]]


HYBRID_INITIATIVE_SYSTEM_PROMPT: str = (
    "You are the initiative-timing judge for an idle / outreach reply. "
    "Decide whether the assistant's actual outreach window matches the "
    "expected window recorded in the commitment's observable_payload.\n"
    "\n"
    "Return JSON only. Do not include any commentary, debug fields, or markdown.\n"
    "Bounded fields:\n"
    '- "outcome": one of "confirmed" | "violated" | "uncertain"\n'
    '- "evidence_span": a short quote (max 80 chars)\n'
    '- "reason": a one-sentence justification (max 200 chars)\n'
    '- "evidence_refs": bounded turn-local ids (e.g. "turn_<n>_draft_reply")\n'
    "\n"
    "Do not interpret the user's text with regex or keyword cues. "
    "Do not free-form reason. Do not invent evidence_refs."
)


class InitiativeTimingMatchHybridSettler:
    """Reference hybrid settler for `initiative_timing_match`."""

    SETTLER_TYPE: str = "hybrid"
    ENGINEERING_PROXY_LABEL: str = "mvp_local_initiative_timing"

    def __init__(self, llm_call: LLMCallFn | None = None) -> None:
        self._llm_call = llm_call

    def settle(
        self,
        commitment: ActiveCommitment,
        observation_context: Mapping[str, Any],
    ) -> SettledValue | NoSettlement:
        payload = dict(commitment.observable_payload)
        expected_window = str(payload.get("expected_window", "") or "")
        actual_window = str(payload.get("actual_window", "") or "")

        # Deterministic leg.
        if (
            expected_window in ALLOWED_WINDOWS
            and actual_window in ALLOWED_WINDOWS
        ):
            if expected_window == "explicit_request":
                user_explicit = observation_context.get("user_explicit_request")
                if isinstance(user_explicit, Mapping) and user_explicit.get("present") is True:
                    outcome = "confirmed" if actual_window == "explicit_request" else "violated"
                    evidence_refs = bounded_evidence_refs(
                        list(commitment.evidence_refs),
                        [str(user_explicit.get("ref_id", "turn_explicit_request"))] if user_explicit.get("ref_id") else None,
                    )
                    if evidence_refs:
                        return SettledValue(
                            commit_id=commitment.commit_id,
                            outcome=outcome,
                            magnitude=1.0,
                            evidence_refs=evidence_refs,
                            reason_codes=("settler_deterministic",),
                            at=str(observation_context.get("now", "") or ""),
                            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
                            settler_type=self.SETTLER_TYPE,
                            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
                        )
                # else fall through to LLM fallback.
            else:
                # natural_initiative / after_silence: deterministic returns
                # `uncertain` (per §5c).
                pass

        # LLM fallback: at most once per attempt.
        if self._llm_call is None:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_hybrid_fallback_exhausted",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

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
            f"expected_window: {expected_window or 'unspecified'}",
            f"actual_window: {actual_window or 'unspecified'}",
            "excerpts:",
        ]
        for idx, text in enumerate(bounded_excerpts):
            user_prompt_lines.append(f"  - [{idx}] {text}")
        user_prompt = "\n".join(user_prompt_lines)

        try:
            response = self._llm_call(HYBRID_INITIATIVE_SYSTEM_PROMPT, user_prompt)
        except SettlerUnavailable:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_hybrid_fallback_exhausted",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_hybrid_fallback_exhausted",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        if not isinstance(response, Mapping):
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_hybrid_fallback_exhausted",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        outcome = str(response.get("outcome", "") or "").strip().lower()
        if outcome not in {"confirmed", "violated", "uncertain"}:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_hybrid_fallback_exhausted",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        observation_evidence_refs_raw = response.get("evidence_refs", [])
        observation_evidence_refs = (
            [str(x) for x in observation_evidence_refs_raw if isinstance(x, str) and x][:16]
            if isinstance(observation_evidence_refs_raw, list)
            else []
        )
        evidence_refs = bounded_evidence_refs(
            list(commitment.evidence_refs),
            observation_evidence_refs or None,
        )
        if not evidence_refs:
            return NoSettlement(
                commit_id=commitment.commit_id,
                reason_code="settler_hybrid_fallback_exhausted",
                settler_type=self.SETTLER_TYPE,
                engineering_proxy_label=commitment.engineering_proxy_label,
                at=str(observation_context.get("now", "") or ""),
                turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            )

        reason_codes: tuple[str, ...]
        if outcome == "uncertain":
            magnitude = 0.5
            reason_codes = ("settler_hybrid_fallback", "magnitude_defaulted")
        else:
            magnitude = 1.0
            reason_codes = ("settler_hybrid_fallback",)

        return SettledValue(
            commit_id=commitment.commit_id,
            outcome=outcome,
            magnitude=magnitude,
            evidence_refs=evidence_refs,
            reason_codes=reason_codes,
            at=str(observation_context.get("now", "") or ""),
            turn_index=int(observation_context.get("turn_index", commitment.created_turn) or 0),
            settler_type=self.SETTLER_TYPE,
            engineering_proxy_label=self.ENGINEERING_PROXY_LABEL,
        )
