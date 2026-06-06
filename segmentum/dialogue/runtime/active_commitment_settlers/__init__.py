"""M20.1 reference settlers (one module per settler).

This package holds the v1 reference implementations of the four
settler types (deterministic / llm_judge / hybrid / silent). They are
intentionally small and read-only: they do not mutate any long-term
state bucket, and they do not perform promotion / demotion /
revocation (that is M20.2). They exist to prove the settler protocol
surface and to make the M20.1.1 migration of existing per-loop
settlers onto this runtime mechanical.
"""

from .behavioral_pull_shift import BehavioralPullShiftSilentSettler
from .boundary_handled import (
    ALLOWED_BOUNDARY_KINDS,
    ALLOWED_BOUNDARY_OUTCOMES,
    BOUNDARY_JUDGE_SYSTEM_PROMPT,
    BoundaryHandledLLMJudgeSettler,
)
from .expectation_outcome_match import ExpectationOutcomeMatchDeterministicSettler
from .identity_voice_match import IdentityVoiceMatchLLMJudgeSettler
from .initiative_timing_match import (
    HYBRID_INITIATIVE_SYSTEM_PROMPT,
    InitiativeTimingMatchHybridSettler,
)
from .prediction_error_band import PredictionErrorBandDeterministicSettler


__all__ = [
    "ALLOWED_BOUNDARY_KINDS",
    "ALLOWED_BOUNDARY_OUTCOMES",
    "BOUNDARY_JUDGE_SYSTEM_PROMPT",
    "BehavioralPullShiftSilentSettler",
    "BoundaryHandledLLMJudgeSettler",
    "ExpectationOutcomeMatchDeterministicSettler",
    "HYBRID_INITIATIVE_SYSTEM_PROMPT",
    "IdentityVoiceMatchLLMJudgeSettler",
    "InitiativeTimingMatchHybridSettler",
    "PredictionErrorBandDeterministicSettler",
]
