"""M12.2 reciprocal role cognition package."""

from .evidence_cards import (
    ReciprocalEvidenceCard,
    evidence_cards_from_candidates,
    evidence_cards_from_model,
    prompt_safe_cards,
    reconcile_hints,
)
from .hyperparams import DEFAULT_HYPERPARAMS, M122Hyperparams
from .information_gain import no_action_candidate, rank_information_gain_candidates, rank_or_no_action
from .m12_2_runtime import M122RuntimeConfig, M122RuntimeState, M122TurnResult, run_m12_2_tick
from .plain_language_linter import PlainLanguageFinding, lint_text, lint_user_facing_fields, passes_plain_language
from .reciprocal_model import (
    EvidenceRef,
    InformationGainCandidate,
    ReciprocalClaim,
    ReciprocalClaimGroup,
    ReciprocalRoleModel,
    UncertaintyPoint,
    apply_model_patch,
    mark_group_contradicted,
    promote_claim_with_evidence,
)
from .safety_linter import SafetyFinding, apply_safety_linter, lint_candidate
from .second_order_extractor import (
    SecondOrderExtractorValidationError,
    build_extractor_prompt,
    insufficient_output,
    validate_first_order_output,
    validate_second_order_output,
)
from .trigger_policy import TriggerDecision, TriggerPolicyInput, decide_trigger
from .turn_assessment import ReciprocalTurnAssessment, ReplyPolicyHint, assess_turn_light

__all__ = [
    "DEFAULT_HYPERPARAMS",
    "EvidenceRef",
    "InformationGainCandidate",
    "M122Hyperparams",
    "M122RuntimeConfig",
    "M122RuntimeState",
    "M122TurnResult",
    "PlainLanguageFinding",
    "ReciprocalClaim",
    "ReciprocalClaimGroup",
    "ReciprocalEvidenceCard",
    "ReciprocalRoleModel",
    "ReciprocalTurnAssessment",
    "ReplyPolicyHint",
    "SafetyFinding",
    "SecondOrderExtractorValidationError",
    "TriggerDecision",
    "TriggerPolicyInput",
    "UncertaintyPoint",
    "apply_model_patch",
    "apply_safety_linter",
    "assess_turn_light",
    "build_extractor_prompt",
    "decide_trigger",
    "evidence_cards_from_candidates",
    "evidence_cards_from_model",
    "insufficient_output",
    "lint_candidate",
    "lint_text",
    "lint_user_facing_fields",
    "mark_group_contradicted",
    "no_action_candidate",
    "passes_plain_language",
    "promote_claim_with_evidence",
    "prompt_safe_cards",
    "rank_information_gain_candidates",
    "rank_or_no_action",
    "reconcile_hints",
    "run_m12_2_tick",
    "validate_first_order_output",
    "validate_second_order_output",
]
