"""M12.0 user identity continuity module."""

from .evidence_cards import IdentityEvidenceCard, cards_to_prompt_safe_memory_evidence
from .hyperparams import DEFAULT_HYPERPARAMS, M12Hyperparams
from .identity_claim_ledger import IdentityClaimLedger, IdentityClaimLedgerEntry
from .identity_conflict_detector import ConflictRecord, detect_identity_conflicts
from .identity_profile import (
    AliasObservation,
    ContinuityCue,
    IdentityProfile,
    profile_state_hash,
)
from .llm_identity_extractor import (
    ExtractorValidationError,
    noop_extraction,
    validate_extractor_output,
)
from .m12_runtime import (
    M12RuntimeConfig,
    M12RuntimeState,
    M12TurnResult,
    run_m12_turn,
)
from .reply_policy import ReplyPolicyDecision, select_reply_policy
from .strangeness_signal import IdentityStrangenessSignal, build_strangeness_signal

__all__ = [
    "AliasObservation",
    "ConflictRecord",
    "ContinuityCue",
    "DEFAULT_HYPERPARAMS",
    "ExtractorValidationError",
    "IdentityClaimLedger",
    "IdentityClaimLedgerEntry",
    "IdentityEvidenceCard",
    "IdentityProfile",
    "IdentityStrangenessSignal",
    "M12Hyperparams",
    "M12RuntimeConfig",
    "M12RuntimeState",
    "M12TurnResult",
    "ReplyPolicyDecision",
    "build_strangeness_signal",
    "cards_to_prompt_safe_memory_evidence",
    "detect_identity_conflicts",
    "noop_extraction",
    "profile_state_hash",
    "run_m12_turn",
    "select_reply_policy",
    "validate_extractor_output",
]
