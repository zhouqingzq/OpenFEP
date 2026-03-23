"""Project Segmentum: predictive-coding survival primitives and daemon loop."""

from .daemon import HeartbeatDaemon, run_daemon
from .io_bus import (
    ActionBus,
    ActionDispatchRecord,
    ActionEffectAck,
    BusSignal,
    PerceptionBus,
    PerceptionPacket,
)
from .runtime import SegmentRuntime
from .homeostasis import HomeostasisScheduler, MaintenanceAgenda
from .workspace import GlobalWorkspace, GlobalWorkspaceState, WorkspaceContent
from .social_model import OtherModel, SocialMemory
from .subject_state import SubjectState, ActiveTension, DominantNeed, SubjectBinding, SubjectPriority
from .prediction_ledger import (
    DiscrepancySource,
    LedgerDiscrepancy,
    LedgerPriority,
    PredictionHypothesis,
    PredictionLedger,
    PredictionLedgerUpdate,
    VerificationStatus,
)
from .verification import (
    FalsificationRecord,
    PredictionUpdateResult,
    VerificationEvidence,
    VerificationEvidenceSource,
    VerificationLoop,
    VerificationLoopUpdate,
    VerificationOutcome,
    VerificationPlan,
    VerificationTarget,
    VerificationTargetStatus,
)
from .slow_learning import (
    ConsolidationUpdate,
    DriftBudget as SlowLearningDriftBudget,
    IdentityStabilityState,
    LearningPressure,
    PlasticityWindow,
    ProtectedAnchor,
    SlowLearningState,
    SlowTraitState,
    SlowUpdateAudit,
    SlowVariableLearner,
    ValueStabilityState,
)
from .governance import CapabilityDescriptor, GovernanceController, GovernanceState
from .memory import AutobiographicalMemory
from .preferences import Goal, GoalStack, ValueHierarchy
from .sleep_consolidator import SleepConsolidation
from .counterfactual import CounterfactualLearning, ReviewFamily
from .narrative_compiler import NarrativeCompiler
from .narrative_ingestion import NarrativeIngestionService
from .narrative_types import (
    AppraisalVector,
    CompiledNarrativeEvent,
    EmbodiedNarrativeEpisode,
    NarrativeEpisode,
)
from .self_model import (
    BodySchema,
    CapabilityModel,
    ContinuityAudit,
    DriftBudget,
    ErrorClassifier,
    IdentityCommitment,
    NarrativePriors,
    PersonalityProfile,
    PersonalitySignal,
    ResourceState,
    SelfModel,
    ThreatModel,
    build_default_self_model,
)
from .state import AgentState, Strategy
from .precision_manipulation import PrecisionManipulator, ManipulationType
from .defense_strategy import DefenseStrategy, DefenseStrategySelector, IdentityPE
from .metacognitive import MetaCognitiveLayer
from .therapeutic import TherapeuticAgent, SimulatedPersonalityState
from .via_projection import VIAProjection, VIAProfile
from .analysis_types import (
    ConfidenceRated,
    PersonalityAnalysisResult,
)
from .personality_analyzer import PersonalityAnalyzer

__all__ = [
    "AgentState",
    "Strategy",
    "HeartbeatDaemon",
    "SegmentRuntime",
    "BusSignal",
    "PerceptionPacket",
    "PerceptionBus",
    "ActionEffectAck",
    "ActionDispatchRecord",
    "ActionBus",
    "MaintenanceAgenda",
    "HomeostasisScheduler",
    "WorkspaceContent",
    "GlobalWorkspaceState",
    "GlobalWorkspace",
    "OtherModel",
    "SocialMemory",
    "SubjectState",
    "ActiveTension",
    "DominantNeed",
    "SubjectBinding",
    "SubjectPriority",
    "PredictionLedger",
    "PredictionHypothesis",
    "LedgerDiscrepancy",
    "PredictionLedgerUpdate",
    "VerificationStatus",
    "DiscrepancySource",
    "LedgerPriority",
    "VerificationLoop",
    "VerificationLoopUpdate",
    "VerificationPlan",
    "VerificationTarget",
    "VerificationTargetStatus",
    "VerificationEvidence",
    "VerificationEvidenceSource",
    "VerificationOutcome",
    "FalsificationRecord",
    "PredictionUpdateResult",
    "SlowVariableLearner",
    "SlowLearningState",
    "SlowTraitState",
    "ValueStabilityState",
    "IdentityStabilityState",
    "SlowLearningDriftBudget",
    "LearningPressure",
    "ConsolidationUpdate",
    "SlowUpdateAudit",
    "ProtectedAnchor",
    "PlasticityWindow",
    "CapabilityDescriptor",
    "GovernanceController",
    "GovernanceState",
    "AutobiographicalMemory",
    "Goal",
    "GoalStack",
    "ValueHierarchy",
    "SleepConsolidation",
    "CounterfactualLearning",
    "NarrativeCompiler",
    "NarrativeIngestionService",
    "NarrativeEpisode",
    "CompiledNarrativeEvent",
    "AppraisalVector",
    "EmbodiedNarrativeEpisode",
    "ReviewFamily",
    "run_daemon",
    "BodySchema",
    "CapabilityModel",
    "ContinuityAudit",
    "DriftBudget",
    "ErrorClassifier",
    "IdentityCommitment",
    "NarrativePriors",
    "PersonalityProfile",
    "PersonalitySignal",
    "ResourceState",
    "SelfModel",
    "ThreatModel",
    "build_default_self_model",
    # M2.7
    "PrecisionManipulator",
    "ManipulationType",
    "DefenseStrategy",
    "DefenseStrategySelector",
    "IdentityPE",
    "MetaCognitiveLayer",
    "TherapeuticAgent",
    "SimulatedPersonalityState",
    "VIAProjection",
    "VIAProfile",
    # Personality Analysis API
    "ConfidenceRated",
    "PersonalityAnalysisResult",
    "PersonalityAnalyzer",
]
