"""Project Segmentum: predictive-coding survival primitives and daemon loop."""

from .daemon import HeartbeatDaemon, run_daemon
from .runtime import SegmentRuntime
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
    ErrorClassifier,
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

__all__ = [
    "AgentState",
    "Strategy",
    "HeartbeatDaemon",
    "SegmentRuntime",
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
    "ErrorClassifier",
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
]
