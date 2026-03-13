"""Project Segmentum: predictive-coding survival primitives and daemon loop."""

from .daemon import HeartbeatDaemon, run_daemon
from .runtime import SegmentRuntime
from .memory import AutobiographicalMemory
from .preferences import Goal, GoalStack, ValueHierarchy
from .sleep_consolidator import SleepConsolidation
from .counterfactual import CounterfactualLearning, ReviewFamily
from .self_model import (
    BodySchema,
    CapabilityModel,
    ErrorClassifier,
    ResourceState,
    SelfModel,
    ThreatModel,
    build_default_self_model,
)
from .state import AgentState, Strategy

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
    "ReviewFamily",
    "run_daemon",
    "BodySchema",
    "CapabilityModel",
    "ErrorClassifier",
    "ResourceState",
    "SelfModel",
    "ThreatModel",
    "build_default_self_model",
]
