"""Project Segmentum: predictive-coding survival primitives and daemon loop."""

from .daemon import HeartbeatDaemon, run_daemon
from .runtime import SegmentRuntime
from .state import AgentState, Strategy

__all__ = ["AgentState", "Strategy", "HeartbeatDaemon", "SegmentRuntime", "run_daemon"]
