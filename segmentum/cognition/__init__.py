"""Small cognition-loop adapters layered over the existing dialogue runtime."""

from .attention_gate import AttentionGate, AttentionGateConfig, AttentionGateResult
from .cognitive_loop import CognitiveLoop, CognitiveLoopResult

__all__ = [
    "AttentionGate",
    "AttentionGateConfig",
    "AttentionGateResult",
    "CognitiveLoop",
    "CognitiveLoopResult",
]
