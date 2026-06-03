"""Small cognition-loop adapters layered over the existing dialogue runtime."""

from .attention_gate import AttentionGate, AttentionGateConfig, AttentionGateResult

__all__ = [
    "AttentionGate",
    "AttentionGateConfig",
    "AttentionGateResult",
]
