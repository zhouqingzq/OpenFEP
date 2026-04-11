from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACCEPT = "ACCEPT"
PARTIAL_ACCEPT = "PARTIAL_ACCEPT"
NOT_ACCEPTED = "NOT_ACCEPTED"


@dataclass(frozen=True)
class LayerConclusion:
    formal_acceptance_conclusion: str
    three_layer_accept_ready: bool
    missing_layers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "formal_acceptance_conclusion": self.formal_acceptance_conclusion,
            "three_layer_accept_ready": self.three_layer_accept_ready,
            "missing_layers": list(self.missing_layers),
        }


def layer_passed(value: Any) -> bool:
    return value is True


def final_conclusion(
    *,
    structural_pass: Any,
    behavioral_pass: Any,
    phenomenological_pass: Any,
) -> LayerConclusion:
    """Return the memory-milestone three-layer acceptance verdict."""

    layer_values = {
        "structural_pass": structural_pass,
        "behavioral_pass": behavioral_pass,
        "phenomenological_pass": phenomenological_pass,
    }
    missing = tuple(
        name for name, value in layer_values.items() if not layer_passed(value)
    )
    if not missing:
        return LayerConclusion(ACCEPT, True, ())
    if layer_passed(structural_pass) and layer_passed(behavioral_pass):
        return LayerConclusion(PARTIAL_ACCEPT, False, missing)
    return LayerConclusion(NOT_ACCEPTED, False, missing)


MEMORY_MILESTONE_LAYER_STATUS: dict[str, dict[str, object]] = {
    "M4.5": {
        "structural_pass": True,
        "behavioral_pass": False,
        "phenomenological_pass": False,
        "status_note": "Layer (a) only: structural self-consistency.",
    },
    "M4.6": {
        "structural_pass": True,
        "behavioral_pass": False,
        "phenomenological_pass": False,
        "status_note": "Layer (a) only: structural self-consistency.",
    },
    "M4.7": {
        "structural_pass": True,
        "behavioral_pass": False,
        "phenomenological_pass": False,
        "status_note": "Layer (a) only: structural self-consistency.",
    },
    "M4.8": {
        "structural_pass": True,
        "behavioral_pass": True,
        "phenomenological_pass": False,
        "status_note": "Layer (b): default-path behavioral causation via ablation contrast.",
    },
    "M4.9": {
        "structural_pass": True,
        "behavioral_pass": False,
        "phenomenological_pass": False,
        "status_note": "Representational recall mechanism; layer (b) remains conservative unless separately accepted.",
    },
    "M4.10": {
        "structural_pass": True,
        "behavioral_pass": True,
        "phenomenological_pass": "pending(M4.11)",
        "status_note": "Layer (b) upstream of recall; layer (c) pending M4.11.",
    },
    "M4.11": {
        "structural_pass": "inherits(M4.10)",
        "behavioral_pass": "inherits(M4.8)",
        "phenomenological_pass": "target",
        "status_note": "Layer (c): natural-rollout phenomenological fit.",
    },
}


def memory_milestone_layer_status() -> dict[str, dict[str, object]]:
    return {
        milestone: dict(payload)
        for milestone, payload in MEMORY_MILESTONE_LAYER_STATUS.items()
    }
