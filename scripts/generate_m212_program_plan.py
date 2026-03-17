from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "reports" / "m212_program_plan.json"

REQUIRED_EVIDENCE = (
    "schema",
    "determinism",
    "causality",
    "ablation",
    "stress",
    "regression",
    "artifact_freshness",
)

AUDIT_STATUSES = (
    "PASS",
    "PASS_WITH_RESIDUAL_RISK",
    "FAIL",
    "BLOCKED",
)


@dataclass(frozen=True)
class MilestonePlan:
    milestone_id: str
    title: str
    objective: str
    acceptance_focus: tuple[str, ...]
    required_evidence: tuple[str, ...] = REQUIRED_EVIDENCE

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["acceptance_focus"] = list(self.acceptance_focus)
        payload["required_evidence"] = list(self.required_evidence)
        return payload


def build_plan() -> dict[str, object]:
    milestones = [
        MilestonePlan(
            milestone_id="M2.12",
            title="Embodied I/O Bus and World Adapters",
            objective="Unify multi-source input and output channels behind causal, traced adapters.",
            acceptance_focus=(
                "adapter causality",
                "effect acknowledgment",
                "snapshot/schema stability",
            ),
        ),
        MilestonePlan(
            milestone_id="M2.13",
            title="Global Access Workspace",
            objective="Turn scarce attentional winners into a causal broadcast layer for action, report, and learning.",
            acceptance_focus=(
                "workspace capacity",
                "broadcast causality",
                "no report leakage",
            ),
        ),
        MilestonePlan(
            milestone_id="M2.14",
            title="Autonomous Homeostasis and Survival Scheduler",
            objective="Promote survival maintenance into an always-on scheduler across multiple debt timescales.",
            acceptance_focus=(
                "maintenance autonomy",
                "interrupt recovery",
                "anti-collapse scheduling",
            ),
        ),
        MilestonePlan(
            milestone_id="M2.15",
            title="Unified Self-Narrative and Identity Commitments",
            objective="Bind action policy and continuity preservation to explicit identity commitments.",
            acceptance_focus=(
                "identity causality",
                "coherent chapter transitions",
                "evidence-grounded self-report",
            ),
        ),
        MilestonePlan(
            milestone_id="M2.16",
            title="Persistent Others and Social Continuity",
            objective="Introduce stable models of other agents across repeated encounters.",
            acceptance_focus=(
                "counterpart separation",
                "social carryover",
                "rupture and repair",
            ),
        ),
        MilestonePlan(
            milestone_id="M2.17",
            title="Open-World Action and Self-Maintenance Governance",
            objective="Open constrained real-world action channels with explicit governance and rollback discipline.",
            acceptance_focus=(
                "capability declaration",
                "policy enforcement",
                "effect acknowledgment",
            ),
        ),
        MilestonePlan(
            milestone_id="M2.18",
            title="Lifelong Learning, Anti-Collapse, and Continuity Preservation",
            objective="Protect long-run continuity while allowing ongoing adaptation and consolidation.",
            acceptance_focus=(
                "continuity preservation",
                "anti-forgetting",
                "restart consistency",
            ),
        ),
        MilestonePlan(
            milestone_id="M3.0",
            title="Closed-Loop Life Trial",
            objective="Validate the integrated organism under mixed long-horizon conditions.",
            acceptance_focus=(
                "end-to-end survival",
                "cross-context continuity",
                "independent audit readiness",
            ),
        ),
    ]
    return {
        "program_id": "M2.12-M3.0",
        "audit_statuses": list(AUDIT_STATUSES),
        "required_evidence_categories": list(REQUIRED_EVIDENCE),
        "milestones": [item.to_dict() for item in milestones],
    }


def main() -> None:
    payload = build_plan()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
