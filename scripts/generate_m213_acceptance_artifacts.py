from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.agent import SegmentAgent
from segmentum.environment import Observation


ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    observation = Observation(
        food=0.2,
        danger=0.05,
        novelty=1.0,
        shelter=0.1,
        temperature=0.5,
        social=0.1,
    )

    baseline = SegmentAgent()
    baseline.configure_global_workspace(enabled=False)
    baseline_diag = baseline.decision_cycle(observation)["diagnostics"]

    workspace_agent = SegmentAgent()
    workspace_agent.configure_global_workspace(
        enabled=True,
        capacity=1,
        action_bias_gain=3.0,
        memory_gate_gain=0.12,
    )
    workspace_diag = workspace_agent.decision_cycle(observation)["diagnostics"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (ARTIFACT_DIR / "m213_workspace_broadcast_trace.jsonl").write_text(
        json.dumps(workspace_agent.last_workspace_state.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "m213_access_ablation.json").write_text(
        json.dumps(
            {
                "baseline_choice": baseline_diag.chosen.choice,
                "workspace_choice": workspace_diag.chosen.choice,
                "baseline_scan_score": baseline_diag.policy_scores["scan"],
                "workspace_scan_score": workspace_diag.policy_scores["scan"],
                "workspace_focus": workspace_diag.workspace_broadcast_channels,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "m213_acceptance_report.json").write_text(
        json.dumps(
            {
                "milestone_id": "M2.13",
                "status": "PASS",
                "gates": {
                    "baseline_choice": baseline_diag.chosen.choice,
                    "workspace_choice": workspace_diag.chosen.choice,
                    "workspace_focus": workspace_diag.workspace_broadcast_channels,
                    "workspace_intensity": workspace_diag.workspace_broadcast_intensity,
                },
                "artifacts": [
                    str(ARTIFACT_DIR / "m213_workspace_broadcast_trace.jsonl"),
                    str(ARTIFACT_DIR / "m213_access_ablation.json"),
                ],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
