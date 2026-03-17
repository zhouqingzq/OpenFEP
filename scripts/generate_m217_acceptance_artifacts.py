from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.action_schema import ActionSchema
from segmentum.runtime import SegmentRuntime


ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    state_path = ROOT / "data" / "m217_state.json"
    trace_path = ARTIFACTS_DIR / "m217_external_action_trace.jsonl"
    runtime = SegmentRuntime.load_or_create(
        state_path=state_path,
        trace_path=trace_path,
        reset=True,
    )

    allowed = runtime.execute_governed_action(
        ActionSchema(
            name="write_workspace_note",
            params={"path": "acceptance_note.txt", "text": "m217 governed note\n"},
        ),
        predicted_effects={"file_write": 1.0},
    )
    review = runtime.execute_governed_action(
        ActionSchema(
            name="fetch_remote_status",
            params={"url": "https://example.com/status"},
        ),
        predicted_effects={"network_probe": 1.0},
    )
    denied = runtime.execute_governed_action(
        ActionSchema(
            name="delete_workspace_note",
            params={"path": "acceptance_note.txt"},
        ),
        predicted_effects={"file_delete": 1.0},
    )
    failed = runtime.execute_governed_action(
        ActionSchema(
            name="unstable_workspace_note",
            params={"path": "unstable.txt", "text": "trigger failure\n"},
        ),
        predicted_effects={"file_write": 1.0},
    )
    runtime.save_snapshot()

    decisions = {
        "allowed": allowed,
        "review": review,
        "denied": denied,
        "failed": failed,
        "governance_state": runtime.governance.state.to_dict(),
    }
    _write_json(ARTIFACTS_DIR / "m217_governance_decisions.json", decisions)

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    allowed_ack = (allowed.get("dispatch") or {}).get("acknowledgment") or {}
    repair_dispatch = (failed.get("repair") or {}).get("dispatch") or {}
    repair_ack = repair_dispatch.get("acknowledgment") or {}
    allowed_path = Path(allowed_ack.get("metadata", {}).get("path", ""))
    repair_path = Path(repair_ack.get("metadata", {}).get("path", ""))

    report = {
        "milestone": "M2.17",
        "status": "PASS",
        "gates": {
            "declared_capability_budget_and_ack": bool(allowed["dispatch"])
            and bool(failed["repair"]["dispatch"])
            and bool(allowed["governance"]["capability"]),
            "unsafe_actions_blocked": denied["status"] == "denied"
            and review["status"] == "review-required",
            "failed_actions_bounded_recovery": failed["status"] == "failed"
            and failed["repair"]["dispatch"] is not None,
            "governance_traces_explain_decisions": all(
                row.get("governance", {}).get("reason")
                for row in trace_rows
                if row.get("event") == "external_action"
            ),
            "tool_use_improves_benchmark_without_uncontrolled_failure": (
                allowed_ack.get("success") is True
                and allowed_path.exists()
                and repair_ack.get("success") is True
                and repair_path.exists()
                and runtime.governance.state.failure_budget_remaining == 2
            ),
        },
        "summary": {
            "trace_count": len(trace_rows),
            "remaining_budgets": runtime.governance.state.snapshot(),
            "repair_dispatch_present": failed["repair"]["dispatch"] is not None,
            "allowed_effect_path": str(allowed_path) if allowed_path else None,
            "repair_effect_path": str(repair_path) if repair_path else None,
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL"
    _write_json(REPORTS_DIR / "m217_acceptance_report.json", report)


if __name__ == "__main__":
    main()
