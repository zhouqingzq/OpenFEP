"""Extract a one-line summary from a P0-7 run output JSON.

Usage:
    python -m segmentum.tools.extract_p0_7_run_summary <path-to-output>

The output JSON is the one written to stdout by
`scripts/run_m18_7_1_real_llm_calibration.py`. This script
prints a compact one-line-per-key summary that the P0-7
report can ingest directly.

This is a read-only inspector (no I/O side effects).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def extract(out: dict) -> dict:
    addr = out["addressee"]
    react = out["reaction"]
    m20_4 = out.get("m20_4_attribution_diagnostics", {}) or {}
    threshold = out["threshold_recommendation"]
    acb = addr.get("addressee_class_breakdown", {}) or {}
    rjb = react.get("reaction_joint_breakdown", {}) or {}

    return {
        "n_fixtures": out["n_fixtures"],
        "scoring_mode": out["scoring_mode"],
        "verdict": out["verdict"],
        # Addressee
        "addr_n_present": addr["n_present"],
        "addr_n_correct": addr["n_correct"],
        "addr_accuracy": _fmt(addr["accuracy"]),
        "addr_brier": _fmt(addr["brier"]),
        "addr_ece": _fmt(addr["ece"]),
        "addr_precision_on_not_addressed": acb.get("precision_on_not_addressed"),
        "addr_recall_on_addressed": acb.get("recall_on_addressed"),
        # Reaction joint
        "react_n_present": react["n_present"],
        "react_n_correct": react["n_correct"],
        "react_accuracy": _fmt(react["accuracy"]),
        "react_brier": _fmt(react["brier"]),
        "react_ece": _fmt(react["ece"]),
        "react_acc_joint_all_decidable": rjb.get("acc_joint_all_decidable"),
        "react_acc_joint_emit_subset": rjb.get("acc_joint_emit_subset"),
        "react_n_joint_no_emit_wrong": rjb.get("n_joint_no_emit_wrong"),
        # Threshold
        "candidate_admit_min": threshold.get("candidate_admit_min"),
        "candidate_tie_breaker_min": threshold.get("candidate_tie_breaker_min"),
        # M20.4 diagnostics (sorted for diff stability)
        "m20_4": {k: m20_4.get(k, 0) for k in sorted(m20_4.keys())},
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: extract_p0_7_run_summary <output.json>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8-sig")
    # The runner prints "{json}" to stdout, possibly with leading log lines.
    start = text.find("{")
    if start < 0:
        print(f"no JSON object found in {path}", file=sys.stderr)
        return 2
    data = json.loads(text[start:])
    summary = extract(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
