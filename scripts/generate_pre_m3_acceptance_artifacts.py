from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m211_readiness import (
    ARTIFACTS_DIR,
    build_attention_summary,
    build_personality_summary,
    build_pre_m3_readiness_report,
    build_snapshot_compatibility_payload,
    build_transfer_summary,
    run_soak_regression,
    write_json,
)


def generate_artifacts() -> dict[str, Path]:
    attention = build_attention_summary()
    transfer = build_transfer_summary()
    personality = build_personality_summary()
    soak = run_soak_regression()
    snapshot = build_snapshot_compatibility_payload()
    readiness = build_pre_m3_readiness_report(
        attention_summary=attention,
        transfer_summary=transfer,
        personality_summary=personality,
        soak_regression=soak,
        snapshot_compatibility=snapshot,
    ).payload
    return {
        "attention": write_json(ARTIFACTS_DIR / "pre_m3_attention_summary.json", attention),
        "transfer": write_json(ARTIFACTS_DIR / "pre_m3_transfer_summary.json", transfer),
        "personality": write_json(ARTIFACTS_DIR / "pre_m3_personality_summary.json", personality),
        "readiness": write_json(ARTIFACTS_DIR / "pre_m3_readiness_report.json", readiness),
    }


def main() -> None:
    generate_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
