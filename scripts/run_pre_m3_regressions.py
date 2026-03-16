from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentum.m211_readiness import ARTIFACTS_DIR, run_pre_m3_regression_suite, write_json


def main() -> None:
    payload = run_pre_m3_regression_suite()
    write_json(ARTIFACTS_DIR / "pre_m3_regression_summary.json", payload)
    print(json.dumps(payload["acceptance"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
