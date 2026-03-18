from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentum.m225_benchmarks import (
    M225_PYTEST_LOG,
    clear_m225_persisted_test_execution_log,
    load_m225_test_execution_log,
)
from segmentum.m226_maturity_audit import SEED_SET, write_m226_maturity_audit_artifacts


REQUIRED_PYTEST_COMMANDS = [
    ["py", "-m", "pytest", "-q", "tests/test_m225_freshness_guards.py"],
    ["py", "-m", "pytest", "-q", "tests/test_self_model.py", "tests/test_m2_targeted_repair.py", "tests/test_baseline_regressions.py"],
]


def _run_required_pytest_suites() -> list[dict[str, object]]:
    clear_m225_persisted_test_execution_log(M225_PYTEST_LOG)
    for index, command in enumerate(REQUIRED_PYTEST_COMMANDS):
        env = dict(os.environ)
        env["SEGMENTUM_M225_TEST_LOG"] = str(M225_PYTEST_LOG)
        env["SEGMENTUM_M225_CLEAR_LOG"] = "1" if index == 0 else "0"
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    return load_m225_test_execution_log(M225_PYTEST_LOG)


if __name__ == "__main__":
    pytest_evidence = _run_required_pytest_suites()
    paths = write_m226_maturity_audit_artifacts(seed_set=list(SEED_SET), pytest_evidence=pytest_evidence)
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "mode": "write",
                "status": report["status"],
                "final_status": report["final_status"],
                "default_mature": report["default_mature"],
                "paths": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
