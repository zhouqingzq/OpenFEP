from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from segmentum.benchmark_registry import benchmark_status, validate_benchmark_bundle
from segmentum.igt_external_bundle import build_igt_external_bundle
from segmentum.m4_benchmarks import run_iowa_gambling_benchmark
from segmentum.m4_cognitive_style import CognitiveStyleParameters
import os


class TestIgtExternalBundle(unittest.TestCase):
    def test_builds_acceptance_ready_igt_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "igt dataset"
            source_dir.mkdir(parents=True)
            for subject_id, rows in {
                "s-01": [
                    "iteration,EEG sample,decision,win,lose,balance",
                    "1,1000,B,100,0,2100",
                    "2,2000,D,50,0,2150",
                ],
                "s-02": [
                    "iteration,EEG sample,decision,win,lose,balance",
                    "1,1000,A,100,250,1850",
                    "2,2000,C,50,0,1900",
                ],
                "s-03": [
                    "iteration,EEG sample,decision,win,lose,balance",
                    "1,1000,D,50,0,2050",
                    "2,2000,C,50,0,2100",
                ],
            }.items():
                subject_dir = source_dir / subject_id
                subject_dir.mkdir()
                (subject_dir / "IGT.csv").write_text("\n".join(rows), encoding="utf-8")
            destination_root = Path(tmp) / "registry"
            report = build_igt_external_bundle(source_dir, destination_root)
            self.assertEqual(report.included_subjects, 3)
            validation = validate_benchmark_bundle("iowa_gambling_task", root=destination_root)
            self.assertTrue(validation.ok)
            status = benchmark_status("iowa_gambling_task", root=destination_root)
            self.assertEqual(status.benchmark_state, "acceptance_ready")
            manifest = json.loads((destination_root / "iowa_gambling_task" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_type"], "external_bundle")
            previous_root = os.environ.get("SEGMENTUM_BENCHMARK_ROOT")
            try:
                os.environ["SEGMENTUM_BENCHMARK_ROOT"] = str(destination_root)
                run = run_iowa_gambling_benchmark(CognitiveStyleParameters(), seed=44)
            finally:
                if previous_root is None:
                    os.environ.pop("SEGMENTUM_BENCHMARK_ROOT", None)
                else:
                    os.environ["SEGMENTUM_BENCHMARK_ROOT"] = previous_root
            self.assertEqual(run["benchmark_status"]["benchmark_state"], "acceptance_ready")
            self.assertEqual(run["claim_envelope"], "benchmark_eval")
            self.assertEqual(run["trial_count"], 6)


if __name__ == "__main__":
    unittest.main()
