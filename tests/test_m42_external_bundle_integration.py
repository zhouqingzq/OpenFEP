from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import os

from segmentum.confidence_external_bundle import build_confidence_external_bundle
from segmentum.igt_external_bundle import build_igt_external_bundle
from segmentum.benchmark_registry import benchmark_status
from segmentum.m4_benchmarks import run_confidence_database_benchmark, run_iowa_gambling_benchmark
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM42ExternalBundleIntegration(unittest.TestCase):
    def test_both_external_bundles_reach_acceptance_ready_and_run_basic_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            confidence_source = workspace / "Confidence Database"
            confidence_source.mkdir()
            (confidence_source / "data_demo.csv").write_text(
                "\n".join(
                    [
                        "Subj_idx,Stimulus,Response,Confidence,RT_decConf,Difficulty,Orientation,Task",
                        "1,2,2,4,0.60,1,12.5,A",
                        "1,1,1,2,0.70,2,-5.0,A",
                        "2,2,1,3,0.55,1,8.0,B",
                    ]
                ),
                encoding="utf-8",
            )

            igt_source = workspace / "igt dataset"
            igt_source.mkdir()
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
                subject_dir = igt_source / subject_id
                subject_dir.mkdir()
                (subject_dir / "IGT.csv").write_text("\n".join(rows), encoding="utf-8")

            registry_root = workspace / "registry"
            build_confidence_external_bundle(confidence_source, registry_root)
            build_igt_external_bundle(igt_source, registry_root)

            self.assertEqual(benchmark_status("confidence_database", root=registry_root).benchmark_state, "acceptance_ready")
            self.assertEqual(benchmark_status("iowa_gambling_task", root=registry_root).benchmark_state, "acceptance_ready")

            previous_root = os.environ.get("SEGMENTUM_BENCHMARK_ROOT")
            try:
                os.environ["SEGMENTUM_BENCHMARK_ROOT"] = str(registry_root)
                confidence_run = run_confidence_database_benchmark(
                    CognitiveStyleParameters(),
                    seed=42,
                    split=None,
                    include_subject_summary=False,
                    include_predictions=False,
                    max_trials=2,
                )
                igt_run = run_iowa_gambling_benchmark(
                    CognitiveStyleParameters(),
                    seed=44,
                    include_subject_summary=False,
                    include_predictions=False,
                    max_trials=6,
                )
            finally:
                if previous_root is None:
                    os.environ.pop("SEGMENTUM_BENCHMARK_ROOT", None)
                else:
                    os.environ["SEGMENTUM_BENCHMARK_ROOT"] = previous_root

            self.assertEqual(confidence_run["benchmark_status"]["benchmark_state"], "acceptance_ready")
            self.assertEqual(confidence_run["claim_envelope"], "benchmark_eval")
            self.assertGreaterEqual(confidence_run["trial_count"], 1)
            self.assertEqual(igt_run["benchmark_status"]["benchmark_state"], "acceptance_ready")
            self.assertEqual(igt_run["claim_envelope"], "benchmark_eval")
            self.assertEqual(igt_run["trial_count"], 6)


if __name__ == "__main__":
    unittest.main()
