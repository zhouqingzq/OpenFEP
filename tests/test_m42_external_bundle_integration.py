from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from segmentum.confidence_external_bundle import build_confidence_external_bundle
from segmentum.igt_external_bundle import build_igt_external_bundle
from segmentum.benchmark_registry import benchmark_status
from segmentum.m4_benchmarks import run_confidence_database_benchmark, run_iowa_gambling_benchmark
from segmentum.m4_cognitive_style import CognitiveStyleParameters


class TestM42ExternalBundleIntegration(unittest.TestCase):
    @staticmethod
    def _igt_rows(*, start_offset: int) -> list[str]:
        rows = ["iteration,EEG sample,decision,win,lose,balance"]
        balance = 2000
        decks = "ABCD"
        for index in range(1, 101):
            deck = decks[(index - 1 + start_offset) % len(decks)]
            reward = 100 if deck in {"A", "B"} else 50
            penalty = 0
            if deck == "A" and index % 5 == 0:
                penalty = 250
            elif deck == "B" and index % 10 == 0:
                penalty = 125
            elif deck == "C" and index % 5 == 0:
                penalty = 25
            elif deck == "D" and index % 4 == 0:
                penalty = 50
            balance += reward - penalty
            rows.append(f"{index},{1000 * index},{deck},{reward},{penalty},{balance}")
        return rows

    def test_both_external_bundles_reach_acceptance_ready_and_run_formal_acceptance_flow(self) -> None:
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
                "s-01": self._igt_rows(start_offset=0),
                "s-02": self._igt_rows(start_offset=1),
                "s-03": self._igt_rows(start_offset=2),
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
                )
                igt_run = run_iowa_gambling_benchmark(
                    CognitiveStyleParameters(),
                    seed=44,
                    selected_subject_id="s-01",
                    include_subject_summary=False,
                    include_predictions=False,
                    protocol_mode="standard_100",
                )
            finally:
                if previous_root is None:
                    os.environ.pop("SEGMENTUM_BENCHMARK_ROOT", None)
                else:
                    os.environ["SEGMENTUM_BENCHMARK_ROOT"] = previous_root

            self.assertEqual(confidence_run["benchmark_status"]["benchmark_state"], "acceptance_ready")
            self.assertEqual(confidence_run["claim_envelope"], "benchmark_eval")
            self.assertGreaterEqual(confidence_run["trial_count"], 1)
            self.assertTrue(confidence_run["leakage_check"]["ok"])
            self.assertEqual(igt_run["benchmark_status"]["benchmark_state"], "acceptance_ready")
            self.assertEqual(igt_run["claim_envelope"], "benchmark_eval")
            self.assertEqual(igt_run["trial_count"], 100)
            self.assertEqual(igt_run["protocol_validation"]["protocol_mode"], "standard_100")
            self.assertEqual(igt_run["protocol_validation"]["standard_trial_count"], 100)
            self.assertEqual(igt_run["trial_trace"][-1]["trial_index"], 100)
            self.assertTrue(igt_run["leakage_check"]["ok"])


if __name__ == "__main__":
    unittest.main()
