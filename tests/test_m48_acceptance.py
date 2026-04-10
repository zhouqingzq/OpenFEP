from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from segmentum.m48_audit import (
    GATE_ORDER,
    build_m48_acceptance_report,
    write_m48_acceptance_artifacts,
)


class TestM48Acceptance(unittest.TestCase):
    def _read_json(self, path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_acceptance_report_passes_all_six_gates(self) -> None:
        report, evidence = build_m48_acceptance_report(seed=42, cycles=20)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["formal_acceptance_conclusion"], "ACCEPT")
        self.assertEqual(report["artifact_lineage"], "official_runtime_evidence")
        self.assertEqual(set(report["gate_summaries"]), set(GATE_ORDER))
        self.assertEqual(report["failed_gates"], [])
        self.assertEqual(len(evidence["enabled"]["trace"]), 20)
        self.assertEqual(len(evidence["disabled"]["trace"]), 20)

    def test_writer_emits_ablation_evidence_report_and_summary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            outputs = write_m48_acceptance_artifacts(
                output_root=tmp_dir,
                round_started_at="2026-04-10T00:00:00+00:00",
                seed=42,
                cycles=20,
            )
            report = self._read_json(outputs["report"])
            ablation_evidence = self._read_json(outputs["ablation_evidence"])
            summary = Path(outputs["summary"]).read_text(encoding="utf-8")

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["formal_acceptance_conclusion"], "ACCEPT")
        self.assertTrue(report["freshness"]["generated_in_this_run"])
        self.assertEqual(len(ablation_evidence["enabled"]["trace"]), 20)
        self.assertEqual(len(ablation_evidence["disabled"]["trace"]), 20)
        self.assertIn("M4.8 Official Acceptance Summary", summary)
        self.assertIn("Formal Acceptance Conclusion: `ACCEPT`", summary)
        self.assertIn("G3 `ablation_contrast`: `PASS`", summary)


if __name__ == "__main__":
    unittest.main()
