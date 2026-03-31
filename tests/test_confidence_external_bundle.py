from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from segmentum.benchmark_registry import benchmark_status, validate_benchmark_bundle
from segmentum.confidence_external_bundle import build_confidence_external_bundle


class TestConfidenceExternalBundle(unittest.TestCase):
    def test_builds_acceptance_ready_bundle_from_supported_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "Confidence Database"
            source_dir.mkdir(parents=True)
            (source_dir / "data_demo.csv").write_text(
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
            (source_dir / "data_skip.csv").write_text(
                "\n".join(
                    [
                        "Subject,Answer",
                        "1,yes",
                    ]
                ),
                encoding="utf-8",
            )
            destination_root = Path(tmp) / "registry"
            report = build_confidence_external_bundle(source_dir, destination_root)
            self.assertEqual(report.included_files, 1)
            self.assertEqual(report.skipped_files, 1)
            validation = validate_benchmark_bundle("confidence_database", root=destination_root)
            self.assertTrue(validation.ok)
            status = benchmark_status("confidence_database", root=destination_root)
            self.assertEqual(status.benchmark_state, "acceptance_ready")
            manifest = json.loads((destination_root / "confidence_database" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_type"], "external_bundle")
            self.assertFalse(manifest["smoke_test_only"])
            self.assertIn("data_skip.csv", manifest["conversion_summary"]["skipped_reasons"])


if __name__ == "__main__":
    unittest.main()
