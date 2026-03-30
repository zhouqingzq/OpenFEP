from __future__ import annotations

import unittest

from segmentum.m4_benchmarks import ConfidenceDatabaseAdapter, IowaGamblingTaskAdapter, preprocess_confidence_database


class TestM42BenchmarkAdapter(unittest.TestCase):
    def test_preprocess_produces_all_required_splits(self) -> None:
        payload = preprocess_confidence_database()
        splits = {row["split"] for row in payload["trials"]}
        self.assertEqual(payload["manifest"]["benchmark_id"], "confidence_database")
        self.assertEqual(splits, {"train", "validation", "heldout"})

    def test_protocol_schemas_exist_for_confidence_and_igt(self) -> None:
        confidence_schema = ConfidenceDatabaseAdapter().schema()
        igt_schema = IowaGamblingTaskAdapter().schema()
        self.assertEqual(confidence_schema["benchmark_id"], "confidence_database")
        self.assertEqual(igt_schema["benchmark_id"], "iowa_gambling_task")
        self.assertEqual(igt_schema["status"], "adapter_skeleton_only")


if __name__ == "__main__":
    unittest.main()
