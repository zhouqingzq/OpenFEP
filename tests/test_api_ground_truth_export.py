from __future__ import annotations

import json

from segmentum.api import AnalysisRequest, analyze_ground_truth, export_ground_truth_jsonl


SCHEMA_MATERIALS = [
    "A close friend found me when I was lost, stayed nearby, and shared food until I felt safe again.",
    "Another ally welcomed me back, made room for me, and remained beside me until I calmed down.",
    "When I froze, my companion reassured me, kept watch, and helped me reconnect with the group.",
]


def test_ground_truth_export_contains_structured_provenance() -> None:
    request = AnalysisRequest(materials=SCHEMA_MATERIALS)
    payload = analyze_ground_truth(request)
    entry = next(item for item in payload["ground_truth"] if item["path"] == "core_priors.other_reliability")
    assert entry["evidence"]
    assert entry["evidence_details"]
    assert entry["evidence_details"][0]["kind"] in {"episode", "schema"}


def test_ground_truth_jsonl_export_roundtrip() -> None:
    request = AnalysisRequest(materials=SCHEMA_MATERIALS)
    line = export_ground_truth_jsonl(request)
    payload = json.loads(line)
    assert payload["materials"] == SCHEMA_MATERIALS
    assert payload["ground_truth"]
    entry = next(item for item in payload["ground_truth"] if item["path"] == "core_priors.other_reliability")
    assert entry["evidence_details"]
