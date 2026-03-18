from __future__ import annotations

from segmentum.m221_benchmarks import run_m221_open_narrative_benchmark


AUDIT_SEED_SET = [226, 245, 323, 342, 420, 439]


def test_m221_behavior_causality_passes_on_m226_audit_seed_set() -> None:
    payload = run_m221_open_narrative_benchmark(seed_set=list(AUDIT_SEED_SET), cycles=24)

    assert payload["gates"]["behavior_causality"] is True
    assert payload["causality_breakdown"]["threat_hardened"]["passed"] is True
    assert payload["causality_breakdown"]["exploratory_adaptive"]["passed"] is True
