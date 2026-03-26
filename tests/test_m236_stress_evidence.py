from __future__ import annotations

from segmentum.m236_open_continuity_trial import build_m236_stress_payload


def test_stress_payload_detects_failure_injections_without_silent_pass() -> None:
    payload = build_m236_stress_payload()
    checks = payload["stress_checks"]

    assert checks["maintenance_overload_is_rejected"] is True
    assert checks["restart_corruption_is_rejected"] is True
    assert checks["maintenance_overload_detects_pressure"] is True
    assert checks["restart_corruption_detects_identity_break"] is True
    assert checks["failure_injections_not_silent"] is True
