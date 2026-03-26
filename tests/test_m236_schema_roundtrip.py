from __future__ import annotations

from segmentum.m236_open_continuity_trial import SCHEMA_VERSION, build_m236_schema_payload


def test_schema_payload_is_versioned_and_roundtrippable() -> None:
    payload = build_m236_schema_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["payload_kind"] == "m236_acceptance_bundle"
    assert payload["roundtrip_ok"] is True
    assert payload["canonical_fields_present"] is True
    assert payload["determinism_signature_preserved"] is True
    assert payload["payload_size_bytes"] > 0
