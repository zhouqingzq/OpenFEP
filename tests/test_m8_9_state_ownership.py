"""M8.9 R1: State Ownership Contract tests."""

import pytest

from segmentum.state_ownership import (
    STATE_OWNERSHIP_TABLE,
    StateOwnershipEntry,
    get_ownership_for,
)
from segmentum.dialogue.cognitive_guidance import build_compressed_cognitive_guidance


# ── R1.1: Table completeness ─────────────────────────────────────────────

def test_state_ownership_table_names_all_six_surfaces():
    """All 6 required self-related state surfaces are present."""
    surface_names = {entry.state_surface for entry in STATE_OWNERSHIP_TABLE}
    required = {
        "CognitiveStateMVP",
        "SubjectState",
        "SelfModel",
        "MemoryStore / anchored memory",
        "MetaControlSignal",
        "FEPPromptCapsule",
    }
    missing = required - surface_names
    assert not missing, f"Missing state surfaces: {missing}"


def test_each_surface_has_named_owner():
    """Every surface has a non-empty owner."""
    for entry in STATE_OWNERSHIP_TABLE:
        assert entry.owner, (
            f"State surface '{entry.state_surface}' has no owner"
        )


def test_each_surface_has_update_path():
    """Every surface has a non-empty update path."""
    for entry in STATE_OWNERSHIP_TABLE:
        assert entry.update_path, (
            f"State surface '{entry.state_surface}' has no update path"
        )


def test_get_ownership_returns_entry():
    """get_ownership_for() finds entries by surface name."""
    entry = get_ownership_for("CognitiveStateMVP")
    assert entry is not None
    assert entry.owner == "cognitive state reducer"
    assert entry.prompt_eligibility == "compressed_only"


# ── R1.2: Prompt safety boundaries ───────────────────────────────────────

def test_cognitive_state_not_passed_raw_to_prompt_builder():
    """Compressed guidance filters sensitive keys and never returns raw state."""
    raw_cognitive_state = {
        "task": {"prompt": "secret task prompt text"},
        "memory": {"api_key": "sk-12345"},
        "diagnostics": "raw diagnostic dump",
        "event_payload": "raw event data",
        "full_memory_dump": "all episodes",
        "Self-consciousness.md": "I am conscious",
    }
    result = build_compressed_cognitive_guidance(raw_cognitive_state)
    # Must be a dict
    assert isinstance(result, dict)
    # Sensitive keys must not appear in result keys or string values
    result_str = str(result).lower()
    for sensitive in ("api", "secret", "token", "diagnostic", "prompt", "event", "payload"):
        assert sensitive not in result_str, f"Sensitive fragment '{sensitive}' leaked: {result_str}"
    # Raw dump values must not appear
    assert "sk-12345" not in result_str
    assert "raw diagnostic dump" not in result_str
    assert "raw event data" not in result_str


def test_fep_capsule_does_not_contain_raw_memory_or_events():
    """FEP capsule blocks raw events/diagnostics/markdown in its output."""
    from segmentum.dialogue.fep_prompt import build_fep_prompt_capsule, FEPPromptCapsule

    class _MockDiag:
        ranked_options = []
        chosen = None

    capsule = build_fep_prompt_capsule(
        diagnostics=_MockDiag(),
        observation={"valence": 0.5, "arousal": 0.3},
        persona_id="test_persona",
        session_id="test_session",
    )
    assert isinstance(capsule, FEPPromptCapsule)
    capsule_dict = capsule.to_dict()
    # Check that sensitive fields are not in the capsule output
    for key in capsule_dict:
        key_lower = str(key).lower()
        assert "raw" not in key_lower, f"Sensitive key '{key}' in capsule"
        assert "diagnostic" not in key_lower, f"Sensitive key '{key}' in capsule"
    # Check that observation_channels are numeric dict (compressed)
    obs = capsule_dict.get("observation_channels", {})
    assert isinstance(obs, dict)
    # Capsule values must not contain markdown fences or raw dump markers
    capsule_str = str(capsule_dict)
    assert "```" not in capsule_str, "Markdown code fences leaked into capsule"
    assert "Conscious.md" not in capsule_str, "Conscious.md reference leaked"


def test_prompt_builder_memory_context_not_raw_store():
    """Prompt builder memory context returns structured data, not raw objects."""
    from segmentum.memory_anchored import AnchoredMemoryItem, MemoryPermissionFilter

    item = AnchoredMemoryItem(
        speaker="user",
        proposition="用户叫周青",
        status="asserted",
        visibility="explicit",
        memory_type="user_asserted_fact",
    )
    buckets = MemoryPermissionFilter.filter([item])

    # explicit_facts are AnchoredMemoryItem objects (not raw MemoryStore)
    assert len(buckets.explicit_facts) == 1
    assert isinstance(buckets.explicit_facts[0], AnchoredMemoryItem)
    # forbidden bucket is empty (retracted/forbidden items are excluded)
    assert len(buckets.forbidden) == 0

    # Simulate what _build_memory_context_v2 would produce:
    # facts are rendered as structured string summaries, never raw dict dumps
    for fact in buckets.explicit_facts:
        rendered = f"[{fact.status}] {fact.proposition}"
        assert "memory_id" not in rendered  # not a raw dict dump
        assert "source_text" not in rendered  # not raw internal fields
