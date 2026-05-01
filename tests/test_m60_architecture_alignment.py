from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_NOTE = ROOT / "reports" / "m6_architecture_alignment.md"
SCHEMA_DRAFT = ROOT / "reports" / "m6_event_state_schema_draft.md"


REQUIRED_EXISTING_FILES = (
    "segmentum/dialogue/conversation_loop.py",
    "segmentum/dialogue/observer.py",
    "segmentum/agent.py",
    "segmentum/dialogue/fep_prompt.py",
    "segmentum/dialogue/generator.py",
    "segmentum/dialogue/outcome.py",
    "segmentum/attention.py",
    "segmentum/workspace.py",
    "segmentum/metacognitive.py",
    "segmentum/types.py",
    "segmentum/dialogue/runtime/chat.py",
    "segmentum/dialogue/runtime/prompts.py",
)


REQUIRED_CHAIN_ANCHORS = (
    "DialogueObserver.observe",
    "SegmentAgent.decision_cycle_from_dict",
    "build_fep_prompt_capsule",
    "ResponseGenerator.generate",
    "classify_dialogue_outcome",
    "SegmentAgent.integrate_outcome",
)


REQUIRED_BOUNDARY_CONSTRAINTS = (
    "Do not build a parallel cognition runtime.",
    "Do not replace `SegmentAgent.decision_cycle`.",
    "Do not replace `AttentionBottleneck` or `GlobalWorkspace`.",
    "Do not replace `MetaCognitiveLayer`.",
    "Do not create a second prompt capsule type.",
    "Do not insert raw event streams, full memory dumps, full diagnostics, or",
    "Do not let `CognitiveEventBus` become write-only logging",
    "Do not let `CognitiveStateMVP` become a second truth source duplicating",
    "Do not let `AffectiveStateMVP` become a full emotion simulator",
    "Do not let `Conscious.md` or `Self-consciousness.md` become the source of truth",
    "Do not let multiple personas share one `Self-consciousness.md`.",
    "Do not let `Self-consciousness.md` update every turn without",
    "Do not treat outcome correlation as causal proof without ablation.",
    "Do not compute previous outcome in multiple owners without a single source of",
)


def test_m60_architecture_note_exists_and_names_current_chain() -> None:
    assert ARCHITECTURE_NOTE.exists()
    text = ARCHITECTURE_NOTE.read_text(encoding="utf-8")

    assert "M6 augments the M5 dialogue runtime rather than replacing it" in text
    for anchor in REQUIRED_CHAIN_ANCHORS:
        assert anchor in text


def test_m60_schema_draft_exists_and_defines_mvp_objects() -> None:
    assert SCHEMA_DRAFT.exists()
    text = SCHEMA_DRAFT.read_text(encoding="utf-8")

    for schema_name in (
        "CognitiveEvent",
        "CognitiveStateMVP",
        "AffectiveStateMVP",
        "CognitivePath",
        "MetaControlGuidance",
        "TurnTrace",
        "Conscious.md",
        "Self-consciousness.md",
    ):
        assert schema_name in text

    for contract_field in (
        "Source",
        "Consumer",
        "JSON safety",
        "Prompt entry",
        "Scope",
    ):
        assert contract_field in text


def test_m60_conscious_artifact_layout_and_identity_rules_are_documented() -> None:
    text = ARCHITECTURE_NOTE.read_text(encoding="utf-8")

    for layout_line in (
        "artifacts/conscious/",
        "personas/",
        "{persona_id}/",
        "profile.json",
        "Self-consciousness.md",
        "sessions/",
        "{session_id}/",
        "Conscious.md",
        "conscious_trace.jsonl",
        "turn_summaries/",
        "turn_0001.md",
    ):
        assert layout_line in text

    for identity_rule in (
        "`persona_id` must be stable",
        "No persona may read or write another persona's `Self-consciousness.md`.",
        "Prompt assembly must resolve `persona_id` before reading any conscious artifact.",
        "Display names may change; directory identity must not.",
    ):
        assert identity_rule in text


def test_m60_architecture_note_names_real_repository_anchors() -> None:
    text = ARCHITECTURE_NOTE.read_text(encoding="utf-8")

    for relative_path in REQUIRED_EXISTING_FILES:
        assert relative_path in text
        assert (ROOT / relative_path).exists(), relative_path


def test_m60_boundary_constraints_name_required_non_goals() -> None:
    text = ARCHITECTURE_NOTE.read_text(encoding="utf-8")

    assert "## Non-goals And Boundary Constraints" in text
    for constraint in REQUIRED_BOUNDARY_CONSTRAINTS:
        assert constraint in text
