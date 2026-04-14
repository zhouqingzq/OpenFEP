from __future__ import annotations

from segmentum.agent import SegmentAgent
from segmentum.dialogue.actions import DIALOGUE_ACTION_NAMES, is_dialogue_action
from segmentum.dialogue.conversation_loop import run_conversation
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.dialogue.observer import DialogueObserver, normalize_conversation_history
from segmentum.dialogue.outcome import (
    DialogueOutcomeType,
    classify_dialogue_outcome,
    inject_outcome_semantics,
)
from segmentum.dialogue.policy import DialoguePolicyEvaluator
from segmentum.dialogue.prediction_bridge import register_dialogue_actions as register_from_bridge
from segmentum.dialogue.types import TranscriptUtterance


def test_dialogue_policy_evaluator_lists_registered_actions() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    pol = DialoguePolicyEvaluator(agent)
    assert pol.registered_dialogue_actions() == DIALOGUE_ACTION_NAMES
    assert pol.strategy_for("ask_question") == "explore"
    assert pol.strategy_for("disagree") == "escape"


def test_detect_dialogue_patterns_withdrawal_and_question_reward() -> None:
    from segmentum.dialogue.cognitive_style_bridge import detect_dialogue_patterns

    hist = [{"action": "minimal_response"}] * 4
    w = detect_dialogue_patterns([], hist)
    assert "withdrawal_pattern" in w
    batch = [
        {
            "dialogue_outcome_semantic": "epistemic_gain",
            "action_taken": "ask_question",
        }
    ]
    q = detect_dialogue_patterns(batch, [])
    assert "question_reward_pattern" in q


def test_llm_generator_is_placeholder() -> None:
    from segmentum.dialogue.generator import LLMGenerator

    gen = LLMGenerator()
    try:
        gen.generate(
            "agree",
            {},
            {},
            [],
            master_seed=1,
            turn_index=0,
        )
    except NotImplementedError:
        return
    raise AssertionError("LLMGenerator.generate should raise NotImplementedError")


def test_register_dialogue_actions_count() -> None:
    from segmentum.action_registry import ActionRegistry

    reg = ActionRegistry()
    register_from_bridge(reg)
    names = {a.name for a in reg.get_all() if is_dialogue_action(a.name)}
    assert names == set(DIALOGUE_ACTION_NAMES)


def test_policy_bias_nonzero_for_dialogue() -> None:
    from segmentum.self_model import PersonalityProfile

    profile = PersonalityProfile()
    b = profile.policy_bias("ask_question", danger=0.4)
    assert abs(b) > 1e-6


def test_slow_action_bias_nonzero_for_dialogue() -> None:
    from segmentum.slow_learning import SlowVariableLearner

    learner = SlowVariableLearner()
    b = learner.action_bias("disagree")
    assert abs(b) > 1e-6


def test_observer_accepts_transcript_utterances() -> None:
    obs = DialogueObserver()
    hist: list[TranscriptUtterance] = [
        TranscriptUtterance(role="interlocutor", text="你好"),
        TranscriptUtterance(role="agent", text="我在。"),
    ]
    flat = normalize_conversation_history(hist)
    assert flat == ["你好", "我在。"]
    out = obs.observe(
        current_turn="为什么？",
        conversation_history=hist,
        partner_uid=1,
        session_context={},
        session_id="s",
        turn_index=0,
        speaker_uid=1,
    )
    assert len(out.channels) == 6


def _six(
    *,
    sem: float = 0.5,
    topic: float = 0.5,
    emo: float = 0.5,
    conflict: float = 0.5,
    rel: float = 0.5,
    hid: float = 0.5,
) -> dict[str, float]:
    return {
        "semantic_content": sem,
        "topic_novelty": topic,
        "emotional_tone": emo,
        "conflict_tension": conflict,
        "relationship_depth": rel,
        "hidden_intent": hid,
    }


def test_classify_all_dialogue_outcome_types() -> None:
    base_prev = _six()
    assert (
        classify_dialogue_outcome("disagree", _six(conflict=0.85), {}, previous_observation=base_prev)
        == DialogueOutcomeType.SOCIAL_THREAT
    )
    assert (
        classify_dialogue_outcome("share_opinion", _six(conflict=0.55), {}, previous_observation=base_prev)
        == DialogueOutcomeType.IDENTITY_THREAT
    )
    assert (
        classify_dialogue_outcome(
            "share_opinion",
            _six(emo=0.65, conflict=0.4, sem=0.5),
            {},
            previous_observation=base_prev,
        )
        == DialogueOutcomeType.SOCIAL_REWARD
    )
    assert (
        classify_dialogue_outcome(
            "agree",
            _six(emo=0.7, rel=0.6, conflict=0.35),
            {},
            previous_observation=base_prev,
        )
        == DialogueOutcomeType.IDENTITY_AFFIRM
    )
    assert (
        classify_dialogue_outcome("ask_question", _six(sem=0.6), {}, previous_observation=base_prev)
        == DialogueOutcomeType.EPISTEMIC_GAIN
    )
    assert (
        classify_dialogue_outcome(
            "elaborate",
            _six(hid=0.75, conflict=0.55),
            {},
            previous_observation=base_prev,
        )
        == DialogueOutcomeType.EPISTEMIC_LOSS
    )
    assert (
        classify_dialogue_outcome("ask_question", _six(), {}, previous_observation=base_prev)
        == DialogueOutcomeType.NEUTRAL
    )
    assert (
        classify_dialogue_outcome(
            "empathize",
            _six(rel=0.62, conflict=0.45, emo=0.55),
            {},
            previous_observation=_six(rel=0.55, conflict=0.5),
        )
        == DialogueOutcomeType.SOCIAL_REWARD
    )


def test_inject_outcome_semantics_bumps_relevance() -> None:
    ep: dict[str, object] = {"relevance_threat": 0.1, "valence": 0.2}
    inject_outcome_semantics(ep, DialogueOutcomeType.SOCIAL_THREAT)
    assert ep["dialogue_outcome_semantic"] == "social_threat"
    assert ep["predicted_outcome"] == "dialogue_threat"
    assert float(ep["relevance_threat"]) >= 0.18


def test_classify_uses_delta_for_escalation() -> None:
    prev = {"conflict_tension": 0.4, "emotional_tone": 0.5, "semantic_content": 0.5, "topic_novelty": 0.5, "relationship_depth": 0.5, "hidden_intent": 0.5}
    nxt = {"conflict_tension": 0.85, "emotional_tone": 0.4, "semantic_content": 0.5, "topic_novelty": 0.5, "relationship_depth": 0.5, "hidden_intent": 0.6}
    o = classify_dialogue_outcome(
        "disagree",
        nxt,
        {},
        previous_observation=prev,
    )
    assert o == DialogueOutcomeType.SOCIAL_THREAT


def test_run_conversation_deterministic_actions() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    observer = DialogueObserver()
    lines = ["你还好吗？", "我有点担心。"]
    seed = 42
    t1 = run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=seed,
        partner_uid=9,
        session_id="t",
    )
    agent2 = SegmentAgent()
    register_from_bridge(agent2.action_registry)
    t2 = run_conversation(
        agent2,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=seed,
        partner_uid=9,
        session_id="t",
    )
    assert [x.action for x in t1] == [x.action for x in t2]
    assert all(x.text for x in t1)
    assert all(x.action and is_dialogue_action(x.action) for x in t1)


def test_run_conversation_twenty_turns_no_error() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    observer = DialogueObserver()
    lines = [f"第{i}轮对方台词。" for i in range(20)]
    turns = run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=99,
        partner_uid=3,
        session_id="long",
    )
    assert len(turns) == 20
    assert all(is_dialogue_action(t.action or "") for t in turns)


def test_personality_extremes_can_diverge_dialogue_actions() -> None:
    """Sanity check: Big Five extremes should not always yield identical dialogue policies."""
    lines = ["你好。", "我很焦虑，你觉得呢？", "再深入说说。"]
    observer = DialogueObserver()

    def _run_with_openness(o: float) -> list[str]:
        agent = SegmentAgent()
        register_from_bridge(agent.action_registry)
        agent.self_model.personality_profile.openness = float(o)
        agent.self_model.personality_profile.neuroticism = 0.2 if o > 0.5 else 0.85
        turns = run_conversation(
            agent,
            lines,
            observer=observer,
            generator=RuleBasedGenerator(),
            master_seed=1001,
            partner_uid=2,
            session_id="p",
        )
        return [t.action or "" for t in turns]

    hi = _run_with_openness(0.92)
    lo = _run_with_openness(0.08)
    assert hi == hi and lo == lo
    assert hi != lo


def test_m53_acceptance_record_shape() -> None:
    """Locks the acceptance artifact contract (see artifacts/m53_acceptance.json)."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "artifacts" / "m53_acceptance.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("milestone") == "M5.3"
    assert data.get("dialogue_action_subspace") is True
    assert data.get("prediction_ledger", {}).get("dialogue_hypothesis_maintenance") is True
    assert "outcome_types_synthetic_coverage" in data
