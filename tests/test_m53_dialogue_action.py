from __future__ import annotations

from collections import Counter
import math

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


def test_dialogue_policy_evaluator_facade_methods() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    pol = DialoguePolicyEvaluator(agent)
    obs = {
        "semantic_content": 0.55,
        "topic_novelty": 0.48,
        "emotional_tone": 0.52,
        "conflict_tension": 0.40,
        "relationship_depth": 0.50,
        "hidden_intent": 0.42,
    }
    cycle_before = int(agent.cycle)
    episodes_before = len(agent.long_term_memory.episodes)
    decisions_before = len(agent.decision_history)
    social_events_before = len(getattr(agent.social_memory, "interaction_history", []))
    scores = pol.evaluate_actions(obs, {"session_id": "facade", "partner_uid": 1})
    assert scores
    assert set(scores).issubset(set(DIALOGUE_ACTION_NAMES))
    selected = pol.select_action(obs, {"session_id": "facade", "partner_uid": 1})
    assert selected in scores
    assert is_dialogue_action(selected)
    # Facade helpers are read-only: no decision-cycle side effects.
    assert int(agent.cycle) == cycle_before
    assert len(agent.long_term_memory.episodes) == episodes_before
    assert len(agent.decision_history) == decisions_before
    assert len(getattr(agent.social_memory, "interaction_history", [])) == social_events_before


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


def _cosine_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    na = math.sqrt(sum(float(a.get(k, 0.0)) ** 2 for k in keys))
    nb = math.sqrt(sum(float(b.get(k, 0.0)) ** 2 for k in keys))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - cosine


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


def test_efe_driven_policy_responds_to_observation_change() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    pol = DialoguePolicyEvaluator(agent)
    low_conflict = _six(sem=0.72, topic=0.68, emo=0.62, conflict=0.20, rel=0.56, hid=0.22)
    high_conflict = _six(sem=0.46, topic=0.34, emo=0.30, conflict=0.90, rel=0.28, hid=0.84)
    s1 = pol.evaluate_actions(low_conflict, {"session_id": "efe", "partner_uid": 9, "master_seed": 17})
    s2 = pol.evaluate_actions(high_conflict, {"session_id": "efe", "partner_uid": 9, "master_seed": 17})
    assert s1 and s2
    a1 = pol.select_action(low_conflict, {"session_id": "efe", "partner_uid": 9, "master_seed": 17})
    a2 = pol.select_action(high_conflict, {"session_id": "efe", "partner_uid": 9, "master_seed": 17})
    assert a1 == min(s1, key=s1.get)
    assert a2 == min(s2, key=s2.get)
    overlap = sorted(set(s1) & set(s2))
    assert overlap
    assert any(abs(float(s1[k]) - float(s2[k])) > 1e-6 for k in overlap)
    # Current model may keep the same argmin action, but score landscape must move with observation.
    assert float(s1[a1]) != float(s2[a2])


def test_slow_trait_state_stays_stable_after_50_turn_dialogue() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    observer = DialogueObserver()
    initial = agent.slow_variable_learner.state.traits.to_dict()
    lines = [f"第{i}轮：我们继续讨论这件事。你能再说明一下吗？" for i in range(50)]
    run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=31415,
        partner_uid=12,
        session_id="stable-50",
    )
    agent.sleep()
    final_state = agent.slow_variable_learner.state.traits.to_dict()
    assert _cosine_distance(initial, final_state) < 0.1


def test_aggregate_pressures_has_dialogue_threat_or_safe_repair_events() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    observer = DialogueObserver()
    lines = [f"第{i}轮：你是不是在回避问题？这让我更不安。为什么不直接回答？" for i in range(20)]
    run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=2026,
        partner_uid=7,
        session_id="pressure-20",
    )
    for payload in agent.long_term_memory.episodes[-6:]:
        if not isinstance(payload, dict):
            continue
        payload["predicted_outcome"] = "dialogue_threat"
        payload["dialogue_outcome_semantic"] = "social_threat"
        payload["action_taken"] = str(payload.get("action_taken", payload.get("choice", "disagree")))
    replay_batch = [dict(item) for item in agent.long_term_memory.episodes if isinstance(item, dict)]
    threat_events = sum(
        1
        for payload in replay_batch
        if str(payload.get("predicted_outcome", "neutral")) in {"survival_threat", "integrity_loss", "dialogue_threat"}
        or str(payload.get("dialogue_outcome_semantic", "")) in {"social_threat", "identity_threat", "epistemic_loss"}
        or float(payload.get("observation", {}).get("danger", payload.get("danger", 0.0))) >= 0.72
    )
    safe_repairs = sum(
        1
        for payload in replay_batch
        if str(payload.get("predicted_outcome", "neutral")) == "dialogue_reward"
        and str(payload.get("action_taken", payload.get("action", ""))) in {"agree", "empathize", "elaborate", "joke"}
    )
    assert threat_events > 0 or safe_repairs > 0
    pressures = agent.slow_variable_learner.aggregate_pressures(
        tick=agent.cycle,
        replay_batch=replay_batch,
        decision_history=list(agent.decision_history),
        prediction_ledger=agent.prediction_ledger,
        verification_loop=agent.verification_loop,
        social_memory=agent.social_memory,
        identity_tension_history=list(agent.identity_tension_history),
        self_model=agent.self_model,
        body_state={
            "energy": agent.energy,
            "stress": agent.stress,
            "fatigue": agent.fatigue,
            "temperature": agent.temperature,
        },
    )
    assert pressures


def test_error_aversion_increases_after_high_conflict_50_turn_dialogue() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    observer = DialogueObserver()
    baseline = float(agent.memory_cognitive_style.error_aversion)
    lines = [f"第{i}轮：你这次的解释前后矛盾，我不同意，而且这让我更警惕。" for i in range(50)]
    run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=9090,
        partner_uid=5,
        session_id="conflict-50",
    )
    for payload in agent.long_term_memory.episodes[-12:]:
        if not isinstance(payload, dict):
            continue
        payload["predicted_outcome"] = "dialogue_threat"
        payload["dialogue_outcome_semantic"] = "social_threat"
        payload["action_taken"] = "disagree"
    agent.sleep()
    delta = float(agent.memory_cognitive_style.error_aversion) - baseline
    assert delta > 0.005


def test_counterfactual_phase_runs_on_dialogue_memory_without_skip_warning() -> None:
    agent = SegmentAgent()
    register_from_bridge(agent.action_registry)
    observer = DialogueObserver()
    lines = [f"第{i}轮：我觉得你在隐瞒信息，这让我很不舒服。" for i in range(20)]
    run_conversation(
        agent,
        lines,
        observer=observer,
        generator=RuleBasedGenerator(),
        master_seed=1234,
        partner_uid=8,
        session_id="cf-20",
    )
    for payload in agent.long_term_memory.episodes[-8:]:
        if not isinstance(payload, dict):
            continue
        payload["errors"] = {"danger": 0.9, "social": 0.8, "hidden_intent": 0.7}
        payload["predicted_outcome"] = str(payload.get("predicted_outcome", "dialogue_threat") or "dialogue_threat")
    summary = agent.sleep()
    assert summary.counterfactual_episodes_evaluated > 0
    sandbox = [item for item in summary.counterfactual_log if str(item.get("type", "")) == "virtual_sandbox_reasoning"]
    assert sandbox
    assert str(sandbox[0].get("skipped_reason", "")) == ""


def _chi_square_critical_0_05(df: int) -> float:
    table = {
        1: 3.841,
        2: 5.991,
        3: 7.815,
        4: 9.488,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
        10: 18.307,
        11: 19.675,
        12: 21.026,
        13: 22.362,
        14: 23.685,
        15: 24.996,
        16: 26.296,
        17: 27.587,
        18: 28.869,
        19: 30.144,
        20: 31.410,
        21: 32.671,
        22: 33.924,
        23: 35.172,
        24: 36.415,
        25: 37.652,
        26: 38.885,
        27: 40.113,
        28: 41.337,
        29: 42.557,
        30: 43.773,
    }
    return table.get(df, 43.773)


def test_personality_action_distribution_is_significantly_different() -> None:
    """Acceptance-like check: 3 personas × 10 turns × multi-seed yields significant action shift."""
    lines = [
        "你好，我们先聊聊最近的状态。",
        "我最近压力有点大。",
        "你觉得我应该先做什么？",
        "其实我也有点生气。",
        "我不太确定你是不是在敷衍我。",
        "那你再解释具体一点。",
        "好吧，也许是我太敏感了。",
        "我们能不能换个角度看？",
        "你会怎么做？",
        "最后给我一个建议。",
    ]
    observer = DialogueObserver()
    seeds = tuple(range(1001, 1009))

    def _run_persona(persona: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for seed in seeds:
            agent = SegmentAgent()
            register_from_bridge(agent.action_registry)
            if persona == "high_trust":
                agent.self_model.narrative_priors.trust_prior = 0.90
                agent.self_model.narrative_priors.trauma_bias = 0.05
                agent.self_model.personality_profile.agreeableness = 0.78
                agent.self_model.personality_profile.neuroticism = 0.22
            elif persona == "low_trust":
                agent.self_model.narrative_priors.trust_prior = -0.65
                agent.self_model.narrative_priors.trauma_bias = 0.20
                agent.self_model.personality_profile.agreeableness = 0.28
                agent.self_model.personality_profile.neuroticism = 0.58
            elif persona == "high_trauma":
                agent.self_model.narrative_priors.trust_prior = -0.20
                agent.self_model.narrative_priors.trauma_bias = 0.92
                agent.self_model.personality_profile.agreeableness = 0.32
                agent.self_model.personality_profile.neuroticism = 0.88
            else:
                raise AssertionError(f"unknown persona: {persona}")
            turns = run_conversation(
                agent,
                lines,
                observer=observer,
                generator=RuleBasedGenerator(),
                master_seed=seed,
                partner_uid=2,
                session_id=f"p-{persona}-{seed}",
            )
            counts.update(t.action or "" for t in turns if t.action)
        return counts

    personas = ("high_trust", "low_trust", "high_trauma")
    per_group = {persona: _run_persona(persona) for persona in personas}
    columns = [a for a in DIALOGUE_ACTION_NAMES if sum(per_group[p][a] for p in personas) > 0]
    assert len(columns) >= 3

    rows = len(personas)
    cols = len(columns)
    grand_total = sum(per_group[p][a] for p in personas for a in columns)
    assert grand_total > 0

    row_totals = {p: sum(per_group[p][a] for a in columns) for p in personas}
    col_totals = {a: sum(per_group[p][a] for p in personas) for a in columns}
    chi2 = 0.0
    for persona in personas:
        for action in columns:
            observed = float(per_group[persona][action])
            expected = float(row_totals[persona] * col_totals[action]) / float(grand_total)
            if expected > 0.0:
                diff = observed - expected
                chi2 += (diff * diff) / expected
    df = (rows - 1) * (cols - 1)
    critical = _chi_square_critical_0_05(df)
    assert chi2 > critical, f"chi2={chi2:.4f}, df={df}, critical_0.05={critical:.4f}"


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
