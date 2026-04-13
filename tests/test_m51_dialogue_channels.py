from __future__ import annotations

from segmentum.attention import AttentionBottleneck
from segmentum.predictive_coding import BayesianBeliefState
from segmentum.dialogue.channel_registry import (
    DIALOGUE_CHANNEL_NAMES,
    DIALOGUE_CHANNELS,
    ObservabilityTier,
    get_channel_spec,
)
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.precision_bounds import ChannelPrecisionBounds
from segmentum.dialogue.signal_extractors import (
    ConflictTensionExtractor,
    EmotionalToneExtractor,
    HiddenIntentExtractor,
    RelationshipDepthExtractor,
    SemanticContentExtractor,
    TopicNoveltyExtractor,
)


def test_channel_registry_has_expected_six_channels() -> None:
    assert len(DIALOGUE_CHANNELS) == 6
    assert set(DIALOGUE_CHANNEL_NAMES) == {
        "semantic_content",
        "topic_novelty",
        "emotional_tone",
        "conflict_tension",
        "relationship_depth",
        "hidden_intent",
    }
    assert get_channel_spec("semantic_content").tier == ObservabilityTier.HIGH
    assert get_channel_spec("hidden_intent").tier == ObservabilityTier.LOW


def test_signal_extractors_rule_based_ranges() -> None:
    history = ["我们聊过旅行", "你喜欢海边吗"]
    context: dict[str, object] = {"conflict_trend": 0.2}
    text = "你为什么现在必须回复我？"
    partner_uid = 7
    values = {
        "semantic_content": SemanticContentExtractor().extract(text, history, partner_uid, context),
        "topic_novelty": TopicNoveltyExtractor().extract(text, history, partner_uid, context),
        "emotional_tone": EmotionalToneExtractor().extract(text, history, partner_uid, context),
        "conflict_tension": ConflictTensionExtractor().extract(text, history, partner_uid, context),
        "relationship_depth": RelationshipDepthExtractor().extract(text, history, partner_uid, context),
        "hidden_intent": HiddenIntentExtractor().extract(text, history, partner_uid, context),
    }
    for value in values.values():
        assert 0.0 <= value <= 1.0


def test_dialogue_observer_and_bus_signal_confidence() -> None:
    observer = DialogueObserver()
    obs = observer.observe(
        current_turn="谢谢你今天愿意聊聊。",
        conversation_history=["最近怎么样？"],
        partner_uid=101,
        session_context={"conflict_trend": -0.2},
        session_id="s1",
        turn_index=0,
        speaker_uid=101,
    )
    assert set(obs.channels) == set(DIALOGUE_CHANNEL_NAMES)
    signals = obs.to_bus_signals()
    assert len(signals) == 6
    for signal in signals:
        spec = get_channel_spec(signal.channel)
        assert signal.confidence == spec.default_precision


def test_precision_bounds_anomaly_report() -> None:
    bounds = ChannelPrecisionBounds.from_dialogue_channels()
    report = bounds.anomaly_report(
        {
            "hidden_intent": 0.50,
            "relationship_depth": 0.01,
            "emotional_tone": 0.60,
            "conflict_tension": 0.01,
        }
    )
    assert report["hidden_intent"] == "paranoid"
    assert report["relationship_depth"] == "naive"
    assert report["emotional_tone"] == "anxious"
    assert report["conflict_tension"] == "numb"


def test_tier3_extractors_change_slowly() -> None:
    rel = RelationshipDepthExtractor()
    intent = HiddenIntentExtractor()
    history: list[str] = []
    context = {"conflict_trend": 0.0}
    rel_values: list[float] = []
    intent_values: list[float] = []
    for i in range(10):
        msg = f"第{i}轮 我觉得我们可以继续聊。"
        rel_values.append(rel.extract(msg, history, partner_uid=1, session_context=context))
        intent_values.append(intent.extract(msg, history, partner_uid=1, session_context=context))
        history.append(msg)
    rel_deltas = [abs(rel_values[i] - rel_values[i - 1]) for i in range(1, len(rel_values))]
    intent_deltas = [abs(intent_values[i] - intent_values[i - 1]) for i in range(1, len(intent_values))]
    assert max(rel_deltas) < 0.05
    assert max(intent_deltas) < 0.05


def test_attention_bottleneck_accepts_dialogue_observation() -> None:
    observer = DialogueObserver()
    obs = observer.observe(
        current_turn="你是不是在回避这个问题？",
        conversation_history=["我们先讨论计划", "我不同意这个方案"],
        partner_uid=2,
        session_context={"conflict_trend": 0.5},
        session_id="s2",
        turn_index=2,
        speaker_uid=2,
    )
    bottleneck = AttentionBottleneck(capacity=3)
    prediction = {channel: 0.5 for channel in obs.channels}
    errors = {channel: obs.channels[channel] - prediction[channel] for channel in obs.channels}
    scores = bottleneck.score_channels(
        observation=obs.channels,
        prediction=prediction,
        errors=errors,
        narrative_priors={"trauma_bias": 0.3, "trust_prior": -0.2, "controllability_prior": 0.1},
    )
    assert set(scores) == set(obs.channels)


def test_predictive_coding_dynamic_modalities_and_bounds() -> None:
    state = BayesianBeliefState(
        layer_name="dialogue",
        beliefs={
            "semantic_content": 0.5,
            "topic_novelty": 0.5,
            "emotional_tone": 0.5,
            "conflict_tension": 0.5,
            "relationship_depth": 0.1,
            "hidden_intent": 0.1,
        },
        initial_precision=0.2,
        channel_precision_bounds={"hidden_intent": (0.05, 0.20)},
        min_precision=0.01,
        max_precision=3.0,
    )
    state.ensure_modalities(("semantic_content", "hidden_intent", "new_channel"))
    assert "new_channel" in state.beliefs
    update = state.posterior_update(
        incoming_observation={"hidden_intent": 1.0},
        top_down_prediction={"hidden_intent": 0.1},
    )
    assert update.error_precision["hidden_intent"] <= 0.20
