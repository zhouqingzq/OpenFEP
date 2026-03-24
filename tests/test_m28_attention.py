from __future__ import annotations

import json
import tempfile
from pathlib import Path

from segmentum.agent import SegmentAgent
from segmentum.attention import AttentionAllocation, AttentionBottleneck
from segmentum.environment import Observation
from segmentum.runtime import SegmentRuntime


def test_salience_scoring_is_deterministic() -> None:
    bottleneck = AttentionBottleneck(capacity=2)
    observation = {
        "food": 0.40,
        "danger": 0.92,
        "novelty": 0.80,
        "shelter": 0.25,
        "temperature": 0.48,
        "social": 0.15,
    }
    prediction = {
        "food": 0.45,
        "danger": 0.30,
        "novelty": 0.35,
        "shelter": 0.45,
        "temperature": 0.50,
        "social": 0.35,
    }
    errors = {
        key: observation[key] - prediction[key]
        for key in observation
    }
    priors = {
        "trauma_bias": 0.8,
        "contamination_sensitivity": 0.2,
        "trust_prior": -0.1,
    }

    first = bottleneck.score_channels(observation, prediction, errors, priors)
    second = bottleneck.score_channels(observation, prediction, errors, priors)

    assert first == second
    assert first["danger"] > first["food"]
    assert first["novelty"] > first["shelter"]


def test_top_k_selection_is_stable() -> None:
    bottleneck = AttentionBottleneck(capacity=2)
    trace = bottleneck.allocate(
        observation={"danger": 0.9, "novelty": 0.7, "food": 0.4, "social": 0.2},
        prediction={"danger": 0.2, "novelty": 0.4, "food": 0.5, "social": 0.2},
        errors={"danger": 0.7, "novelty": 0.3, "food": -0.1, "social": 0.0},
        narrative_priors={"trauma_bias": 0.7},
        tick=3,
    )

    assert trace.allocation.selected_channels == ("danger", "novelty")
    assert trace.allocation.dropped_channels == ("food", "social")


def test_filter_keeps_selected_and_attenuates_dropped() -> None:
    bottleneck = AttentionBottleneck(capacity=2)
    allocation = AttentionAllocation(
        selected_channels=("danger", "novelty"),
        dropped_channels=("food", "social"),
        weights={"danger": 1.0, "novelty": 1.0, "food": 0.35, "social": 0.35},
        bottleneck_load=1.0,
    )
    observation = {"danger": 0.9, "novelty": 0.8, "food": 0.2, "social": 0.1}
    prediction = {"danger": 0.3, "novelty": 0.4, "food": 0.6, "social": 0.5}

    filtered = bottleneck.filter_observation(observation, allocation, prediction=prediction)

    assert filtered["danger"] == observation["danger"]
    assert filtered["novelty"] == observation["novelty"]
    assert prediction["food"] > filtered["food"] > observation["food"]
    assert prediction["social"] > filtered["social"] > observation["social"]


def test_attention_serialization_round_trip() -> None:
    runtime = SegmentRuntime.load_or_create(seed=23, reset=True)
    runtime.agent.configure_attention_bottleneck(enabled=True, capacity=2)
    runtime.step(verbose=False)

    payload = runtime.agent.to_dict()
    restored = SegmentAgent.from_dict(payload, rng=runtime.world.rng)

    assert restored.attention_bottleneck.to_dict() == runtime.agent.attention_bottleneck.to_dict()
    assert restored.attention_state()["last_trace"] == runtime.agent.attention_state()["last_trace"]


def test_old_snapshot_without_attention_still_loads() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        runtime = SegmentRuntime.load_or_create(state_path=state_path, seed=17, reset=True)
        runtime.run(cycles=2, verbose=False)

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["agent"].pop("attention_bottleneck", None)
        payload["agent"].pop("last_attention_trace", None)
        payload["agent"].pop("last_attention_filtered_observation", None)
        state_path.write_text(json.dumps(payload), encoding="utf-8")

        restored = SegmentRuntime.load_or_create(state_path=state_path, seed=99)
        assert restored.agent.attention_bottleneck.enabled is True
        assert restored.agent.cycle == 2


def test_runtime_trace_contains_attention_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = Path(tmp_dir) / "segment_state.json"
        trace_path = Path(tmp_dir) / "segment_trace.jsonl"
        runtime = SegmentRuntime.load_or_create(
            state_path=state_path,
            trace_path=trace_path,
            seed=19,
            reset=True,
        )
        runtime.agent.configure_attention_bottleneck(enabled=True, capacity=2)
        runtime.step(verbose=False)

        record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        assert "attention" in record
        assert "attention_selected_channels" in record["decision_loop"]
        assert len(record["decision_loop"]["attention_selected_channels"]) == 2


def test_agent_perceive_records_attention_trace() -> None:
    agent = SegmentAgent()
    agent.configure_attention_bottleneck(enabled=True, capacity=2)
    observation = Observation(
        food=0.15,
        danger=0.93,
        novelty=0.84,
        shelter=0.25,
        temperature=0.46,
        social=0.18,
    )

    observed, prediction, _errors, _free_energy, _hierarchy = agent.perceive(observation)

    assert observed["danger"] == 0.93
    assert agent.last_attention_trace is not None
    assert tuple(agent.last_attention_trace.allocation.selected_channels) == ("danger", "novelty")
    assert agent.last_attention_filtered_observation["danger"] == observed["danger"]
    assert agent.last_attention_filtered_observation["food"] != observed["food"]


def test_memory_sensitive_pattern_bias_can_promote_threat_channel() -> None:
    bottleneck = AttentionBottleneck(capacity=1)

    baseline = bottleneck.allocate(
        observation={"danger": 0.32, "novelty": 0.55, "food": 0.52},
        prediction={"danger": 0.28, "novelty": 0.30, "food": 0.50},
        errors={"danger": 0.04, "novelty": 0.25, "food": 0.02},
        narrative_priors={"trauma_bias": 0.0},
        tick=1,
    )
    biased = bottleneck.allocate(
        observation={"danger": 0.32, "novelty": 0.55, "food": 0.52},
        prediction={"danger": 0.28, "novelty": 0.30, "food": 0.50},
        errors={"danger": 0.04, "novelty": 0.25, "food": 0.02},
        narrative_priors={"trauma_bias": 0.0},
        tick=2,
        memory_context={
            "aggregate": {
                "chronic_threat_bias": 0.95,
                "protected_anchor_bias": 0.85,
            },
            "sensitive_channels": ["danger"],
            "attention_biases": {"danger": 0.25},
        },
    )

    assert baseline.allocation.selected_channels == ("novelty",)
    assert biased.allocation.selected_channels == ("danger",)
    assert biased.salience_scores["danger"] > baseline.salience_scores["danger"]
