from __future__ import annotations

from datetime import datetime, timedelta
import random

from segmentum.chat_pipeline.parser import parse_line
from segmentum.chat_pipeline.session_builder import build_sessions
from segmentum.agent import SegmentAgent
from segmentum.dialogue.lifecycle import ImplantationConfig, implant_personality
from segmentum.dialogue.memory_bridge import dialogue_observation_to_memory_fields
from segmentum.dialogue.maturity import PersonalitySnapshot, is_mature
from segmentum.dialogue.observer import DialogueObserver
from segmentum.dialogue.world import DialogueWorld
from segmentum.environment import Observation


def _dataset() -> dict[str, object]:
    uid = 11
    base = datetime(2024, 1, 1, 9, 0, 0)
    sessions: list[dict[str, object]] = []
    session_specs = [
        ("s1", 21, ["你为什么不回复我？", "我觉得你在回避", "我们再聊聊"]),
        ("s2", 22, ["谢谢你愿意听我说", "我很开心", "我们可以一起解决"]),
        ("s3", 21, ["这不对", "你必须解释", "现在就说清楚"]),
        ("s4", 23, ["最近怎么样", "我想分享一个计划", "谢谢反馈"]),
        ("s5", 22, ["我有点担心", "你别急", "我们慢慢来"]),
    ]
    cursor = base
    for idx, (sid, partner, turns) in enumerate(session_specs):
        session_turns: list[dict[str, object]] = []
        for body in turns:
            session_turns.append(
                {
                    "timestamp": cursor.isoformat(),
                    "msg_type": 0,
                    "sender_uid": partner,
                    "receiver_uid": uid,
                    "body": body,
                }
            )
            cursor += timedelta(minutes=1)
            session_turns.append(
                {
                    "timestamp": cursor.isoformat(),
                    "msg_type": 0,
                    "sender_uid": uid,
                    "receiver_uid": partner,
                    "body": "收到",
                }
            )
            cursor += timedelta(minutes=1)
        sessions.append(
            {
                "session_id": sid,
                "uid_a": min(uid, partner),
                "uid_b": max(uid, partner),
                "start_time": session_turns[0]["timestamp"],
                "end_time": session_turns[-1]["timestamp"],
                "metadata": {"turn_count": len(session_turns)},
                "turns": session_turns,
            }
        )
        cursor += timedelta(hours=2 + idx)
    return {"uid": uid, "profile": {}, "sessions": sessions}


def _line(ts: str, sender: int, receiver: int, body: str, msg_type: int = 0) -> str:
    return (
        f"{ts} INFO   MessageSender::OnData message type: {msg_type}, "
        f"sender uid: {sender}, reciever uid: {receiver}, body: {body}"
    )


def _dataset_from_lines(uid: int, lines: list[str]) -> dict[str, object]:
    messages = [parse_line(line) for line in lines]
    parsed = [message for message in messages if message is not None and message.msg_type == 0]
    sessions = build_sessions(parsed, gap_threshold_minutes=30)
    session_rows: list[dict[str, object]] = []
    for pair_sessions in sessions.values():
        for session in pair_sessions:
            session_rows.append(
                {
                    "session_id": session.session_id,
                    "uid_a": session.uid_a,
                    "uid_b": session.uid_b,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat(),
                    "metadata": dict(session.metadata),
                    "turns": [
                        {
                            "timestamp": turn.timestamp.isoformat(),
                            "msg_type": turn.msg_type,
                            "sender_uid": turn.sender_uid,
                            "receiver_uid": turn.receiver_uid,
                            "body": turn.body,
                        }
                        for turn in session.turns
                    ],
                }
            )
    session_rows.sort(key=lambda item: (str(item["start_time"]), str(item["session_id"])))
    return {"uid": uid, "profile": {"source": "m50_like_logs"}, "sessions": session_rows}


def test_m52_implantation_basic_flow_and_social_memory() -> None:
    world = DialogueWorld(_dataset(), DialogueObserver(), seed=42)
    agent = SegmentAgent(rng=random.Random(42))
    result = implant_personality(
        agent,
        world,
        ImplantationConfig(sleep_every_n_sessions=1, maturity_window=2, max_ticks=200),
    )
    assert result.total_ticks > 0
    assert result.total_sleep_cycles >= 1
    assert len(result.final_agent_state.get("episodes", [])) > 0
    traits = result.final_agent_state["slow_variable_learner"]["state"]["traits"]
    assert set(traits) == {
        "caution_bias",
        "threat_sensitivity",
        "trust_stance",
        "exploration_posture",
        "social_approach",
    }
    others = set(result.final_agent_state["social_memory"]["others"])
    assert {"21", "22"}.issubset(others)
    assert len(others) >= 2
    assert result.snapshots
    pv0 = result.snapshots[0].prediction_verification
    assert set(pv0.keys()) >= {"confirmed", "falsified", "total", "falsification_rate"}
    assert int(pv0["total"]) == int(pv0["confirmed"]) + int(pv0["falsified"])


def test_m52_prior_self_body_on_replay() -> None:
    world = DialogueWorld(_dataset(), DialogueObserver(), seed=42)
    assert world.current_turn.get("prior_self_body") == ""
    world.advance()
    assert world.current_turn.get("prior_self_body") == "收到"


def test_m52_memory_bridge_prior_self_question_bumps_relevance_self() -> None:
    obs = {
        "semantic_content": 0.5,
        "topic_novelty": 0.5,
        "emotional_tone": 0.5,
        "conflict_tension": 0.0,
        "relationship_depth": 0.5,
        "hidden_intent": 0.5,
    }
    base = dialogue_observation_to_memory_fields(obs, None)
    bumped = dialogue_observation_to_memory_fields(obs, {"prior_self_body": "你在吗？"})
    assert bumped["relevance_self"] > base["relevance_self"]


def test_m52_sleep_trigger_and_determinism() -> None:
    dataset = _dataset()
    cfg = ImplantationConfig(sleep_every_n_sessions=1, maturity_window=3, max_ticks=200)
    result_a = implant_personality(
        SegmentAgent(rng=random.Random(7)),
        DialogueWorld(dataset, DialogueObserver(), seed=7),
        cfg,
    )
    result_b = implant_personality(
        SegmentAgent(rng=random.Random(7)),
        DialogueWorld(dataset, DialogueObserver(), seed=7),
        cfg,
    )
    assert result_a.total_sleep_cycles >= 1
    assert result_a.final_agent_state == result_b.final_agent_state


def test_m52_maturity_detection_converged_and_not_converged() -> None:
    base = PersonalitySnapshot(
        sleep_cycle=1,
        tick=10,
        slow_traits={"a": 0.5},
        narrative_priors={"b": 0.5},
        precision_debt={"c": 0.1},
        defense_distribution={"suppress": 2},
        memory_stats={"episodic": 10},
        maturity_distance=0.03,
    )
    near = PersonalitySnapshot(**{**base.to_dict(), "sleep_cycle": 2, "maturity_distance": 0.01})
    near2 = PersonalitySnapshot(**{**base.to_dict(), "sleep_cycle": 3, "maturity_distance": 0.015})
    far = PersonalitySnapshot(**{**base.to_dict(), "sleep_cycle": 3, "maturity_distance": 0.08})
    cfg = ImplantationConfig(maturity_threshold=0.02, maturity_window=2)
    assert is_mature([base, near, near2], cfg)
    assert not is_mature([base, near, far], cfg)


def test_m52_decision_cycle_from_dict_matches_observation_path() -> None:
    agent_a = SegmentAgent(rng=random.Random(5))
    agent_b = SegmentAgent(rng=random.Random(5))
    obs = Observation(
        food=0.45,
        danger=0.31,
        novelty=0.62,
        shelter=0.51,
        temperature=0.48,
        social=0.44,
    )
    ref = agent_a.decision_cycle(obs)
    candidate = agent_b.decision_cycle_from_dict(
        {
            "food": 0.45,
            "danger": 0.31,
            "novelty": 0.62,
            "shelter": 0.51,
            "temperature": 0.48,
            "social": 0.44,
        }
    )
    assert ref["observed"] == candidate["observed"]
    assert ref["prediction"] == candidate["prediction"]


def test_m52_implantation_with_m50_like_sessions() -> None:
    uid = 31
    lines = [
        _line("2024-02-01-10:00:00", uid, 77, "早上好"),
        _line("2024-02-01-10:00:40", 77, uid, "今天会议你会来吗？"),
        _line("2024-02-01-10:01:10", uid, 77, "会来"),
        _line("2024-02-01-10:02:20", 77, uid, "太好了"),
        _line("2024-02-01-12:10:00", uid, 88, "你在吗"),
        _line("2024-02-01-12:10:20", 88, uid, "在，怎么了"),
        _line("2024-02-01-12:10:45", uid, 88, "帮我看下文档"),
        _line("2024-02-01-12:11:30", 88, uid, "好的"),
    ]
    dataset = _dataset_from_lines(uid, lines)
    result = implant_personality(
        SegmentAgent(rng=random.Random(19)),
        DialogueWorld(dataset, DialogueObserver(), seed=19),
        ImplantationConfig(sleep_every_n_sessions=1, maturity_window=2, max_ticks=200),
    )
    assert result.total_ticks >= 2
    assert result.total_sleep_cycles >= 1
    assert result.final_agent_state["prediction_ledger"]["archived_predictions"]
    assert result.maturity["window"] == 2


def test_m52_world_handles_malformed_dialogue_payload() -> None:
    malformed_dataset = {
        "uid": "11",
        "sessions": [
            {"session_id": "bad", "uid_a": "11", "uid_b": "xx", "turns": "bad"},
            {
                "session_id": "s_ok",
                "uid_a": "11",
                "uid_b": "22",
                "metadata": {"turn_count": 3},
                "turns": [
                    None,
                    {
                        "timestamp": "not-a-time",
                        "msg_type": 0,
                        "sender_uid": "22x",
                        "receiver_uid": "11",
                        "body": "你在吗？",
                    },
                    {
                        "timestamp": "2024-01-01T09:01:00",
                        "msg_type": 0,
                        "sender_uid": "22",
                        "receiver_uid": "11",
                        "body": "我们聊聊",
                    },
                ],
            },
        ],
    }
    result = implant_personality(
        SegmentAgent(rng=random.Random(23)),
        DialogueWorld(malformed_dataset, DialogueObserver(), seed=23),
        ImplantationConfig(sleep_every_n_sessions=1, maturity_window=2, max_ticks=50),
    )
    assert result.total_ticks >= 1
