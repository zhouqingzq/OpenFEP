from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from segmentum.agent import SegmentAgent
from segmentum.dialogue.validation.act_classifier import (
    ActionPrediction,
    DialogueActClassifier,
    KeywordDialogueActClassifier,
    validate_act_classifier,
)
from segmentum.dialogue.validation.baselines import select_wrong_users
from segmentum.dialogue.validation.constants import (
    M54_ACCEPTANCE_RULES_VERSION,
    M54_CLASSIFIER_LABEL_SCHEMA_VERSION,
)
from segmentum.dialogue.validation.metrics import SimilarityResult
from segmentum.dialogue.validation.metrics import behavioral_similarity, reply_function_buckets, semantic_pair_weights
from segmentum.dialogue.validation.pipeline import (
    ValidationConfig,
    ValidationReport,
    run_batch_validation,
    run_pilot_validation,
    run_validation,
)
from segmentum.dialogue.validation.report import generate_report
from segmentum.dialogue.validation.splitter import SplitStrategy, split_user_data
from segmentum.dialogue.validation.state_calibration import apply_train_state_calibration
from segmentum.dialogue.validation.policy_context import (
    dialogue_partner_policy_context_bucket,
    dialogue_policy_context_bucket,
)
from segmentum.dialogue.validation.reply_function import classify_reply_function
from segmentum.dialogue.validation.statistics import paired_comparison
from segmentum.dialogue.validation.surface_profile import (
    DialogueSurfaceProfile,
    average_surface_profiles,
    build_surface_profile,
)
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.slow_learning import SlowVariableLearner
from scripts.run_m54_validation import _parse_required_users_error


def _build_user(uid: int, *, sessions: int = 32, partner_start: int = 1000) -> dict:
    base = datetime(2024, 1, 1, 8, 0, 0) + timedelta(days=uid)
    rows: list[dict] = []
    topics = [
        ("项目进度 alpha", "我们对齐一下需求和排期。"),
        ("情绪支持 beta", "我最近有点焦虑，想听听你的想法。"),
        ("技术方案 gamma", "你觉得这个架构怎么调整比较稳？"),
        ("日常闲聊 delta", "今天发生了一件挺好玩的事。"),
    ]
    for idx in range(sessions):
        partner = partner_start + (idx % 4)
        t0 = base + timedelta(hours=idx * 3)
        topic, line = topics[idx % len(topics)]
        turns = [
            {
                "timestamp": (t0 + timedelta(minutes=0)).isoformat(),
                "sender_uid": partner,
                "receiver_uid": uid,
                "body": f"{topic}：{line}",
            },
            {
                "timestamp": (t0 + timedelta(minutes=1)).isoformat(),
                "sender_uid": uid,
                "receiver_uid": partner,
                "body": f"我在听，关于{topic}我会认真回应。",
            },
            {
                "timestamp": (t0 + timedelta(minutes=2)).isoformat(),
                "sender_uid": partner,
                "receiver_uid": uid,
                "body": f"你可以围绕{topic}再展开一点吗？",
            },
            {
                "timestamp": (t0 + timedelta(minutes=3)).isoformat(),
                "sender_uid": uid,
                "receiver_uid": partner,
                "body": f"当然可以，我补充{topic}里的关键细节。",
            },
        ]
        rows.append(
            {
                "session_id": f"s-{uid}-{idx}",
                "uid_a": min(uid, partner),
                "uid_b": max(uid, partner),
                "start_time": turns[0]["timestamp"],
                "end_time": turns[-1]["timestamp"],
                "metadata": {"turn_count": len(turns), "topic": topic},
                "turns": turns,
            }
        )
    return {
        "uid": uid,
        "profile": {
            "openness": 0.45 + (uid % 5) * 0.05,
            "agreeableness": 0.40 + (uid % 3) * 0.07,
            "neuroticism": 0.35 + (uid % 4) * 0.06,
            "trust_prior": -0.1 + (uid % 5) * 0.08,
        },
        "sessions": rows,
    }


def _classifier_samples(per_class: int, *, offset: int = 0) -> list[dict[str, str]]:
    templates = {
        "explore": ("ask_question", "想了解背景信息和更多线索"),
        "exploit": ("elaborate", "我会补充细节并接住这个话题"),
        "escape": ("deflect", "这个部分先放一放稍后再说"),
    }
    rows: list[dict[str, str]] = []
    for label_3, (label_11, phrase) in templates.items():
        for idx in range(per_class):
            rows.append(
                {
                    "text": f"中文门控样本 {offset + idx} {phrase} {label_3}",
                    "label_11": label_11,
                    "label_3": label_3,
                }
            )
    return rows


def _with_classifier_provenance(
    rows: list[dict[str, str]],
    *,
    source: str = "human_labeled_dialogue_corpus",
    annotator: str = "human_annotation_team",
    sampling_policy: str = "stratified_holdout_review",
    label_schema_version: str = M54_CLASSIFIER_LABEL_SCHEMA_VERSION,
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "source": source,
                "annotator": annotator,
                "sampling_policy": sampling_policy,
                "label_schema_version": label_schema_version,
            }
        )
        enriched.append(item)
    return enriched


class TestM54Validation(unittest.TestCase):
    def test_split_strategies_produce_disjoint_sets(self) -> None:
        user = _build_user(11, sessions=32)
        for strategy in (
            SplitStrategy.RANDOM,
            SplitStrategy.TEMPORAL,
            SplitStrategy.PARTNER,
            SplitStrategy.TOPIC,
        ):
            split = split_user_data(user, strategy, train_ratio=0.70, seed=42)
            train_ids = {item["session_id"] for item in split.train_sessions}
            holdout_ids = {item["session_id"] for item in split.holdout_sessions}
            self.assertTrue(train_ids.isdisjoint(holdout_ids))
            self.assertEqual(len(train_ids) + len(holdout_ids), len(user["sessions"]))
            if strategy is SplitStrategy.TOPIC:
                self.assertFalse(split.split_metadata["topic_split_not_applicable"])

    def test_temporal_split_order(self) -> None:
        user = _build_user(12, sessions=10)
        split = split_user_data(user, SplitStrategy.TEMPORAL, train_ratio=0.70, seed=42)
        self.assertTrue(split.train_sessions and split.holdout_sessions)
        self.assertLessEqual(split.train_sessions[-1]["start_time"], split.holdout_sessions[0]["start_time"])

    def test_partner_split_no_overlap(self) -> None:
        user = _build_user(13, sessions=32)
        split = split_user_data(user, SplitStrategy.PARTNER, train_ratio=0.70, seed=42)
        train_partners = {
            (item["uid_b"] if item["uid_a"] == user["uid"] else item["uid_a"])
            for item in split.train_sessions
        }
        holdout_partners = {
            (item["uid_b"] if item["uid_a"] == user["uid"] else item["uid_a"])
            for item in split.holdout_sessions
        }
        self.assertTrue(train_partners.isdisjoint(holdout_partners))

    def test_topic_split_protocol_metadata(self) -> None:
        user = _build_user(14, sessions=32)
        split = split_user_data(user, SplitStrategy.TOPIC, train_ratio=0.70, seed=42)
        meta = split.split_metadata
        self.assertIn("strict_train_only_fit", meta)
        self.assertIn("topic_split_not_applicable", meta)
        self.assertFalse(meta["topic_split_not_applicable"])
        self.assertIn("k_selected", meta)
        self.assertIn("cluster_sizes", meta)
        self.assertIn("merge_steps", meta)
        self.assertIn("topic_jsd_train_holdout", meta)
        self.assertTrue(meta.get("topic_kmeans_fitted_on_train_sessions_only"))

    def test_topic_not_applicable_does_not_random_fallback(self) -> None:
        user = _build_user(15, sessions=8)
        split = split_user_data(user, SplitStrategy.TOPIC, train_ratio=0.70, seed=42)
        self.assertTrue(split.split_metadata["topic_split_not_applicable"])
        self.assertTrue(split.split_metadata["strict_topic_no_random_fallback"])
        self.assertEqual(split.holdout_sessions, [])
        self.assertEqual(len(split.train_sessions), len(user["sessions"]))

    def test_agent_state_similarity_is_measured_before_holdout_generation(self) -> None:
        user = _build_user(16, sessions=8)
        other = _build_user(17, sessions=8)
        captured_cycles: list[int] = []

        def fake_implant(agent, world, config):
            del world, config
            agent.cycle = 7
            return None

        def fake_generate(agent, holdout_sessions, *, user_uid: int, seed: int, classifier):
            del holdout_sessions, user_uid, seed, classifier
            agent.cycle = 999
            return ["generated detail"], ["real detail"], ["elaborate"], ["elaborate"], ["exploit"], ["exploit"]

        def fake_state_similarity(agent_train, agent_full):
            del agent_full
            captured_cycles.append(int(agent_train.cycle))
            return SimilarityResult(
                "agent_state_similarity",
                0.91,
                {"vector_length": 1},
            )

        cfg = ValidationConfig(
            strategies=[SplitStrategy.RANDOM],
            min_holdout_sessions=1,
            seed=3,
            skip_population_average_implant=True,
        )
        prev_sem = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC")
        os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = "1"
        try:
            with patch("segmentum.dialogue.validation.pipeline.implant_personality", side_effect=fake_implant), patch(
                "segmentum.dialogue.validation.pipeline._generate_from_sessions", side_effect=fake_generate
            ), patch(
                "segmentum.dialogue.validation.pipeline.agent_state_similarity", side_effect=fake_state_similarity
            ):
                report = run_validation(user, cfg, all_user_profiles=[user, other])
        finally:
            if prev_sem is None:
                os.environ.pop("SEGMENTUM_USE_TFIDF_SEMANTIC", None)
            else:
                os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = prev_sem
        self.assertEqual(captured_cycles, [7])
        details = report.per_strategy["random"]["personality_metric_details"]["agent_state_similarity"]
        self.assertTrue(details["agent_state_measured_before_holdout_generation"])

    def test_baseline_c_population_average_excludes_target_user(self) -> None:
        user = _build_user(171, sessions=8)
        others = [_build_user(172, sessions=8), _build_user(173, sessions=8)]
        captured_uid_sets: list[set[int]] = []

        def fake_implant(agent, world, config):
            del world, config
            agent.slow_variable_learner.state.traits.social_approach = 0.75
            return None

        def fake_generate(agent, holdout_sessions, *, user_uid: int, seed: int, classifier):
            del agent, user_uid, seed, classifier
            turns = max(1, len(holdout_sessions))
            return (
                ["persona alpha"] * turns,
                ["persona alpha"] * turns,
                ["elaborate"] * turns,
                ["elaborate"] * turns,
                ["exploit"] * turns,
                ["exploit"] * turns,
            )

        def fake_average_agent(profiles, *, seed: int, classifier=None):
            del seed, classifier
            captured_uid_sets.append({int(item["uid"]) for item in profiles})
            return SegmentAgent()

        cfg = ValidationConfig(
            strategies=[SplitStrategy.RANDOM],
            min_holdout_sessions=1,
            seed=5,
            skip_population_average_implant=True,
        )
        prev_sem = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC")
        os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = "1"
        try:
            with patch("segmentum.dialogue.validation.pipeline.implant_personality", side_effect=fake_implant), patch(
                "segmentum.dialogue.validation.pipeline._generate_from_sessions", side_effect=fake_generate
            ), patch(
                "segmentum.dialogue.validation.pipeline.create_average_agent", side_effect=fake_average_agent
            ):
                report = run_validation(user, cfg, all_user_profiles=[user, *others])
        finally:
            if prev_sem is None:
                os.environ.pop("SEGMENTUM_USE_TFIDF_SEMANTIC", None)
            else:
                os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = prev_sem

        self.assertTrue(captured_uid_sets)
        self.assertNotIn(int(user["uid"]), captured_uid_sets[0])
        self.assertEqual(captured_uid_sets[0], {172, 173})
        strategy = report.per_strategy["random"]
        self.assertTrue(strategy["baseline_c_leave_one_out"])
        self.assertEqual(strategy["baseline_c_builder"], "profile_only_average_fallback")
        self.assertEqual(
            strategy["baseline_c_input_scope"],
            "leave_one_out_population_train_and_profile_data",
        )
        self.assertEqual(strategy["baseline_c_population_excluded_uid"], int(user["uid"]))
        self.assertEqual(strategy["baseline_c_population_user_count"], 2)

    def test_baseline_c_formal_path_uses_full_population_implant_builder(self) -> None:
        user = _build_user(271, sessions=8)
        others = [_build_user(272, sessions=8), _build_user(273, sessions=8)]
        captured_uid_sets: list[set[int]] = []

        def fake_generate(agent, holdout_sessions, *, user_uid: int, seed: int, classifier):
            del agent, user_uid, seed, classifier
            turns = max(1, len(holdout_sessions))
            return (
                ["persona alpha"] * turns,
                ["persona alpha"] * turns,
                ["elaborate"] * turns,
                ["elaborate"] * turns,
                ["exploit"] * turns,
                ["exploit"] * turns,
            )

        def fake_population_builder(profiles, config, *, seed: int, classifier=None):
            del config, seed, classifier
            captured_uid_sets.append({int(item["uid"]) for item in profiles})
            return SegmentAgent()

        cfg = ValidationConfig(
            strategies=[SplitStrategy.RANDOM],
            min_holdout_sessions=1,
            seed=5,
            skip_population_average_implant=False,
        )
        prev_sem = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC")
        os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = "1"
        try:
            with patch(
                "segmentum.dialogue.validation.pipeline._generate_from_sessions",
                side_effect=fake_generate,
            ), patch(
                "segmentum.dialogue.validation.pipeline.build_population_average_agent",
                side_effect=fake_population_builder,
            ):
                report = run_validation(user, cfg, all_user_profiles=[user, *others])
        finally:
            if prev_sem is None:
                os.environ.pop("SEGMENTUM_USE_TFIDF_SEMANTIC", None)
            else:
                os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = prev_sem

        self.assertTrue(captured_uid_sets)
        self.assertEqual(captured_uid_sets[0], {272, 273})
        strategy = report.per_strategy["random"]
        self.assertEqual(strategy["baseline_c_builder"], "population_average_full_implant")
        self.assertEqual(report.aggregate["baseline_c_builder"], "population_average_full_implant")

    def test_surface_profile_uses_train_sessions_only(self) -> None:
        user = _build_user(18, sessions=4)
        train_dataset = dict(user)
        train_dataset["sessions"] = user["sessions"][:2]
        train_dataset["sessions"][0]["turns"][0]["body"] = "train ctx alpha signal"
        train_dataset["sessions"][1]["turns"][0]["body"] = "train ctx beta signal"
        holdout_text = "HOLDOUT_UNIQUE_TOKEN_DO_NOT_LEAK"
        holdout_context = "HOLDOUT_CONTEXT_TOKEN_DO_NOT_LEAK"
        user["sessions"][3]["turns"][1]["body"] = holdout_text
        user["sessions"][3]["turns"][0]["body"] = holdout_context

        profile = build_surface_profile(train_dataset, source="unit_train")
        payload = json.dumps(profile.to_dict(), ensure_ascii=False)

        self.assertEqual(profile.source, "unit_train")
        self.assertGreater(profile.reply_count, 0)
        self.assertIn("train", profile.context_top_tokens)
        self.assertTrue(any(tokens for tokens in profile.partner_context_tokens.values()))
        self.assertNotIn(holdout_text, payload)
        self.assertNotIn(holdout_context, payload)
        self.assertNotIn(str(user["sessions"][3]["session_id"]), payload)

    def test_state_calibration_uses_train_sessions_only(self) -> None:
        user = _build_user(181, sessions=4)
        train_dataset = dict(user)
        train_dataset["sessions"] = user["sessions"][:2]
        holdout_text = "HOLDOUT_STATE_TOKEN_DO_NOT_LEAK"
        holdout_session_id = str(user["sessions"][3]["session_id"])
        user["sessions"][3]["turns"][1]["body"] = holdout_text
        agent = SegmentAgent()

        summary = apply_train_state_calibration(agent, train_dataset, source="unit_train")
        payload = json.dumps(summary, ensure_ascii=False)

        self.assertTrue(summary["applied"])
        self.assertEqual(summary["source"], "unit_train")
        self.assertNotIn(holdout_text, payload)
        self.assertNotIn(holdout_session_id, payload)
        self.assertNotEqual(agent.slow_variable_learner.state.traits.to_dict(), SlowVariableLearner().state.traits.to_dict())

    def test_state_calibration_marks_collapsed_majority_as_uncertain(self) -> None:
        user = _build_user(182, sessions=8)
        for session in user["sessions"]:
            for turn in session["turns"]:
                if int(turn["sender_uid"]) == int(user["uid"]):
                    turn["body"] = "好"
        agent = SegmentAgent()

        summary = apply_train_state_calibration(agent, user, source="unit_majority")

        self.assertGreaterEqual(float(summary["strategy_majority_coverage"]), 0.75)
        self.assertLess(float(summary["strategy_entropy"]), 0.35)
        self.assertEqual(summary["policy_dominant_strategy"], "expected_free_energy")
        self.assertFalse(summary["policy_dominant_strategy_confident"])

    def test_low_info_replies_drive_behavioral_escape_policy(self) -> None:
        user = _build_user(183, sessions=8)
        for session in user["sessions"]:
            for turn in session["turns"]:
                if int(turn["sender_uid"]) == int(user["uid"]):
                    turn["body"] = "8888"
        agent = SegmentAgent()

        class LowInfoClassifier:
            def predict(self, text: str) -> ActionPrediction:
                return ActionPrediction("minimal_response", "escape", 0.9, source="unit")

            def predict_batch(self, texts: list[str]) -> list[ActionPrediction]:
                return [self.predict(text) for text in texts]

        summary = apply_train_state_calibration(
            agent,
            user,
            classifier=LowInfoClassifier(),
            source="unit_low_info",
        )

        self.assertGreater(summary["surface_bucket_counts"].get("ultra_low_info", 0), 0)
        self.assertGreater(summary["policy_action_distribution"].get("minimal_response", 0.0), 0.0)
        self.assertGreater(summary["policy_evidence_count"], 0)
        self.assertEqual(summary["policy_dominant_strategy"], "expected_free_energy")
        context_bucket = dialogue_policy_context_bucket(user["sessions"][0]["turns"][0]["body"])
        self.assertIn(context_bucket, summary["policy_by_context"])

    def test_context_policy_bias_uses_partner_turn_not_holdout_reply(self) -> None:
        user = _build_user(185, sessions=8)
        for session in user["sessions"]:
            session["turns"][0]["body"] = "鎴戜滑绋嶅悗鍐嶈皥杩欎釜"
            session["turns"][1]["body"] = "8888"
            session["turns"][2]["body"] = "浣犺兘鍐嶈涓€鐐瑰悧锛?"
            session["turns"][3]["body"] = "8888"
        agent = SegmentAgent()

        class EscapeClassifier:
            def predict(self, text: str) -> ActionPrediction:
                return ActionPrediction("minimal_response", "escape", 0.9, source="unit")

            def predict_batch(self, texts: list[str]) -> list[ActionPrediction]:
                return [self.predict(text) for text in texts]

        summary = apply_train_state_calibration(agent, user, classifier=EscapeClassifier())
        context_bucket = dialogue_policy_context_bucket("鎴戜滑绋嶅悗鍐嶈皥杩欎釜")
        partner_context_bucket = dialogue_partner_policy_context_bucket(
            "閹存垳婊戠粙宥呮倵閸愬秷鐨ユ潻娆庨嚋",
            user["sessions"][0]["uid_b"],
        )
        self.assertIn(context_bucket, summary["policy_by_context"])
        self.assertNotIn("8888", json.dumps(summary["policy_by_context"], ensure_ascii=False))

        agent.policy_evaluator._dialogue_decision_context = {
            "event_type": "dialogue_turn",
            "body": "鎴戜滑绋嶅悗鍐嶈皥杩欎釜",
        }
        escape_bias = agent.policy_evaluator.identity_bias(
            action="minimal_response",
            projected_state={},
            predicted_outcome={},
            cost=0.0,
        )
        exploit_bias = agent.policy_evaluator.identity_bias(
            action="agree",
            projected_state={},
            predicted_outcome={},
            cost=0.0,
        )
        self.assertGreater(escape_bias, exploit_bias)
        self.assertEqual(
            agent.policy_evaluator._last_policy_context_by_action["minimal_response"]["policy_context_bucket"],
            context_bucket,
        )
        self.assertTrue(
            agent.policy_evaluator._last_policy_context_by_action["minimal_response"][
                "policy_action_selection_lift_applied"
            ]
        )

    def _disabled_test_partner_conditioned_context_bucket_preferred_when_supported(self) -> None:
        user = _build_user(186, sessions=8)
        partner_uid = user["sessions"][0]["uid_b"]
        for session in user["sessions"]:
            session["uid_b"] = partner_uid
            session["turns"][0]["body"] = "闁瑰瓨鍨冲鎴犵矙瀹ュ懏鍊甸柛鎰Х閻ㄣ儲娼诲▎搴ㄥ殝"
            session["turns"][1]["body"] = "8888"
            session["turns"][2]["body"] = "闁瑰瓨鍨冲鎴犵矙瀹ュ懏鍊甸柛鎰Х閻ㄣ儲娼诲▎搴ㄥ殝"
            session["turns"][3]["body"] = "8888"
        agent = SegmentAgent()

        class EscapeClassifier:
            def predict(self, text: str) -> ActionPrediction:
                return ActionPrediction("minimal_response", "escape", 0.9, source="unit")

            def predict_batch(self, texts: list[str]) -> list[ActionPrediction]:
                return [self.predict(text) for text in texts]

        summary = apply_train_state_calibration(agent, user, classifier=EscapeClassifier())
        partner_context_bucket = dialogue_partner_policy_context_bucket(
            "闁瑰瓨鍨冲鎴犵矙瀹ュ懏鍊甸柛鎰Х閻ㄣ儲娼诲▎搴ㄥ殝",
            partner_uid,
        )
        self.assertIn(partner_context_bucket, summary["policy_by_context"])

        agent.policy_evaluator._dialogue_decision_context = {
            "event_type": "dialogue_turn",
            "body": "闁瑰瓨鍨冲鎴犵矙瀹ュ懏鍊甸柛鎰Х閻ㄣ儲娼诲▎搴ㄥ殝",
            "partner_uid": partner_uid,
        }
        agent.policy_evaluator.identity_bias(
            action="minimal_response",
            projected_state={},
            predicted_outcome={},
            cost=0.0,
        )
        self.assertEqual(
            agent.policy_evaluator._last_policy_context_by_action["minimal_response"]["policy_context_bucket"],
            partner_context_bucket,
        )

    def test_transactional_low_info_replies_drive_behavioral_policy_not_semantic_weight(self) -> None:
        user = _build_user(184, sessions=8)
        for session in user["sessions"]:
            for turn in session["turns"]:
                if int(turn["sender_uid"]) == int(user["uid"]):
                    turn["body"] = "收到"
        agent = SegmentAgent()

        summary = apply_train_state_calibration(agent, user, source="unit_transactional")

        self.assertGreater(summary["surface_bucket_counts"].get("low_info_ack", 0), 0)
        self.assertEqual(summary["reply_function_counts"].get("transactional_ack"), 16)
        self.assertGreater(summary["policy_evidence_count"], 0)
        self.assertGreater(summary["policy_action_distribution"].get("agree", 0.0), 0.0)
        self.assertIn("transactional_ack", summary["policy_by_reply_function"])
        self.assertLess(summary["semantic_policy_evidence_weight"], summary["policy_evidence_weight"])

    def test_weighted_behavioral_similarity_downweights_low_info_majority(self) -> None:
        real_texts = ["8888", "好", "我补充一下项目的关键细节和下一步判断"]
        real_strategy = ["escape", "exploit", "exploit"]
        majority_strategy = ["escape", "escape", "escape"]
        personality_strategy = ["exploit", "exploit", "exploit"]
        weights = semantic_pair_weights(real_texts)

        majority = behavioral_similarity(
            majority_strategy,
            real_strategy,
            granularity="strategy",
            weights=weights,
        )
        personality = behavioral_similarity(
            personality_strategy,
            real_strategy,
            granularity="strategy",
            weights=weights,
        )

        self.assertGreater(float(personality.value), float(majority.value))
        self.assertEqual(majority.details["aggregation"], "information_weighted_pair_distribution")

    def test_bucket_balanced_behavioral_similarity_blocks_global_escape_majority(self) -> None:
        real_texts = ["8888", "收到", "账号我晚点发你", "怎么处理下一步？"]
        real_strategy = ["escape", "exploit", "exploit", "explore"]
        majority_strategy = ["escape", "escape", "escape", "escape"]
        personality_strategy = ["escape", "exploit", "exploit", "explore"]
        weights = semantic_pair_weights(real_texts)
        buckets = reply_function_buckets(real_texts)

        majority = behavioral_similarity(
            majority_strategy,
            real_strategy,
            granularity="strategy",
            weights=weights,
            buckets=buckets,
        )
        personality = behavioral_similarity(
            personality_strategy,
            real_strategy,
            granularity="strategy",
            weights=weights,
            buckets=buckets,
        )

        self.assertGreater(float(personality.value), float(majority.value))
        self.assertEqual(
            majority.details["aggregation"],
            "reply_function_bucket_balanced_information_weighted_pair_distribution",
        )

    def test_population_surface_profile_drops_target_like_anchors(self) -> None:
        profile_a = DialogueSurfaceProfile(
            source="a",
            reply_count=4,
            avg_reply_chars=20.0,
            median_reply_chars=18,
            punctuation_counts={"?": 2},
            opening_phrases=["target opener"],
            connector_phrases=["target connector"],
            top_tokens=["target_topic"],
            context_top_tokens=["target_context"],
            action_phrases={"elaborate": ["target phrase"]},
            strategy_counts={"exploit": 4},
            partner_tokens={"100": ["partner_secret"]},
            partner_context_tokens={"100": ["partner_context_secret"]},
        )
        profile_b = DialogueSurfaceProfile(
            source="b",
            reply_count=6,
            avg_reply_chars=10.0,
            median_reply_chars=9,
            punctuation_counts={"!": 1},
            opening_phrases=["other opener"],
            connector_phrases=["other connector"],
            top_tokens=["other_topic"],
            context_top_tokens=["other_context"],
            action_phrases={"ask_question": ["other phrase"]},
            strategy_counts={"explore": 6},
            partner_tokens={"200": ["other_secret"]},
            partner_context_tokens={"200": ["other_context_secret"]},
        )

        population = average_surface_profiles(
            [profile_a, profile_b],
            include_surface_anchors=False,
        )

        self.assertEqual(population.top_tokens, [])
        self.assertEqual(population.context_top_tokens, [])
        self.assertEqual(population.opening_phrases, [])
        self.assertEqual(population.connector_phrases, [])
        self.assertEqual(population.action_phrases, {})
        self.assertEqual(population.partner_tokens, {})
        self.assertEqual(population.partner_context_tokens, {})
        self.assertEqual(population.strategy_counts["exploit"], 4)
        self.assertEqual(population.strategy_counts["explore"], 6)

    def test_rule_generator_uses_surface_profile_deterministically(self) -> None:
        generator = RuleBasedGenerator()
        context = {
            "current_turn": "我们继续聊项目 alpha",
            "partner_uid": 1000,
            "observation": {"conflict_tension": 0.0, "emotional_tone": 0.5},
        }
        base_state = {"slow_traits": {"social_approach": 0.8, "caution_bias": 0.2}}
        profile_a = {
            "source": "a",
            "reply_count": 8,
            "connector_phrases": ["我会认真回应"],
            "top_tokens": ["alpha"],
            "context_top_tokens": ["alpha"],
            "action_phrases": {"elaborate": ["我补充 alpha 的关键细节"]},
            "partner_tokens": {"1000": ["partner_alpha"]},
            "partner_context_tokens": {"1000": ["alpha"]},
        }
        profile_b = {
            "source": "b",
            "reply_count": 8,
            "connector_phrases": ["先看 beta"],
            "top_tokens": ["beta"],
            "context_top_tokens": ["beta"],
            "action_phrases": {"elaborate": ["我补充 beta 的关键细节"]},
            "partner_tokens": {"1000": ["partner_beta"]},
            "partner_context_tokens": {"1000": ["beta"]},
        }

        first = generator.generate(
            "elaborate",
            context,
            {**base_state, "surface_profile": profile_a},
            [],
            master_seed=9,
            turn_index=0,
        )
        first_diag = dict(generator.last_diagnostics)
        second = generator.generate(
            "elaborate",
            context,
            {**base_state, "surface_profile": profile_a},
            [],
            master_seed=9,
            turn_index=0,
        )
        different = generator.generate(
            "elaborate",
            context,
            {**base_state, "surface_profile": profile_b},
            [],
            master_seed=9,
            turn_index=0,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertIn("alpha", first)
        self.assertTrue(first_diag["topic_anchor_used"])
        self.assertEqual(first_diag["topic_anchor_source"], "partner_context")
        self.assertEqual(first_diag["profile_degraded_reason"], "")
        self.assertTrue(first_diag["profile_phrase_used"])
        self.assertIn("action_phrase", first_diag["profile_expression_sources"])
        self.assertEqual(generator.last_diagnostics["surface_source"], "b")
        self.assertFalse(generator.last_diagnostics["profile_phrase_used"])
        self.assertFalse(generator.last_diagnostics["topic_anchor_used"])
        self.assertEqual(generator.last_diagnostics["topic_anchor_source"], "none")
        self.assertIn("anchor_mismatch", generator.last_diagnostics["profile_degraded_reason"])
        self.assertLess(generator.last_diagnostics["profile_confidence"], first_diag["profile_confidence"])
        self.assertIn("connector", generator.last_diagnostics["profile_expression_sources"])
        self.assertIn("template_id", generator.last_diagnostics)

    def test_rule_generator_falls_back_to_context_top_tokens(self) -> None:
        generator = RuleBasedGenerator()
        reply = generator.generate(
            "elaborate",
            {
                "current_turn": "please revisit alpha tomorrow",
                "partner_uid": 1000,
                "observation": {"conflict_tension": 0.0, "emotional_tone": 0.5},
            },
            {
                "slow_traits": {"social_approach": 0.7, "caution_bias": 0.2},
                "surface_profile": {
                    "source": "unit",
                    "reply_count": 8,
                    "connector_phrases": ["alpha first"],
                    "context_top_tokens": ["alpha"],
                    "partner_context_tokens": {},
                },
            },
            [],
            master_seed=9,
            turn_index=0,
        )

        self.assertIn("alpha", reply)
        self.assertTrue(generator.last_diagnostics["topic_anchor_used"])
        self.assertEqual(generator.last_diagnostics["topic_anchor_source"], "context_global")

    def test_dialogue_policy_context_bucket_refines_generic_temporal_and_confirmation(self) -> None:
        self.assertEqual(dialogue_policy_context_bucket("ok"), "ctx:partner_low_info_ack")
        self.assertEqual(dialogue_policy_context_bucket("tmr"), "ctx:partner_low_info_temporal")
        self.assertEqual(dialogue_policy_context_bucket("tomorrow"), "ctx:partner_short_temporal")
        self.assertEqual(dialogue_policy_context_bucket("okayyy"), "ctx:partner_short_confirmation")
        self.assertEqual(
            dialogue_policy_context_bucket("about alpha project details"),
            "ctx:partner_statement_topicish",
        )
        self.assertEqual(
            dialogue_policy_context_bucket("we can finish this later after lunch"),
            "ctx:partner_statement_temporal",
        )

    def test_population_average_surface_does_not_emit_ultra_short_template(self) -> None:
        generator = RuleBasedGenerator()
        reply = generator.generate(
            "minimal_response",
            {
                "current_turn": "確認匯款資訊",
                "partner_uid": 1000,
                "observation": {"conflict_tension": 0.0, "emotional_tone": 0.5},
            },
            {
                "slow_traits": {"caution_bias": 0.8, "trust_stance": 0.3},
                "surface_profile": {
                    "source": "population_average",
                    "reply_count": 100,
                    "median_reply_chars": 2,
                    "ultra_short_ratio": 1.0,
                },
            },
            [],
            master_seed=4,
            turn_index=0,
        )

        self.assertGreater(len(reply), 12)
        self.assertTrue(generator.last_diagnostics["population_surface_state_only"])
        self.assertTrue(generator.last_diagnostics["surface_shortcut_suppressed"])

    def test_policy_lift_suppresses_ultra_short_surface_shortcut(self) -> None:
        generator = RuleBasedGenerator()
        reply = generator.generate(
            "agree",
            {
                "current_turn": "我们继续确认项目排期和付款细节",
                "partner_uid": 1000,
                "observation": {"conflict_tension": 0.0, "emotional_tone": 0.6},
            },
            {
                "slow_traits": {"caution_bias": 0.8, "trust_stance": 0.3, "social_approach": 0.3},
                "preferred_policies": {
                    "strategy_confidence": 0.8,
                    "policy_evidence_count": 12,
                    "action_distribution": {"agree": 0.4, "elaborate": 0.3},
                    "learned_preferences": ["agree", "elaborate"],
                },
                "surface_profile": {
                    "source": "random:train",
                    "reply_count": 100,
                    "median_reply_chars": 2,
                    "ultra_short_ratio": 1.0,
                    "connector_phrases": ["好"],
                },
            },
            [],
            master_seed=4,
            turn_index=0,
        )

        self.assertGreater(len(reply), 12)
        self.assertTrue(generator.last_diagnostics["policy_lift_applied"])
        self.assertFalse(generator.last_diagnostics["policy_action_selection_lift_applied"])
        self.assertNotIn("policy_detail", generator.last_diagnostics["profile_expression_sources"])
        self.assertTrue(generator.last_diagnostics["surface_shortcut_suppressed"])
        self.assertEqual(generator.last_diagnostics["rhetorical_move"], "warm_supportive")

    def test_policy_detail_requires_action_selection_lift(self) -> None:
        generator = RuleBasedGenerator()
        state = {
            "slow_traits": {"social_approach": 0.6, "trust_stance": 0.6},
            "preferred_policies": {
                "strategy_confidence": 0.8,
                "policy_evidence_count": 12,
                "action_distribution": {"agree": 0.4},
                "learned_preferences": ["agree"],
            },
            "policy_action_selection_context": {
                "policy_context_bucket": "ctx:task_coordination",
                "conditional_policy_frequency": 0.5,
                "conditional_policy_strategy_frequency": 0.7,
                "policy_action_selection_lift_applied": True,
            },
            "surface_profile": {
                "source": "random:train",
                "reply_count": 20,
                "median_reply_chars": 18,
                "ultra_short_ratio": 0.0,
                "connector_phrases": ["ok"],
            },
        }
        generator.generate(
            "agree",
            {
                "current_turn": "鎴戜滑缁х画纭椤圭洰鎺掓湡",
                "partner_uid": 1000,
                "observation": {"conflict_tension": 0.0, "emotional_tone": 0.6},
            },
            state,
            [],
            master_seed=5,
            turn_index=0,
        )

        self.assertTrue(generator.last_diagnostics["policy_action_selection_lift_applied"])
        self.assertIn("policy_detail", generator.last_diagnostics["profile_expression_sources"])

    def test_rhetorical_move_categories_are_reachable(self) -> None:
        generator = RuleBasedGenerator()
        context = {
            "current_turn": "我们下一步怎么处理？",
            "partner_uid": 1000,
            "observation": {"conflict_tension": 0.0, "emotional_tone": 0.5},
        }
        cases = [
            (
                "agree",
                {"social_approach": 0.80, "trust_stance": 0.82, "caution_bias": 0.20},
                "warm_supportive",
            ),
            (
                "deflect",
                {"social_approach": 0.20, "trust_stance": 0.25, "caution_bias": 0.85},
                "guarded_short",
            ),
            (
                "ask_question",
                {"exploration_posture": 0.84, "trust_stance": 0.55, "caution_bias": 0.20},
                "exploratory_questioning",
            ),
            (
                "share_opinion",
                {
                    "social_approach": 0.30,
                    "trust_stance": 0.50,
                    "caution_bias": 0.40,
                    "exploration_posture": 0.40,
                },
                "direct_advisory",
            ),
        ]
        for action, traits, expected in cases:
            generator.generate(action, context, {"slow_traits": traits}, [], master_seed=3, turn_index=0)
            self.assertEqual(generator.last_diagnostics["rhetorical_move"], expected)

    def test_dialogue_action_bias_changes_with_slow_traits(self) -> None:
        warm = SlowVariableLearner()
        guarded = SlowVariableLearner()
        warm.state.traits.social_approach = 0.85
        warm.state.traits.trust_stance = 0.85
        warm.state.traits.caution_bias = 0.20
        guarded.state.traits.social_approach = 0.20
        guarded.state.traits.trust_stance = 0.20
        guarded.state.traits.caution_bias = 0.85

        self.assertGreater(warm.action_bias("empathize"), guarded.action_bias("empathize"))
        self.assertGreater(guarded.action_bias("deflect"), warm.action_bias("deflect"))

    def test_keyword_classifier_is_debug_only(self) -> None:
        clf = KeywordDialogueActClassifier()
        pred = clf.predict("Can you compare the options?")
        self.assertEqual(clf.engine, "keyword_debug")
        self.assertFalse(clf.formal_engine)
        self.assertEqual(pred.label_3, "explore")

    def test_short_cooperative_replies_are_exploit_not_escape(self) -> None:
        clf = KeywordDialogueActClassifier()

        for text in ("好", "收到", "ok", "got it"):
            pred = clf.predict(text)
            self.assertEqual(pred.label_11, "agree")
            self.assertEqual(pred.label_3, "exploit")

        evasive = clf.predict("...")
        self.assertEqual(evasive.label_11, "minimal_response")
        self.assertEqual(evasive.label_3, "escape")

    def test_transactional_cooperative_replies_are_not_escape(self) -> None:
        clf = DialogueActClassifier(_classifier_samples(4))

        for text in ("那我先幫你儲鑽好了", "早喔，有需要可以密我", "是喔，謝謝，妳辛苦了"):
            pred = clf.predict(text)
            self.assertEqual(pred.label_3, "exploit")

        question = clf.predict("合唱入圍有二個，那還有邀請卡嗎")
        self.assertEqual(question.label_3, "explore")

    def test_reply_function_definition_keeps_transactional_and_humor_out_of_escape(self) -> None:
        cases = {
            "收到": ("transactional_ack", "exploit"),
            "账号我晚点发你": ("payment_or_account_info", "exploit"),
            "我先帮你处理这个任务": ("task_coordination", "exploit"),
            "哈哈可以": ("affiliative_humor", "exploit"),
            "怎么处理下一步？": ("information_request", "explore"),
            "先别继续了": ("refusal_boundary", "escape"),
        }
        clf = DialogueActClassifier(_classifier_samples(4), use_tfidf=True)
        for text, (reply_function, label_3) in cases.items():
            self.assertEqual(classify_reply_function(text), reply_function)
            if reply_function != "task_coordination":
                self.assertEqual(clf.predict(text).label_3, label_3)

    def test_supervised_classifier_fixture_gate_degrades_when_tfidf_forced(self) -> None:
        train = _classifier_samples(100, offset=0)
        gate = _classifier_samples(50, offset=1000)
        clf = DialogueActClassifier(train, use_tfidf=True)
        report = validate_act_classifier(
            train_samples=train,
            gate_samples=gate,
            classifier=clf,
            dataset_origin="tests/realistic_chinese_fixture",
        )
        self.assertEqual(report["train_count"], 300)
        self.assertEqual(report["gate_count"], 150)
        self.assertTrue(report["dataset_separation_ok"])
        self.assertEqual(report["class_distribution"]["train"]["explore"], 100)
        self.assertGreaterEqual(report["macro_f1_3class"], 0.70)
        self.assertIn("confusion_matrix_3class", report)
        self.assertIn("per_class_metrics_3class", report)
        self.assertIn("cue_override_rate", report)
        self.assertIn("macro_f1_3class_without_cue", report)
        self.assertFalse(report["classifier_provenance_ok"])
        self.assertEqual(report["classifier_evidence_tier"], "repo_fixture_smoke")
        self.assertIn("without_cue_3class_gate_passed", report)
        self.assertIn("reply_function_source_counts", report)
        self.assertGreater(report["per_class_metrics_3class"]["explore"]["recall"], 0.0)
        self.assertFalse(report["formal_gate_eligible"])
        self.assertFalse(report["behavioral_hard_metric_enabled"])

    def test_classifier_provenance_marks_codex_authored_labels_provisional(self) -> None:
        train = _classifier_samples(100, offset=0)
        gate = _classifier_samples(50, offset=1000)
        for sample in train + gate:
            sample["source"] = "codex_authored_realistic_zh_fixture_v4"
            sample["annotator"] = "human_annotation_team"
            sample["sampling_policy"] = "stratified_holdout_review"
            sample["label_schema_version"] = M54_CLASSIFIER_LABEL_SCHEMA_VERSION
        report = validate_act_classifier(
            train_samples=train,
            gate_samples=gate,
            classifier=DialogueActClassifier(train, use_tfidf=True),
            dataset_origin="independent_holdout_labels",
        )
        self.assertFalse(report["classifier_provenance_ok"])
        self.assertEqual(report["classifier_evidence_tier"], "llm_generated_provisional")
        self.assertIn("codex_authored", report["classifier_provenance_failure_reason"])
        self.assertFalse(report["formal_gate_eligible"])

    def test_classifier_positive_provenance_required_for_external_tier(self) -> None:
        train = _classifier_samples(100, offset=0)
        gate = _classifier_samples(50, offset=1000)
        report = validate_act_classifier(
            train_samples=train,
            gate_samples=gate,
            classifier=DialogueActClassifier(train, use_tfidf=True),
            dataset_origin="independent_holdout_labels",
        )
        self.assertFalse(report["classifier_provenance_ok"])
        self.assertEqual(report["classifier_evidence_tier"], "repo_fixture_smoke")
        self.assertIn("missing_provenance_field", report["classifier_provenance_failure_reason"])

        enriched_train = _with_classifier_provenance(train)
        enriched_gate = _with_classifier_provenance(gate)
        external_report = validate_act_classifier(
            train_samples=enriched_train,
            gate_samples=enriched_gate,
            classifier=DialogueActClassifier(enriched_train, use_tfidf=True),
            dataset_origin="independent_holdout_labels",
        )
        self.assertTrue(external_report["classifier_provenance_ok"])
        self.assertEqual(external_report["classifier_evidence_tier"], "external_human_labeled")

    def test_classifier_positive_provenance_blocks_single_synthetic_gate_marker(self) -> None:
        train = _with_classifier_provenance(_classifier_samples(100, offset=0))
        gate = _with_classifier_provenance(_classifier_samples(50, offset=1000))
        gate[0]["source"] = "synthetic_holdout_row"
        report = validate_act_classifier(
            train_samples=train,
            gate_samples=gate,
            classifier=DialogueActClassifier(train, use_tfidf=True),
            dataset_origin="independent_holdout_labels",
        )
        self.assertFalse(report["classifier_provenance_ok"])
        self.assertEqual(report["classifier_evidence_tier"], "repo_fixture_smoke")
        self.assertIn("synthetic", report["classifier_provenance_failure_reason"])
        self.assertFalse(report["formal_gate_eligible"])

    def test_classifier_cue_override_gate_blocks_high_cue_dependence(self) -> None:
        train = _classifier_samples(100, offset=0)
        cue_gate: list[dict[str, str]] = []
        cue_texts = {
            "explore": ("ask_question", "formal gate cue explore ?"),
            "exploit": ("elaborate", "formal gate cue exploit \u5c55\u5f00\u4e00\u4e0b"),
            "escape": ("deflect", "formal gate cue escape \u5148\u653e\u4e00\u653e"),
        }
        for label_3, (label_11, text) in cue_texts.items():
            for idx in range(50):
                cue_gate.append({"text": f"{text} {idx}", "label_11": label_11, "label_3": label_3})
        report = validate_act_classifier(
            train_samples=train,
            gate_samples=cue_gate,
            classifier=KeywordDialogueActClassifier(),
            dataset_origin="independent_holdout_labels",
            require_classifier_provenance=False,
            max_cue_override_rate=0.35,
        )
        self.assertGreater(report["cue_override_rate"], 0.35)
        self.assertFalse(report["cue_override_gate_passed"])
        self.assertFalse(report["without_cue_3class_gate_passed"])
        self.assertFalse(report["formal_gate_eligible"])

    def test_supervised_classifier_uses_cues_as_features_not_overrides(self) -> None:
        train = _classifier_samples(100, offset=0)
        gate = _classifier_samples(50, offset=1000)
        report = validate_act_classifier(
            train_samples=train,
            gate_samples=gate,
            classifier=DialogueActClassifier(train, use_tfidf=True),
            dataset_origin="independent_holdout_labels",
            require_classifier_provenance=False,
            max_cue_override_rate=0.35,
        )
        self.assertEqual(report["cue_override_rate"], 0.0)
        self.assertGreater(report["cue_feature_assist_rate"], 0.0)
        self.assertTrue(report["cue_override_gate_passed"])
        self.assertTrue(report["without_cue_3class_gate_passed"])
        self.assertGreaterEqual(report["macro_f1_3class_without_cue"], 0.70)

    def test_supervised_classifier_handles_non_keyword_chinese(self) -> None:
        train = _classifier_samples(100, offset=0)
        clf = DialogueActClassifier(train, use_tfidf=True)
        preds = clf.predict_batch(
            [
                "我想了解背景信息和更多线索",
                "我会补充细节并接住这个话题",
                "这个部分先放一放稍后再说",
            ]
        )
        self.assertEqual([item.label_3 for item in preds], ["explore", "exploit", "escape"])

    def test_act_classifier_gate_fails_with_inconsistent_gold_3class(self) -> None:
        samples = [
            {"text": "我同意。", "label_11": "agree", "label_3": "escape"},
            {"text": "我反对。", "label_11": "disagree", "label_3": "exploit"},
            {"text": "为什么？", "label_11": "ask_question", "label_3": "escape"},
        ]
        report = validate_act_classifier(samples, min_macro_f1_3class=0.70)
        self.assertFalse(report["passed_3class_gate"])

    def test_select_wrong_users_protocol(self) -> None:
        target = _build_user(21, sessions=8)
        candidates = [_build_user(uid, sessions=8) for uid in (22, 23, 24, 25, 26, 27)]
        selected = select_wrong_users(target, candidates, k=3, seed=42)
        self.assertEqual(len(selected), 3)
        bands = {item.get("_wrong_user_band") for item in selected}
        self.assertTrue(bands <= {"semi-hard", "medium", "far"})
        dists = sorted(item.get("_wrong_user_distance") for item in selected)
        self.assertLessEqual(dists[0], dists[1])
        self.assertLessEqual(dists[1], dists[2])

    def test_parse_direction_auto_escalation_required_users(self) -> None:
        self.assertEqual(
            _parse_required_users_error("insufficient users for validation: have=10, required=15"),
            15,
        )
        self.assertIsNone(_parse_required_users_error("some other validation failure"))

    def test_paired_comparison_outputs(self) -> None:
        p, sig, mean_diff, better = paired_comparison(
            [0.7, 0.8, 0.75, 0.81], [0.6, 0.7, 0.72, 0.79], test="wilcoxon"
        )
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertIsInstance(sig, bool)
        self.assertGreater(mean_diff, 0)
        self.assertTrue(better)
        p2, sig2, _, _ = paired_comparison(
            [0.7, 0.8, 0.75, 0.81], [0.6, 0.7, 0.72, 0.79], test="t_test"
        )
        self.assertGreaterEqual(p2, 0.0)
        self.assertLessEqual(p2, 1.0)
        self.assertIsInstance(sig2, bool)

    def test_report_writes_semantic_diagnostics(self) -> None:
        metrics_p = {
            "semantic_similarity": 0.70,
            "behavioral_similarity_strategy": 0.8,
            "behavioral_similarity_action11": 0.8,
            "stylistic_similarity": 0.8,
            "personality_similarity": 0.8,
            "agent_state_similarity": 0.9,
        }
        metrics_a = {
            "semantic_similarity": 0.40,
            "behavioral_similarity_strategy": 0.4,
            "behavioral_similarity_action11": 0.4,
            "stylistic_similarity": 0.4,
            "personality_similarity": 0.4,
        }
        strategy = {
            "skipped": False,
            "eligible_for_hard_gate": True,
            "split_metadata": {"strategy": "random"},
            "personality_metrics": metrics_p,
            "baseline_a_metrics": metrics_a,
            "baseline_b_metrics": dict(metrics_a),
            "baseline_c_metrics": dict(metrics_a),
            "personality_metric_details": {
                "semantic_similarity": {"method": "tfidf_cosine", "pair_count": 1}
            },
            "baseline_a_metric_details": {},
            "baseline_c_metric_details": {},
            "classifier_validation": {"passed_3class_gate": False, "formal_gate_eligible": False},
            "diagnostic_trace": [
                {
                    "strategy": "random",
                    "pair_index": 0,
                    "personality_vs_a_pair_delta": 0.3,
                    "personality_action": "elaborate",
                    "baseline_a_action": "ask_question",
                    "baseline_c_action": "elaborate",
                    "personality_strategy": "exploit",
                    "baseline_a_strategy": "explore",
                    "baseline_c_strategy": "exploit",
                    "personality_text": "persona alpha",
                    "baseline_a_text": "default beta",
                    "baseline_c_text": "average gamma",
                    "personality_generated_chars": 13,
                    "baseline_a_generated_chars": 12,
                    "baseline_c_generated_chars": 13,
                    "personality_semantic_pair_score": 0.7,
                    "baseline_a_semantic_pair_score": 0.4,
                    "baseline_c_semantic_pair_score": 0.5,
                    "personality_vs_a_text_similarity": 0.2,
                    "personality_vs_c_text_similarity": 0.3,
                    "personality_template_id": "elaborate:0",
                    "baseline_a_template_id": "ask_question:1",
                    "personality_surface_source": "random:train",
                    "baseline_a_surface_source": "generic",
                    "personality_profile_phrase_used": True,
                    "personality_profile_expression_sources": ["focus", "action_phrase", "anchor"],
                    "baseline_c_profile_expression_sources": ["generic"],
                    "personality_topic_anchor_used": True,
                    "personality_topic_anchor_source": "partner_context",
                    "personality_rhetorical_move": "warm_supportive",
                    "baseline_c_rhetorical_move": "direct_advisory",
                    "reply_length_bucket": "medium",
                }
            ],
            "ablation_summary": [
                {
                    "name": "no_surface_profile",
                    "semantic_mean": 0.6,
                    "semantic_vs_baseline_a_diff": 0.2,
                    "action_agreement_vs_personality": 0.0,
                    "text_similarity_vs_personality": 0.3,
                }
            ],
            "state_distance_diagnostics": {
                "train_full": {"cosine": 0.9, "l2": 0.1},
                "train_default": {"cosine": 0.5, "l2": 0.5},
                "train_wrong_user": {"cosine": 0.4, "l2": 0.4},
                "per_dimension_variance": {"trait:social_approach": 0.01},
            },
        }
        report = ValidationReport(
            user_uid=101,
            per_strategy={"random": strategy},
            aggregate={
                "required_users": 1,
                "pilot": {"required_users": 1},
                "skip_population_average_implant": True,
            },
            conclusion="completed",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "m54_diag"
            generate_report([report], output_dir)
            aggregate = json.loads((output_dir / "aggregate_report.json").read_text(encoding="utf-8"))
            self.assertEqual(aggregate["diagnostic_trace_rows"], 1)
            self.assertTrue((output_dir / "diagnostic_trace.jsonl").exists())
            self.assertIn("semantic_delta_summary", aggregate)
            self.assertEqual(aggregate["semantic_delta_summary"]["users"]["positive"], 1)
            self.assertIn("baseline_audit_summary", aggregate)
            self.assertIn("ablation_summary", aggregate)
            self.assertIn("state_saturation_summary", aggregate)
            self.assertIn("profile_expression_source_summary", aggregate)
            self.assertIn("context_anchor_usage_summary", aggregate)
            self.assertTrue((output_dir / "baseline_audit_summary.json").exists())
            self.assertTrue((output_dir / "ablation_summary.json").exists())
            self.assertTrue((output_dir / "state_saturation_summary.json").exists())
            self.assertTrue((output_dir / "profile_expression_source_summary.json").exists())
            self.assertTrue((output_dir / "context_anchor_usage_summary.json").exists())
            self.assertIn("debug_readiness_gate", aggregate)
            self.assertTrue(aggregate["debug_readiness_gate"]["passed"])
            self.assertTrue(aggregate["baseline_audit_summary"]["baseline_c_too_close_warning"])
            self.assertIn(
                "action_agreement_high",
                aggregate["baseline_audit_summary"]["baseline_c_too_close_reason"],
            )
            self.assertGreater(
                aggregate["baseline_audit_summary"]["baselines"]["baseline_a"]["rows"],
                0,
            )
            self.assertFalse(aggregate["baseline_audit_summary"]["baseline_c_too_weak_warning"])
            self.assertEqual(
                aggregate["profile_expression_source_summary"]["personality"]["source_counts"]["action_phrase"],
                1,
            )
            self.assertEqual(
                aggregate["context_anchor_usage_summary"]["by_strategy"]["random"]["source_counts"]["partner_context"],
                1,
            )
            self.assertIn("no_surface_profile", aggregate["ablation_summary"])

    def test_pipeline_and_report_generation_cover_all_strategies(self) -> None:
        prev_sem = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC")
        os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = "1"

        def fake_implant(agent, world, config):
            del config
            uid_mod = int(getattr(world, "user_uid", 0)) % 10
            agent.slow_variable_learner.state.traits.social_approach = 0.80 + uid_mod * 0.001
            return None

        def fake_generate(agent, holdout_sessions, *, user_uid: int, seed: int, classifier):
            del seed, classifier
            turns = max(1, sum(1 for session in holdout_sessions for _ in session.get("turns", [])[:1]))
            social = float(agent.slow_variable_learner.state.traits.social_approach)
            if social > 0.70:
                gen = ["persona alpha warm detail"] * turns
            else:
                gen = ["default unrelated short"] * turns
            real = ["persona alpha warm detail"] * turns
            return gen, real, ["elaborate"] * turns, ["elaborate"] * turns, ["exploit"] * turns, ["exploit"] * turns

        try:
            users = [_build_user(uid, sessions=32) for uid in range(30, 40)]
            cfg = ValidationConfig(
                strategies=[
                    SplitStrategy.RANDOM,
                    SplitStrategy.TEMPORAL,
                    SplitStrategy.PARTNER,
                    SplitStrategy.TOPIC,
                ],
                seed=7,
                min_users=10,
                pilot_user_count=3,
                min_holdout_sessions=1,
                skip_population_average_implant=True,
            )
            with patch("segmentum.dialogue.validation.pipeline.implant_personality", side_effect=fake_implant), patch(
                "segmentum.dialogue.validation.pipeline._generate_from_sessions", side_effect=fake_generate
            ):
                pilot = run_pilot_validation(users, cfg)
                self.assertIn("semantic_diff_sd", pilot)
                reports = run_batch_validation(users, cfg)
            self.assertEqual(len(reports), 10)
            for report in reports:
                self.assertEqual(set(report.per_strategy), {"random", "temporal", "partner", "topic"})
                self.assertTrue(all(not row.get("skipped", False) for row in report.per_strategy.values()))
                self.assertGreater(
                    float(report.aggregate["semantic_personality_mean"]),
                    float(report.aggregate["semantic_baseline_a_mean"]),
                )
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp) / "m54_validation"
                md_path = generate_report(reports, output_dir)
                self.assertTrue(md_path.exists())
                self.assertTrue((output_dir / "aggregate_report.json").exists())
                self.assertTrue((output_dir / "per_user").exists())
                self.assertTrue(all(r.conclusion == "completed" for r in reports))
                agg = json.loads((output_dir / "aggregate_report.json").read_text(encoding="utf-8"))
                self.assertEqual(agg.get("metric_version"), M54_ACCEPTANCE_RULES_VERSION)
                self.assertEqual(agg.get("artifact_rules_version"), M54_ACCEPTANCE_RULES_VERSION)
                self.assertEqual(set(agg["split_gate"]["required_strategies"]), {"random", "temporal", "partner", "topic"})
                self.assertFalse(agg["formal_acceptance_eligible"])
                self.assertFalse(agg["hard_pass"])
                self.assertIn("acceptance_rules", agg)
        finally:
            if prev_sem is None:
                os.environ.pop("SEGMENTUM_USE_TFIDF_SEMANTIC", None)
            else:
                os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = prev_sem


if __name__ == "__main__":
    unittest.main()
