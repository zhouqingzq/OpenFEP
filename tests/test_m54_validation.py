from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from segmentum.dialogue.validation.act_classifier import (
    DialogueActClassifier,
    KeywordDialogueActClassifier,
    validate_act_classifier,
)
from segmentum.dialogue.validation.baselines import select_wrong_users
from segmentum.dialogue.validation.metrics import SimilarityResult
from segmentum.dialogue.validation.pipeline import (
    ValidationConfig,
    ValidationReport,
    run_batch_validation,
    run_pilot_validation,
    run_validation,
)
from segmentum.dialogue.validation.report import generate_report
from segmentum.dialogue.validation.splitter import SplitStrategy, split_user_data
from segmentum.dialogue.validation.statistics import paired_comparison
from segmentum.dialogue.validation.surface_profile import build_surface_profile
from segmentum.dialogue.generator import RuleBasedGenerator
from segmentum.slow_learning import SlowVariableLearner


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

    def test_surface_profile_uses_train_sessions_only(self) -> None:
        user = _build_user(18, sessions=4)
        train_dataset = dict(user)
        train_dataset["sessions"] = user["sessions"][:2]
        holdout_text = "HOLDOUT_UNIQUE_TOKEN_DO_NOT_LEAK"
        user["sessions"][3]["turns"][1]["body"] = holdout_text

        profile = build_surface_profile(train_dataset, source="unit_train")
        payload = json.dumps(profile.to_dict(), ensure_ascii=False)

        self.assertEqual(profile.source, "unit_train")
        self.assertGreater(profile.reply_count, 0)
        self.assertNotIn(holdout_text, payload)
        self.assertNotIn(str(user["sessions"][3]["session_id"]), payload)

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
            "reply_count": 5,
            "connector_phrases": ["我会认真回应"],
            "top_tokens": ["alpha"],
            "action_phrases": {"elaborate": ["我补充 alpha 的关键细节"]},
            "partner_tokens": {"1000": ["partner_alpha"]},
        }
        profile_b = {
            "source": "b",
            "reply_count": 5,
            "connector_phrases": ["先看 beta"],
            "top_tokens": ["beta"],
            "action_phrases": {"elaborate": ["我补充 beta 的关键细节"]},
            "partner_tokens": {"1000": ["partner_beta"]},
        }

        first = generator.generate(
            "elaborate",
            context,
            {**base_state, "surface_profile": profile_a},
            [],
            master_seed=9,
            turn_index=0,
        )
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
        self.assertFalse(report["formal_gate_eligible"])
        self.assertFalse(report["behavioral_hard_metric_enabled"])

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
                    "reply_length_bucket": "medium",
                }
            ],
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
                self.assertEqual(agg.get("metric_version"), "m54_v3")
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
