from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from segmentum.dialogue.validation.act_classifier import DialogueActClassifier, validate_act_classifier
from segmentum.dialogue.validation.baselines import select_wrong_users
from segmentum.dialogue.validation.pipeline import ValidationConfig, run_batch_validation, run_pilot_validation
from segmentum.dialogue.validation.report import generate_report
from segmentum.dialogue.validation.splitter import SplitStrategy, split_user_data
from segmentum.dialogue.validation.statistics import paired_comparison


def _build_user(uid: int, *, sessions: int = 12, partner_start: int = 1000) -> dict:
    base = datetime(2024, 1, 1, 8, 0, 0) + timedelta(days=uid)
    rows: list[dict] = []
    topics = [
        ("项目进度", "我们来对齐一下需求。"),
        ("情绪支持", "我最近有点焦虑。"),
        ("技术方案", "你觉得这个架构怎么样？"),
        ("日常闲聊", "今天吃了什么？"),
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
                "body": "我在听。",
            },
            {
                "timestamp": (t0 + timedelta(minutes=2)).isoformat(),
                "sender_uid": partner,
                "receiver_uid": uid,
                "body": "你可以再解释一下吗？",
            },
            {
                "timestamp": (t0 + timedelta(minutes=3)).isoformat(),
                "sender_uid": uid,
                "receiver_uid": partner,
                "body": "当然可以，我补充一下。",
            },
        ]
        rows.append(
            {
                "session_id": f"s-{uid}-{idx}",
                "uid_a": min(uid, partner),
                "uid_b": max(uid, partner),
                "start_time": turns[0]["timestamp"],
                "end_time": turns[-1]["timestamp"],
                "metadata": {"turn_count": len(turns)},
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


class TestM54Validation(unittest.TestCase):
    def test_split_strategies_produce_disjoint_sets(self) -> None:
        user = _build_user(11, sessions=12)
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

    def test_temporal_split_order(self) -> None:
        user = _build_user(12, sessions=10)
        split = split_user_data(user, SplitStrategy.TEMPORAL, train_ratio=0.70, seed=42)
        self.assertTrue(split.train_sessions and split.holdout_sessions)
        self.assertLessEqual(split.train_sessions[-1]["start_time"], split.holdout_sessions[0]["start_time"])

    def test_partner_split_no_overlap(self) -> None:
        user = _build_user(13, sessions=12)
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
        user = _build_user(14, sessions=20)
        split = split_user_data(user, SplitStrategy.TOPIC, train_ratio=0.70, seed=42)
        meta = split.split_metadata
        self.assertIn("strict_train_only_fit", meta)
        self.assertIn("topic_split_not_applicable", meta)
        if not bool(meta["topic_split_not_applicable"]):
            self.assertIn("k_selected", meta)
            self.assertIn("cluster_sizes", meta)
            self.assertIn("merge_steps", meta)
            self.assertIn("topic_jsd_train_holdout", meta)
            self.assertTrue(meta.get("topic_kmeans_fitted_on_train_sessions_only"))

    def test_act_classifier_and_gate(self) -> None:
        clf = DialogueActClassifier()
        pred = clf.predict("你能再具体说一下吗？")
        self.assertIn(
            pred.label_11,
            {
                "ask_question",
                "elaborate",
                "agree",
                "disagree",
                "empathize",
                "joke",
                "deflect",
                "minimal_response",
                "disengage",
                "introduce_topic",
                "share_opinion",
            },
        )
        samples = [
            {"text": "你能解释一下吗？", "label_11": "ask_question", "label_3": "explore"},
            {"text": "我同意", "label_11": "agree", "label_3": "exploit"},
            {"text": "这一点不对", "label_11": "disagree", "label_3": "escape"},
        ]
        report = validate_act_classifier(samples, min_macro_f1_3class=0.0)
        self.assertEqual(report["sample_count"], 3)
        self.assertIn("macro_f1_3class", report)

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

    def test_pipeline_and_report_generation(self) -> None:
        # min_holdout_sessions default 3 → need enough sessions so holdout >= 3
        # TF-IDF semantic path avoids loading sentence-transformers (slow / CI-unfriendly).
        # skip_population_average_implant avoids N× full-data implants for Baseline C (tests only).
        prev_sem = os.environ.get("SEGMENTUM_USE_TFIDF_SEMANTIC")
        os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = "1"
        try:
            users = [_build_user(uid, sessions=12) for uid in range(30, 40)]
            cfg = ValidationConfig(
                strategies=[SplitStrategy.RANDOM],
                seed=7,
                min_users=10,
                pilot_user_count=3,
                skip_population_average_implant=True,
            )
            pilot = run_pilot_validation(users, cfg)
            self.assertIn("semantic_diff_sd", pilot)
            reports = run_batch_validation(users, cfg)
            self.assertEqual(len(reports), 10)
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp) / "m54_validation"
                md_path = generate_report(reports, output_dir)
                self.assertTrue(md_path.exists())
                self.assertTrue((output_dir / "aggregate_report.json").exists())
                self.assertTrue((output_dir / "per_user").exists())
                self.assertTrue(all(r.conclusion == "completed" for r in reports))
                agg = json.loads((output_dir / "aggregate_report.json").read_text(encoding="utf-8"))
                self.assertEqual(agg.get("metric_version"), "m54_v3")
                self.assertIn("hard_pass", agg)
                self.assertIn("acceptance_rules", agg)
        finally:
            if prev_sem is None:
                os.environ.pop("SEGMENTUM_USE_TFIDF_SEMANTIC", None)
            else:
                os.environ["SEGMENTUM_USE_TFIDF_SEMANTIC"] = prev_sem


if __name__ == "__main__":
    unittest.main()
