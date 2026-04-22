from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from segmentum.dialogue.validation.metrics import (
    semantic_pair_info_bucket,
    semantic_pair_weights,
    semantic_similarity,
    weighted_semantic_mean,
)


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestM54SemanticEmbedding(unittest.TestCase):
    def test_empty_pairs(self) -> None:
        r = semantic_similarity([], [])
        self.assertEqual(r.value, 0.0)
        self.assertEqual(r.details.get("method"), "none")

    @patch.dict(os.environ, {"SEGMENTUM_USE_TFIDF_SEMANTIC": "1"}, clear=False)
    def test_tfidf_forced_same_text(self) -> None:
        r = semantic_similarity(["你好，对齐需求。"], ["你好，对齐需求。"])
        self.assertEqual(r.details.get("method"), "tfidf_cosine")
        self.assertEqual(r.details.get("aggregation"), "information_weighted_pair_mean")
        self.assertIn("raw_pair_mean", r.details)
        self.assertIn("weighted_pair_mean", r.details)
        self.assertGreaterEqual(r.value, 0.99)

    def test_weighted_semantic_downweights_low_information_replies(self) -> None:
        scores = [1.0, 0.0]
        weights = semantic_pair_weights(["好", "我们需要确认汇款金额和时间"])

        self.assertEqual(semantic_pair_info_bucket("好"), "low_info_ack")
        self.assertLess(weights[0], weights[1])
        self.assertLess(weighted_semantic_mean(scores, weights), 0.50)

    @unittest.skipUnless(_has_sentence_transformers(), "install segmentum[validation] for embedding path")
    @patch.dict(os.environ, {"SEGMENTUM_USE_TFIDF_SEMANTIC": ""}, clear=False)
    def test_sentence_embedding_same_text_high(self) -> None:
        r = semantic_similarity(
            ["We should align on requirements."],
            ["We should align on requirements."],
        )
        self.assertEqual(r.details.get("method"), "sentence_embedding_cosine")
        self.assertIn("model", r.details)
        self.assertGreaterEqual(r.value, 0.99)


if __name__ == "__main__":
    unittest.main()
