from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from segmentum.dialogue.validation.metrics import semantic_similarity


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
        self.assertGreaterEqual(r.value, 0.99)

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
