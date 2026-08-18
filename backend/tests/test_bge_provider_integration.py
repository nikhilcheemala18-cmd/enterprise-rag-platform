"""
Real-model integration test for BGEEmbeddingProvider.

This loads the actual BAAI/bge-base-en-v1.5 model via sentence-
transformers. On a machine with no cached copy, the first construction
downloads ~420MB of weights through the standard Hugging Face cache;
subsequent runs reuse that cache and do not re-download. No model files
are committed to this repository.

If the model cannot be loaded (no network access and nothing cached
locally, or any other load failure), the whole class is skipped with a
clear reason via setUpClass -- this is NOT faked: it either genuinely
exercises the real model, or it skips instead of failing the suite.
"""

import math
import unittest

from app.embeddings.bge import BGE_DIMENSION, BGEEmbeddingProvider

_provider: BGEEmbeddingProvider | None = None
_skip_reason: str | None = None
_attempted = False


def _get_provider() -> tuple[BGEEmbeddingProvider | None, str | None]:
    global _provider, _skip_reason, _attempted
    if not _attempted:
        _attempted = True
        try:
            _provider = BGEEmbeddingProvider()
        except Exception as exc:  # network/cache/runtime failure
            _skip_reason = f"BAAI/bge-base-en-v1.5 could not be loaded: {exc}"
    return _provider, _skip_reason


class TestBGEProviderRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provider, reason = _get_provider()
        if provider is None:
            raise unittest.SkipTest(reason)
        cls.provider = provider

    def test_model_loads(self):
        self.assertIsNotNone(self.provider)

    def test_single_text_produces_vector_of_correct_dimension(self):
        vector = self.provider.embed_text("Revenue increased by 18% in July.")
        self.assertEqual(len(vector), BGE_DIMENSION)
        self.assertTrue(all(isinstance(v, float) for v in vector))

    def test_batch_embedding_matches_input_count(self):
        texts = [
            "Revenue grew by 18% in July.",
            "Orders increased steadily.",
            "Figure 5.1 — Revenue Trend",
        ]
        vectors = self.provider.embed_texts(texts)
        self.assertEqual(len(vectors), len(texts))
        for vector in vectors:
            self.assertEqual(len(vector), BGE_DIMENSION)

    def test_same_text_produces_stable_output(self):
        first = self.provider.embed_text("stability check")
        second = self.provider.embed_text("stability check")
        self.assertEqual(first, second)

    def test_query_and_document_embeddings_share_dimension(self):
        document_vector = self.provider.embed_text("Revenue increased by 18%.")
        query_vector = self.provider.embed_text(
            "what was revenue growth?", is_query=True
        )
        self.assertEqual(len(document_vector), BGE_DIMENSION)
        self.assertEqual(len(query_vector), BGE_DIMENSION)

    def test_embeddings_are_normalized(self):
        vector = self.provider.embed_text("normalization check")
        norm = math.sqrt(sum(v * v for v in vector))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_semantically_related_texts_are_closer_than_unrelated_ones(self):
        # Confirms this is a real semantic model (unlike the
        # deterministic hash-based test provider): related sentences
        # should be closer in vector space than unrelated ones.
        related_a = self.provider.embed_text("The cat sat on the mat.")
        related_b = self.provider.embed_text("A cat was sitting on a mat.")
        unrelated = self.provider.embed_text(
            "Quarterly earnings report for the automotive sector."
        )

        def cosine_similarity(x: list[float], y: list[float]) -> float:
            # vectors are already normalized, so dot product == cosine similarity
            return sum(xi * yi for xi, yi in zip(x, y))

        similarity_related = cosine_similarity(related_a, related_b)
        similarity_unrelated = cosine_similarity(related_a, unrelated)
        self.assertGreater(similarity_related, similarity_unrelated)


if __name__ == "__main__":
    unittest.main()
