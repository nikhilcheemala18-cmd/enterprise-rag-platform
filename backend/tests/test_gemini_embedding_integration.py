"""
Real API integration test for GeminiEmbeddingProvider.

Skipped entirely unless GEMINI_API_KEY is set in the environment -- this
is NOT faked. Every test in this class makes a real network call to
Google's Gemini API and will incur usage against your account (free
tier or billed, depending on your plan). See
https://ai.google.dev/gemini-api/docs/pricing for current pricing.

Local setup:

    # bash
    export GEMINI_API_KEY=your-real-key
    # PowerShell
    $env:GEMINI_API_KEY = "your-real-key"

    python -m unittest tests.test_gemini_embedding_integration -v

This environment has no GEMINI_API_KEY configured, so this class is
expected to skip when run here.
"""

import math
import os
import unittest

from app.embeddings.gemini import GEMINI_API_KEY_ENV_VAR, GeminiEmbeddingProvider

_HAS_API_KEY = bool(os.environ.get(GEMINI_API_KEY_ENV_VAR) or os.environ.get("GOOGLE_API_KEY"))


@unittest.skipUnless(
    _HAS_API_KEY,
    f"requires a real Gemini API key; set {GEMINI_API_KEY_ENV_VAR} to run "
    "(see module docstring for local setup)",
)
class TestGeminiEmbeddingProviderRealAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = GeminiEmbeddingProvider()

    def test_authenticates_and_embeds_single_text(self):
        vector = self.provider.embed_text("Revenue increased by 18% in July.")
        self.assertEqual(len(vector), self.provider.config.dimension)
        self.assertTrue(all(isinstance(v, float) for v in vector))

    def test_batch_embedding_matches_input_count_and_order(self):
        texts = [
            "Revenue grew by 18% in July.",
            "Orders increased steadily.",
            "Figure 5.1 — Revenue Trend",
        ]
        vectors = self.provider.embed_texts(texts)
        self.assertEqual(len(vectors), len(texts))
        for vector in vectors:
            self.assertEqual(len(vector), self.provider.config.dimension)

    def test_expected_dimension(self):
        vector = self.provider.embed_text("dimension check")
        self.assertEqual(len(vector), 1536)

    def test_query_and_document_embeddings_are_both_produced(self):
        document_vector = self.provider.embed_text("Revenue increased by 18%.")
        query_vector = self.provider.embed_text(
            "what was revenue growth?", is_query=True
        )
        self.assertEqual(len(document_vector), self.provider.config.dimension)
        self.assertEqual(len(query_vector), self.provider.config.dimension)

    def test_output_is_normalized(self):
        # gemini-embedding-2 documents automatic renormalization of its
        # output, including truncated (non-default) dimensions.
        vector = self.provider.embed_text("normalization check")
        norm = math.sqrt(sum(v * v for v in vector))
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_real_provider_satisfies_embedding_provider_contract(self):
        from app.embeddings.base import EmbeddingProvider

        self.assertIsInstance(self.provider, EmbeddingProvider)
        # exercised via the shared base-class validation (count + dimension)
        self.provider.embed_texts(["a", "b"])

    def test_semantically_related_texts_are_closer_than_unrelated_ones(self):
        related_a = self.provider.embed_text("The cat sat on the mat.")
        related_b = self.provider.embed_text("A cat was sitting on a mat.")
        unrelated = self.provider.embed_text(
            "Quarterly earnings report for the automotive sector."
        )

        def cosine_similarity(x: list[float], y: list[float]) -> float:
            return sum(xi * yi for xi, yi in zip(x, y))

        self.assertGreater(
            cosine_similarity(related_a, related_b),
            cosine_similarity(related_a, unrelated),
        )


if __name__ == "__main__":
    unittest.main()
