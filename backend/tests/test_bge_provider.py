"""
Dependency-light unit tests for BGEEmbeddingProvider.

These must NOT trigger a real model load/download. Importing
app.embeddings.bge only imports the sentence-transformers library (fast,
no network); constructing a SentenceTransformer instance is what would
hit the Hugging Face Hub / local cache, and this file never does that --
BGEEmbeddingProvider validates model_name/dimension against a bad
EmbeddingConfig and raises BEFORE ever reaching that constructor call.

The real-model behavior (loading BAAI/bge-base-en-v1.5 and actually
encoding text) is covered separately in test_bge_provider_integration.py,
which is environment-gated and skips cleanly if the model can't be
loaded offline in this environment.
"""

import unittest

import numpy as np

from app.embeddings.base import EmbeddingProvider
from app.embeddings.bge import (
    BGE_DIMENSION,
    BGE_MAX_TOKENS,
    BGE_MODEL_NAME,
    BGE_QUERY_INSTRUCTION,
    BGEEmbeddingProvider,
    _to_float_matrix,
    default_bge_config,
)
from app.embeddings.config import EmbeddingConfig


class TestBGEConfiguration(unittest.TestCase):
    def test_default_config_matches_expected_values(self):
        config = default_bge_config()
        self.assertEqual(config.provider_name, "huggingface")
        self.assertEqual(config.model_name, "BAAI/bge-base-en-v1.5")
        self.assertEqual(config.dimension, 768)

    def test_model_name_constant(self):
        self.assertEqual(BGE_MODEL_NAME, "BAAI/bge-base-en-v1.5")

    def test_dimension_constant(self):
        self.assertEqual(BGE_DIMENSION, 768)

    def test_max_tokens_constant(self):
        self.assertEqual(BGE_MAX_TOKENS, 512)

    def test_query_instruction_matches_documented_text(self):
        self.assertEqual(
            BGE_QUERY_INSTRUCTION,
            "Represent this sentence for searching relevant passages: ",
        )

    def test_config_uses_existing_embedding_config_model(self):
        self.assertIsInstance(default_bge_config(), EmbeddingConfig)


class TestBGEProviderConstructionValidation(unittest.TestCase):
    def test_rejects_wrong_model_name_without_loading_model(self):
        bad_config = EmbeddingConfig(
            provider_name="huggingface", model_name="some-other-model", dimension=768
        )
        with self.assertRaises(ValueError):
            BGEEmbeddingProvider(config=bad_config)

    def test_rejects_wrong_dimension_without_loading_model(self):
        bad_config = EmbeddingConfig(
            provider_name="huggingface", model_name=BGE_MODEL_NAME, dimension=384
        )
        with self.assertRaises(ValueError):
            BGEEmbeddingProvider(config=bad_config)

    def test_error_message_names_the_expected_model(self):
        bad_config = EmbeddingConfig(
            provider_name="huggingface", model_name="wrong-model", dimension=768
        )
        with self.assertRaises(ValueError) as ctx:
            BGEEmbeddingProvider(config=bad_config)
        self.assertIn(BGE_MODEL_NAME, str(ctx.exception))


class TestBGEInterfaceCompliance(unittest.TestCase):
    def test_is_subclass_of_embedding_provider(self):
        self.assertTrue(issubclass(BGEEmbeddingProvider, EmbeddingProvider))

    def test_exposes_embed_text_and_embed_texts(self):
        self.assertTrue(hasattr(BGEEmbeddingProvider, "embed_text"))
        self.assertTrue(hasattr(BGEEmbeddingProvider, "embed_texts"))

    def test_implements_embed_texts_hook(self):
        self.assertTrue(hasattr(BGEEmbeddingProvider, "_embed_texts"))


class TestOutputConversion(unittest.TestCase):
    def test_numpy_array_converted_to_plain_python_floats(self):
        array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = _to_float_matrix(array)

        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        for row in result:
            self.assertIsInstance(row, list)
            for value in row:
                self.assertIsInstance(value, float)
                self.assertNotIsInstance(value, np.floating)

    def test_empty_array_converts_to_empty_list(self):
        self.assertEqual(_to_float_matrix(np.empty((0, 8))), [])


if __name__ == "__main__":
    unittest.main()
