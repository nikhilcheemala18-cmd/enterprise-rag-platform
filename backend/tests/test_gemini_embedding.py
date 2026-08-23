"""
Dependency-light unit tests for GeminiEmbeddingProvider.

Every test here uses a FakeGeminiClient -- none of them make a real
network call or require GEMINI_API_KEY. The real API is only exercised
by tests/test_gemini_embedding_integration.py, which is environment-
gated and skips cleanly without a key.
"""

import os
import unittest

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig
from app.embeddings.gemini import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_DOCUMENT_NO_TITLE,
    GEMINI_MAX_DIMENSION,
    GEMINI_MIN_DIMENSION,
    GEMINI_MODEL_NAME,
    GeminiEmbeddingProvider,
    _format_document_input,
    _format_query_input,
    default_gemini_config,
)


class FakeEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class FakeEmbedContentResult:
    def __init__(self, values: list[float]):
        self.embeddings = [FakeEmbedding(values)]


class FakeModels:
    """Stands in for genai.Client().models -- no network involved."""

    def __init__(self, dimension: int = 1536, raise_error: Exception | None = None):
        self.dimension = dimension
        self.raise_error = raise_error
        self.calls: list[tuple[str, str, str, int]] = []

    def embed_content(self, *, model, contents, config):
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append((model, contents, config.task_type, config.output_dimensionality))
        # deterministic fake vector so batch/order tests can assert on
        # content. Uses self.dimension (not config.output_dimensionality)
        # so a test can deliberately simulate the API returning a
        # different dimension than what was requested.
        seed = sum(ord(c) for c in contents) % 97
        values = [float((seed + i) % 11) for i in range(self.dimension)]
        return FakeEmbedContentResult(values)


class FakeGeminiClient:
    def __init__(self, dimension: int = 1536, raise_error: Exception | None = None):
        self.models = FakeModels(dimension=dimension, raise_error=raise_error)


def make_config(dimension: int = 1536) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider_name="google", model_name=GEMINI_MODEL_NAME, dimension=dimension
    )


class TestGeminiProviderConstruction(unittest.TestCase):
    def test_constructs_with_fake_client_no_network(self):
        provider = GeminiEmbeddingProvider(client=FakeGeminiClient())
        self.assertIsInstance(provider, EmbeddingProvider)

    def test_default_config_used_when_none_provided(self):
        provider = GeminiEmbeddingProvider(client=FakeGeminiClient())
        self.assertEqual(provider.config.model_name, GEMINI_MODEL_NAME)
        self.assertEqual(provider.config.provider_name, "google")

    def test_default_config_dimension_is_1536(self):
        config = default_gemini_config()
        self.assertEqual(config.dimension, 1536)

    def test_explicit_config_is_used(self):
        config = make_config(dimension=768)
        provider = GeminiEmbeddingProvider(config=config, client=FakeGeminiClient(dimension=768))
        self.assertEqual(provider.config.dimension, 768)


class TestGeminiApiKeyValidation(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)
        os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
        os.environ.pop("GOOGLE_API_KEY", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_missing_api_key_raises_clear_error_without_network(self):
        with self.assertRaises(RuntimeError) as ctx:
            GeminiEmbeddingProvider()
        self.assertIn(GEMINI_API_KEY_ENV_VAR, str(ctx.exception))

    def test_client_injection_bypasses_api_key_requirement(self):
        # a fake client is enough -- no key needed when client is supplied
        provider = GeminiEmbeddingProvider(client=FakeGeminiClient())
        self.assertIsNotNone(provider)

    def test_explicit_api_key_avoids_env_lookup_error(self):
        # we don't assert a real Client was built without hitting network;
        # we only assert this path doesn't raise the "missing key" error.
        try:
            GeminiEmbeddingProvider(api_key="fake-key-for-construction-only")
        except RuntimeError as exc:
            self.fail(f"should not raise the missing-key error: {exc}")
        except Exception:
            # genai.Client() itself may validate key format eagerly in some
            # SDK versions -- that's a different failure mode than "missing
            # key", which is what this test targets.
            pass


class TestGeminiModelAndDimension(unittest.TestCase):
    def test_model_name_constant(self):
        self.assertEqual(GEMINI_MODEL_NAME, "gemini-embedding-2")

    def test_rejects_wrong_model_name_without_network(self):
        bad_config = EmbeddingConfig(
            provider_name="google", model_name="not-gemini", dimension=1536
        )
        with self.assertRaises(ValueError):
            GeminiEmbeddingProvider(config=bad_config, client=FakeGeminiClient())

    def test_rejects_dimension_below_minimum(self):
        bad_config = EmbeddingConfig(
            provider_name="google", model_name=GEMINI_MODEL_NAME, dimension=GEMINI_MIN_DIMENSION - 1
        )
        with self.assertRaises(ValueError):
            GeminiEmbeddingProvider(config=bad_config, client=FakeGeminiClient())

    def test_rejects_dimension_above_maximum(self):
        bad_config = EmbeddingConfig(
            provider_name="google", model_name=GEMINI_MODEL_NAME, dimension=GEMINI_MAX_DIMENSION + 1
        )
        with self.assertRaises(ValueError):
            GeminiEmbeddingProvider(config=bad_config, client=FakeGeminiClient())

    def test_accepts_boundary_dimensions(self):
        for dim in (GEMINI_MIN_DIMENSION, GEMINI_MAX_DIMENSION, 1536):
            config = make_config(dimension=dim)
            provider = GeminiEmbeddingProvider(config=config, client=FakeGeminiClient(dimension=dim))
            self.assertEqual(provider.config.dimension, dim)


class TestGeminiSingleAndBatchEmbedding(unittest.TestCase):
    def setUp(self):
        self.fake_client = FakeGeminiClient(dimension=1536)
        self.provider = GeminiEmbeddingProvider(client=self.fake_client)

    def test_single_text_embedding(self):
        vector = self.provider.embed_text("Revenue increased by 18% in July.")
        self.assertEqual(len(vector), 1536)
        self.assertTrue(all(isinstance(v, float) for v in vector))

    def test_batch_embedding_returns_one_vector_per_text(self):
        texts = ["alpha", "beta", "gamma"]
        vectors = self.provider.embed_texts(texts)
        self.assertEqual(len(vectors), 3)
        for v in vectors:
            self.assertEqual(len(v), 1536)

    def test_batch_preserves_input_order(self):
        texts = ["first chunk", "second chunk", "third chunk"]
        vectors = self.provider.embed_texts(texts)
        for text, vector in zip(texts, vectors):
            self.assertEqual(vector, self.provider.embed_text(text))

    def test_batch_calls_are_one_per_text_not_one_multi_item_call(self):
        # documented gemini-embedding-2 behavior: passing a list to
        # `contents` aggregates into one embedding, which would break our
        # per-item contract -- so the provider must call embed_content
        # once per text, never once with a list.
        self.provider.embed_texts(["a", "b", "c"])
        self.assertEqual(len(self.fake_client.models.calls), 3)
        for call in self.fake_client.models.calls:
            self.assertIsInstance(call[1], str)  # contents is a single string

    def test_empty_input_returns_empty_list_without_calling_api(self):
        result = self.provider.embed_texts([])
        self.assertEqual(result, [])
        self.assertEqual(len(self.fake_client.models.calls), 0)


class TestGeminiDimensionOutputValidation(unittest.TestCase):
    def test_base_class_rejects_mismatched_returned_dimension(self):
        # fake client configured to return 999-dim vectors while
        # EmbeddingConfig declares 1536 -- the inherited base-class
        # validation (not duplicated here) must catch this.
        provider = GeminiEmbeddingProvider(
            config=make_config(dimension=1536), client=FakeGeminiClient(dimension=999)
        )
        with self.assertRaises(ValueError):
            provider.embed_text("x")


class TestGeminiQueryDocumentTextFormatting(unittest.TestCase):
    """gemini-embedding-2 does not support task_type (confirmed against
    current official docs: "You cannot use the task_type field for the
    gemini-embedding-2 model. Instead, include the task as an
    instruction in your prompt."). These tests verify the exact text
    sent to the fake client -- not a task_type parameter, which must
    never be set for this model.
    """

    def setUp(self):
        self.fake_client = FakeGeminiClient(dimension=1536)
        self.provider = GeminiEmbeddingProvider(client=self.fake_client)

    def test_document_input_uses_documented_title_text_format(self):
        self.provider.embed_text("Revenue increased by 18%.", is_query=False)
        sent_contents = self.fake_client.models.calls[0][1]
        self.assertEqual(sent_contents, "title: none | text: Revenue increased by 18%.")

    def test_query_input_uses_documented_task_query_format(self):
        self.provider.embed_text("what was revenue growth?", is_query=True)
        sent_contents = self.fake_client.models.calls[0][1]
        self.assertEqual(
            sent_contents, "task: search result | query: what was revenue growth?"
        )

    def test_default_is_query_false_uses_document_format(self):
        self.provider.embed_text("no is_query passed")
        sent_contents = self.fake_client.models.calls[0][1]
        self.assertTrue(sent_contents.startswith("title: "))

    def test_task_type_is_never_set_for_gemini_embedding_2(self):
        self.provider.embed_text("document text", is_query=False)
        self.provider.embed_text("query text", is_query=True)
        for call in self.fake_client.models.calls:
            self.assertIsNone(call[2])  # config.task_type must stay None

    def test_document_uses_no_title_placeholder_never_a_fabricated_title(self):
        self.assertEqual(GEMINI_DOCUMENT_NO_TITLE, "none")
        self.assertEqual(
            _format_document_input("some content"), "title: none | text: some content"
        )

    def test_format_helpers_match_documented_templates_exactly(self):
        self.assertEqual(
            _format_query_input("search terms"), "task: search result | query: search terms"
        )
        self.assertEqual(
            _format_document_input("passage body", title="Q3 Report"),
            "title: Q3 Report | text: passage body",
        )

    def test_batch_texts_are_each_formatted_consistently(self):
        self.provider.embed_texts(["doc one", "doc two"], is_query=False)
        contents_sent = [call[1] for call in self.fake_client.models.calls]
        self.assertEqual(
            contents_sent,
            ["title: none | text: doc one", "title: none | text: doc two"],
        )


class TestGeminiErrorHandling(unittest.TestCase):
    def test_api_error_is_wrapped_and_does_not_leak_the_api_key(self):
        secret_key = "AIzaSyFAKESECRETVALUEFORTESTINGONLY123"
        underlying_error = Exception(f"401 Unauthorized: bad key {secret_key}")
        fake_client = FakeGeminiClient(raise_error=underlying_error)
        provider = GeminiEmbeddingProvider(client=fake_client)

        with self.assertRaises(RuntimeError) as ctx:
            provider.embed_text("x")

        self.assertNotIn(secret_key, str(ctx.exception))

    def test_api_error_message_is_still_clear(self):
        fake_client = FakeGeminiClient(raise_error=Exception("boom"))
        provider = GeminiEmbeddingProvider(client=fake_client)
        with self.assertRaises(RuntimeError) as ctx:
            provider.embed_text("x")
        self.assertIn(GEMINI_MODEL_NAME, str(ctx.exception))

    def test_underlying_exception_is_chained_not_discarded(self):
        original = Exception("boom")
        fake_client = FakeGeminiClient(raise_error=original)
        provider = GeminiEmbeddingProvider(client=fake_client)
        with self.assertRaises(RuntimeError) as ctx:
            provider.embed_text("x")
        self.assertIs(ctx.exception.__cause__, original)


if __name__ == "__main__":
    unittest.main()
