import unittest

from pydantic import ValidationError

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig
from app.embeddings.local import DeterministicTestEmbeddingProvider
from app.embeddings.query import embed_query
from app.embeddings.service import ChunkEmbeddingService
from app.models.chunk import Chunk


def make_config(dimension: int = 8) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider_name="deterministic-test",
        model_name="deterministic-test-v1",
        dimension=dimension,
    )


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_element_ids=["el-1"],
        content="Revenue increased by 18% in July for Client A.",
        token_count=9,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


class TestEmbeddingConfig(unittest.TestCase):
    def test_valid_config(self):
        config = make_config(dimension=384)
        self.assertEqual(config.dimension, 384)
        self.assertEqual(config.provider_name, "deterministic-test")

    def test_rejects_non_positive_dimension(self):
        with self.assertRaises(ValidationError):
            EmbeddingConfig(provider_name="x", model_name="y", dimension=0)
        with self.assertRaises(ValidationError):
            EmbeddingConfig(provider_name="x", model_name="y", dimension=-1)

    def test_rejects_empty_provider_or_model_name(self):
        with self.assertRaises(ValidationError):
            EmbeddingConfig(provider_name="", model_name="y", dimension=8)
        with self.assertRaises(ValidationError):
            EmbeddingConfig(provider_name="x", model_name="", dimension=8)


class TestEmbeddingProviderInterface(unittest.TestCase):
    def test_provider_is_abstract(self):
        with self.assertRaises(TypeError):
            EmbeddingProvider(make_config())  # type: ignore[abstract]

    def test_concrete_provider_exposes_embed_text_and_embed_texts(self):
        provider = DeterministicTestEmbeddingProvider(make_config())
        self.assertTrue(hasattr(provider, "embed_text"))
        self.assertTrue(hasattr(provider, "embed_texts"))


class TestSingleTextEmbedding(unittest.TestCase):
    def test_embed_text_returns_list_of_floats_with_configured_dimension(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=16))
        vector = provider.embed_text("hello world")
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 16)
        self.assertTrue(all(isinstance(v, float) for v in vector))


class TestBatchEmbedding(unittest.TestCase):
    def test_embed_texts_returns_one_vector_per_text(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        texts = ["alpha", "beta", "gamma"]
        vectors = provider.embed_texts(texts)
        self.assertEqual(len(vectors), 3)
        for v in vectors:
            self.assertEqual(len(v), 8)

    def test_batch_preserves_input_order(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        texts = ["first text", "second text", "third text"]
        vectors = provider.embed_texts(texts)
        for text, vector in zip(texts, vectors):
            self.assertEqual(vector, provider.embed_text(text))

    def test_batch_matches_single_text_embedding(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        single = provider.embed_text("consistent text")
        batched = provider.embed_texts(["consistent text"])[0]
        self.assertEqual(single, batched)


class TestEmptyInput(unittest.TestCase):
    def test_embed_texts_with_empty_list_returns_empty_list(self):
        provider = DeterministicTestEmbeddingProvider(make_config())
        self.assertEqual(provider.embed_texts([]), [])


class TestDimensionValidation(unittest.TestCase):
    class BrokenProvider(EmbeddingProvider):
        """Deliberately returns the wrong dimension to prove the base
        class's validation catches it."""

        def _embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * (self.config.dimension + 1) for _ in texts]

    class MiscountingProvider(EmbeddingProvider):
        """Deliberately returns the wrong number of vectors."""

        def _embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.config.dimension for _ in texts[:-1]] if texts else []

    def test_dimension_mismatch_raises(self):
        provider = self.BrokenProvider(make_config(dimension=8))
        with self.assertRaises(ValueError):
            provider.embed_text("x")

    def test_batch_count_mismatch_raises(self):
        provider = self.MiscountingProvider(make_config(dimension=8))
        with self.assertRaises(ValueError):
            provider.embed_texts(["a", "b", "c"])

    def test_correct_dimension_does_not_raise(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        provider.embed_text("fine")  # must not raise


class TestDeterministicTestProviderBehavior(unittest.TestCase):
    def test_same_text_produces_same_vector(self):
        provider = DeterministicTestEmbeddingProvider(make_config())
        self.assertEqual(provider.embed_text("hello"), provider.embed_text("hello"))

    def test_different_text_produces_different_vector(self):
        provider = DeterministicTestEmbeddingProvider(make_config())
        self.assertNotEqual(
            provider.embed_text("hello"), provider.embed_text("goodbye")
        )

    def test_no_network_dependency(self):
        # purely a documentation-style assertion: constructing and using
        # the provider must not require any I/O. If it did, this call
        # would be slow/flaky/failing in a sandboxed test environment
        # rather than completing instantly.
        provider = DeterministicTestEmbeddingProvider(make_config())
        provider.embed_texts(["a", "b", "c"])

    def test_vector_values_within_expected_range(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=32))
        vector = provider.embed_text("range check")
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in vector))


class TestChunkEmbeddingService(unittest.TestCase):
    def setUp(self):
        self.provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        self.service = ChunkEmbeddingService(self.provider)

    def test_returns_chunk_id_to_vector_mapping(self):
        chunks = [
            make_chunk(chunk_id="c1", content="Revenue increased by 18%."),
            make_chunk(chunk_id="c2", content="Orders grew steadily."),
        ]
        result = self.service.embed_chunks(chunks)
        self.assertEqual(set(result.keys()), {"c1", "c2"})
        self.assertEqual(len(result["c1"]), 8)

    def test_vectors_match_direct_provider_calls(self):
        chunks = [make_chunk(chunk_id="c1", content="Deterministic content.")]
        result = self.service.embed_chunks(chunks)
        self.assertEqual(result["c1"], self.provider.embed_text("Deterministic content."))

    def test_document_id_preserved_on_source_chunks(self):
        chunks = [make_chunk(chunk_id="c1", document_id="doc-42")]
        self.service.embed_chunks(chunks)
        self.assertEqual(chunks[0].document_id, "doc-42")

    def test_chunk_order_preserved_via_batched_call(self):
        chunks = [
            make_chunk(chunk_id="c1", content="first"),
            make_chunk(chunk_id="c2", content="second"),
            make_chunk(chunk_id="c3", content="third"),
        ]
        result = self.service.embed_chunks(chunks)
        expected = self.provider.embed_texts(["first", "second", "third"])
        self.assertEqual(
            [result["c1"], result["c2"], result["c3"]], expected
        )

    def test_empty_chunk_list_returns_empty_mapping(self):
        self.assertEqual(self.service.embed_chunks([]), {})

    def test_does_not_mutate_chunk_model(self):
        chunk = make_chunk(chunk_id="c1")
        original_dump = chunk.model_dump()
        self.service.embed_chunks([chunk])
        self.assertEqual(chunk.model_dump(), original_dump)

    def test_compatible_with_indexing_service_embeddings_argument(self):
        # IndexingService.index_chunks(chunks, embeddings=...) expects
        # dict[str, list[float]] -- this proves embed_chunks() returns
        # exactly that shape without any adapter.
        chunks = [make_chunk(chunk_id="c1")]
        embeddings = self.service.embed_chunks(chunks)
        self.assertIsInstance(embeddings, dict)
        self.assertIsInstance(embeddings["c1"], list)
        self.assertTrue(all(isinstance(x, float) for x in embeddings["c1"]))


class TestQueryEmbedding(unittest.TestCase):
    def test_embed_query_returns_vector(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        vector = embed_query(provider, "what was revenue in July?")
        self.assertEqual(len(vector), 8)

    def test_query_and_document_embeddings_share_the_same_provider_and_space(self):
        provider = DeterministicTestEmbeddingProvider(make_config(dimension=8))
        service = ChunkEmbeddingService(provider)
        chunk = make_chunk(chunk_id="c1", content="what was revenue in July?")

        query_vector = embed_query(provider, "what was revenue in July?")
        chunk_vectors = service.embed_chunks([chunk])

        # same text through both paths -> identical vector, because both
        # paths route through the exact same EmbeddingProvider/config.
        self.assertEqual(query_vector, chunk_vectors["c1"])


if __name__ == "__main__":
    unittest.main()
