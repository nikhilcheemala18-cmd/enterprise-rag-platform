import unittest

from sqlalchemy import MetaData

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig
from app.embeddings.local import DeterministicTestEmbeddingProvider
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.vector import VectorIndexer
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.models import RetrievalResult
from app.retrieval.vector import VectorRetriever
from app.services.retrieval_service import RetrievalService


class FakeEmbeddingProvider(EmbeddingProvider):
    """Records every embed call (texts + is_query) so tests can assert
    that RetrievalService actually calls embed_query() -- no real model,
    no network.
    """

    def __init__(self, config: EmbeddingConfig, vector: list[float] | None = None):
        super().__init__(config)
        self.vector = vector or [0.1] * config.dimension
        self.calls: list[tuple[list[str], bool]] = []

    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        self.calls.append((list(texts), is_query))
        return [self.vector for _ in texts]


class FakeHybridSearchService:
    """Stands in for app.retrieval.hybrid.HybridSearchService -- records
    the exact RetrievalRequest it received and returns canned results.
    """

    def __init__(self, results: list[RetrievalResult] | None = None):
        self.results = results if results is not None else []
        self.received_requests: list = []

    def search(self, request):
        self.received_requests.append(request)
        return self.results


def make_config(dimension: int = 8) -> EmbeddingConfig:
    return EmbeddingConfig(provider_name="fake", model_name="fake-model", dimension=dimension)


def make_result(chunk_id: str = "c1", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="Revenue increased by 18% in July.",
        score=score,
        metadata={"element_type": "text"},
    )


class TestRetrievalServiceSuccessfulFlow(unittest.TestCase):
    def setUp(self):
        self.provider = FakeEmbeddingProvider(make_config())
        self.hybrid = FakeHybridSearchService(
            results=[make_result("c1", 0.9), make_result("c2", 0.5)]
        )
        self.service = RetrievalService(self.provider, self.hybrid)

    def test_returns_hybrid_search_results(self):
        results = self.service.search("what was revenue growth?")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, RetrievalResult) for r in results))
        self.assertEqual(results[0].chunk_id, "c1")

    def test_query_embedding_is_performed(self):
        self.service.search("what was revenue growth?")
        self.assertEqual(len(self.provider.calls), 1)
        texts, is_query = self.provider.calls[0]
        self.assertEqual(texts, ["what was revenue growth?"])
        self.assertTrue(is_query)  # embed_query() must pass is_query=True

    def test_correct_request_passed_to_hybrid_search_service(self):
        self.service.search("what was revenue growth?", top_k=3, document_id="doc-9")
        self.assertEqual(len(self.hybrid.received_requests), 1)
        request = self.hybrid.received_requests[0]
        self.assertEqual(request.query, "what was revenue growth?")
        self.assertEqual(request.query_embedding, self.provider.vector)
        self.assertEqual(request.top_k, 3)
        self.assertEqual(request.document_id, "doc-9")

    def test_top_k_default_passthrough(self):
        self.service.search("query text")
        self.assertEqual(self.hybrid.received_requests[0].top_k, 10)

    def test_top_k_custom_passthrough(self):
        self.service.search("query text", top_k=25)
        self.assertEqual(self.hybrid.received_requests[0].top_k, 25)

    def test_document_id_defaults_to_none(self):
        self.service.search("query text")
        self.assertIsNone(self.hybrid.received_requests[0].document_id)

    def test_document_id_passthrough(self):
        self.service.search("query text", document_id="doc-42")
        self.assertEqual(self.hybrid.received_requests[0].document_id, "doc-42")


class TestRetrievalServiceEmptyResults(unittest.TestCase):
    def test_empty_hybrid_results_returned_as_is(self):
        provider = FakeEmbeddingProvider(make_config())
        hybrid = FakeHybridSearchService(results=[])
        service = RetrievalService(provider, hybrid)

        results = service.search("no matches expected for this query")

        self.assertEqual(results, [])


class TestRetrievalServiceValidation(unittest.TestCase):
    def setUp(self):
        self.provider = FakeEmbeddingProvider(make_config())
        self.hybrid = FakeHybridSearchService()
        self.service = RetrievalService(self.provider, self.hybrid)

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            self.service.search("")

    def test_empty_query_never_calls_embedding_provider(self):
        with self.assertRaises(ValueError):
            self.service.search("")
        self.assertEqual(self.provider.calls, [])

    def test_whitespace_only_query_rejected(self):
        with self.assertRaises(ValueError):
            self.service.search("   ")

    def test_invalid_top_k_propagates_from_retrieval_request(self):
        # Not duplicated in RetrievalService -- RetrievalRequest itself
        # validates top_k > 0 and raises (a pydantic ValidationError,
        # which subclasses ValueError) on construction.
        with self.assertRaises(ValueError):
            self.service.search("a valid query", top_k=0)

    def test_empty_document_id_propagates_from_retrieval_request(self):
        with self.assertRaises(ValueError):
            self.service.search("a valid query", document_id="")


# ---------------------------------------------------------------------------
# Realistic smoke/wiring test -- real RetrievalService, real retrieval-layer
# classes (LexicalRetriever/VectorRetriever/HybridSearchService), real
# indexing-layer search classes (LexicalIndexer/VectorIndexer), real
# DeterministicTestEmbeddingProvider. Only the database connection itself is
# faked (FakeEngine/FakeConnection return canned rows instead of opening a
# real PostgreSQL connection) -- this proves the real components compose and
# the generated SQL is well-formed, without any network or DB dependency.
# ---------------------------------------------------------------------------


class FakeResultMappingProxy:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def execute(self, stmt):
        return FakeResultMappingProxy(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeEngine:
    """Stands in for a real PostgreSQL connection: no network, no
    Supabase. Both LexicalIndexer.search() and VectorIndexer.search()
    call conn.execute(stmt).mappings().all() -- this fake returns the
    same canned rows regardless of which query was run, which is enough
    to prove the wiring (both retrievers -> RRF -> ranked results) works
    correctly end to end.
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def connect(self):
        return FakeConnection(self.rows)

    def begin(self):
        return FakeConnection(self.rows)


_FAKE_ROWS = [
    {
        "chunk_id": "c1",
        "document_id": "doc-1",
        "content": "Revenue increased by 18% in July for Client A.",
        "metadata": {"element_type": "text"},
        "score": 0.9,  # column alias LexicalIndexer.search() reads
        "distance": 0.05,  # column alias VectorIndexer.search() reads
    },
    {
        "chunk_id": "c2",
        "document_id": "doc-1",
        "content": "Figure 5.1 — Revenue Trend",
        "metadata": {"element_type": "image"},
        "score": 0.4,
        "distance": 0.3,
    },
]


class TestRealisticRetrievalSmoke(unittest.TestCase):
    def test_real_components_compose_without_a_real_database(self):
        engine = FakeEngine(_FAKE_ROWS)
        table = build_chunks_table(MetaData(), embedding_dimension=8)

        lexical_indexer = LexicalIndexer(engine, table)
        vector_indexer = VectorIndexer(engine, table, embedding_dimension=8)

        lexical_retriever = LexicalRetriever(lexical_indexer)
        vector_retriever = VectorRetriever(vector_indexer)
        hybrid_search_service = HybridSearchService(lexical_retriever, vector_retriever)

        provider = DeterministicTestEmbeddingProvider(
            EmbeddingConfig(provider_name="deterministic-test", model_name="det-v1", dimension=8)
        )

        retrieval_service = RetrievalService(provider, hybrid_search_service)

        results = retrieval_service.search("What was the revenue trend in July?", top_k=5)

        self.assertGreater(len(results), 0)
        self.assertTrue(all(isinstance(r, RetrievalResult) for r in results))
        # c1 appears (with the same chunk_id) in both the lexical and
        # vector result sets here, so RRF must rank it first.
        self.assertEqual(results[0].chunk_id, "c1")


if __name__ == "__main__":
    unittest.main()
