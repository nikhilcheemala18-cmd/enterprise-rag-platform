import unittest

from app.retrieval.models import RetrievalRequest
from app.retrieval.vector import VectorRetriever


class FakeSearchHit:
    def __init__(self, chunk_id, document_id, content, score, metadata=None):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = metadata or {}


class FakeVectorBackend:
    """Stands in for app.indexing.vector.VectorIndexer -- no database or
    pgvector involved. Records the call so tests can assert delegation
    boundaries.
    """

    def __init__(self, hits):
        self.hits = hits
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_embedding, limit: int = 10):
        self.calls.append((query_embedding, limit))
        return self.hits


class TestVectorRetrieverDelegation(unittest.TestCase):
    def test_delegates_embedding_and_limit_to_backend(self):
        backend = FakeVectorBackend([])
        retriever = VectorRetriever(backend)
        retriever.search(RetrievalRequest(query_embedding=[1.0, 0.0, 0.0], top_k=7))
        self.assertEqual(backend.calls, [([1.0, 0.0, 0.0], 7)])

    def test_adapts_backend_hits_to_retrieval_results(self):
        backend = FakeVectorBackend(
            [FakeSearchHit("c3", "doc-1", "Figure 5.1 — Revenue Trend", 0.87, {"element_type": "image"})]
        )
        retriever = VectorRetriever(backend)
        results = retriever.search(RetrievalRequest(query_embedding=[0.1, 0.2]))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c3")
        self.assertEqual(results[0].score, 0.87)
        self.assertEqual(results[0].metadata, {"element_type": "image"})

    def test_raises_when_request_has_no_embedding(self):
        backend = FakeVectorBackend([])
        retriever = VectorRetriever(backend)
        request = RetrievalRequest(query="text only")
        with self.assertRaises(ValueError):
            retriever.search(request)

    def test_does_not_generate_an_embedding_itself(self):
        # VectorRetriever must only ever forward request.query_embedding
        # verbatim -- it has no code path that could compute one.
        backend = FakeVectorBackend([])
        retriever = VectorRetriever(backend)
        embedding = [0.5, -0.5, 0.25]
        retriever.search(RetrievalRequest(query_embedding=embedding))
        self.assertEqual(backend.calls[0][0], embedding)

    def test_document_id_filter_applied_after_backend_returns(self):
        backend = FakeVectorBackend(
            [
                FakeSearchHit("c1", "doc-1", "x", 1.0),
                FakeSearchHit("c2", "doc-2", "y", 0.9),
            ]
        )
        retriever = VectorRetriever(backend)
        results = retriever.search(
            RetrievalRequest(query_embedding=[0.1], document_id="doc-2")
        )
        self.assertEqual([r.chunk_id for r in results], ["c2"])


if __name__ == "__main__":
    unittest.main()
