import unittest

from app.retrieval.lexical import LexicalRetriever
from app.retrieval.models import RetrievalRequest


class FakeSearchHit:
    def __init__(self, chunk_id, document_id, content, score, metadata=None):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = metadata or {}


class FakeLexicalBackend:
    """Stands in for app.indexing.lexical.LexicalIndexer -- no database
    or SQL involved. Records the call so tests can assert delegation
    boundaries (what LexicalRetriever passed through, unchanged).
    """

    def __init__(self, hits):
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10):
        self.calls.append((query, limit))
        return self.hits


class TestLexicalRetrieverDelegation(unittest.TestCase):
    def test_delegates_query_and_limit_to_backend(self):
        backend = FakeLexicalBackend([])
        retriever = LexicalRetriever(backend)
        retriever.search(RetrievalRequest(query="INV-2026-01847", top_k=5))
        self.assertEqual(backend.calls, [("INV-2026-01847", 5)])

    def test_adapts_backend_hits_to_retrieval_results(self):
        backend = FakeLexicalBackend(
            [FakeSearchHit("c1", "doc-1", "Invoice text", 0.9, {"element_type": "text"})]
        )
        retriever = LexicalRetriever(backend)
        results = retriever.search(RetrievalRequest(query="invoice"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].document_id, "doc-1")
        self.assertEqual(results[0].content, "Invoice text")
        self.assertEqual(results[0].score, 0.9)
        self.assertEqual(results[0].metadata, {"element_type": "text"})

    def test_raises_when_request_has_no_query(self):
        backend = FakeLexicalBackend([])
        retriever = LexicalRetriever(backend)
        request = RetrievalRequest(query_embedding=[0.1, 0.2])
        with self.assertRaises(ValueError):
            retriever.search(request)

    def test_document_id_filter_applied_after_backend_returns(self):
        backend = FakeLexicalBackend(
            [
                FakeSearchHit("c1", "doc-1", "x", 1.0),
                FakeSearchHit("c2", "doc-2", "y", 0.9),
            ]
        )
        retriever = LexicalRetriever(backend)
        results = retriever.search(RetrievalRequest(query="x", document_id="doc-1"))
        self.assertEqual([r.chunk_id for r in results], ["c1"])

    def test_never_touches_a_database_object(self):
        # the fake backend has no engine/connection attribute at all --
        # if LexicalRetriever tried to reach past it for a live query,
        # this test would fail with an AttributeError instead of passing.
        backend = FakeLexicalBackend([])
        retriever = LexicalRetriever(backend)
        retriever.search(RetrievalRequest(query="x"))


if __name__ == "__main__":
    unittest.main()
