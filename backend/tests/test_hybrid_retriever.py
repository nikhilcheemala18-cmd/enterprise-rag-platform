import unittest

from app.retrieval.hybrid import HybridSearchService
from app.retrieval.models import RetrievalRequest, RetrievalResult
from app.retrieval.rrf import RRFConfig


def result(chunk_id: str, score: float = 1.0, document_id: str = "doc-1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        score=score,
        metadata={},
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.calls: list[RetrievalRequest] = []

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        self.calls.append(request)
        return self.results


class TestHybridSearchServiceDelegation(unittest.TestCase):
    def test_query_only_request_calls_lexical_not_vector(self):
        lexical = FakeRetriever([result("c1")])
        vector = FakeRetriever([result("c2")])
        service = HybridSearchService(lexical, vector)

        service.search(RetrievalRequest(query="invoice"))

        self.assertEqual(len(lexical.calls), 1)
        self.assertEqual(len(vector.calls), 0)

    def test_embedding_only_request_calls_vector_not_lexical(self):
        lexical = FakeRetriever([result("c1")])
        vector = FakeRetriever([result("c2")])
        service = HybridSearchService(lexical, vector)

        service.search(RetrievalRequest(query_embedding=[0.1, 0.2]))

        self.assertEqual(len(lexical.calls), 0)
        self.assertEqual(len(vector.calls), 1)

    def test_hybrid_request_calls_both(self):
        lexical = FakeRetriever([result("c1")])
        vector = FakeRetriever([result("c2")])
        service = HybridSearchService(lexical, vector)

        service.search(RetrievalRequest(query="invoice", query_embedding=[0.1, 0.2]))

        self.assertEqual(len(lexical.calls), 1)
        self.assertEqual(len(vector.calls), 1)

    def test_same_request_object_passed_to_both_retrievers(self):
        lexical = FakeRetriever([])
        vector = FakeRetriever([])
        service = HybridSearchService(lexical, vector)
        request = RetrievalRequest(query="invoice", query_embedding=[0.1, 0.2], top_k=3)

        service.search(request)

        self.assertIs(lexical.calls[0], request)
        self.assertIs(vector.calls[0], request)


class TestHybridSearchServiceFusion(unittest.TestCase):
    def test_results_are_fused_via_rrf(self):
        lexical = FakeRetriever([result("shared"), result("only_lexical")])
        vector = FakeRetriever([result("shared"), result("only_vector")])
        service = HybridSearchService(lexical, vector)

        fused = service.search(
            RetrievalRequest(query="x", query_embedding=[0.1], top_k=10)
        )

        chunk_ids = {r.chunk_id for r in fused}
        self.assertEqual(chunk_ids, {"shared", "only_lexical", "only_vector"})
        # the chunk present in both lists must rank first
        self.assertEqual(fused[0].chunk_id, "shared")

    def test_lexical_only_results_pass_through_rrf_unchanged_in_membership(self):
        lexical = FakeRetriever([result("c1"), result("c2")])
        vector = FakeRetriever([])
        service = HybridSearchService(lexical, vector)

        fused = service.search(RetrievalRequest(query="x"))

        self.assertEqual({r.chunk_id for r in fused}, {"c1", "c2"})

    def test_custom_rrf_config_is_honored(self):
        lexical = FakeRetriever([result(f"c{i}") for i in range(5)])
        vector = FakeRetriever([])
        service = HybridSearchService(lexical, vector, rrf_config=RRFConfig(k=60, top_k=2))

        fused = service.search(RetrievalRequest(query="x"))

        self.assertEqual(len(fused), 2)

    def test_provenance_fields_survive_hybrid_fusion(self):
        lexical = FakeRetriever(
            [
                RetrievalResult(
                    chunk_id="c1",
                    document_id="doc-9",
                    content="Invoice INV-2026-01847 was issued in July.",
                    score=0.5,
                    metadata={"element_type": "text", "page_number": 4},
                )
            ]
        )
        vector = FakeRetriever([])
        service = HybridSearchService(lexical, vector)

        fused = service.search(RetrievalRequest(query="invoice"))

        self.assertEqual(fused[0].document_id, "doc-9")
        self.assertEqual(fused[0].content, "Invoice INV-2026-01847 was issued in July.")
        self.assertEqual(fused[0].metadata, {"element_type": "text", "page_number": 4})


if __name__ == "__main__":
    unittest.main()
