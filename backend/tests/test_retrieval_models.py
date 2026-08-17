import unittest

from pydantic import ValidationError

from app.retrieval.models import RetrievalRequest, RetrievalResult


class TestRetrievalRequest(unittest.TestCase):
    def test_query_only_is_valid(self):
        request = RetrievalRequest(query="revenue trend")
        self.assertEqual(request.query, "revenue trend")
        self.assertIsNone(request.query_embedding)

    def test_query_embedding_only_is_valid(self):
        request = RetrievalRequest(query_embedding=[0.1, 0.2, 0.3])
        self.assertEqual(request.query_embedding, [0.1, 0.2, 0.3])
        self.assertIsNone(request.query)

    def test_both_query_and_embedding_is_valid(self):
        request = RetrievalRequest(query="revenue trend", query_embedding=[0.1, 0.2])
        self.assertEqual(request.query, "revenue trend")
        self.assertEqual(request.query_embedding, [0.1, 0.2])

    def test_rejects_neither_query_nor_embedding(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest()

    def test_rejects_empty_query_string(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(query="")

    def test_rejects_empty_embedding_list(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(query_embedding=[])

    def test_default_top_k(self):
        request = RetrievalRequest(query="x")
        self.assertEqual(request.top_k, 10)

    def test_rejects_non_positive_top_k(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(query="x", top_k=0)

    def test_document_id_filter_optional(self):
        request = RetrievalRequest(query="x")
        self.assertIsNone(request.document_id)
        scoped = RetrievalRequest(query="x", document_id="doc-1")
        self.assertEqual(scoped.document_id, "doc-1")

    def test_rejects_empty_document_id(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(query="x", document_id="")

    def test_no_tenant_field_exists(self):
        self.assertNotIn("tenant_id", RetrievalRequest.model_fields)


class TestRetrievalResult(unittest.TestCase):
    def test_valid_construction(self):
        result = RetrievalResult(
            chunk_id="c1",
            document_id="doc-1",
            content="Revenue increased by 18% in July.",
            score=0.83,
            metadata={"element_type": "text"},
        )
        self.assertEqual(result.chunk_id, "c1")
        self.assertEqual(result.score, 0.83)

    def test_metadata_defaults_to_empty_dict(self):
        result = RetrievalResult(
            chunk_id="c1", document_id="doc-1", content="x", score=1.0
        )
        self.assertEqual(result.metadata, {})

    def test_rejects_empty_chunk_id(self):
        with self.assertRaises(ValidationError):
            RetrievalResult(chunk_id="", document_id="doc-1", content="x", score=1.0)

    def test_rejects_empty_document_id(self):
        with self.assertRaises(ValidationError):
            RetrievalResult(chunk_id="c1", document_id="", content="x", score=1.0)

    def test_rejects_empty_content(self):
        with self.assertRaises(ValidationError):
            RetrievalResult(chunk_id="c1", document_id="doc-1", content="", score=1.0)

    def test_score_accepts_negative_and_zero(self):
        # RRF/cosine-derived scores are not guaranteed positive; the model
        # itself should not impose a domain-specific bound.
        RetrievalResult(chunk_id="c1", document_id="doc-1", content="x", score=0.0)
        RetrievalResult(chunk_id="c1", document_id="doc-1", content="x", score=-0.5)


if __name__ == "__main__":
    unittest.main()
