import unittest

from fastapi.testclient import TestClient

from app.api.dependencies import get_retrieval_service
from app.main import app
from app.retrieval.models import RetrievalResult


def make_result(chunk_id: str = "c1", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="Revenue increased by 18% in July for Client A.",
        score=score,
        metadata={"element_type": "text"},
    )


class FakeRetrievalService:
    def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None):
        self.results = results if results is not None else [make_result()]
        self.error = error
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query: str, top_k: int = 10, document_id: str | None = None):
        self.calls.append((query, top_k, document_id))
        if self.error is not None:
            raise self.error
        return self.results


def override_retrieval_service(fake: FakeRetrievalService):
    app.dependency_overrides[get_retrieval_service] = lambda: fake


class BaseQueryTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Successful retrieval
# ---------------------------------------------------------------------------


class TestSuccessfulQuery(BaseQueryTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeRetrievalService(
            results=[make_result("c1", 0.9), make_result("c2", 0.5)]
        )
        override_retrieval_service(self.fake)

    def test_returns_200(self):
        response = self.client.post("/query", json={"query": "what was revenue growth?"})
        self.assertEqual(response.status_code, 200)

    def test_response_shape(self):
        response = self.client.post("/query", json={"query": "what was revenue growth?"})
        body = response.json()
        self.assertEqual(body["query"], "what was revenue growth?")
        self.assertEqual(body["result_count"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["chunk_id"], "c1")
        self.assertEqual(body["results"][0]["score"], 0.9)
        self.assertEqual(body["results"][0]["metadata"], {"element_type": "text"})

    def test_retrieval_service_was_called(self):
        self.client.post("/query", json={"query": "what was revenue growth?"})
        self.assertEqual(len(self.fake.calls), 1)

    def test_default_top_k_passed_through(self):
        self.client.post("/query", json={"query": "some query"})
        query, top_k, document_id = self.fake.calls[0]
        self.assertEqual(top_k, 10)

    def test_custom_top_k_passed_through(self):
        self.client.post("/query", json={"query": "some query", "top_k": 3})
        _, top_k, _ = self.fake.calls[0]
        self.assertEqual(top_k, 3)

    def test_document_id_passed_through(self):
        self.client.post(
            "/query", json={"query": "some query", "document_id": "doc-42"}
        )
        _, _, document_id = self.fake.calls[0]
        self.assertEqual(document_id, "doc-42")

    def test_document_id_defaults_to_none(self):
        self.client.post("/query", json={"query": "some query"})
        _, _, document_id = self.fake.calls[0]
        self.assertIsNone(document_id)


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults(BaseQueryTestCase):
    def test_empty_results_returns_200_with_empty_list(self):
        fake = FakeRetrievalService(results=[])
        override_retrieval_service(fake)

        response = self.client.post("/query", json={"query": "no matches for this"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["result_count"], 0)
        self.assertEqual(body["results"], [])


# ---------------------------------------------------------------------------
# Invalid / empty query
# ---------------------------------------------------------------------------


class TestInvalidQuery(BaseQueryTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeRetrievalService()
        override_retrieval_service(self.fake)

    def test_missing_query_field_returns_422(self):
        response = self.client.post("/query", json={})
        self.assertEqual(response.status_code, 422)

    def test_empty_query_string_returns_422(self):
        response = self.client.post("/query", json={"query": ""})
        self.assertEqual(response.status_code, 422)

    def test_whitespace_only_query_maps_to_400(self):
        # Passes the schema's min_length=1 check (length 3) but is
        # rejected deeper by RetrievalService itself -- proving the
        # route correctly maps that ValueError to a 400, not a 500.
        fake = FakeRetrievalService(error=ValueError("query must not be empty"))
        override_retrieval_service(fake)

        response = self.client.post("/query", json={"query": "   "})

        self.assertEqual(response.status_code, 400)

    def test_invalid_query_never_reaches_retrieval_service_for_schema_failures(self):
        self.client.post("/query", json={})
        self.assertEqual(self.fake.calls, [])


# ---------------------------------------------------------------------------
# top_k validation
# ---------------------------------------------------------------------------


class TestTopKValidation(BaseQueryTestCase):
    def setUp(self):
        super().setUp()
        override_retrieval_service(FakeRetrievalService())

    def test_zero_top_k_rejected(self):
        response = self.client.post("/query", json={"query": "x", "top_k": 0})
        self.assertEqual(response.status_code, 422)

    def test_negative_top_k_rejected(self):
        response = self.client.post("/query", json={"query": "x", "top_k": -5})
        self.assertEqual(response.status_code, 422)

    def test_excessive_top_k_rejected(self):
        response = self.client.post("/query", json={"query": "x", "top_k": 100000})
        self.assertEqual(response.status_code, 422)

    def test_top_k_at_upper_bound_accepted(self):
        response = self.client.post("/query", json={"query": "x", "top_k": 100})
        self.assertEqual(response.status_code, 200)

    def test_top_k_of_one_accepted(self):
        response = self.client.post("/query", json={"query": "x", "top_k": 1})
        self.assertEqual(response.status_code, 200)


# ---------------------------------------------------------------------------
# document_id validation
# ---------------------------------------------------------------------------


class TestDocumentIdValidation(BaseQueryTestCase):
    def setUp(self):
        super().setUp()
        override_retrieval_service(FakeRetrievalService())

    def test_empty_document_id_rejected(self):
        response = self.client.post("/query", json={"query": "x", "document_id": ""})
        self.assertEqual(response.status_code, 422)


# ---------------------------------------------------------------------------
# Service-level failure -> 500, no internal details leaked
# ---------------------------------------------------------------------------


class TestServiceFailure(BaseQueryTestCase):
    def test_unexpected_exception_maps_to_500(self):
        secret = "postgresql+psycopg://raguser:supersecret@db.example.com/prod"
        fake = FakeRetrievalService(error=RuntimeError(f"connection failed: {secret}"))
        override_retrieval_service(fake)

        response = self.client.post("/query", json={"query": "some query"})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("supersecret", response.text)
        self.assertNotIn(secret, response.text)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestQueryRouteRegistration(BaseQueryTestCase):
    def test_query_path_registered_in_openapi_schema(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/query", schema["paths"])

    def test_query_is_a_post_operation(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("post", schema["paths"]["/query"])

    def test_health_and_upload_still_registered(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/health", schema["paths"])
        self.assertIn("/upload", schema["paths"])


if __name__ == "__main__":
    unittest.main()
