import unittest

from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_service
from app.main import app
from app.services.rag_service import (
    GenerationFailedError,
    InvalidQuestionError,
    RAGAnswer,
    RAGSource,
    RetrievalFailedError,
)


def make_source(chunk_id: str = "c1", score: float = 0.9) -> RAGSource:
    return RAGSource(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="Revenue increased by 18% in July for Client A.",
        score=score,
        metadata={"element_type": "text", "page_number": 3},
    )


def make_answer(sources: list[RAGSource] | None = None) -> RAGAnswer:
    source_items = sources if sources is not None else [make_source()]
    return RAGAnswer(
        question="What was revenue growth?",
        answer="Revenue grew 18% in July.",
        retrieved_count=len(source_items),
        sources=source_items,
    )


class FakeRAGService:
    def __init__(self, result: RAGAnswer | None = None, error: Exception | None = None):
        self.result = result if result is not None else make_answer()
        self.error = error
        self.calls: list[tuple[str, int, str | None]] = []

    def answer(self, question: str, top_k: int = 10, document_id: str | None = None):
        self.calls.append((question, top_k, document_id))
        if self.error is not None:
            raise self.error
        return self.result


def override_rag_service(fake: FakeRAGService):
    app.dependency_overrides[get_rag_service] = lambda: fake


class BaseAskTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Successful answer generation
# ---------------------------------------------------------------------------


class TestSuccessfulAsk(BaseAskTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeRAGService(
            result=make_answer([make_source("c1", 0.9), make_source("c2", 0.5)])
        )
        override_rag_service(self.fake)

    def test_returns_200(self):
        response = self.client.post("/ask", json={"question": "What was revenue growth?"})
        self.assertEqual(response.status_code, 200)

    def test_response_shape(self):
        response = self.client.post("/ask", json={"question": "What was revenue growth?"})
        body = response.json()
        self.assertEqual(body["question"], "What was revenue growth?")
        self.assertEqual(body["answer"], "Revenue grew 18% in July.")
        self.assertEqual(body["retrieved_count"], 2)
        self.assertEqual(len(body["sources"]), 2)
        self.assertEqual(body["sources"][0]["chunk_id"], "c1")
        self.assertEqual(body["sources"][0]["score"], 0.9)
        self.assertEqual(body["sources"][0]["metadata"], {"element_type": "text", "page_number": 3})

    def test_rag_service_was_called(self):
        self.client.post("/ask", json={"question": "What was revenue growth?"})
        self.assertEqual(len(self.fake.calls), 1)

    def test_default_top_k_passed_through(self):
        self.client.post("/ask", json={"question": "some question"})
        _, top_k, _ = self.fake.calls[0]
        self.assertEqual(top_k, 10)

    def test_custom_top_k_passed_through(self):
        self.client.post("/ask", json={"question": "some question", "top_k": 3})
        _, top_k, _ = self.fake.calls[0]
        self.assertEqual(top_k, 3)

    def test_document_id_passed_through(self):
        self.client.post(
            "/ask", json={"question": "some question", "document_id": "doc-42"}
        )
        _, _, document_id = self.fake.calls[0]
        self.assertEqual(document_id, "doc-42")

    def test_document_id_defaults_to_none(self):
        self.client.post("/ask", json={"question": "some question"})
        _, _, document_id = self.fake.calls[0]
        self.assertIsNone(document_id)


# ---------------------------------------------------------------------------
# Empty retrieval results
# ---------------------------------------------------------------------------


class TestAskWithNoContext(BaseAskTestCase):
    def test_empty_sources_return_200_with_empty_sources(self):
        fake = FakeRAGService(
            result=RAGAnswer(
                question="no matches",
                answer="I couldn't find any relevant information in the indexed documents to answer this question.",
                retrieved_count=0,
                sources=[],
            )
        )
        override_rag_service(fake)

        response = self.client.post("/ask", json={"question": "no matches"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["retrieved_count"], 0)
        self.assertEqual(body["sources"], [])


# ---------------------------------------------------------------------------
# Invalid question / request validation
# ---------------------------------------------------------------------------


class TestInvalidAskRequest(BaseAskTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeRAGService()
        override_rag_service(self.fake)

    def test_missing_question_field_returns_422(self):
        response = self.client.post("/ask", json={})
        self.assertEqual(response.status_code, 422)

    def test_empty_question_string_returns_422(self):
        response = self.client.post("/ask", json={"question": ""})
        self.assertEqual(response.status_code, 422)

    def test_whitespace_only_question_maps_to_400(self):
        fake = FakeRAGService(error=InvalidQuestionError("question must not be empty"))
        override_rag_service(fake)

        response = self.client.post("/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 400)

    def test_schema_failures_never_reach_rag_service(self):
        self.client.post("/ask", json={})
        self.assertEqual(self.fake.calls, [])


# ---------------------------------------------------------------------------
# top_k and document_id validation
# ---------------------------------------------------------------------------


class TestAskParameterValidation(BaseAskTestCase):
    def setUp(self):
        super().setUp()
        override_rag_service(FakeRAGService())

    def test_zero_top_k_rejected(self):
        response = self.client.post("/ask", json={"question": "x", "top_k": 0})
        self.assertEqual(response.status_code, 422)

    def test_negative_top_k_rejected(self):
        response = self.client.post("/ask", json={"question": "x", "top_k": -5})
        self.assertEqual(response.status_code, 422)

    def test_excessive_top_k_rejected(self):
        response = self.client.post("/ask", json={"question": "x", "top_k": 100000})
        self.assertEqual(response.status_code, 422)

    def test_top_k_at_upper_bound_accepted(self):
        response = self.client.post("/ask", json={"question": "x", "top_k": 100})
        self.assertEqual(response.status_code, 200)

    def test_empty_document_id_rejected(self):
        response = self.client.post("/ask", json={"question": "x", "document_id": ""})
        self.assertEqual(response.status_code, 422)


# ---------------------------------------------------------------------------
# Service-level failures -> mapped responses, no internal details leaked
# ---------------------------------------------------------------------------


class TestAskServiceFailures(BaseAskTestCase):
    def test_retrieval_failure_maps_to_500_without_leaking_details(self):
        secret = "postgresql+psycopg://raguser:supersecret@db.example.com/prod"
        fake = FakeRAGService(error=RetrievalFailedError(f"connection failed: {secret}"))
        override_rag_service(fake)

        response = self.client.post("/ask", json={"question": "some question"})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("supersecret", response.text)
        self.assertNotIn(secret, response.text)

    def test_generation_failure_maps_to_500_without_leaking_details(self):
        secret = "AIzaSyFAKESECRETVALUEFORTESTINGONLY123"
        fake = FakeRAGService(error=GenerationFailedError(f"401 unauthorized, key={secret}"))
        override_rag_service(fake)

        response = self.client.post("/ask", json={"question": "some question"})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(secret, response.text)

    def test_unexpected_exception_maps_to_500_without_leaking_details(self):
        secret = "hunter2-db-password"
        fake = FakeRAGService(error=RuntimeError(f"unexpected failure: {secret}"))
        override_rag_service(fake)

        response = self.client.post("/ask", json={"question": "some question"})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(secret, response.text)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestAskRouteRegistration(BaseAskTestCase):
    def test_ask_path_registered_in_openapi_schema(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/ask", schema["paths"])

    def test_ask_is_a_post_operation(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("post", schema["paths"]["/ask"])

    def test_existing_routes_still_registered(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/health", schema["paths"])
        self.assertIn("/upload", schema["paths"])
        self.assertIn("/query", schema["paths"])


if __name__ == "__main__":
    unittest.main()
