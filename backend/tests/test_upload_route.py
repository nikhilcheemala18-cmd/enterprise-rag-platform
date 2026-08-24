import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_ingestion_service
from app.main import app
from app.services.ingestion_service import (
    EmbeddingGenerationError,
    IndexingFailedError,
    IngestionResult,
    UnsupportedFileTypeError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_SUPPORTED_LOADERS = {".csv": object(), ".pdf": object(), ".docx": object(), ".xlsx": object(), ".xls": object()}


class FakeIngestionService:
    """Records every ingest() call (including the actual temp file path
    used, while it still exists) so tests can assert on cleanup timing
    without needing a real loader/embedding/indexing pipeline.
    """

    def __init__(self, result: IngestionResult | None = None, error: Exception | None = None):
        self.loaders = dict(_SUPPORTED_LOADERS)
        self.result = result or IngestionResult(
            document_id="doc-abc",
            filename="some-internal-temp-name.csv",  # deliberately != original upload name
            file_type="csv",
            element_count=1,
            element_counts_by_type={"table": 1},
            chunk_count=2,
            chunk_counts_by_type={"table": 2},
            embedding_model="fake-model",
            indexed_chunk_count=2,
        )
        self.error = error
        self.calls: list[tuple[Path, str | None]] = []
        self.path_existed_during_call: bool | None = None

    def ingest(self, file_path, uploaded_by=None) -> IngestionResult:
        path = Path(file_path)
        self.calls.append((path, uploaded_by))
        self.path_existed_during_call = path.exists()
        if self.error is not None:
            raise self.error
        return self.result


def override_ingestion_service(fake: FakeIngestionService):
    app.dependency_overrides[get_ingestion_service] = lambda: fake


class BaseUploadTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Health endpoint still works
# ---------------------------------------------------------------------------


class TestHealthStillWorks(BaseUploadTestCase):
    def test_health_unaffected_by_upload_route(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})


# ---------------------------------------------------------------------------
# 2 & 3. Successful CSV upload, and it actually calls IngestionService
# ---------------------------------------------------------------------------


class TestSuccessfulUpload(BaseUploadTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeIngestionService()
        override_ingestion_service(self.fake)

    def test_successful_csv_upload_returns_201(self):
        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 201)

    def test_response_body_shape(self):
        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["document_id"], "doc-abc")
        self.assertEqual(body["chunk_count"], 2)
        self.assertEqual(body["indexed_chunk_count"], 2)

    def test_response_filename_is_original_upload_name_not_temp_name(self):
        response = self.client.post(
            "/upload",
            files={"file": ("my_original_report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(response.json()["filename"], "my_original_report.csv")
        self.assertNotEqual(response.json()["filename"], self.fake.result.filename)

    def test_ingestion_service_was_actually_called(self):
        self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(len(self.fake.calls), 1)

    def test_temp_file_path_has_original_suffix(self):
        self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        temp_path, _ = self.fake.calls[0]
        self.assertEqual(temp_path.suffix, ".csv")

    def test_uploaded_by_is_passed_through(self):
        self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
            data={"uploaded_by": "user-42"},
        )
        _, uploaded_by = self.fake.calls[0]
        self.assertEqual(uploaded_by, "user-42")

    def test_uploaded_by_optional(self):
        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 201)
        _, uploaded_by = self.fake.calls[0]
        self.assertIsNone(uploaded_by)


# ---------------------------------------------------------------------------
# 4. Unsupported file type
# ---------------------------------------------------------------------------


class TestUnsupportedFileType(BaseUploadTestCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeIngestionService()
        override_ingestion_service(self.fake)

    def test_unsupported_extension_returns_415(self):
        response = self.client.post(
            "/upload",
            files={"file": ("slides.pptx", b"not really pptx", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 415)

    def test_unsupported_extension_never_reaches_ingestion_service(self):
        self.client.post(
            "/upload",
            files={"file": ("slides.pptx", b"data", "application/octet-stream")},
        )
        self.assertEqual(self.fake.calls, [])

    def test_ingestion_service_raising_unsupported_file_type_error_maps_to_415(self):
        fake = FakeIngestionService(error=UnsupportedFileTypeError("no loader"))
        override_ingestion_service(fake)
        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 415)


# ---------------------------------------------------------------------------
# 5. Missing / empty upload
# ---------------------------------------------------------------------------


class TestMissingOrEmptyUpload(BaseUploadTestCase):
    def setUp(self):
        super().setUp()
        override_ingestion_service(FakeIngestionService())

    def test_no_file_field_at_all_is_rejected(self):
        response = self.client.post("/upload")
        self.assertIn(response.status_code, (400, 422))

    def test_empty_filename_is_rejected(self):
        # An empty filename makes Starlette parse the multipart part as a
        # plain form field rather than a file, so FastAPI's own request
        # validation rejects it (422) before our handler's `if not
        # file.filename` check ever runs. Either way it's a clean 4xx,
        # never a 500 and never an accepted upload.
        response = self.client.post(
            "/upload",
            files={"file": ("", b"some content", "text/csv")},
        )
        self.assertIn(response.status_code, (400, 422))


# ---------------------------------------------------------------------------
# 6 & 7. Temp file cleanup on success and on failure
# ---------------------------------------------------------------------------


class TestTempFileCleanup(BaseUploadTestCase):
    def test_temp_file_exists_during_call_and_is_removed_after_success(self):
        fake = FakeIngestionService()
        override_ingestion_service(fake)

        self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertTrue(fake.path_existed_during_call)
        temp_path, _ = fake.calls[0]
        self.assertFalse(temp_path.exists())

    def test_temp_file_removed_after_embedding_failure(self):
        fake = FakeIngestionService(error=EmbeddingGenerationError("boom"))
        override_ingestion_service(fake)

        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 500)
        self.assertTrue(fake.path_existed_during_call)
        temp_path, _ = fake.calls[0]
        self.assertFalse(temp_path.exists())

    def test_temp_file_removed_after_indexing_failure(self):
        fake = FakeIngestionService(error=IndexingFailedError("boom"))
        override_ingestion_service(fake)

        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 500)
        temp_path, _ = fake.calls[0]
        self.assertFalse(temp_path.exists())

    def test_temp_file_removed_after_unexpected_exception(self):
        fake = FakeIngestionService(error=RuntimeError("totally unexpected"))
        override_ingestion_service(fake)

        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 500)
        temp_path, _ = fake.calls[0]
        self.assertFalse(temp_path.exists())


# ---------------------------------------------------------------------------
# 8. Sensitive information is not exposed
# ---------------------------------------------------------------------------


class TestSensitiveInformationNotExposed(BaseUploadTestCase):
    def test_indexing_failure_does_not_leak_connection_string(self):
        secret = "postgresql+psycopg://raguser:supersecret123@db.example.com/prod"
        fake = FakeIngestionService(error=IndexingFailedError(f"connection failed: {secret}"))
        override_ingestion_service(fake)

        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("supersecret123", response.text)
        self.assertNotIn(secret, response.text)

    def test_embedding_failure_does_not_leak_api_key(self):
        secret = "AIzaSyFAKESECRETVALUEFORTESTINGONLY123"
        fake = FakeIngestionService(error=EmbeddingGenerationError(f"401 unauthorized, key={secret}"))
        override_ingestion_service(fake)

        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(secret, response.text)

    def test_unexpected_exception_does_not_leak_raw_message(self):
        secret = "hunter2-db-password"
        fake = FakeIngestionService(error=RuntimeError(f"unexpected failure: {secret}"))
        override_ingestion_service(fake)

        response = self.client.post(
            "/upload",
            files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(secret, response.text)


# ---------------------------------------------------------------------------
# 9. Route registration
# ---------------------------------------------------------------------------


class TestRouteRegistration(BaseUploadTestCase):
    def test_upload_path_registered_in_openapi_schema(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/upload", schema["paths"])

    def test_upload_is_a_post_operation(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("post", schema["paths"]["/upload"])

    def test_health_still_registered_alongside_upload(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/health", schema["paths"])


if __name__ == "__main__":
    unittest.main()
