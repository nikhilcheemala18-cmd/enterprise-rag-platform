import unittest
from pathlib import Path

from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig
from app.embeddings.local import DeterministicTestEmbeddingProvider
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.indexing.vector import VectorIndexer
from app.ingestion.base import DocumentLoader
from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.service import ChunkingService
from app.ingestion.loaders.csv import CSVLoader
from app.models.document import (
    CSVLocation,
    DocumentMetadata,
    NormalizedDocument,
    SourceReference,
    TableElement,
    TextElement,
)
from app.services.ingestion_service import (
    EmbeddingGenerationError,
    IndexingFailedError,
    IngestionError,
    IngestionResult,
    IngestionService,
    InvalidInputError,
    UnsupportedFileTypeError,
    default_loaders,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLoader(DocumentLoader):
    def __init__(self, document: NormalizedDocument | None = None, error: Exception | None = None):
        self.document = document
        self.error = error
        self.calls: list[tuple] = []

    def load(self, file_path, uploaded_by=None) -> NormalizedDocument:
        self.calls.append((file_path, uploaded_by))
        if self.error is not None:
            raise self.error
        return self.document


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, error: Exception | None = None):
        super().__init__(config)
        self.error = error
        self.calls: list[list[str]] = []

    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        self.calls.append(list(texts))
        return [[0.1] * self.config.dimension for _ in texts]


class FakeIndexingService:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.indexed_chunks = None
        self.indexed_embeddings = None

    def index_chunks(self, chunks, embeddings=None):
        if self.error is not None:
            raise self.error
        self.indexed_chunks = chunks
        self.indexed_embeddings = embeddings


class FakeConnection:
    def __init__(self, log: list):
        self.log = log

    def execute(self, stmt):
        self.log.append(stmt)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeEngine:
    """Same pattern as tests/test_indexing_service.py: records every
    statement without opening a real PostgreSQL connection.
    """

    def __init__(self):
        self.log: list = []

    def begin(self):
        return FakeConnection(self.log)

    def connect(self):
        return FakeConnection(self.log)


def make_config(dimension: int = 8, model_name: str = "fake-model") -> EmbeddingConfig:
    return EmbeddingConfig(provider_name="fake", model_name=model_name, dimension=dimension)


def make_normalized_document(document_id: str = "doc-1") -> NormalizedDocument:
    def source() -> SourceReference:
        return SourceReference(
            document_id=document_id,
            source_type="csv",
            location=CSVLocation(row_start=1, row_end=2),
        )

    return NormalizedDocument(
        document_id=document_id,
        metadata=DocumentMetadata(
            filename="report.csv", file_type="csv", file_size=123, content_hash="abc123"
        ),
        elements=[
            TextElement(
                element_id="el-1",
                element_index=0,
                source=source(),
                content="Revenue increased by 18% in July for Client A.",
            ),
            TableElement(
                element_id="el-2",
                element_index=1,
                source=source(),
                columns=["Month", "Revenue"],
                rows=[["June", 72000], ["July", 85000]],
            ),
        ],
    )


def make_ingestion_service(
    document: NormalizedDocument | None = None,
    loader_error: Exception | None = None,
    embedding_error: Exception | None = None,
    indexing_error: Exception | None = None,
    dimension: int = 8,
    extension: str = ".csv",
    chunking_service: ChunkingService | None = None,
) -> tuple[IngestionService, FakeLoader, FakeEmbeddingProvider, FakeIndexingService]:
    loader = FakeLoader(document=document or make_normalized_document(), error=loader_error)
    provider = FakeEmbeddingProvider(make_config(dimension=dimension), error=embedding_error)
    indexing_service = FakeIndexingService(error=indexing_error)
    service = IngestionService(
        embedding_provider=provider,
        indexing_service=indexing_service,
        chunking_service=chunking_service,
        loaders={extension: loader},
    )
    return service, loader, provider, indexing_service


# ---------------------------------------------------------------------------
# 1. Successful end-to-end orchestration
# ---------------------------------------------------------------------------


class TestSuccessfulOrchestration(unittest.TestCase):
    def setUp(self):
        self.document = make_normalized_document()
        self.service, self.loader, self.provider, self.indexing = make_ingestion_service(
            document=self.document
        )
        self.result = self.service.ingest("some/path/report.csv", uploaded_by="user-1")

    def test_returns_ingestion_result(self):
        self.assertIsInstance(self.result, IngestionResult)
        self.assertEqual(self.result.status, "success")

    def test_document_identifier_matches(self):
        self.assertEqual(self.result.document_id, self.document.document_id)

    def test_filename_and_file_type_propagated(self):
        self.assertEqual(self.result.filename, "report.csv")
        self.assertEqual(self.result.file_type, "csv")

    def test_element_and_chunk_counts_populated(self):
        self.assertEqual(self.result.element_count, 2)
        self.assertGreater(self.result.chunk_count, 0)
        self.assertEqual(self.result.indexed_chunk_count, self.result.chunk_count)

    def test_embedding_model_reflects_injected_provider(self):
        self.assertEqual(self.result.embedding_model, "fake-model")

    def test_loader_called_with_correct_path_and_uploaded_by(self):
        self.assertEqual(self.loader.calls, [(Path("some/path/report.csv"), "user-1")])

    def test_indexing_service_received_chunks(self):
        self.assertIsNotNone(self.indexing.indexed_chunks)
        self.assertEqual(len(self.indexing.indexed_chunks), self.result.chunk_count)


# ---------------------------------------------------------------------------
# 2. Loader selection
# ---------------------------------------------------------------------------


class TestLoaderSelection(unittest.TestCase):
    def test_selects_loader_matching_extension(self):
        document = make_normalized_document()
        csv_loader = FakeLoader(document=document)
        docx_loader = FakeLoader(document=document)
        service = IngestionService(
            embedding_provider=FakeEmbeddingProvider(make_config()),
            indexing_service=FakeIndexingService(),
            loaders={".csv": csv_loader, ".docx": docx_loader},
        )

        service.ingest("report.csv")

        self.assertEqual(len(csv_loader.calls), 1)
        self.assertEqual(len(docx_loader.calls), 0)

    def test_extension_matching_is_case_insensitive(self):
        document = make_normalized_document()
        loader = FakeLoader(document=document)
        service = IngestionService(
            embedding_provider=FakeEmbeddingProvider(make_config()),
            indexing_service=FakeIndexingService(),
            loaders={".csv": loader},
        )

        service.ingest("REPORT.CSV")

        self.assertEqual(len(loader.calls), 1)

    def test_default_loaders_cover_all_supported_extensions(self):
        loaders = default_loaders()
        self.assertEqual(
            set(loaders.keys()), {".csv", ".docx", ".xlsx", ".xls", ".pdf"}
        )


# ---------------------------------------------------------------------------
# 3. Unsupported file handling
# ---------------------------------------------------------------------------


class TestUnsupportedFileHandling(unittest.TestCase):
    def setUp(self):
        self.provider = FakeEmbeddingProvider(make_config())
        self.indexing = FakeIndexingService()
        self.service = IngestionService(
            embedding_provider=self.provider,
            indexing_service=self.indexing,
            loaders={".csv": FakeLoader(document=make_normalized_document())},
        )

    def test_raises_unsupported_file_type_error(self):
        with self.assertRaises(UnsupportedFileTypeError):
            self.service.ingest("presentation.pptx")

    def test_unsupported_file_type_error_is_also_value_error(self):
        with self.assertRaises(ValueError):
            self.service.ingest("presentation.pptx")

    def test_unsupported_file_type_error_is_also_ingestion_error(self):
        with self.assertRaises(IngestionError):
            self.service.ingest("presentation.pptx")

    def test_message_names_supported_extensions(self):
        with self.assertRaises(UnsupportedFileTypeError) as ctx:
            self.service.ingest("presentation.pptx")
        self.assertIn(".csv", str(ctx.exception))
        self.assertIn(".pptx", str(ctx.exception))

    def test_no_embedding_or_indexing_attempted(self):
        with self.assertRaises(UnsupportedFileTypeError):
            self.service.ingest("presentation.pptx")
        self.assertEqual(self.provider.calls, [])
        self.assertIsNone(self.indexing.indexed_chunks)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestInvalidInput(unittest.TestCase):
    def setUp(self):
        self.service, *_ = make_ingestion_service()

    def test_empty_string_path_rejected(self):
        with self.assertRaises(InvalidInputError):
            self.service.ingest("")

    def test_whitespace_only_path_rejected(self):
        with self.assertRaises(InvalidInputError):
            self.service.ingest("   ")

    def test_invalid_input_error_is_also_value_error(self):
        with self.assertRaises(ValueError):
            self.service.ingest("")


# ---------------------------------------------------------------------------
# 4. Data propagation between stages
# ---------------------------------------------------------------------------


class TestDataPropagation(unittest.TestCase):
    def setUp(self):
        self.document = make_normalized_document()
        self.service, _, _, self.indexing = make_ingestion_service(document=self.document)
        self.result = self.service.ingest("report.csv")

    def test_every_chunk_carries_the_documents_id(self):
        for chunk in self.indexing.indexed_chunks:
            self.assertEqual(chunk.document_id, self.document.document_id)

    def test_source_element_ids_trace_back_to_real_elements(self):
        real_element_ids = {e.element_id for e in self.document.elements}
        for chunk in self.indexing.indexed_chunks:
            for element_id in chunk.source_element_ids:
                self.assertIn(element_id, real_element_ids)

    def test_element_counts_by_type(self):
        self.assertEqual(self.result.element_counts_by_type, {"text": 1, "table": 1})

    def test_chunk_counts_by_type_sum_to_chunk_count(self):
        total = sum(self.result.chunk_counts_by_type.values())
        self.assertEqual(total, self.result.chunk_count)


# ---------------------------------------------------------------------------
# 5. Embeddings generated and passed to indexing
# ---------------------------------------------------------------------------


class TestEmbeddingsReachIndexing(unittest.TestCase):
    def setUp(self):
        self.service, _, self.provider, self.indexing = make_ingestion_service(dimension=8)
        self.result = self.service.ingest("report.csv")

    def test_provider_was_called_with_chunk_content(self):
        embedded_texts = [text for call in self.provider.calls for text in call]
        indexed_contents = [c.content for c in self.indexing.indexed_chunks]
        for content in indexed_contents:
            self.assertTrue(
                any(content in text or text in content for text in embedded_texts)
            )

    def test_embeddings_dict_has_one_vector_per_chunk(self):
        self.assertEqual(
            set(self.indexing.indexed_embeddings.keys()),
            {c.chunk_id for c in self.indexing.indexed_chunks},
        )

    def test_embedding_vectors_have_configured_dimension(self):
        for vector in self.indexing.indexed_embeddings.values():
            self.assertEqual(len(vector), 8)


# ---------------------------------------------------------------------------
# 6. Dependency injection / provider flexibility
# ---------------------------------------------------------------------------


class TestDependencyInjectionFlexibility(unittest.TestCase):
    def test_swapping_embedding_provider_changes_reported_model_and_dimension(self):
        document = make_normalized_document()

        service_a = IngestionService(
            embedding_provider=FakeEmbeddingProvider(
                make_config(dimension=8, model_name="provider-a")
            ),
            indexing_service=FakeIndexingService(),
            loaders={".csv": FakeLoader(document=document)},
        )
        result_a = service_a.ingest("report.csv")

        service_b = IngestionService(
            embedding_provider=FakeEmbeddingProvider(
                make_config(dimension=32, model_name="provider-b")
            ),
            indexing_service=FakeIndexingService(),
            loaders={".csv": FakeLoader(document=document)},
        )
        result_b = service_b.ingest("report.csv")

        self.assertEqual(result_a.embedding_model, "provider-a")
        self.assertEqual(result_b.embedding_model, "provider-b")

    def test_custom_chunking_service_is_respected(self):
        document = make_normalized_document()
        document.elements[0].content = "word " * 50  # long enough to split

        default_service, _, _, default_indexing = make_ingestion_service(document=document)
        default_service.ingest("report.csv")

        tiny_chunking = ChunkingService(ChunkingConfig(text_chunk_size=5, text_chunk_overlap=1))
        small_service, _, _, small_indexing = make_ingestion_service(
            document=document, chunking_service=tiny_chunking
        )
        small_service.ingest("report.csv")

        self.assertGreater(len(small_indexing.indexed_chunks), len(default_indexing.indexed_chunks))

    def test_custom_loader_registry_is_respected(self):
        document = make_normalized_document()
        custom_loader = FakeLoader(document=document)
        service = IngestionService(
            embedding_provider=FakeEmbeddingProvider(make_config()),
            indexing_service=FakeIndexingService(),
            loaders={".weird": custom_loader},
        )
        service.ingest("file.weird")
        self.assertEqual(len(custom_loader.calls), 1)


# ---------------------------------------------------------------------------
# 7. Failure behavior at pipeline boundaries
# ---------------------------------------------------------------------------


class TestFailureBoundaries(unittest.TestCase):
    def test_loader_failure_propagates_unwrapped(self):
        service, _, provider, indexing = make_ingestion_service(
            loader_error=FileNotFoundError("no such file")
        )
        with self.assertRaises(FileNotFoundError):
            service.ingest("report.csv")
        # nothing downstream was attempted
        self.assertEqual(provider.calls, [])
        self.assertIsNone(indexing.indexed_chunks)

    def test_embedding_failure_is_wrapped_and_chained(self):
        original = ValueError("embedding dimension mismatch")
        service, _, _, indexing = make_ingestion_service(embedding_error=original)

        with self.assertRaises(EmbeddingGenerationError) as ctx:
            service.ingest("report.csv")

        self.assertIs(ctx.exception.__cause__, original)
        self.assertIsNone(indexing.indexed_chunks)  # indexing never reached

    def test_embedding_failure_is_also_runtime_error(self):
        service, *_ = make_ingestion_service(embedding_error=ValueError("bad"))
        with self.assertRaises(RuntimeError):
            service.ingest("report.csv")

    def test_indexing_failure_is_wrapped_and_chained(self):
        original = RuntimeError("connection refused")
        service, _, provider, _ = make_ingestion_service(indexing_error=original)

        with self.assertRaises(IndexingFailedError) as ctx:
            service.ingest("report.csv")

        self.assertIs(ctx.exception.__cause__, original)
        # embeddings WERE generated before the indexing stage failed
        self.assertTrue(len(provider.calls) > 0)

    def test_wrapped_errors_do_not_include_raw_exception_text(self):
        secret_looking_message = "connection to postgresql://user:hunter2@host failed"
        service, *_ = make_ingestion_service(indexing_error=RuntimeError(secret_looking_message))
        with self.assertRaises(IndexingFailedError) as ctx:
            service.ingest("report.csv")
        self.assertNotIn("hunter2", str(ctx.exception))


# ---------------------------------------------------------------------------
# 8. Realistic smoke test -- real loader, real chunker, real (deterministic)
# embedding provider, real IndexingService against a fake DB engine.
# ---------------------------------------------------------------------------


class TestRealisticPipelineSmoke(unittest.TestCase):
    def test_real_csv_file_through_the_real_pipeline(self):
        provider = DeterministicTestEmbeddingProvider(
            EmbeddingConfig(provider_name="deterministic-test", model_name="det-v1", dimension=8)
        )

        engine = FakeEngine()
        table = build_chunks_table(MetaData(), embedding_dimension=8)
        indexing_service = IndexingService(
            repository=ChunkRepository(engine, table),
            lexical_indexer=LexicalIndexer(engine, table),
            vector_indexer=VectorIndexer(engine, table, embedding_dimension=8),
        )

        service = IngestionService(
            embedding_provider=provider,
            indexing_service=indexing_service,
            loaders={".csv": CSVLoader()},
        )

        result = service.ingest(FIXTURES_DIR / "sample.csv", uploaded_by="smoke-test")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.filename, "sample.csv")
        self.assertEqual(result.file_type, "csv")
        self.assertEqual(result.element_count, 1)  # one TableElement for the whole CSV
        self.assertGreater(result.chunk_count, 0)
        self.assertEqual(result.embedding_model, "det-v1")

        # 3 SQL statements per chunk: repository upsert, lexical update, vector update
        self.assertEqual(len(engine.log), result.chunk_count * 3)
        compiled_upsert = str(engine.log[0].compile(dialect=postgresql.dialect()))
        self.assertIn("ON CONFLICT", compiled_upsert)


if __name__ == "__main__":
    unittest.main()
