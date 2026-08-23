from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import ChunkEmbeddingService
from app.ingestion.base import DocumentLoader
from app.ingestion.chunkers.service import ChunkingService
from app.ingestion.loaders.csv import CSVLoader
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.excel import ExcelLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.models.chunk import Chunk
from app.models.document import NormalizedDocument


class IngestionError(Exception):
    """Base class for failures raised by IngestionService.ingest()."""


class InvalidInputError(IngestionError, ValueError):
    """The arguments passed to ingest() were invalid (e.g. empty path)."""


class UnsupportedFileTypeError(IngestionError, ValueError):
    """No loader is registered for the input file's extension."""


class EmbeddingGenerationError(IngestionError, RuntimeError):
    """Generating embeddings for the document's chunks failed."""


class IndexingFailedError(IngestionError, RuntimeError):
    """Persisting/indexing the document's chunks failed."""


class IndexingServiceLike(Protocol):
    """The only surface IngestionService needs from an indexing service --
    matches app.indexing.service.IndexingService structurally, so real and
    fake indexing services are interchangeable without inheritance.
    """

    def index_chunks(
        self, chunks: list[Chunk], embeddings: dict[str, list[float]] | None = None
    ) -> None: ...


class IngestionResult(BaseModel):
    """Summary of one ingest() call. Every field here is derived from
    objects the pipeline already produced -- nothing is recomputed or
    re-derived from the database.
    """

    document_id: str
    filename: str
    file_type: str
    element_count: int
    element_counts_by_type: dict[str, int]
    chunk_count: int
    chunk_counts_by_type: dict[str, int]
    embedding_model: str
    indexed_chunk_count: int
    status: Literal["success"] = "success"


def default_loaders() -> dict[str, DocumentLoader]:
    """The built-in extension -> loader registry. File-type dispatch is
    core, stable domain logic (not an external policy like which
    database or embedding model to use), so it lives here rather than
    being mandatory to inject -- but IngestionService still accepts a
    `loaders` override for tests or a custom PDFLoader(asset_dir=...).
    """
    return {
        ".csv": CSVLoader(),
        ".docx": DocxLoader(),
        ".xlsx": ExcelLoader(),
        ".xls": ExcelLoader(),
        ".pdf": PDFLoader(),
    }


class IngestionService:
    """Orchestrates the full ingestion pipeline end to end:

        file
          -> DocumentLoader                  (existing, per file type)
          -> NormalizedDocument               (existing models)
          -> ChunkingService                  (existing)
          -> Chunk[]
          -> ChunkEmbeddingService(EmbeddingProvider)   (existing)
          -> {chunk_id: vector}
          -> IndexingService.index_chunks(chunks, embeddings)   (existing)

    Every stage is an already-existing, independently tested component.
    This class only sequences them and adds stage-labeled error context
    where that context doesn't already exist -- it does not reimplement
    loading, normalization, chunking, embedding, or indexing, and it
    never talks to PostgreSQL directly (all persistence goes through the
    injected IndexingService).

    Dependency injection: embedding_provider, indexing_service,
    chunking_service, and loaders are all constructor arguments. Nothing
    about a specific embedding model, provider, database URL, or API key
    is hardcoded here -- swapping DeterministicTestEmbeddingProvider for
    BGEEmbeddingProvider or GeminiEmbeddingProvider (or any future
    EmbeddingProvider) requires no change to this class.

    Error handling: loader exceptions (FileNotFoundError, ValueError for
    corrupt/invalid files, etc.) propagate unchanged -- each loader
    already raises clear, specific, well-tested exceptions, and wrapping
    them here would only discard that specificity for no benefit.
    Embedding and indexing failures ARE wrapped (EmbeddingGenerationError
    / IndexingFailedError), because the underlying exceptions there
    (ValueError, RuntimeError, raw SQLAlchemy errors) carry no indication
    of which document or which pipeline stage failed on their own; the
    original exception is always preserved via `raise ... from exc`.
    Wrapped messages never include str(exc) -- only the exception's type
    name -- since some underlying errors (SQLAlchemy connection errors,
    provider SDK errors) can otherwise echo connection strings or request
    details.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        indexing_service: IndexingServiceLike,
        chunking_service: ChunkingService | None = None,
        loaders: dict[str, DocumentLoader] | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.indexing_service = indexing_service
        self.chunking_service = chunking_service or ChunkingService()
        self.loaders = loaders if loaders is not None else default_loaders()
        self._chunk_embedding_service = ChunkEmbeddingService(embedding_provider)

    def ingest(self, file_path: str | Path, uploaded_by: str | None = None) -> IngestionResult:
        if not str(file_path).strip():
            raise InvalidInputError("file_path must not be empty")

        path = Path(file_path)
        loader = self._select_loader(path)

        # Loader exceptions (FileNotFoundError, ValueError, ...) propagate
        # unchanged -- see class docstring.
        document = loader.load(path, uploaded_by=uploaded_by)

        chunks = self.chunking_service.chunk_document(document)

        embeddings = self._embed_chunks(chunks, document.document_id)
        self._index_chunks(chunks, embeddings, document.document_id)

        return self._build_result(document, chunks)

    def _select_loader(self, path: Path) -> DocumentLoader:
        suffix = path.suffix.lower()
        loader = self.loaders.get(suffix)
        if loader is None:
            supported = ", ".join(sorted(self.loaders)) or "(none configured)"
            raise UnsupportedFileTypeError(
                f"No loader configured for file extension {suffix!r} "
                f"(file: {path.name}). Supported extensions: {supported}"
            )
        return loader

    def _embed_chunks(self, chunks: list[Chunk], document_id: str) -> dict[str, list[float]]:
        try:
            return self._chunk_embedding_service.embed_chunks(chunks)
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings for document {document_id!r} "
                f"({type(exc).__name__})"
            ) from exc

    def _index_chunks(
        self, chunks: list[Chunk], embeddings: dict[str, list[float]], document_id: str
    ) -> None:
        try:
            self.indexing_service.index_chunks(chunks, embeddings=embeddings)
        except Exception as exc:
            raise IndexingFailedError(
                f"Failed to index document {document_id!r} ({type(exc).__name__})"
            ) from exc

    def _build_result(
        self, document: NormalizedDocument, chunks: list[Chunk]
    ) -> IngestionResult:
        element_counts_by_type: dict[str, int] = {}
        for element in document.elements:
            element_counts_by_type[element.element_type] = (
                element_counts_by_type.get(element.element_type, 0) + 1
            )

        chunk_counts_by_type: dict[str, int] = {}
        for chunk in chunks:
            element_type = chunk.metadata.get("element_type", "unknown")
            chunk_counts_by_type[element_type] = chunk_counts_by_type.get(element_type, 0) + 1

        return IngestionResult(
            document_id=document.document_id,
            filename=document.metadata.filename,
            file_type=document.metadata.file_type,
            element_count=len(document.elements),
            element_counts_by_type=element_counts_by_type,
            chunk_count=len(chunks),
            chunk_counts_by_type=chunk_counts_by_type,
            embedding_model=self.embedding_provider.config.model_name,
            # IndexingService.index_chunks() is currently all-or-nothing
            # (it raises rather than returning partial results), so this
            # always equals chunk_count today. Kept as its own field since
            # it represents a distinct concept (what got indexed vs. what
            # got chunked) that may diverge if partial-failure indexing
            # is introduced later.
            indexed_chunk_count=len(chunks),
        )
