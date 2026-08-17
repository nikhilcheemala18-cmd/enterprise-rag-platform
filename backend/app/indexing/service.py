from app.indexing.lexical import LexicalIndexer
from app.indexing.repository import ChunkRepository
from app.indexing.vector import VectorIndexer
from app.models.chunk import Chunk


class IndexingService:
    """Coordinates persisting Chunks and updating their search indexes.

    Does not generate embeddings -- an embedding vector may optionally be
    supplied per chunk_id by the caller (a future EmbeddingProvider owns
    that). Does not perform retrieval, hybrid search, RRF, or reranking;
    this stops at Chunk[] -> persisted + indexed.
    """

    def __init__(
        self,
        repository: ChunkRepository,
        lexical_indexer: LexicalIndexer,
        vector_indexer: VectorIndexer,
    ):
        self.repository = repository
        self.lexical_indexer = lexical_indexer
        self.vector_indexer = vector_indexer

    def index_chunk(self, chunk: Chunk, embedding: list[float] | None = None) -> None:
        self.repository.upsert(chunk)
        self.lexical_indexer.index(chunk)
        self.vector_indexer.index(chunk, embedding=embedding)

    def index_chunks(
        self,
        chunks: list[Chunk],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        embeddings = embeddings or {}
        for chunk in chunks:
            self.index_chunk(chunk, embedding=embeddings.get(chunk.chunk_id))

    def reindex_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Delete all chunks for document_id, then insert the new set.

        Chunks belonging to other documents are untouched -- the delete
        is scoped by document_id, matching the id already carried by
        every Chunk.
        """
        self.repository.delete_by_document_id(document_id)
        self.index_chunks(chunks, embeddings)
