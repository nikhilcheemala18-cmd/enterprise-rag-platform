from typing import Any

from sqlalchemy import Engine, Table, select, update

from app.indexing.base import BaseIndexer
from app.indexing.models import SearchResult
from app.models.chunk import Chunk


class VectorIndexer(BaseIndexer):
    """pgvector-backed semantic index over the `embedding` column.

    Does not generate embeddings -- this phase has no EmbeddingProvider.
    Callers supply a precomputed vector (or none at all, in which case
    index() is a no-op for that chunk). If embedding_dimension was
    configured, every vector is validated against it in plain Python
    before any database call.
    """

    def __init__(
        self,
        engine: Engine,
        table: Table,
        embedding_dimension: int | None = None,
    ):
        self.engine = engine
        self.table = table
        self.embedding_dimension = embedding_dimension

    def _validate(self, embedding: list[float]) -> None:
        if not embedding:
            raise ValueError("embedding must be a non-empty vector")
        if self.embedding_dimension is not None and len(embedding) != self.embedding_dimension:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, "
                f"expected {self.embedding_dimension}"
            )

    def index(self, chunk: Chunk, embedding: list[float] | None = None, **kwargs: Any) -> None:
        if embedding is None:
            return
        self._validate(embedding)
        stmt = (
            update(self.table)
            .where(self.table.c.chunk_id == chunk.chunk_id)
            .values(embedding=embedding)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def search(self, query_embedding: list[float], limit: int = 10) -> list[SearchResult]:
        self._validate(query_embedding)
        distance = self.table.c.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(
                self.table.c.chunk_id,
                self.table.c.document_id,
                self.table.c.content,
                self.table.c.metadata,
                distance,
            )
            .where(self.table.c.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=1.0 - float(r["distance"]),
                metadata=r["metadata"],
            )
            for r in rows
        ]
