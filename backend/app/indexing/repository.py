from typing import Any

from sqlalchemy import Engine, Table, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.chunk import Chunk


def chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    """Map a Chunk onto the core `chunks` table columns.

    Deliberately excludes search_vector/embedding -- those are owned by
    LexicalIndexer/VectorIndexer, not the repository.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "token_count": chunk.token_count,
        "chunk_index": chunk.chunk_index,
        "embedding_model": chunk.embedding_model,
        "source_element_ids": chunk.source_element_ids,
        "metadata": chunk.metadata,
    }


class ChunkRepository:
    """Persists the core Chunk fields into the `chunks` table.

    Upserts are keyed on chunk_id (the table's primary key) via
    PostgreSQL's ON CONFLICT DO UPDATE, so indexing the same chunk twice
    updates the existing row instead of creating a duplicate.
    """

    def __init__(self, engine: Engine, table: Table):
        self.engine = engine
        self.table = table

    def upsert(self, chunk: Chunk) -> None:
        row = chunk_to_row(chunk)
        stmt = pg_insert(self.table).values(**row)
        update_columns = {col: stmt.excluded[col] for col in row if col != "chunk_id"}
        stmt = stmt.on_conflict_do_update(index_elements=["chunk_id"], set_=update_columns)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def upsert_many(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.upsert(chunk)

    def delete_by_document_id(self, document_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(self.table).where(self.table.c.document_id == document_id))

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(self.table).where(self.table.c.chunk_id == chunk_id))
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def get_by_document_id(self, document_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(self.table)
                    .where(self.table.c.document_id == document_id)
                    .order_by(self.table.c.chunk_index)
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]
