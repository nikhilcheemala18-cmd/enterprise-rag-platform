from typing import Any

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Index, Integer, MetaData, String, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR


def build_chunks_table(metadata: MetaData, embedding_dimension: int | None = None) -> Table:
    """Define the `chunks` table.

    embedding_dimension is a parameter, not a hardcoded constant, so the
    pgvector column width is a deployment/config concern rather than
    something baked into the schema module at import time. Passing None
    (the default) creates a dimension-agnostic `vector` column, which is
    what this phase uses since no embedding model has been selected yet.

    search_vector and embedding are populated by LexicalIndexer and
    VectorIndexer respectively, not by ChunkRepository -- the repository
    only owns the core Chunk fields.
    """
    return Table(
        "chunks",
        metadata,
        Column("chunk_id", String, primary_key=True),
        Column("document_id", String, nullable=False),
        Column("content", Text, nullable=False),
        Column("token_count", Integer, nullable=False),
        Column("chunk_index", Integer, nullable=False),
        Column("embedding_model", String, nullable=True),
        Column("source_element_ids", JSONB, nullable=False),
        Column("metadata", JSONB, nullable=False),
        Column("search_vector", TSVECTOR, nullable=True),
        Column("embedding", Vector(embedding_dimension), nullable=True),
        Column(
            "created_at", DateTime(timezone=True), server_default=func.now(), nullable=False
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        ),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )


class SearchResult(BaseModel):
    """Return shape shared by LexicalIndexer.search() and
    VectorIndexer.search(). Not the Chunk model itself -- this is a
    minimal read projection; full provenance requires looking up
    source_element_ids via chunk_id.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any]
