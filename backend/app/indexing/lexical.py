from typing import Any

from sqlalchemy import Engine, Table, func, select, update

from app.indexing.base import BaseIndexer
from app.indexing.models import SearchResult
from app.models.chunk import Chunk

DEFAULT_FTS_CONFIG = "english"


class LexicalIndexer(BaseIndexer):
    """PostgreSQL full-text search over Chunk.content.

    This is Postgres lexical search (to_tsvector / websearch_to_tsquery /
    ts_rank) -- explicitly NOT BM25. It is kept behind this narrow
    index()/search() surface so a real BM25Index (a different backend
    entirely, e.g. Elasticsearch/Tantivy) can later implement the same
    shape without IndexingService or ChunkRepository changing.
    """

    def __init__(self, engine: Engine, table: Table, fts_config: str = DEFAULT_FTS_CONFIG):
        self.engine = engine
        self.table = table
        self.fts_config = fts_config

    def index(self, chunk: Chunk, **kwargs: Any) -> None:
        stmt = (
            update(self.table)
            .where(self.table.c.chunk_id == chunk.chunk_id)
            .values(search_vector=func.to_tsvector(self.fts_config, self.table.c.content))
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        tsquery = func.websearch_to_tsquery(self.fts_config, query)
        rank = func.ts_rank(self.table.c.search_vector, tsquery).label("score")
        stmt = (
            select(
                self.table.c.chunk_id,
                self.table.c.document_id,
                self.table.c.content,
                self.table.c.metadata,
                rank,
            )
            .where(self.table.c.search_vector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                metadata=r["metadata"],
            )
            for r in rows
        ]
