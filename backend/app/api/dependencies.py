"""
Composition root for request-time dependencies. Everything here is built
lazily (on first use, via FastAPI's Depends()) and cached with lru_cache
so the embedding model/engine are constructed once per process, not once
per request -- constructing a fresh SQLAlchemy engine or reloading BGE's
weights on every upload would be wasteful and, for BGE, slow.

Reuses existing configuration/construction exactly as-is:
DatabaseConfig.from_env() (already loads backend/.env), create_db_engine()
(already the only place a SQLAlchemy engine is built), build_chunks_table()
(already the only Table() definition) -- nothing here parses DATABASE_URL
itself or defines a second database abstraction. No DDL is issued (no
init_schema()/create_all() call) -- the `chunks` table is assumed to
already exist; this module only ever builds Python-side Table objects
for querying it.
"""

from functools import lru_cache

from sqlalchemy import Engine, MetaData

from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import build_embedding_provider
from app.indexing.config import DatabaseConfig
from app.indexing.database import create_db_engine
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.indexing.vector import VectorIndexer
from app.services.ingestion_service import IngestionService


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider()


@lru_cache
def get_db_engine() -> Engine:
    config = DatabaseConfig.from_env()
    return create_db_engine(config.database_url)


@lru_cache
def get_indexing_service() -> IndexingService:
    db_config = DatabaseConfig.from_env()
    embedding_provider = get_embedding_provider()
    # Explicit EMBEDDING_DIMENSION always wins (that's its documented
    # purpose); otherwise fall back to whichever provider is actually
    # configured, since a real model has now been chosen.
    dimension = db_config.embedding_dimension or embedding_provider.config.dimension

    engine = get_db_engine()
    table = build_chunks_table(MetaData(), embedding_dimension=dimension)

    return IndexingService(
        repository=ChunkRepository(engine, table),
        lexical_indexer=LexicalIndexer(engine, table),
        vector_indexer=VectorIndexer(engine, table, embedding_dimension=dimension),
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_provider=get_embedding_provider(),
        indexing_service=get_indexing_service(),
    )
