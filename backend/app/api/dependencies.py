"""
Composition root for request-time dependencies. Everything here is built
lazily (on first use, via FastAPI's Depends()) and cached with lru_cache
so the embedding model/engine are constructed once per process, not once
per request -- constructing a fresh SQLAlchemy engine or reloading BGE's
weights on every request would be wasteful and, for BGE, slow.

Reuses existing configuration/construction exactly as-is:
DatabaseConfig.from_env() (already loads backend/.env), create_db_engine()
(already the only place a SQLAlchemy engine is built), build_chunks_table()
(already the only Table() definition) -- nothing here parses DATABASE_URL
itself or defines a second database abstraction. No DDL is issued (no
init_schema()/create_all() call) -- the `chunks` table is assumed to
already exist; this module only ever builds Python-side Table objects
for querying it.

get_embedding_dimension()/get_chunks_table()/get_lexical_indexer()/
get_vector_indexer() are shared building blocks: both the write path
(get_indexing_service()) and the read path (get_lexical_retriever()/
get_vector_retriever()) consume the exact same cached engine, table, and
resolved dimension -- the dimension-resolution logic
(EMBEDDING_DIMENSION override vs. the configured provider's own
dimension) exists in exactly one place.
"""

from functools import lru_cache

from sqlalchemy import Engine, MetaData, Table

from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import build_embedding_provider
from app.indexing.config import DatabaseConfig
from app.indexing.database import create_db_engine
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.indexing.vector import VectorIndexer
from app.rag.llm import GeminiLLMProvider, LLMProvider
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.vector import VectorRetriever
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider()


@lru_cache
def get_db_engine() -> Engine:
    config = DatabaseConfig.from_env()
    return create_db_engine(config.database_url)


@lru_cache
def get_embedding_dimension() -> int:
    db_config = DatabaseConfig.from_env()
    embedding_provider = get_embedding_provider()
    # Explicit EMBEDDING_DIMENSION always wins (that's its documented
    # purpose); otherwise fall back to whichever provider is actually
    # configured, since a real model has now been chosen.
    return db_config.embedding_dimension or embedding_provider.config.dimension


@lru_cache
def get_chunks_table() -> Table:
    return build_chunks_table(MetaData(), embedding_dimension=get_embedding_dimension())


@lru_cache
def get_lexical_indexer() -> LexicalIndexer:
    return LexicalIndexer(get_db_engine(), get_chunks_table())


@lru_cache
def get_vector_indexer() -> VectorIndexer:
    return VectorIndexer(
        get_db_engine(), get_chunks_table(), embedding_dimension=get_embedding_dimension()
    )


@lru_cache
def get_indexing_service() -> IndexingService:
    return IndexingService(
        repository=ChunkRepository(get_db_engine(), get_chunks_table()),
        lexical_indexer=get_lexical_indexer(),
        vector_indexer=get_vector_indexer(),
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_provider=get_embedding_provider(),
        indexing_service=get_indexing_service(),
    )


@lru_cache
def get_lexical_retriever() -> LexicalRetriever:
    return LexicalRetriever(get_lexical_indexer())


@lru_cache
def get_vector_retriever() -> VectorRetriever:
    return VectorRetriever(get_vector_indexer())


@lru_cache
def get_hybrid_search_service() -> HybridSearchService:
    return HybridSearchService(get_lexical_retriever(), get_vector_retriever())


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_embedding_provider(), get_hybrid_search_service())


@lru_cache
def get_llm_provider() -> LLMProvider:
    return GeminiLLMProvider()


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(get_retrieval_service(), get_llm_provider())
