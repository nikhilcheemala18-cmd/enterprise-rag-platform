"""
Pipeline smoke test: Chunk -> BGEEmbeddingProvider -> 768-dim vector ->
VectorIndexer-compatible embedding.

Uses the real BGE model (so it's gated the same way as
test_bge_provider_integration.py -- skips cleanly if the model can't be
loaded) but a FAKE database engine, exactly like
tests/test_indexing_service.py's FakeEngine: no real PostgreSQL
connection is opened, only the compiled SQL is inspected.
"""

import unittest

from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql

from app.embeddings.bge import BGE_DIMENSION, BGEEmbeddingProvider
from app.embeddings.service import ChunkEmbeddingService
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.indexing.vector import VectorIndexer
from app.models.chunk import Chunk


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
    def __init__(self):
        self.log: list = []

    def begin(self):
        return FakeConnection(self.log)

    def connect(self):
        return FakeConnection(self.log)


def _load_provider() -> BGEEmbeddingProvider | None:
    try:
        return BGEEmbeddingProvider()
    except Exception:
        return None


class TestBGEPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = _load_provider()
        if cls.provider is None:
            raise unittest.SkipTest(
                "BAAI/bge-base-en-v1.5 could not be loaded in this environment"
            )

    def test_chunk_to_vector_indexer_compatible_embedding(self):
        chunk = Chunk(
            chunk_id="c1",
            document_id="doc-1",
            source_element_ids=["el-1"],
            content="Revenue increased by 18% in July for Client A.",
            token_count=10,
            chunk_index=0,
            metadata={"element_type": "text"},
        )

        # Chunk -> ChunkEmbeddingService(BGEEmbeddingProvider) -> {chunk_id: vector}
        embedding_service = ChunkEmbeddingService(self.provider)
        embeddings = embedding_service.embed_chunks([chunk])

        self.assertIn("c1", embeddings)
        self.assertEqual(len(embeddings["c1"]), BGE_DIMENSION)
        self.assertTrue(all(isinstance(v, float) for v in embeddings["c1"]))

        # -> feed straight into the existing, unmodified IndexingService/
        # VectorIndexer against a fake DB engine (no PostgreSQL required)
        engine = FakeEngine()
        table = build_chunks_table(MetaData(), embedding_dimension=BGE_DIMENSION)
        indexing_service = IndexingService(
            repository=ChunkRepository(engine, table),
            lexical_indexer=LexicalIndexer(engine, table),
            vector_indexer=VectorIndexer(engine, table, embedding_dimension=BGE_DIMENSION),
        )

        indexing_service.index_chunk(chunk, embedding=embeddings["c1"])

        self.assertEqual(len(engine.log), 3)  # upsert, lexical update, vector update
        vector_update_stmt = engine.log[2]
        compiled = vector_update_stmt.compile(dialect=postgresql.dialect())
        self.assertIn("UPDATE chunks SET embedding", str(compiled))
        self.assertEqual(len(compiled.params["embedding"]), BGE_DIMENSION)


if __name__ == "__main__":
    unittest.main()
