"""
Pipeline smoke test: Chunk -> GeminiEmbeddingProvider -> 1536-dim vector
-> VectorIndexer-compatible embedding.

Uses a FAKE Gemini client (no real API call, no GEMINI_API_KEY required,
no network/cost) and a FAKE database engine (no PostgreSQL required) --
this proves the wiring/shape compatibility with the existing indexing
layer, not real API behavior (that's test_gemini_embedding_integration.py's
job).
"""

import unittest

from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql

from app.embeddings.gemini import GEMINI_DEFAULT_DIMENSION, GeminiEmbeddingProvider
from app.embeddings.service import ChunkEmbeddingService
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.indexing.vector import VectorIndexer
from app.models.chunk import Chunk


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeEmbedContentResult:
    def __init__(self, values):
        self.embeddings = [FakeEmbedding(values)]


class FakeModels:
    def embed_content(self, *, model, contents, config):
        return FakeEmbedContentResult([0.01 * i for i in range(config.output_dimensionality)])


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeModels()


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


class TestGeminiPipelineSmoke(unittest.TestCase):
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

        provider = GeminiEmbeddingProvider(client=FakeGeminiClient())
        self.assertEqual(provider.config.dimension, GEMINI_DEFAULT_DIMENSION)

        # Chunk -> ChunkEmbeddingService(GeminiEmbeddingProvider) -> {chunk_id: vector}
        embedding_service = ChunkEmbeddingService(provider)
        embeddings = embedding_service.embed_chunks([chunk])

        self.assertIn("c1", embeddings)
        self.assertEqual(len(embeddings["c1"]), GEMINI_DEFAULT_DIMENSION)
        self.assertTrue(all(isinstance(v, float) for v in embeddings["c1"]))

        # -> feed straight into the existing, unmodified IndexingService/
        # VectorIndexer against a fake DB engine (no PostgreSQL required)
        engine = FakeEngine()
        table = build_chunks_table(MetaData(), embedding_dimension=GEMINI_DEFAULT_DIMENSION)
        indexing_service = IndexingService(
            repository=ChunkRepository(engine, table),
            lexical_indexer=LexicalIndexer(engine, table),
            vector_indexer=VectorIndexer(
                engine, table, embedding_dimension=GEMINI_DEFAULT_DIMENSION
            ),
        )

        indexing_service.index_chunk(chunk, embedding=embeddings["c1"])

        self.assertEqual(len(engine.log), 3)  # upsert, lexical update, vector update
        vector_update_stmt = engine.log[2]
        compiled = vector_update_stmt.compile(dialect=postgresql.dialect())
        self.assertIn("UPDATE chunks SET embedding", str(compiled))
        self.assertEqual(len(compiled.params["embedding"]), GEMINI_DEFAULT_DIMENSION)


if __name__ == "__main__":
    unittest.main()
