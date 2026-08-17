"""
End-to-end integration tests against a real PostgreSQL + pgvector instance.

These are SKIPPED by default -- this environment has no PostgreSQL server
(no psql/pg_ctl, no running Docker daemon, no DATABASE_URL). Do not fake
these; they only run when explicitly pointed at a real database.

Local setup:

    1. Start PostgreSQL 15+ with the pgvector extension available, e.g.:
         docker run --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=rag_test \\
             -p 5432:5432 pgvector/pgvector:pg16

    2. Point the test run at it:
         # bash
         export DATABASE_URL=postgresql+psycopg://postgres:test@localhost:5432/rag_test
         # PowerShell
         $env:DATABASE_URL = "postgresql+psycopg://postgres:test@localhost:5432/rag_test"

    3. Run:
         python -m unittest tests.test_indexing_integration -v

The extension (CREATE EXTENSION vector) and the `chunks` table are created
by init_schema() at the start of each test via setUp -- no manual schema
setup is required beyond having a reachable, empty database.
"""

import os
import unittest

from sqlalchemy import MetaData

from app.indexing.database import create_db_engine, init_schema
from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.indexing.vector import VectorIndexer
from app.models.chunk import Chunk

DATABASE_URL = os.environ.get("DATABASE_URL")

# Deterministic 4-dimensional test vectors -- not real embeddings, just
# fixed floats so cosine similarity ordering is predictable.
_VEC_C1 = [1.0, 0.0, 0.0, 0.0]
_VEC_C2 = [0.0, 1.0, 0.0, 0.0]
_VEC_C3 = [0.0, 0.0, 1.0, 0.0]
_VEC_C4 = [0.9, 0.1, 0.0, 0.0]  # deliberately close to C1


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c1",
            document_id="doc-1",
            source_element_ids=["el-1"],
            content="Revenue increased by 18% in July for Client A.",
            token_count=10,
            chunk_index=0,
            metadata={"element_type": "text", "page_number": 1},
        ),
        Chunk(
            chunk_id="c2",
            document_id="doc-1",
            source_element_ids=["el-2"],
            content="Month | Revenue | Orders\nJune | 72000 | 1120\nJuly | 85000 | 1350",
            token_count=12,
            chunk_index=1,
            metadata={"element_type": "table", "page_number": 2},
        ),
        Chunk(
            chunk_id="c3",
            document_id="doc-1",
            source_element_ids=["el-3"],
            content="Figure 5.1 — Revenue Trend",
            token_count=4,
            chunk_index=2,
            metadata={"element_type": "image", "page_number": 2},
        ),
        Chunk(
            chunk_id="c4",
            document_id="doc-2",
            source_element_ids=["el-4"],
            content="Invoice INV-2026-01847 was issued in July.",
            token_count=7,
            chunk_index=0,
            metadata={"element_type": "text", "page_number": 1},
        ),
    ]


@unittest.skipUnless(
    DATABASE_URL,
    "requires a live PostgreSQL + pgvector instance; set DATABASE_URL to run "
    "(see module docstring for local setup)",
)
class TestIndexingIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(DATABASE_URL)
        self.metadata = MetaData()
        self.table = build_chunks_table(self.metadata, embedding_dimension=4)
        init_schema(self.engine, self.metadata)
        with self.engine.begin() as conn:
            conn.execute(self.table.delete())

        self.repository = ChunkRepository(self.engine, self.table)
        self.lexical_indexer = LexicalIndexer(self.engine, self.table)
        self.vector_indexer = VectorIndexer(self.engine, self.table, embedding_dimension=4)
        self.service = IndexingService(
            self.repository, self.lexical_indexer, self.vector_indexer
        )

    def tearDown(self):
        with self.engine.begin() as conn:
            conn.execute(self.table.delete())
        self.engine.dispose()

    def test_insert_and_lexical_search_find_distinctive_term(self):
        self.service.index_chunks(
            _sample_chunks(),
            embeddings={"c1": _VEC_C1, "c2": _VEC_C2, "c3": _VEC_C3, "c4": _VEC_C4},
        )

        results = self.lexical_indexer.search("INV-2026-01847")

        self.assertTrue(results)
        self.assertEqual(results[0].chunk_id, "c4")
        self.assertIn("INV-2026-01847", results[0].content)

    def test_vector_search_returns_nearest_chunk(self):
        self.service.index_chunks(
            _sample_chunks(),
            embeddings={"c1": _VEC_C1, "c2": _VEC_C2, "c3": _VEC_C3, "c4": _VEC_C4},
        )

        results = self.vector_indexer.search(_VEC_C1, limit=2)

        self.assertTrue(results)
        self.assertEqual(results[0].chunk_id, "c1")
        # c4 was constructed to be the second-closest vector to c1
        self.assertEqual(results[1].chunk_id, "c4")

    def test_metadata_and_provenance_survive_round_trip(self):
        self.service.index_chunk(_sample_chunks()[1])  # c2, the table chunk

        stored = self.repository.get_by_chunk_id("c2")

        self.assertIsNotNone(stored)
        self.assertEqual(stored["document_id"], "doc-1")
        self.assertEqual(stored["source_element_ids"], ["el-2"])
        self.assertEqual(stored["metadata"]["element_type"], "table")
        self.assertEqual(stored["metadata"]["page_number"], 2)

    def test_idempotent_indexing_does_not_duplicate_rows(self):
        chunk = _sample_chunks()[0]
        self.service.index_chunk(chunk, embedding=_VEC_C1)
        self.service.index_chunk(chunk, embedding=_VEC_C1)

        rows = self.repository.get_by_document_id("doc-1")
        matching = [r for r in rows if r["chunk_id"] == "c1"]
        self.assertEqual(len(matching), 1)

    def test_document_isolation_across_documents(self):
        self.service.index_chunks(_sample_chunks())

        doc1_rows = self.repository.get_by_document_id("doc-1")
        doc2_rows = self.repository.get_by_document_id("doc-2")

        self.assertEqual({r["chunk_id"] for r in doc1_rows}, {"c1", "c2", "c3"})
        self.assertEqual({r["chunk_id"] for r in doc2_rows}, {"c4"})

    def test_reindex_document_replaces_only_that_documents_chunks(self):
        self.service.index_chunks(_sample_chunks())

        replacement = [
            Chunk(
                chunk_id="c1-new",
                document_id="doc-1",
                source_element_ids=["el-1"],
                content="Revenue increased by 22% in August for Client A.",
                token_count=10,
                chunk_index=0,
            )
        ]
        self.service.reindex_document("doc-1", replacement)

        doc1_rows = self.repository.get_by_document_id("doc-1")
        doc2_rows = self.repository.get_by_document_id("doc-2")

        self.assertEqual({r["chunk_id"] for r in doc1_rows}, {"c1-new"})
        self.assertEqual({r["chunk_id"] for r in doc2_rows}, {"c4"})


if __name__ == "__main__":
    unittest.main()
