import unittest

from sqlalchemy import MetaData

from app.indexing.models import build_chunks_table
from app.indexing.vector import VectorIndexer
from app.models.chunk import Chunk

"""
Dependency-light tests for VectorIndexer.

Dimension validation happens in plain Python before any database call,
so it can be verified without a live PostgreSQL/pgvector instance. A
PoisonEngine is used in place of a real SQLAlchemy Engine to prove that
validation failures short-circuit before touching the database at all.
Actually storing/searching vectors against pgvector requires a live
instance and is covered by test_indexing_integration.py, which is
skipped unless DATABASE_URL is set.
"""


class PoisonEngine:
    """Raises if any connection is opened -- proves a code path never
    reaches the database."""

    def begin(self):
        raise AssertionError("VectorIndexer touched the database unexpectedly")

    def connect(self):
        raise AssertionError("VectorIndexer touched the database unexpectedly")


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_element_ids=["el-1"],
        content="Figure 5.1 — Revenue Trend",
        token_count=4,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


class TestVectorIndexerDimensionValidation(unittest.TestCase):
    def setUp(self):
        self.table = build_chunks_table(MetaData())

    def test_no_embedding_is_a_noop_and_never_touches_db(self):
        indexer = VectorIndexer(engine=PoisonEngine(), table=self.table, embedding_dimension=3)
        indexer.index(make_chunk(), embedding=None)  # must not raise

    def test_rejects_empty_embedding(self):
        indexer = VectorIndexer(engine=PoisonEngine(), table=self.table)
        with self.assertRaises(ValueError):
            indexer.index(make_chunk(), embedding=[])

    def test_rejects_mismatched_dimension(self):
        indexer = VectorIndexer(engine=PoisonEngine(), table=self.table, embedding_dimension=5)
        with self.assertRaises(ValueError):
            indexer.index(make_chunk(), embedding=[0.1, 0.2, 0.3])

    def test_matching_dimension_passes_validation(self):
        indexer = VectorIndexer(engine=PoisonEngine(), table=self.table, embedding_dimension=3)
        # validation alone must not raise; PoisonEngine will raise if
        # index() proceeds to touch the database, proving this test only
        # exercises the validation path.
        indexer._validate([0.1, 0.2, 0.3])

    def test_no_configured_dimension_accepts_any_length(self):
        indexer = VectorIndexer(engine=PoisonEngine(), table=self.table, embedding_dimension=None)
        indexer._validate([0.1] * 7)
        indexer._validate([0.1] * 3)

    def test_search_rejects_empty_query_vector(self):
        indexer = VectorIndexer(engine=PoisonEngine(), table=self.table)
        with self.assertRaises(ValueError):
            indexer.search([])


if __name__ == "__main__":
    unittest.main()
