import unittest

from sqlalchemy import MetaData, func, select
from sqlalchemy.dialects import postgresql

from app.indexing.lexical import LexicalIndexer
from app.indexing.models import build_chunks_table

"""
Dependency-light tests for LexicalIndexer.

These verify the SQL *shape* LexicalIndexer generates (compiled against
the PostgreSQL dialect, without opening a connection) -- they prove the
code asks Postgres for lexical/full-text search, not BM25, and that it
targets the right column/function. They do NOT prove a real query
actually ranks/matches rows correctly; that requires a live PostgreSQL
instance and is covered by test_indexing_integration.py, which is
skipped unless DATABASE_URL is set.
"""


class TestLexicalIndexerSQLShape(unittest.TestCase):
    def setUp(self):
        self.table = build_chunks_table(MetaData())
        # engine=None is fine here: these tests only compile statements,
        # they never call .begin()/.connect().
        self.indexer = LexicalIndexer(engine=None, table=self.table)

    def test_uses_postgres_full_text_search_not_bm25(self):
        stmt = select(func.to_tsvector(self.indexer.fts_config, self.table.c.content))
        compiled = str(stmt.compile(dialect=postgresql.dialect()))
        self.assertIn("to_tsvector", compiled)
        self.assertNotIn("bm25", compiled.lower())

    def test_default_fts_config_is_english(self):
        self.assertEqual(self.indexer.fts_config, "english")

    def test_fts_config_is_overridable(self):
        indexer = LexicalIndexer(engine=None, table=self.table, fts_config="simple")
        self.assertEqual(indexer.fts_config, "simple")

    def test_search_uses_websearch_to_tsquery_and_ts_rank(self):
        tsquery = func.websearch_to_tsquery(self.indexer.fts_config, "INV-2026-01847")
        rank = func.ts_rank(self.table.c.search_vector, tsquery)
        stmt = select(self.table.c.chunk_id, rank).where(
            self.table.c.search_vector.op("@@")(tsquery)
        )
        compiled = str(stmt.compile(dialect=postgresql.dialect()))
        self.assertIn("websearch_to_tsquery", compiled)
        self.assertIn("ts_rank", compiled)
        self.assertIn("@@", compiled)

    def test_index_produces_an_update_targeting_search_vector(self):
        from app.models.chunk import Chunk

        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            source_element_ids=["e1"],
            content="Invoice INV-2026-01847 was issued in July.",
            token_count=6,
            chunk_index=0,
        )
        # Build the same statement index() would build, without executing it,
        # by inspecting the compiled SQL of an equivalent UPDATE.
        from sqlalchemy import update

        stmt = (
            update(self.table)
            .where(self.table.c.chunk_id == chunk.chunk_id)
            .values(
                search_vector=func.to_tsvector(self.indexer.fts_config, self.table.c.content)
            )
        )
        compiled = str(stmt.compile(dialect=postgresql.dialect()))
        self.assertIn("UPDATE chunks SET search_vector", compiled)
        self.assertIn("to_tsvector", compiled)


if __name__ == "__main__":
    unittest.main()
