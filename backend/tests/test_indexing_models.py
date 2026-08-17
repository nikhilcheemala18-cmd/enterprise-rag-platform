import os
import unittest

from sqlalchemy import MetaData

from app.indexing.config import DatabaseConfig
from app.indexing.models import SearchResult, build_chunks_table
from app.indexing.repository import chunk_to_row
from app.models.chunk import Chunk


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_element_ids=["el-1", "el-2"],
        content="Revenue increased by 18% in July for Client A.",
        token_count=9,
        chunk_index=0,
        metadata={
            "element_type": "text",
            "section_path": ["Monthly Performance"],
            "page_number": 3,
        },
    )
    defaults.update(overrides)
    return Chunk(**defaults)


class TestBuildChunksTable(unittest.TestCase):
    def test_expected_columns_present(self):
        table = build_chunks_table(MetaData())
        column_names = {c.name for c in table.columns}
        expected = {
            "chunk_id",
            "document_id",
            "content",
            "token_count",
            "chunk_index",
            "embedding_model",
            "source_element_ids",
            "metadata",
            "search_vector",
            "embedding",
            "created_at",
            "updated_at",
        }
        self.assertEqual(column_names, expected)

    def test_chunk_id_is_primary_key(self):
        table = build_chunks_table(MetaData())
        self.assertEqual(list(table.primary_key.columns.keys()), ["chunk_id"])

    def test_document_id_and_search_vector_are_indexed(self):
        table = build_chunks_table(MetaData())
        indexed_columns: set[str] = set()
        for ix in table.indexes:
            indexed_columns.update(ix.columns.keys())
        self.assertIn("document_id", indexed_columns)
        self.assertIn("search_vector", indexed_columns)

    def test_embedding_dimension_is_configurable_not_hardcoded(self):
        dimensionless = build_chunks_table(MetaData(), embedding_dimension=None)
        fixed = build_chunks_table(MetaData(), embedding_dimension=1536)
        self.assertIsNone(dimensionless.c.embedding.type.dim)
        self.assertEqual(fixed.c.embedding.type.dim, 1536)


class TestChunkToRow(unittest.TestCase):
    def test_all_important_fields_survive_mapping(self):
        chunk = make_chunk()
        row = chunk_to_row(chunk)
        self.assertEqual(row["chunk_id"], chunk.chunk_id)
        self.assertEqual(row["document_id"], chunk.document_id)
        self.assertEqual(row["content"], chunk.content)
        self.assertEqual(row["token_count"], chunk.token_count)
        self.assertEqual(row["chunk_index"], chunk.chunk_index)
        self.assertEqual(row["embedding_model"], chunk.embedding_model)

    def test_metadata_survives_mapping(self):
        chunk = make_chunk(metadata={"element_type": "table", "page_number": 12})
        row = chunk_to_row(chunk)
        self.assertEqual(row["metadata"], {"element_type": "table", "page_number": 12})

    def test_source_element_ids_survive_mapping(self):
        chunk = make_chunk(source_element_ids=["el-a", "el-b", "el-c"])
        row = chunk_to_row(chunk)
        self.assertEqual(row["source_element_ids"], ["el-a", "el-b", "el-c"])

    def test_row_excludes_index_owned_columns(self):
        row = chunk_to_row(make_chunk())
        self.assertNotIn("search_vector", row)
        self.assertNotIn("embedding", row)


class TestSearchResult(unittest.TestCase):
    def test_construction(self):
        result = SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="hello",
            score=0.42,
            metadata={"element_type": "text"},
        )
        self.assertEqual(result.score, 0.42)
        self.assertEqual(result.metadata, {"element_type": "text"})


class TestDatabaseConfig(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_from_env_raises_clearly_when_missing(self):
        os.environ.pop("DATABASE_URL", None)
        with self.assertRaises(RuntimeError) as ctx:
            DatabaseConfig.from_env()
        self.assertIn("DATABASE_URL", str(ctx.exception))

    def test_from_env_reads_database_url_and_optional_dimension(self):
        os.environ["DATABASE_URL"] = "postgresql+psycopg://u:p@localhost/db"
        os.environ["EMBEDDING_DIMENSION"] = "1536"
        config = DatabaseConfig.from_env()
        self.assertEqual(config.database_url, "postgresql+psycopg://u:p@localhost/db")
        self.assertEqual(config.embedding_dimension, 1536)

    def test_embedding_dimension_optional_and_none_by_default(self):
        config = DatabaseConfig(database_url="postgresql+psycopg://u:p@localhost/db")
        self.assertIsNone(config.embedding_dimension)


if __name__ == "__main__":
    unittest.main()
