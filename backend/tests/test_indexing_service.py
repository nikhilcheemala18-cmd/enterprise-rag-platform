import unittest

from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql

from app.indexing.models import build_chunks_table
from app.indexing.repository import ChunkRepository
from app.indexing.service import IndexingService
from app.models.chunk import Chunk


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_element_ids=["el-1"],
        content="Revenue increased by 18% in July for Client A.",
        token_count=9,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


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
    """Records every statement passed to execute() without needing a real
    PostgreSQL connection -- lets ChunkRepository's real upsert()/delete()
    code run end-to-end, with only the DB round-trip itself faked out.
    """

    def __init__(self):
        self.log: list = []

    def begin(self):
        return FakeConnection(self.log)

    def connect(self):
        return FakeConnection(self.log)


class TestChunkRepositoryIdempotency(unittest.TestCase):
    """
    SQL-shape verification: proves ChunkRepository.upsert() always emits
    an INSERT ... ON CONFLICT (chunk_id) DO UPDATE, i.e. indexing the same
    chunk twice cannot create a second row. This does not execute against
    real PostgreSQL (none is available in this environment); true
    behavioral proof (row count stays 1 after two upserts) is covered by
    test_indexing_integration.py, skipped unless DATABASE_URL is set.
    """

    def setUp(self):
        self.engine = FakeEngine()
        self.table = build_chunks_table(MetaData())
        self.repository = ChunkRepository(self.engine, self.table)

    def test_upsert_statement_is_an_on_conflict_upsert(self):
        chunk = make_chunk()
        self.repository.upsert(chunk)
        self.assertEqual(len(self.engine.log), 1)
        compiled = str(self.engine.log[0].compile(dialect=postgresql.dialect()))
        self.assertIn("ON CONFLICT", compiled)
        self.assertIn("DO UPDATE SET", compiled)
        self.assertIn("chunk_id", compiled)

    def test_upserting_same_chunk_twice_emits_two_upserts_not_inserts(self):
        chunk = make_chunk()
        self.repository.upsert(chunk)
        self.repository.upsert(chunk)
        self.assertEqual(len(self.engine.log), 2)
        for stmt in self.engine.log:
            compiled = str(stmt.compile(dialect=postgresql.dialect()))
            self.assertIn("ON CONFLICT", compiled)

    def test_chunk_id_is_the_conflict_target_not_chunk_index(self):
        chunk = make_chunk()
        self.repository.upsert(chunk)
        compiled = str(self.engine.log[0].compile(dialect=postgresql.dialect()))
        self.assertIn("ON CONFLICT (chunk_id)", compiled)


class TestChunkRepositoryDocumentIsolation(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.table = build_chunks_table(MetaData())
        self.repository = ChunkRepository(self.engine, self.table)

    def test_chunks_from_different_documents_carry_distinct_document_ids(self):
        chunk_a = make_chunk(chunk_id="c-a", document_id="doc-A")
        chunk_b = make_chunk(chunk_id="c-b", document_id="doc-B")
        self.repository.upsert(chunk_a)
        self.repository.upsert(chunk_b)

        compiled_a = str(self.engine.log[0].compile(dialect=postgresql.dialect()))
        compiled_b = str(self.engine.log[1].compile(dialect=postgresql.dialect()))
        params_a = self.engine.log[0].compile(dialect=postgresql.dialect()).params
        params_b = self.engine.log[1].compile(dialect=postgresql.dialect()).params
        self.assertEqual(params_a["document_id"], "doc-A")
        self.assertEqual(params_b["document_id"], "doc-B")

    def test_delete_by_document_id_scopes_to_one_document(self):
        self.repository.delete_by_document_id("doc-A")
        compiled = str(self.engine.log[0].compile(dialect=postgresql.dialect()))
        params = self.engine.log[0].compile(dialect=postgresql.dialect()).params
        self.assertIn("DELETE FROM chunks", compiled)
        self.assertEqual(params["document_id_1"], "doc-A")


class FakeLexicalIndexer:
    def __init__(self):
        self.indexed_chunk_ids: list[str] = []

    def index(self, chunk, **kwargs):
        self.indexed_chunk_ids.append(chunk.chunk_id)


class FakeVectorIndexer:
    def __init__(self):
        self.indexed: list[tuple[str, list[float] | None]] = []

    def index(self, chunk, embedding=None, **kwargs):
        self.indexed.append((chunk.chunk_id, embedding))


class FakeRepository:
    def __init__(self):
        self.upserted: list[Chunk] = []
        self.deleted_document_ids: list[str] = []

    def upsert(self, chunk):
        self.upserted.append(chunk)

    def delete_by_document_id(self, document_id):
        self.deleted_document_ids.append(document_id)
        self.upserted = [c for c in self.upserted if c.document_id != document_id]


class TestIndexingServiceCoordination(unittest.TestCase):
    """Pure-Python orchestration tests using fake collaborators -- verify
    IndexingService calls repository/lexical/vector in the right shape,
    independent of any database.
    """

    def setUp(self):
        self.repository = FakeRepository()
        self.lexical = FakeLexicalIndexer()
        self.vector = FakeVectorIndexer()
        self.service = IndexingService(self.repository, self.lexical, self.vector)

    def test_index_chunk_calls_all_three_collaborators(self):
        chunk = make_chunk()
        self.service.index_chunk(chunk, embedding=[0.1, 0.2, 0.3])
        self.assertEqual(self.repository.upserted, [chunk])
        self.assertEqual(self.lexical.indexed_chunk_ids, [chunk.chunk_id])
        self.assertEqual(self.vector.indexed, [(chunk.chunk_id, [0.1, 0.2, 0.3])])

    def test_index_chunk_without_embedding_passes_none_through(self):
        chunk = make_chunk()
        self.service.index_chunk(chunk)
        self.assertEqual(self.vector.indexed, [(chunk.chunk_id, None)])

    def test_index_chunks_preserves_order_and_maps_embeddings_by_id(self):
        c1 = make_chunk(chunk_id="c1", chunk_index=0)
        c2 = make_chunk(chunk_id="c2", chunk_index=1)
        self.service.index_chunks([c1, c2], embeddings={"c2": [1.0, 2.0]})
        self.assertEqual(self.repository.upserted, [c1, c2])
        self.assertEqual(self.vector.indexed, [("c1", None), ("c2", [1.0, 2.0])])

    def test_reindex_document_deletes_then_reinserts(self):
        old_chunk = make_chunk(chunk_id="old", document_id="doc-1")
        self.repository.upserted.append(old_chunk)
        new_chunks = [make_chunk(chunk_id="new", document_id="doc-1")]

        self.service.reindex_document("doc-1", new_chunks)

        self.assertEqual(self.repository.deleted_document_ids, ["doc-1"])
        self.assertEqual([c.chunk_id for c in self.repository.upserted], ["new"])

    def test_reindex_document_does_not_touch_other_documents(self):
        self.repository.upserted.append(make_chunk(chunk_id="keep", document_id="doc-2"))
        self.service.reindex_document("doc-1", [make_chunk(chunk_id="new", document_id="doc-1")])
        remaining_ids = {c.chunk_id for c in self.repository.upserted}
        self.assertIn("keep", remaining_ids)
        self.assertIn("new", remaining_ids)


if __name__ == "__main__":
    unittest.main()
