import unittest

from pydantic import ValidationError

from app.models.chunk import Chunk


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_element_ids=["el-1"],
        content="some retrievable text",
        token_count=5,
        chunk_index=0,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


class TestChunkValid(unittest.TestCase):
    def test_create_valid_chunk(self):
        chunk = make_chunk()
        self.assertEqual(chunk.chunk_id, "chunk-1")
        self.assertEqual(chunk.metadata, {})
        self.assertIsNone(chunk.embedding_model)

    def test_multiple_source_element_ids(self):
        chunk = make_chunk(source_element_ids=["el-1", "el-2", "el-3"])
        self.assertEqual(chunk.source_element_ids, ["el-1", "el-2", "el-3"])

    def test_metadata_storage(self):
        chunk = make_chunk(metadata={"section": "Introduction", "page": 1})
        self.assertEqual(chunk.metadata["section"], "Introduction")
        self.assertEqual(chunk.metadata["page"], 1)

    def test_optional_embedding_model(self):
        chunk = make_chunk(embedding_model="text-embedding-3-small")
        self.assertEqual(chunk.embedding_model, "text-embedding-3-small")


class TestChunkInvalid(unittest.TestCase):
    def test_rejects_empty_content(self):
        with self.assertRaises(ValidationError):
            make_chunk(content="")

    def test_rejects_empty_source_element_ids(self):
        with self.assertRaises(ValidationError):
            make_chunk(source_element_ids=[])

    def test_rejects_negative_token_count(self):
        with self.assertRaises(ValidationError):
            make_chunk(token_count=-1)

    def test_rejects_negative_chunk_index(self):
        with self.assertRaises(ValidationError):
            make_chunk(chunk_index=-1)

    def test_rejects_empty_chunk_id(self):
        with self.assertRaises(ValidationError):
            make_chunk(chunk_id="")

    def test_rejects_empty_document_id(self):
        with self.assertRaises(ValidationError):
            make_chunk(document_id="")


if __name__ == "__main__":
    unittest.main()
