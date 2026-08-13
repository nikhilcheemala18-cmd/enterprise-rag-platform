import unittest

from pydantic import ValidationError

from app.ingestion.chunkers.config import ChunkingConfig


class TestChunkingConfig(unittest.TestCase):
    def test_defaults(self):
        config = ChunkingConfig()
        self.assertEqual(config.text_chunk_size, 500)
        self.assertEqual(config.text_chunk_overlap, 75)
        self.assertEqual(config.table_max_rows, 50)

    def test_rejects_non_positive_text_chunk_size(self):
        with self.assertRaises(ValidationError):
            ChunkingConfig(text_chunk_size=0)

    def test_rejects_negative_overlap(self):
        with self.assertRaises(ValidationError):
            ChunkingConfig(text_chunk_overlap=-1)

    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self):
        with self.assertRaises(ValidationError):
            ChunkingConfig(text_chunk_size=100, text_chunk_overlap=100)
        with self.assertRaises(ValidationError):
            ChunkingConfig(text_chunk_size=100, text_chunk_overlap=150)

    def test_rejects_non_positive_table_max_rows(self):
        with self.assertRaises(ValidationError):
            ChunkingConfig(table_max_rows=0)

    def test_accepts_valid_custom_config(self):
        config = ChunkingConfig(
            text_chunk_size=200, text_chunk_overlap=20, table_max_rows=10
        )
        self.assertEqual(config.text_chunk_size, 200)
        self.assertEqual(config.text_chunk_overlap, 20)
        self.assertEqual(config.table_max_rows, 10)


if __name__ == "__main__":
    unittest.main()
