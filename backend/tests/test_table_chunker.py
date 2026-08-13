import unittest

from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.table import TableChunker
from app.models.document import ExcelLocation, SourceReference, TableElement


def make_table_element(
    columns: list[str],
    rows: list[list],
    element_id: str = "el-table-1",
    section_path: list[str] | None = None,
) -> TableElement:
    source = SourceReference(
        document_id="doc-1",
        source_type="excel",
        location=ExcelLocation(sheet_name="Revenue", row_start=1, row_end=len(rows) + 1),
    )
    return TableElement(
        element_id=element_id,
        element_index=0,
        source=source,
        section_path=section_path,
        columns=columns,
        rows=rows,
    )


class TestTableChunkerSmallTable(unittest.TestCase):
    def setUp(self):
        self.chunker = TableChunker(ChunkingConfig())
        self.columns = ["Month", "Revenue", "Orders"]
        self.rows = [["June", 72000, 1120], ["July", 85000, 1350]]
        self.element = make_table_element(self.columns, self.rows)
        self.chunks = self.chunker.chunk(self.element)

    def test_small_table_produces_one_chunk(self):
        self.assertEqual(len(self.chunks), 1)

    def test_headers_appear_in_chunk_content(self):
        self.assertIn("Month | Revenue | Orders", self.chunks[0].content)

    def test_row_values_appear_in_chunk_content(self):
        self.assertIn("June | 72000 | 1120", self.chunks[0].content)
        self.assertIn("July | 85000 | 1350", self.chunks[0].content)

    def test_source_element_ids_correct(self):
        self.assertEqual(self.chunks[0].source_element_ids, [self.element.element_id])

    def test_original_rows_unchanged(self):
        self.assertEqual(self.element.rows, [["June", 72000, 1120], ["July", 85000, 1350]])
        self.assertIsInstance(self.element.rows[0][1], int)


class TestTableChunkerLargeTable(unittest.TestCase):
    def setUp(self):
        self.config = ChunkingConfig(table_max_rows=50)
        self.chunker = TableChunker(self.config)
        self.columns = ["Index", "Value"]
        self.rows = [[i, i * 10] for i in range(1, 121)]  # 120 rows
        self.element = make_table_element(
            self.columns, self.rows, section_path=["Monthly Performance", "Revenue"]
        )
        self.chunks = self.chunker.chunk(self.element)

    def test_large_table_produces_multiple_chunks(self):
        self.assertEqual(len(self.chunks), 3)  # 50, 50, 20

    def test_every_chunk_contains_headers(self):
        for chunk in self.chunks:
            self.assertIn("Index | Value", chunk.content)

    def test_row_groups_do_not_exceed_max_rows(self):
        for chunk in self.chunks:
            row_lines = [
                line for line in chunk.content.splitlines() if " | " in line
            ][1:]  # drop the header line
            self.assertLessEqual(len(row_lines), self.config.table_max_rows)

    def test_row_range_metadata_correct(self):
        self.assertEqual(self.chunks[0].metadata["row_range"], {"start": 0, "end": 49})
        self.assertEqual(self.chunks[1].metadata["row_range"], {"start": 50, "end": 99})
        self.assertEqual(self.chunks[2].metadata["row_range"], {"start": 100, "end": 119})

    def test_section_path_propagated(self):
        for chunk in self.chunks:
            self.assertEqual(
                chunk.metadata["section_path"], ["Monthly Performance", "Revenue"]
            )
            self.assertTrue(
                chunk.content.startswith("Section: Monthly Performance > Revenue\n\n")
            )

    def test_source_element_ids_correct_for_every_chunk(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.source_element_ids, [self.element.element_id])

    def test_original_rows_unchanged_after_large_split(self):
        self.assertEqual(len(self.element.rows), 120)
        self.assertEqual(self.element.rows[0], [1, 10])
        self.assertEqual(self.element.rows[-1], [120, 1200])


class TestTableChunkerEmptyTable(unittest.TestCase):
    def test_table_with_no_rows_produces_no_chunks(self):
        chunker = TableChunker(ChunkingConfig())
        element = make_table_element(["A", "B"], [])
        self.assertEqual(chunker.chunk(element), [])


if __name__ == "__main__":
    unittest.main()
