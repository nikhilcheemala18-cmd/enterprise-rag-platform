import unittest
from pathlib import Path

from app.ingestion.loaders.csv import CSVLoader
from app.models.document import NormalizedDocument, TableElement

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample.csv"
EMPTY_CSV = FIXTURES_DIR / "empty.csv"


class TestCSVLoaderValidFixture(unittest.TestCase):
    def setUp(self):
        self.loader = CSVLoader()
        self.document = self.loader.load(SAMPLE_CSV, uploaded_by="user-1")

    def test_produces_normalized_document(self):
        self.assertIsInstance(self.document, NormalizedDocument)

    def test_document_id_generated(self):
        self.assertTrue(self.document.document_id)
        self.assertIsInstance(self.document.document_id, str)

    def test_metadata_filename(self):
        self.assertEqual(self.document.metadata.filename, "sample.csv")

    def test_metadata_file_type(self):
        self.assertEqual(self.document.metadata.file_type, "csv")

    def test_metadata_file_size(self):
        expected_size = SAMPLE_CSV.stat().st_size
        self.assertEqual(self.document.metadata.file_size, expected_size)
        self.assertGreater(self.document.metadata.file_size, 0)

    def test_uploaded_by_preserved(self):
        self.assertEqual(self.document.metadata.uploaded_by, "user-1")

    def test_exactly_one_table_element(self):
        self.assertEqual(len(self.document.elements), 1)
        self.assertIsInstance(self.document.elements[0], TableElement)

    def test_columns_preserved(self):
        table = self.document.elements[0]
        self.assertEqual(table.columns, ["name", "leads", "revenue"])

    def test_rows_preserved(self):
        table = self.document.elements[0]
        self.assertEqual(
            table.rows,
            [
                ["Acme", "120", "45000.50"],
                ["Beta", "95", "31000.25"],
                ["Gamma", "80", "27500.00"],
            ],
        )

    def test_values_kept_as_strings(self):
        table = self.document.elements[0]
        for row in table.rows:
            for value in row:
                self.assertIsInstance(value, str)

    def test_element_index_starts_at_zero(self):
        table = self.document.elements[0]
        self.assertEqual(table.element_index, 0)

    def test_source_document_id_matches(self):
        table = self.document.elements[0]
        self.assertEqual(table.source.document_id, self.document.document_id)

    def test_source_type_is_csv(self):
        table = self.document.elements[0]
        self.assertEqual(table.source.source_type, "csv")

    def test_csv_location_row_bounds(self):
        table = self.document.elements[0]
        # header is physical row 1; data rows are physical rows 2-4
        self.assertEqual(table.source.location.row_start, 2)
        self.assertEqual(table.source.location.row_end, 4)

    def test_serializes_to_json(self):
        payload = self.document.model_dump_json()
        self.assertIsInstance(payload, str)
        self.assertIn("sample.csv", payload)


class TestCSVLoaderEmptyFile(unittest.TestCase):
    def test_empty_csv_produces_zero_elements(self):
        loader = CSVLoader()
        document = loader.load(EMPTY_CSV)
        self.assertIsInstance(document, NormalizedDocument)
        self.assertEqual(document.elements, [])
        self.assertIsInstance(document.model_dump_json(), str)


class TestCSVLoaderMissingFile(unittest.TestCase):
    def test_missing_file_raises_clear_error(self):
        loader = CSVLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load(FIXTURES_DIR / "does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
