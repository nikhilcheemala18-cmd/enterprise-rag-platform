import shutil
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from app.ingestion.base import DocumentLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.models.document import ImageElement, NormalizedDocument, TableElement, TextElement

_PAGE1_CONTENT = b"""
BT /F1 18 Tf 50 750 Td (Quarterly Report) Tj ET
BT /F1 10 Tf 50 720 Td (This section discusses revenue performance.) Tj ET
BT /F1 10 Tf 50 700 Td (Additional context follows below.) Tj ET
1 w
50 650 m 250 650 l S
50 630 m 250 630 l S
50 610 m 250 610 l S
50 590 m 250 590 l S
50 590 m 50 650 l S
150 590 m 150 650 l S
250 590 m 250 650 l S
BT /F1 10 Tf 55 636 Td (Metric) Tj ET
BT /F1 10 Tf 155 636 Td (Value) Tj ET
BT /F1 10 Tf 55 616 Td (Revenue) Tj ET
BT /F1 10 Tf 155 616 Td (45000) Tj ET
BT /F1 10 Tf 55 596 Td (Users) Tj ET
BT /F1 10 Tf 155 596 Td (1200) Tj ET
q 100 0 0 100 50 400 cm /Im0 Do Q
BT /F1 9 Tf 50 390 Td (Figure 1: Revenue Chart) Tj ET
"""

_PAGE2_CONTENT = b"""
BT /F1 10 Tf 50 750 Td (Page two content.) Tj ET
q 80 0 0 80 50 600 cm /Im0 Do Q
"""


def _add_font(writer: PdfWriter):
    font_dict = DictionaryObject()
    font_dict[NameObject("/Type")] = NameObject("/Font")
    font_dict[NameObject("/Subtype")] = NameObject("/Type1")
    font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")
    return writer._add_object(font_dict)


def _add_image(writer: PdfWriter, size: int, rgb: tuple[int, int, int]):
    raw = bytes(rgb) * (size * size)
    stream = DecodedStreamObject()
    stream.set_data(raw)
    stream[NameObject("/Type")] = NameObject("/XObject")
    stream[NameObject("/Subtype")] = NameObject("/Image")
    stream[NameObject("/Width")] = NumberObject(size)
    stream[NameObject("/Height")] = NumberObject(size)
    stream[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    stream[NameObject("/BitsPerComponent")] = NumberObject(8)
    return writer._add_object(stream)


def _build_fixture_pdf(path: Path) -> None:
    writer = PdfWriter()
    font_ref = _add_font(writer)

    page1 = writer.add_blank_page(width=400, height=800)
    image1_ref = _add_image(writer, size=4, rgb=(255, 0, 0))
    resources1 = DictionaryObject()
    resources1[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
    resources1[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): image1_ref})
    page1[NameObject("/Resources")] = resources1
    content1 = DecodedStreamObject()
    content1.set_data(_PAGE1_CONTENT)
    page1.replace_contents(content1)

    page2 = writer.add_blank_page(width=400, height=800)
    image2_ref = _add_image(writer, size=4, rgb=(0, 0, 255))
    resources2 = DictionaryObject()
    resources2[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
    resources2[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): image2_ref})
    page2[NameObject("/Resources")] = resources2
    content2 = DecodedStreamObject()
    content2.set_data(_PAGE2_CONTENT)
    page2.replace_contents(content2)

    with open(path, "wb") as f:
        writer.write(f)


def _build_empty_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=800)
    with open(path, "wb") as f:
        writer.write(f)


class TestPDFLoaderValidFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.fixture_path = cls.tmp_dir / "fixture.pdf"
        _build_fixture_pdf(cls.fixture_path)
        cls.asset_dir = cls.tmp_dir / "assets"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def setUp(self):
        self.loader = PDFLoader(asset_dir=self.asset_dir)
        self.document = self.loader.load(self.fixture_path, uploaded_by="user-1")

    def test_produces_normalized_document(self):
        self.assertIsInstance(self.document, NormalizedDocument)

    def test_document_id_generated(self):
        self.assertTrue(self.document.document_id)

    def test_metadata_filename(self):
        self.assertEqual(self.document.metadata.filename, "fixture.pdf")

    def test_metadata_file_type(self):
        self.assertEqual(self.document.metadata.file_type, "pdf")

    def test_metadata_file_size(self):
        self.assertGreater(self.document.metadata.file_size, 0)

    def test_uploaded_by_preserved(self):
        self.assertEqual(self.document.metadata.uploaded_by, "user-1")

    def test_content_hash_present(self):
        self.assertEqual(len(self.document.metadata.content_hash), 64)

    def test_page_count(self):
        self.assertEqual(self.document.metadata.extra["page_count"], 2)

    def test_element_kinds_in_order(self):
        kinds = [type(e).__name__ for e in self.document.elements]
        self.assertEqual(
            kinds,
            [
                "TextElement",  # heading
                "TextElement",  # body 1
                "TextElement",  # body 2
                "TableElement",
                "ImageElement",
                "TextElement",  # caption line, also emitted as text
                "TextElement",  # page 2 body
                "ImageElement",  # page 2 image, no caption
            ],
        )

    def test_heading_detected(self):
        heading = self.document.elements[0]
        self.assertEqual(heading.content, "Quarterly Report")
        self.assertEqual(heading.heading_level, 1)

    def test_body_text_not_flagged_as_heading(self):
        body = self.document.elements[1]
        self.assertEqual(
            body.content, "This section discusses revenue performance."
        )
        self.assertIsNone(body.heading_level)

    def test_text_page_number(self):
        heading = self.document.elements[0]
        self.assertEqual(heading.source.location.page_number, 1)

    def test_text_source_type(self):
        self.assertEqual(self.document.elements[0].source.source_type, "pdf")

    def test_section_path_after_heading(self):
        body = self.document.elements[1]
        self.assertEqual(body.section_path, ["Quarterly Report"])

    def test_table_element_created(self):
        table = self.document.elements[3]
        self.assertIsInstance(table, TableElement)
        self.assertEqual(table.columns, ["Metric", "Value"])
        self.assertEqual(table.rows, [["Revenue", "45000"], ["Users", "1200"]])

    def test_table_page_number(self):
        table = self.document.elements[3]
        self.assertEqual(table.source.location.page_number, 1)

    def test_table_bbox_preserved(self):
        table = self.document.elements[3]
        self.assertIsNotNone(table.source.location.bbox)
        self.assertEqual(len(table.source.location.bbox), 4)

    def test_table_text_not_duplicated(self):
        contents = [
            e.content for e in self.document.elements if isinstance(e, TextElement)
        ]
        for value in ["Metric", "Value", "Revenue", "45000", "Users", "1200"]:
            self.assertNotIn(value, contents)

    def test_image_element_created(self):
        image = self.document.elements[4]
        self.assertIsInstance(image, ImageElement)

    def test_image_has_asset_reference_not_binary(self):
        image = self.document.elements[4]
        self.assertIsInstance(image.image_uri, str)
        self.assertTrue(Path(image.image_uri).is_file())
        # the pydantic model itself carries no binary payload field
        self.assertNotIn("data", type(image).model_fields)
        self.assertNotIn("bytes", type(image).model_fields)

    def test_image_page_number(self):
        image = self.document.elements[4]
        self.assertEqual(image.source.location.page_number, 1)

    def test_image_caption_associated(self):
        image = self.document.elements[4]
        self.assertEqual(image.label, "Figure 1: Revenue Chart")
        self.assertIsNone(image.description)

    def test_image_without_caption_has_no_fabricated_label(self):
        image = self.document.elements[7]
        self.assertIsInstance(image, ImageElement)
        self.assertIsNone(image.label)
        self.assertIsNone(image.description)

    def test_multi_page_order_preserved(self):
        page_numbers = [e.source.location.page_number for e in self.document.elements]
        self.assertEqual(page_numbers, sorted(page_numbers))
        self.assertEqual(page_numbers[-1], 2)
        self.assertEqual(page_numbers[0], 1)

    def test_element_index_sequential(self):
        indices = [e.element_index for e in self.document.elements]
        self.assertEqual(indices, list(range(len(self.document.elements))))

    def test_source_document_id_matches(self):
        for e in self.document.elements:
            self.assertEqual(e.source.document_id, self.document.document_id)

    def test_serializes_to_json(self):
        payload = self.document.model_dump_json()
        self.assertIsInstance(payload, str)
        self.assertIn("Quarterly Report", payload)


class TestPDFLoaderEmptyPDF(unittest.TestCase):
    def test_empty_pdf_returns_valid_document(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            path = tmp_dir / "empty.pdf"
            _build_empty_pdf(path)
            loader = PDFLoader(asset_dir=tmp_dir / "assets")
            document = loader.load(path)
            self.assertIsInstance(document, NormalizedDocument)
            self.assertEqual(document.elements, [])
            self.assertEqual(document.metadata.extra["page_count"], 1)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestPDFLoaderMissingFile(unittest.TestCase):
    def test_missing_file_raises_clear_error(self):
        loader = PDFLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load(Path(tempfile.gettempdir()) / "does_not_exist.pdf")


class TestPDFLoaderInvalidFile(unittest.TestCase):
    def test_invalid_pdf_raises_clear_error(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            path = tmp_dir / "invalid.pdf"
            path.write_bytes(b"this is not a real pdf file")
            loader = PDFLoader(asset_dir=tmp_dir / "assets")
            with self.assertRaises(ValueError):
                loader.load(path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestPDFLoaderContract(unittest.TestCase):
    def test_loader_inherits_document_loader(self):
        self.assertIsInstance(PDFLoader(), DocumentLoader)


if __name__ == "__main__":
    unittest.main()
