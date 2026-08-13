import unittest

from pydantic import TypeAdapter, ValidationError

from app.models.document import (
    ImageElement,
    NormalizedElement,
    PDFLocation,
    SourceReference,
    TableElement,
    TextElement,
)


def make_source(source_type: str = "pdf", location=None) -> SourceReference:
    return SourceReference(
        document_id="doc-1",
        source_type=source_type,
        location=location if location is not None else PDFLocation(page_number=1),
    )


class TestTextElement(unittest.TestCase):
    def test_create(self):
        element = TextElement(
            element_id="el-1",
            element_index=0,
            source=make_source(),
            content="hello world",
        )
        self.assertEqual(element.element_type, "text")
        self.assertEqual(element.content, "hello world")


class TestTableElement(unittest.TestCase):
    def test_create_with_typed_values(self):
        element = TableElement(
            element_id="el-2",
            element_index=1,
            source=make_source(),
            columns=["name", "amount"],
            rows=[["alice", 10], ["bob", 20.5]],
        )
        self.assertEqual(element.element_type, "table")
        self.assertEqual(element.rows[0], ["alice", 10])
        self.assertEqual(element.rows[1], ["bob", 20.5])


class TestImageElement(unittest.TestCase):
    def test_create(self):
        element = ImageElement(
            element_id="el-3",
            element_index=2,
            source=make_source(),
            image_id="img-1",
            image_uri="s3://bucket/img-1.png",
            label="Figure 1",
            description="A bar chart of quarterly revenue",
        )
        self.assertEqual(element.element_type, "image")
        self.assertEqual(element.image_uri, "s3://bucket/img-1.png")

    def test_create_without_optional_fields(self):
        element = ImageElement(
            element_id="el-4",
            element_index=3,
            source=make_source(),
            image_id="img-2",
            image_uri="s3://bucket/img-2.png",
        )
        self.assertIsNone(element.label)
        self.assertIsNone(element.description)


class TestNormalizedElementDiscrimination(unittest.TestCase):
    def setUp(self):
        self.adapter = TypeAdapter(NormalizedElement)

    def test_discriminates_text(self):
        parsed = self.adapter.validate_python(
            {
                "element_id": "el-1",
                "element_index": 0,
                "source": make_source().model_dump(),
                "element_type": "text",
                "content": "hello",
            }
        )
        self.assertIsInstance(parsed, TextElement)

    def test_discriminates_table(self):
        parsed = self.adapter.validate_python(
            {
                "element_id": "el-2",
                "element_index": 1,
                "source": make_source().model_dump(),
                "element_type": "table",
                "columns": ["a"],
                "rows": [[1]],
            }
        )
        self.assertIsInstance(parsed, TableElement)

    def test_discriminates_image(self):
        parsed = self.adapter.validate_python(
            {
                "element_id": "el-3",
                "element_index": 2,
                "source": make_source().model_dump(),
                "element_type": "image",
                "image_id": "img-1",
                "image_uri": "s3://bucket/img-1.png",
            }
        )
        self.assertIsInstance(parsed, ImageElement)


class TestSourceReferenceValidation(unittest.TestCase):
    def test_rejects_mismatched_location(self):
        with self.assertRaises(ValidationError):
            SourceReference(
                document_id="doc-1",
                source_type="csv",
                location=PDFLocation(page_number=1),
            )


if __name__ == "__main__":
    unittest.main()
