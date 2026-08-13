import unittest

from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.image import ImageChunker
from app.models.document import ImageElement, PDFLocation, SourceReference


def make_image_element(
    image_id: str = "img-1",
    image_uri: str = "/assets/img-1.png",
    label: str | None = None,
    description: str | None = None,
    section_path: list[str] | None = None,
) -> ImageElement:
    source = SourceReference(
        document_id="doc-1",
        source_type="pdf",
        location=PDFLocation(page_number=3),
    )
    return ImageElement(
        element_id="el-img-1",
        element_index=0,
        source=source,
        section_path=section_path,
        image_id=image_id,
        image_uri=image_uri,
        label=label,
        description=description,
    )


class TestImageChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = ImageChunker(ChunkingConfig())

    def test_image_with_label_produces_searchable_chunk(self):
        element = make_image_element(label="Figure 5.1 — Revenue Trend")
        chunks = self.chunker.chunk(element)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Figure 5.1 — Revenue Trend")

    def test_image_with_label_and_description(self):
        element = make_image_element(
            label="Figure 5.1 — Revenue Trend",
            description="A line chart showing revenue climbing steadily.",
        )
        chunks = self.chunker.chunk(element)
        content = chunks[0].content
        self.assertIn("Figure 5.1 — Revenue Trend", content)
        self.assertIn("A line chart showing revenue climbing steadily.", content)

    def test_image_without_label_or_description_does_not_fabricate(self):
        element = make_image_element()
        chunks = self.chunker.chunk(element)
        content = chunks[0].content
        self.assertNotIn("chart", content.lower())
        self.assertNotIn("shows", content.lower())
        self.assertIn(element.image_id, content)

    def test_image_uri_preserved_in_metadata(self):
        element = make_image_element(
            image_uri="/assets/doc-1/img-1.png", label="Figure 1"
        )
        chunks = self.chunker.chunk(element)
        self.assertEqual(chunks[0].metadata["image_uri"], "/assets/doc-1/img-1.png")

    def test_content_is_text_not_binary(self):
        element = make_image_element(label="Figure 1")
        chunks = self.chunker.chunk(element)
        self.assertIsInstance(chunks[0].content, str)
        self.assertNotIn("data", type(chunks[0]).model_fields)

    def test_source_element_ids_correct(self):
        element = make_image_element(label="Figure 1")
        chunks = self.chunker.chunk(element)
        self.assertEqual(chunks[0].source_element_ids, [element.element_id])

    def test_section_path_propagated_to_metadata(self):
        element = make_image_element(
            label="Figure 1", section_path=["Monthly Performance"]
        )
        chunks = self.chunker.chunk(element)
        self.assertEqual(chunks[0].metadata["section_path"], ["Monthly Performance"])


if __name__ == "__main__":
    unittest.main()
