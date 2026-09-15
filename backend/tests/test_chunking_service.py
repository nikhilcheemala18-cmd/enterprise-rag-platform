import unittest

from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.service import ChunkingService
from app.models.document import (
    DocumentMetadata,
    ImageElement,
    NormalizedDocument,
    PDFLocation,
    SourceReference,
    TableElement,
    TextElement,
)

DOCUMENT_ID = "doc-mixed-1"


def _source(location=None) -> SourceReference:
    return SourceReference(
        document_id=DOCUMENT_ID,
        source_type="pdf",
        location=location or PDFLocation(page_number=1),
    )


def _text(
    element_id: str,
    index: int,
    content: str,
    section_path: list[str] | None = None,
    heading_level: int | None = None,
    page_number: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
) -> TextElement:
    return TextElement(
        element_id=element_id,
        element_index=index,
        source=_source(PDFLocation(page_number=page_number, bbox=bbox)),
        section_path=section_path,
        content=content,
        heading_level=heading_level,
    )


def _build_mixed_document() -> NormalizedDocument:
    elements = [
        TextElement(
            element_id="el-0",
            element_index=0,
            source=_source(),
            content="Monthly Performance — Client A",
            heading_level=1,
        ),
        TextElement(
            element_id="el-1",
            element_index=1,
            source=_source(),
            section_path=["Monthly Performance — Client A"],
            content="Revenue increased by 18% in July compared to June.",
        ),
        TableElement(
            element_id="el-2",
            element_index=2,
            source=_source(),
            section_path=["Monthly Performance — Client A"],
            columns=["Month", "Revenue", "Orders"],
            rows=[["June", 72000, 1120], ["July", 85000, 1350]],
        ),
        ImageElement(
            element_id="el-3",
            element_index=3,
            source=_source(),
            section_path=["Monthly Performance — Client A"],
            image_id="img-1",
            image_uri="/assets/doc-mixed-1/img-1.png",
            label="Figure 5.1 — Revenue Trend",
        ),
        TextElement(
            element_id="el-4",
            element_index=4,
            source=_source(),
            section_path=["Monthly Performance — Client A"],
            content="Campaign performance is reviewed monthly.",
        ),
    ]
    return NormalizedDocument(
        document_id=DOCUMENT_ID,
        metadata=DocumentMetadata(
            filename="report.pdf",
            file_type="pdf",
            file_size=1234,
            content_hash="abc123",
        ),
        elements=elements,
    )


class TestChunkingServiceMixedDocument(unittest.TestCase):
    def setUp(self):
        self.service = ChunkingService()
        self.document = _build_mixed_document()
        self.chunks = self.service.chunk_document(self.document)

    def test_element_type_order_preserved(self):
        types = [c.metadata["element_type"] for c in self.chunks]
        self.assertEqual(types, ["text", "text", "table", "image", "text"])

    def test_chunk_index_globally_sequential(self):
        indices = [c.chunk_index for c in self.chunks]
        self.assertEqual(indices, list(range(len(self.chunks))))

    def test_document_id_preserved_on_every_chunk(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.document_id, DOCUMENT_ID)

    def test_each_chunk_traceable_to_its_source_element(self):
        expected_source = {
            0: "el-0",
            1: "el-1",
            2: "el-2",
            3: "el-3",
            4: "el-4",
        }
        for i, chunk in enumerate(self.chunks):
            self.assertEqual(chunk.source_element_ids, [expected_source[i]])

    def test_table_chunk_content(self):
        table_chunk = self.chunks[2]
        self.assertIn("Month | Revenue | Orders", table_chunk.content)
        self.assertIn("June | 72000 | 1120", table_chunk.content)

    def test_image_chunk_content(self):
        image_chunk = self.chunks[3]
        self.assertEqual(image_chunk.content, "Figure 5.1 — Revenue Trend")
        self.assertEqual(
            image_chunk.metadata["image_uri"], "/assets/doc-mixed-1/img-1.png"
        )

    def test_chunk_ids_all_unique(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_original_document_not_mutated(self):
        self.assertEqual(len(self.document.elements), 5)
        self.assertEqual(self.document.elements[2].rows, [["June", 72000, 1120], ["July", 85000, 1350]])


class TestChunkingServiceEmptyDocument(unittest.TestCase):
    def test_empty_document_produces_no_chunks(self):
        service = ChunkingService()
        document = NormalizedDocument(
            document_id="doc-empty",
            metadata=DocumentMetadata(
                filename="empty.pdf",
                file_type="pdf",
                file_size=0,
                content_hash="def456",
            ),
            elements=[],
        )
        self.assertEqual(service.chunk_document(document), [])


class TestChunkingServiceCoalescesAdjacentText(unittest.TestCase):
    def _document(self, elements):
        return NormalizedDocument(
            document_id=DOCUMENT_ID,
            metadata=DocumentMetadata(
                filename="coalesce.pdf",
                file_type="pdf",
                file_size=10,
                content_hash="coalesce",
            ),
            elements=elements,
        )

    def test_multiple_small_paragraphs_become_one_larger_chunk(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text("el-1", 0, "First paragraph has a few useful words."),
                _text("el-2", 1, "Second paragraph also has useful words."),
                _text("el-3", 2, "Third paragraph remains part of this chunk."),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_element_ids, ["el-1", "el-2", "el-3"])
        self.assertIn("words.\n\nSecond paragraph", chunks[0].content)

    def test_section_boundaries_are_not_crossed(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text("el-1", 0, "Paragraph in the first section.", ["Section A"]),
                _text("el-2", 1, "Paragraph in the second section.", ["Section B"]),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_element_ids, ["el-1"])
        self.assertEqual(chunks[1].source_element_ids, ["el-2"])

    def test_tables_remain_separate_boundaries(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text("el-1", 0, "First paragraph before the table."),
                _text("el-2", 1, "Second paragraph before the table."),
                TableElement(
                    element_id="el-table",
                    element_index=2,
                    source=_source(),
                    columns=["Metric", "Value"],
                    rows=[["Latency", "200ms"]],
                ),
                _text("el-3", 3, "First paragraph after the table."),
                _text("el-4", 4, "Second paragraph after the table."),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual([c.metadata["element_type"] for c in chunks], ["text", "table", "text"])
        self.assertEqual(chunks[0].source_element_ids, ["el-1", "el-2"])
        self.assertEqual(chunks[1].source_element_ids, ["el-table"])
        self.assertEqual(chunks[2].source_element_ids, ["el-3", "el-4"])

    def test_heading_boundaries_are_not_merged_into_body_text(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text("el-heading", 0, "1. Platform Architecture", heading_level=1),
                _text("el-1", 1, "First body paragraph.", ["1. Platform Architecture"]),
                _text("el-2", 2, "Second body paragraph.", ["1. Platform Architecture"]),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_element_ids, ["el-heading"])
        self.assertEqual(chunks[0].metadata["heading_level"], 1)
        self.assertEqual(chunks[1].source_element_ids, ["el-1", "el-2"])
        self.assertIsNone(chunks[1].metadata["heading_level"])

    def test_page_boundaries_are_not_crossed(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text("el-1", 0, "Last paragraph on page one.", page_number=1),
                _text("el-2", 1, "First paragraph on page two.", page_number=2),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["page_number"], 1)
        self.assertEqual(chunks[1].metadata["page_number"], 2)

    def test_large_vertical_gap_is_not_coalesced(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text(
                    "el-1",
                    0,
                    "Body paragraph near the top of the page.",
                    bbox=(50.0, 100.0, 350.0, 140.0),
                ),
                _text(
                    "el-footer",
                    1,
                    "Synthetic footer Page 1",
                    bbox=(50.0, 760.0, 350.0, 770.0),
                ),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_element_ids, ["el-1"])
        self.assertEqual(chunks[1].source_element_ids, ["el-footer"])

    def test_page_marker_footer_is_not_coalesced(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=50, text_chunk_overlap=5))
        document = self._document(
            [
                _text("el-1", 0, "Body paragraph near the end of the page."),
                _text("el-footer", 1, "Synthetic RAG Evaluation Corpus - Document 01 Page 2"),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_element_ids, ["el-1"])
        self.assertEqual(chunks[1].source_element_ids, ["el-footer"])

    def test_configured_chunk_size_and_overlap_are_respected(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=10, text_chunk_overlap=3))
        document = self._document(
            [
                _text("el-1", 0, "one two three four"),
                _text("el-2", 1, "five six seven eight"),
                _text("el-3", 2, "nine ten eleven twelve"),
            ]
        )

        chunks = service.chunk_document(document)

        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(chunks[0].token_count, 10)
        self.assertLessEqual(chunks[1].token_count, 13)
        self.assertEqual(chunks[0].content.split()[-3:], chunks[1].content.split()[:3])
        self.assertEqual(chunks[0].source_element_ids, ["el-1", "el-2"])
        self.assertEqual(chunks[1].source_element_ids, ["el-2", "el-3"])

    def test_oversized_single_text_element_keeps_existing_split_behavior(self):
        service = ChunkingService(ChunkingConfig(text_chunk_size=5, text_chunk_overlap=2))
        document = self._document(
            [
                _text(
                    "el-long",
                    0,
                    "one two three four five six seven eight nine ten eleven twelve",
                )
            ]
        )

        chunks = service.chunk_document(document)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.source_element_ids, ["el-long"])


class TestChunkingServiceTextSplitProducesMultipleChunksWithSharedIndex(
    unittest.TestCase
):
    def test_index_continues_sequentially_across_a_split_text_element(self):
        config = ChunkingConfig(text_chunk_size=8, text_chunk_overlap=2)
        service = ChunkingService(config)
        long_text = (
            "First short paragraph with several words in it now.\n\n"
            "Second short paragraph with several more words now.\n\n"
            "Third short paragraph with even more words right now."
        )
        document = NormalizedDocument(
            document_id="doc-split",
            metadata=DocumentMetadata(
                filename="split.pdf",
                file_type="pdf",
                file_size=10,
                content_hash="ghi789",
            ),
            elements=[
                TextElement(
                    element_id="el-a",
                    element_index=0,
                    source=_source(),
                    content=long_text,
                ),
                TableElement(
                    element_id="el-b",
                    element_index=1,
                    source=_source(),
                    columns=["A"],
                    rows=[["1"]],
                ),
            ],
        )
        chunks = service.chunk_document(document)
        self.assertGreater(len(chunks), 2)
        indices = [c.chunk_index for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))
        # every text-derived chunk before the table chunk shares the same source element
        text_chunks = [c for c in chunks if c.metadata["element_type"] == "text"]
        for c in text_chunks:
            self.assertEqual(c.source_element_ids, ["el-a"])
        self.assertEqual(chunks[-1].metadata["element_type"], "table")


if __name__ == "__main__":
    unittest.main()
