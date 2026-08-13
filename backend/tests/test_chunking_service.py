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
