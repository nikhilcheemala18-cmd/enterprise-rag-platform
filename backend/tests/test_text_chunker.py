import unittest

from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.text import TextChunker
from app.models.document import PDFLocation, SourceReference, TextElement


def make_text_element(
    content: str,
    element_id: str = "el-1",
    section_path: list[str] | None = None,
    heading_level: int | None = None,
) -> TextElement:
    source = SourceReference(
        document_id="doc-1",
        source_type="pdf",
        location=PDFLocation(page_number=1),
    )
    return TextElement(
        element_id=element_id,
        element_index=0,
        source=source,
        section_path=section_path,
        content=content,
        heading_level=heading_level,
    )


class TestTextChunkerShortText(unittest.TestCase):
    def test_short_text_produces_one_chunk(self):
        chunker = TextChunker(ChunkingConfig())
        element = make_text_element(
            "Our retention program focuses on repeat purchases."
        )
        chunks = chunker.chunk(element)
        self.assertEqual(len(chunks), 1)
        self.assertIn("repeat purchases", chunks[0].content)

    def test_empty_text_produces_no_chunks(self):
        chunker = TextChunker(ChunkingConfig())
        element = make_text_element("   ")
        self.assertEqual(chunker.chunk(element), [])


class TestTextChunkerLongText(unittest.TestCase):
    def setUp(self):
        self.config = ChunkingConfig(text_chunk_size=10, text_chunk_overlap=3)
        self.chunker = TextChunker(self.config)
        self.content = (
            "Paragraph one has exactly eight words here now.\n\n"
            "Paragraph two also has eight words too now.\n\n"
            "Paragraph three has eight words as well now.\n\n"
            "Paragraph four has eight words also right now."
        )
        self.element = make_text_element(self.content)
        self.chunks = self.chunker.chunk(self.element)

    def test_long_text_produces_multiple_chunks(self):
        self.assertGreater(len(self.chunks), 1)

    def test_chunks_respect_configured_size_approximately(self):
        for chunk in self.chunks:
            word_count = len(chunk.content.split())
            # allow overlap carry-over slack on top of the configured size
            self.assertLessEqual(
                word_count, self.config.text_chunk_size + self.config.text_chunk_overlap
            )

    def test_overlap_is_applied_between_consecutive_chunks(self):
        overlap = self.config.text_chunk_overlap
        for i in range(len(self.chunks) - 1):
            tail = self.chunks[i].content.split()[-overlap:]
            head = self.chunks[i + 1].content.split()[:overlap]
            self.assertEqual(tail, head)

    def test_source_element_ids_repeat_same_element(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.source_element_ids, [self.element.element_id])

    def test_original_text_element_not_modified(self):
        self.assertEqual(self.element.content, self.content)

    def test_token_count_populated(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.token_count, len(chunk.content.split()))
            self.assertGreater(chunk.token_count, 0)

    def test_chunk_ids_unique(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))


class TestTextChunkerSectionContext(unittest.TestCase):
    def test_section_path_included_in_content(self):
        chunker = TextChunker(ChunkingConfig())
        element = make_text_element(
            "We send weekly campaigns to re-engage lapsed customers.",
            section_path=["Retention Strategy", "Email Campaigns"],
        )
        chunks = chunker.chunk(element)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(
            chunks[0].content.startswith(
                "Section: Retention Strategy > Email Campaigns\n\n"
            )
        )

    def test_no_section_prefix_when_section_path_is_none(self):
        chunker = TextChunker(ChunkingConfig())
        element = make_text_element("Plain paragraph with no section context.")
        chunks = chunker.chunk(element)
        self.assertFalse(chunks[0].content.startswith("Section:"))

    def test_original_text_element_content_unaffected_by_section_prefix(self):
        chunker = TextChunker(ChunkingConfig())
        original = "We send weekly campaigns to re-engage lapsed customers."
        element = make_text_element(
            original, section_path=["Retention Strategy", "Email Campaigns"]
        )
        chunker.chunk(element)
        self.assertEqual(element.content, original)


if __name__ == "__main__":
    unittest.main()
