import re

from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.image import ImageChunker
from app.ingestion.chunkers.table import TableChunker
from app.ingestion.chunkers.text import TextChunker
from app.models.chunk import Chunk
from app.models.document import ImageElement, NormalizedDocument, TableElement, TextElement

_MAX_TEXT_BLOCK_VERTICAL_GAP = 72.0
_PAGE_MARKER_RE = re.compile(r"\bpage\s+\d+\s*$", re.IGNORECASE)


def _page_number(element: TextElement) -> int | None:
    location = element.source.location
    return getattr(location, "page_number", None)


def _has_large_vertical_gap(previous: TextElement, current: TextElement) -> bool:
    previous_location = previous.source.location
    current_location = current.source.location
    previous_bbox = getattr(previous_location, "bbox", None)
    current_bbox = getattr(current_location, "bbox", None)
    if previous_bbox is None or current_bbox is None:
        return False
    return current_bbox[1] - previous_bbox[3] > _MAX_TEXT_BLOCK_VERTICAL_GAP


def _looks_like_page_marker(element: TextElement) -> bool:
    return bool(_PAGE_MARKER_RE.search(element.content.strip()))


def _can_coalesce_text(previous: TextElement, current: TextElement) -> bool:
    if previous.heading_level is not None or current.heading_level is not None:
        return False
    if _looks_like_page_marker(previous) or _looks_like_page_marker(current):
        return False
    if previous.section_path != current.section_path:
        return False
    if previous.source.source_type != current.source.source_type:
        return False
    if _page_number(previous) != _page_number(current):
        return False
    return not _has_large_vertical_gap(previous, current)


class ChunkingService:
    """Dispatches each element of a NormalizedDocument to the chunker for
    its type, in document element order, then assigns the final
    document-wide sequential chunk_index across the combined result.

    Does not mutate the NormalizedDocument; produces a derived list[Chunk]
    only. Does not embed, index, or retrieve anything.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._text_chunker = TextChunker(self.config)
        self._table_chunker = TableChunker(self.config)
        self._image_chunker = ImageChunker(self.config)

    def chunk_document(self, document: NormalizedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffered_text: list[TextElement] = []

        def flush_text() -> None:
            nonlocal buffered_text
            if buffered_text:
                chunks.extend(self._text_chunker.chunk_many(buffered_text))
                buffered_text = []

        for element in document.elements:
            if isinstance(element, TextElement):
                if element.heading_level is not None:
                    flush_text()
                    chunks.extend(self._text_chunker.chunk(element))
                elif buffered_text and not _can_coalesce_text(buffered_text[-1], element):
                    flush_text()
                    buffered_text.append(element)
                else:
                    buffered_text.append(element)
            elif isinstance(element, TableElement):
                flush_text()
                chunks.extend(self._table_chunker.chunk(element))
            elif isinstance(element, ImageElement):
                flush_text()
                chunks.extend(self._image_chunker.chunk(element))
            else:
                raise TypeError(f"Unsupported element type: {type(element)!r}")

        flush_text()

        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index

        return chunks
