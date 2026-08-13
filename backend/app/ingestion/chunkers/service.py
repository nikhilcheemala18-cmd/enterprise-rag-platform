from app.ingestion.chunkers.config import ChunkingConfig
from app.ingestion.chunkers.image import ImageChunker
from app.ingestion.chunkers.table import TableChunker
from app.ingestion.chunkers.text import TextChunker
from app.models.chunk import Chunk
from app.models.document import ImageElement, NormalizedDocument, TableElement, TextElement


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

        for element in document.elements:
            if isinstance(element, TextElement):
                chunks.extend(self._text_chunker.chunk(element))
            elif isinstance(element, TableElement):
                chunks.extend(self._table_chunker.chunk(element))
            elif isinstance(element, ImageElement):
                chunks.extend(self._image_chunker.chunk(element))
            else:
                raise TypeError(f"Unsupported element type: {type(element)!r}")

        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index

        return chunks
