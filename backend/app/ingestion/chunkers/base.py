from abc import ABC, abstractmethod
from typing import Any

from app.ingestion.chunkers.config import ChunkingConfig
from app.models.chunk import Chunk
from app.models.document import NormalizedElement, SourceReference


def estimate_token_count(text: str) -> int:
    """Whitespace-token approximation of token count.

    This is NOT a real tokenizer. It counts whitespace-separated words as
    a deterministic, dependency-free stand-in until an embedding model is
    chosen and a real tokenizer is wired in.
    """
    return len(text.split())


def location_metadata(source: SourceReference) -> dict[str, Any]:
    """Pull the few location fields that are common across source types.

    Only fields that exist on the element's specific location type are
    included (e.g. page_number for PDF, sheet_name for Excel) -- this does
    NOT copy the whole SourceReference into metadata; the canonical
    provenance path remains source_element_ids -> NormalizedElement.source.
    """
    metadata: dict[str, Any] = {"file_type": source.source_type}
    location = source.location
    if location is None:
        return metadata
    if hasattr(location, "page_number"):
        metadata["page_number"] = location.page_number
    if hasattr(location, "sheet_name"):
        metadata["sheet_name"] = location.sheet_name
    return metadata


class BaseChunker(ABC):
    """Common contract for element-type-specific chunkers.

    Each chunker turns exactly one NormalizedElement into zero or more
    Chunks. It never reaches across elements. chunk_index on the returned
    Chunks is a placeholder (0) -- ChunkingService assigns the final,
    document-wide sequential index after collecting chunks from all
    elements in document order.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, element: NormalizedElement) -> list[Chunk]:
        raise NotImplementedError
