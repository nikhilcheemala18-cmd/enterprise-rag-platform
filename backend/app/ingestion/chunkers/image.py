import uuid

from app.ingestion.chunkers.base import BaseChunker, estimate_token_count, location_metadata
from app.models.chunk import Chunk
from app.models.document import ImageElement


class ImageChunker(BaseChunker):
    """One ImageElement -> one Chunk. No OCR, no vision model, no
    generated description. Chunk.content is built only from what the
    normalized element already carries (label / description); if neither
    is present, a non-semantic identifier placeholder is used instead of
    fabricating meaning -- Chunk.content cannot be empty.
    """

    def chunk(self, element: ImageElement) -> list[Chunk]:
        if element.label and element.description:
            content = f"{element.label}\n\n{element.description}"
        elif element.label:
            content = element.label
        elif element.description:
            content = element.description
        else:
            content = f"[Image: {element.image_id}]"

        metadata = {
            "element_type": "image",
            "section_path": element.section_path,
            "image_uri": element.image_uri,
            "label": element.label,
            **location_metadata(element.source),
        }

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=element.source.document_id,
            source_element_ids=[element.element_id],
            content=content,
            token_count=estimate_token_count(content),
            chunk_index=0,
            metadata=metadata,
        )
        return [chunk]
