import uuid

from app.ingestion.chunkers.base import BaseChunker, estimate_token_count, location_metadata
from app.models.chunk import Chunk
from app.models.document import TextElement

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_AVG_CHARS_PER_TOKEN = 6


def _split_into_pieces(text: str, separators: list[str], token_limit: int) -> list[str]:
    """Recursively break text into pieces each within token_limit.

    Separators are tried in order (paragraph, line, sentence-ish,
    whitespace, character-level), preferring the largest semantic
    boundary that actually fits the piece under the limit before falling
    back to a smaller one.
    """
    text = text.strip()
    if not text:
        return []
    if estimate_token_count(text) <= token_limit:
        return [text]
    if not separators:
        return [text]

    sep, *rest = separators
    if sep == "":
        char_limit = max(token_limit * _AVG_CHARS_PER_TOKEN, 1)
        return [text[i : i + char_limit] for i in range(0, len(text), char_limit)]

    if sep not in text:
        return _split_into_pieces(text, rest, token_limit)

    pieces: list[str] = []
    for part in text.split(sep):
        part = part.strip()
        if part:
            pieces.extend(_split_into_pieces(part, rest, token_limit))
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks up to chunk_size tokens,
    carrying the last `overlap` tokens of each chunk into the next one.
    """
    chunks: list[str] = []
    current_words: list[str] = []

    for piece in pieces:
        piece_words = piece.split()
        if current_words and len(current_words) + len(piece_words) > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap > 0 else []
        current_words.extend(piece_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _with_section_context(text: str, section_path: list[str] | None) -> str:
    if not section_path:
        return text
    return f"Section: {' > '.join(section_path)}\n\n{text}"


class TextChunker(BaseChunker):
    """Splits TextElement.content using a dependency-free recursive
    splitter (paragraph -> line -> sentence-ish -> whitespace -> character
    fallback), sized by the whitespace-token approximation in
    estimate_token_count. TextElement.content itself is never modified;
    section_path is prepended only to the derived Chunk.content.
    """

    def chunk(self, element: TextElement) -> list[Chunk]:
        content = element.content.strip()
        if not content:
            return []

        pieces = _split_into_pieces(
            content, list(_SEPARATORS), self.config.text_chunk_size
        )
        merged = _merge_pieces(
            pieces, self.config.text_chunk_size, self.config.text_chunk_overlap
        )

        chunks: list[Chunk] = []
        for piece in merged:
            chunk_content = _with_section_context(piece, element.section_path)
            metadata = {
                "element_type": "text",
                "section_path": element.section_path,
                "heading_level": element.heading_level,
                **location_metadata(element.source),
            }
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=element.source.document_id,
                    source_element_ids=[element.element_id],
                    content=chunk_content,
                    token_count=estimate_token_count(chunk_content),
                    chunk_index=0,
                    metadata=metadata,
                )
            )
        return chunks
