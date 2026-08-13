import uuid
from typing import Any

from app.ingestion.chunkers.base import BaseChunker, estimate_token_count, location_metadata
from app.models.chunk import Chunk
from app.models.document import TableElement


def _render_row(values: list[Any]) -> str:
    return " | ".join("" if v is None else str(v) for v in values)


def _render_table(columns: list[str], rows: list[list[Any]]) -> str:
    lines = [_render_row(columns)]
    lines.extend(_render_row(row) for row in rows)
    return "\n".join(lines)


def _with_section_context(text: str, section_path: list[str] | None) -> str:
    if not section_path:
        return text
    return f"Section: {' > '.join(section_path)}\n\n{text}"


class TableChunker(BaseChunker):
    """Splits a TableElement into row groups of at most
    config.table_max_rows, repeating the column headers in every chunk so
    each chunk is independently interpretable. TableElement.rows is only
    read, never modified -- the rendered "col | col" text exists solely
    for retrieval and is not a source of truth.
    """

    def chunk(self, element: TableElement) -> list[Chunk]:
        if not element.rows:
            return []

        max_rows = self.config.table_max_rows
        chunks: list[Chunk] = []

        for start in range(0, len(element.rows), max_rows):
            group = element.rows[start : start + max_rows]
            end = start + len(group) - 1

            table_text = "Table:\n" + _render_table(element.columns, group)
            content = _with_section_context(table_text, element.section_path)

            metadata = {
                "element_type": "table",
                "section_path": element.section_path,
                "row_range": {"start": start, "end": end},
                **location_metadata(element.source),
            }

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=element.source.document_id,
                    source_element_ids=[element.element_id],
                    content=content,
                    token_count=estimate_token_count(content),
                    chunk_index=0,
                    metadata=metadata,
                )
            )

        return chunks
