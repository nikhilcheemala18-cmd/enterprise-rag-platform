r"""Temporary diagnostic for inspecting PDF loader output before chunking.

Usage:
    python scripts/debug_pdf_elements.py C:\path\to\document.pdf
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.chunkers.base import estimate_token_count
from app.ingestion.chunkers.service import ChunkingService
from app.ingestion.loaders.pdf import PDFLoader
from app.models.document import TableElement, TextElement


def _preview(text: str, limit: int = 150) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PDF NormalizedDocument elements before chunking."
    )
    parser.add_argument("pdf_path", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="rag_pdf_debug_assets_") as asset_dir:
        document = PDFLoader(asset_dir=asset_dir).load(args.pdf_path)

    text_elements = [e for e in document.elements if isinstance(e, TextElement)]
    table_elements = [e for e in document.elements if isinstance(e, TableElement)]
    text_token_counts = [estimate_token_count(e.content) for e in text_elements]

    print("PDF LOADER OUTPUT")
    print(f"filename: {document.metadata.filename}")
    print(f"document_id: {document.document_id}")
    print(f"file_type: {document.metadata.file_type}")
    print(f"page_count: {document.metadata.extra.get('page_count')}")
    print(f"total_elements: {len(document.elements)}")
    print()

    print("TEXT ELEMENTS BEFORE CHUNKING")
    for element in text_elements:
        location = element.source.location
        page_number = getattr(location, "page_number", None)
        token_count = estimate_token_count(element.content)
        section_path = " > ".join(element.section_path or [])
        print(
            " | ".join(
                [
                    f"element_index={element.element_index}",
                    f"page={page_number}",
                    f"type={element.element_type}",
                    f"heading_level={element.heading_level}",
                    f"section={section_path or '<none>'}",
                    f"chars={len(element.content)}",
                    f"tokens={token_count}",
                    f"preview={_preview(element.content)!r}",
                ]
            )
        )

    print()
    print("TABLE ELEMENTS BEFORE CHUNKING")
    for element in table_elements:
        location = element.source.location
        page_number = getattr(location, "page_number", None)
        section_path = " > ".join(element.section_path or [])
        print(
            " | ".join(
                [
                    f"element_index={element.element_index}",
                    f"page={page_number}",
                    f"type={element.element_type}",
                    f"section={section_path or '<none>'}",
                    f"columns={len(element.columns)}",
                    f"rows={len(element.rows)}",
                    f"header={_preview(' | '.join(str(c) for c in element.columns))!r}",
                ]
            )
        )

    chunks = ChunkingService().chunk_document(document)
    text_chunks = [c for c in chunks if c.metadata.get("element_type") == "text"]
    table_chunks = [c for c in chunks if c.metadata.get("element_type") == "table"]
    chunk_token_counts = [chunk.token_count for chunk in chunks]

    print()
    print("SUMMARY")
    print(f"total_pdf_text_elements: {len(text_elements)}")
    print(f"total_table_elements: {len(table_elements)}")
    print(
        "average_text_element_token_count: "
        f"{statistics.mean(text_token_counts):.1f}" if text_token_counts else "0"
    )
    print(f"minimum_text_element_token_count: {min(text_token_counts) if text_token_counts else 0}")
    print(f"maximum_text_element_token_count: {max(text_token_counts) if text_token_counts else 0}")
    print(f"total_chunks_after_chunking: {len(chunks)}")
    print(f"text_chunks_after_chunking: {len(text_chunks)}")
    print(f"table_chunks_after_chunking: {len(table_chunks)}")
    print(
        "average_final_chunk_token_count: "
        f"{statistics.mean(chunk_token_counts):.1f}" if chunk_token_counts else "0"
    )
    print(f"minimum_final_chunk_token_count: {min(chunk_token_counts) if chunk_token_counts else 0}")
    print(f"maximum_final_chunk_token_count: {max(chunk_token_counts) if chunk_token_counts else 0}")
    print(
        "chunks_close_to_500_token_target: "
        f"{sum(400 <= count <= 575 for count in chunk_token_counts)}"
    )


if __name__ == "__main__":
    main()
