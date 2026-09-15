import hashlib
import re
import statistics
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

from app.ingestion.base import DocumentLoader
from app.models.document import (
    DocumentMetadata,
    ImageElement,
    NormalizedDocument,
    PDFLocation,
    SourceReference,
    TableElement,
    TextElement,
)

_CAPTION_RE = re.compile(r"^(figure|fig\.?|chart|diagram)\b", re.IGNORECASE)
_HEADING_SIZE_RATIO = 1.15
_LINE_TOP_TOLERANCE = 3.0
_FONT_SIZE_TOLERANCE = 0.75
_LEFT_EDGE_TOLERANCE = 12.0
_PARAGRAPH_GAP_MULTIPLIER = 1.75
_MAX_LINE_GAP_TO_FONT_SIZE_RATIO = 0.85
_CAPTION_MAX_VERTICAL_GAP = 40.0
_DEFAULT_ASSET_DIR = Path(tempfile.gettempdir()) / "enterprise_rag_pdf_assets"

BBox = tuple[float, float, float, float]


def _bbox_overlaps(a: BBox, b: BBox) -> bool:
    ax0, atop, ax1, abottom = a
    bx0, btop, bx1, bbottom = b
    return ax0 < bx1 and ax1 > bx0 and atop < bbottom and abottom > btop


def _extract_text_lines(page: "pdfplumber.page.Page") -> list[dict[str, Any]]:
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    groups: list[list[dict[str, Any]]] = []
    for word in words:
        if groups and abs(word["top"] - groups[-1][0]["top"]) <= _LINE_TOP_TOLERANCE:
            groups[-1].append(word)
        else:
            groups.append([word])

    lines: list[dict[str, Any]] = []
    for group in groups:
        group = sorted(group, key=lambda w: w["x0"])
        content = " ".join(w["text"] for w in group).strip()
        if not content:
            continue
        bbox = (
            min(w["x0"] for w in group),
            min(w["top"] for w in group),
            max(w["x1"] for w in group),
            max(w["bottom"] for w in group),
        )
        size = max(w["size"] for w in group)
        lines.append({"content": content, "bbox": bbox, "size": size})
    return lines


def _is_heading_line(line: dict[str, Any], body_size: float | None) -> bool:
    return body_size is not None and line["size"] > body_size * _HEADING_SIZE_RATIO


def _typical_body_line_gap(
    lines: list[dict[str, Any]], body_size: float | None
) -> float | None:
    gaps: list[float] = []
    for previous, current in zip(lines, lines[1:]):
        if _is_heading_line(previous, body_size) or _is_heading_line(current, body_size):
            continue
        if abs(previous["size"] - current["size"]) > _FONT_SIZE_TOLERANCE:
            continue
        if abs(previous["bbox"][0] - current["bbox"][0]) > _LEFT_EDGE_TOLERANCE:
            continue

        gap = current["bbox"][1] - previous["bbox"][3]
        if 0 <= gap <= previous["size"]:
            gaps.append(gap)

    return statistics.median(gaps) if gaps else None


def _has_intervening_table(
    previous: dict[str, Any],
    current: dict[str, Any],
    table_bboxes: list[BBox],
) -> bool:
    previous_bottom = previous["bbox"][3]
    current_top = current["bbox"][1]
    return any(previous_bottom <= table[1] and table[3] <= current_top for table in table_bboxes)


def _should_merge_text_lines(
    previous: dict[str, Any],
    current: dict[str, Any],
    body_size: float | None,
    typical_gap: float | None,
    table_bboxes: list[BBox],
) -> bool:
    if _is_heading_line(previous, body_size) or _is_heading_line(current, body_size):
        return False
    if abs(previous["size"] - current["size"]) > _FONT_SIZE_TOLERANCE:
        return False
    if abs(previous["bbox"][0] - current["bbox"][0]) > _LEFT_EDGE_TOLERANCE:
        return False
    if _has_intervening_table(previous, current, table_bboxes):
        return False

    gap = current["bbox"][1] - previous["bbox"][3]
    if gap < 0:
        return False

    if typical_gap is None:
        max_gap = previous["size"] * 0.5
    else:
        max_gap = min(
            typical_gap * _PARAGRAPH_GAP_MULTIPLIER,
            previous["size"] * _MAX_LINE_GAP_TO_FONT_SIZE_RATIO,
        )
    return gap <= max_gap


def _merge_text_lines_into_blocks(
    lines: list[dict[str, Any]], body_size: float | None, table_bboxes: list[BBox]
) -> list[dict[str, Any]]:
    """Merge adjacent visual body lines into conservative paragraph blocks.

    Headings stay as standalone blocks. Page and table boundaries are honored
    by the caller/passed table boxes; this only groups body lines that look
    like ordinary wrapped text in the same paragraph.
    """
    if not lines:
        return []

    typical_gap = _typical_body_line_gap(lines, body_size)
    blocks: list[dict[str, Any]] = []
    current = dict(lines[0])
    last_line = lines[0]

    for line in lines[1:]:
        if _should_merge_text_lines(last_line, line, body_size, typical_gap, table_bboxes):
            bbox = current["bbox"]
            line_bbox = line["bbox"]
            current["content"] = f"{current['content']} {line['content']}"
            current["bbox"] = (
                min(bbox[0], line_bbox[0]),
                min(bbox[1], line_bbox[1]),
                max(bbox[2], line_bbox[2]),
                max(bbox[3], line_bbox[3]),
            )
            current["size"] = max(current["size"], line["size"])
        else:
            blocks.append(current)
            current = dict(line)
        last_line = line

    blocks.append(current)
    return blocks


def _extract_tables(page: "pdfplumber.page.Page") -> list[dict[str, Any]]:
    results = []
    for table in page.find_tables():
        extracted = table.extract()
        rows_data = [
            ["" if cell is None else cell for cell in row]
            for row in extracted
            if row is not None
        ]
        if not rows_data or not any(any(cell for cell in row) for row in rows_data):
            continue
        results.append(
            {
                "bbox": table.bbox,
                "columns": rows_data[0],
                "rows": rows_data[1:],
            }
        )
    return results


def _nearest_caption(image_bbox: BBox, lines: list[dict[str, Any]]) -> str | None:
    ix0, _itop, ix1, ibottom = image_bbox
    best: tuple[float, str] | None = None
    for line in lines:
        if not _CAPTION_RE.match(line["content"]):
            continue
        lx0, ltop, lx1, _lbottom = line["bbox"]
        gap = ltop - ibottom
        if not (0 <= gap <= _CAPTION_MAX_VERTICAL_GAP):
            continue
        overlap = min(ix1, lx1) - max(ix0, lx0)
        if overlap <= 0:
            continue
        if best is None or gap < best[0]:
            best = (gap, line["content"])
    return best[1] if best else None


class PDFLoader(DocumentLoader):
    """Loads a PDF into a NormalizedDocument covering text, tables, and images.

    Library split: pdfplumber does text extraction (with font-size info),
    layout, and table detection; pypdf extracts the raw bytes of embedded
    images (leveraging its own decoding, e.g. reconstructing a usable PNG
    from a raw XObject). Neither library alone covers both jobs well.

    Coordinate convention: PDFLocation.bbox is (x0, top, x1, bottom) exactly
    as pdfplumber reports it -- a top-down system with the origin at the
    page's top-left corner (top = distance from the top of the page). This
    is NOT the PDF spec's native bottom-up coordinate system; it is used
    as-is from the extraction library rather than converted, to avoid
    inventing a transform.

    Reading order: within a page, text lines, tables, and images are
    merged into one sequence sorted by vertical position (bbox top) to
    approximate reading order. An image with no matching pdfplumber layout
    entry (bbox unknown) sorts after everything else on its page.

    Heading detection (best-effort only): a line's font size is compared
    to the single most common ("mode") line font size across the whole
    document. A line notably larger than that mode is treated as a single
    heading level (heading_level=1) and resets the section_path breadcrumb
    to just that heading's text. This is a coarse heuristic, not a layout
    model -- it does not distinguish multiple heading depths.

    Table/text overlap: a text line whose bbox overlaps a detected table's
    bbox on the same page is dropped from the text stream, so table
    content is not duplicated as a separate TextElement. No such
    deduplication is attempted for image captions -- a caption line is
    both copied into ImageElement.label AND still emitted as its own
    TextElement.

    Image captions: only a nearby line matching an explicit "Figure/Fig/
    Chart/Diagram" prefix, positioned below the image with horizontal
    overlap, is used as ImageElement.label. ImageElement.description is
    always None -- this loader does not generate semantic descriptions.

    Image storage: extracted image bytes are written to
    `<asset_dir>/<document_id>/<image_id><ext>` (asset_dir defaults to a
    directory under the system temp dir) and referenced via image_uri.
    This is a placeholder for a real asset-storage service, which does
    not exist yet in this project.
    """

    def __init__(self, asset_dir: str | Path | None = None):
        self.asset_dir = Path(asset_dir) if asset_dir else _DEFAULT_ASSET_DIR

    def load(
        self,
        file_path: str | Path,
        uploaded_by: str | None = None,
    ) -> NormalizedDocument:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if not path.is_file():
            raise ValueError(f"PDF path is not a file: {path}")

        file_bytes = path.read_bytes()
        file_size = len(file_bytes)
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        document_id = str(uuid.uuid4())

        try:
            with pdfplumber.open(path) as pdf:
                pages_data = []
                for page in pdf.pages:
                    pages_data.append(
                        {
                            "page_number": page.page_number,
                            "lines": _extract_text_lines(page),
                            "tables": _extract_tables(page),
                            "images": list(page.images),
                        }
                    )
                page_count = len(pdf.pages)
        except Exception as exc:
            raise ValueError(f"Failed to parse PDF file {path}: {exc}") from exc

        try:
            pypdf_reader = PdfReader(path)
        except Exception as exc:
            raise ValueError(f"Failed to open PDF file {path}: {exc}") from exc

        for page_data in pages_data:
            table_bboxes = [t["bbox"] for t in page_data["tables"]]
            page_data["lines"] = [
                line
                for line in page_data["lines"]
                if not any(_bbox_overlaps(line["bbox"], tb) for tb in table_bboxes)
            ]

        all_sizes = [
            line["size"] for page_data in pages_data for line in page_data["lines"]
        ]
        body_size = statistics.mode(all_sizes) if all_sizes else None

        for page_data in pages_data:
            table_bboxes = [t["bbox"] for t in page_data["tables"]]
            page_data["text_blocks"] = _merge_text_lines_into_blocks(
                page_data["lines"], body_size, table_bboxes
            )

        metadata = DocumentMetadata(
            filename=path.name,
            file_type="pdf",
            file_size=file_size,
            content_hash=content_hash,
            uploaded_by=uploaded_by,
            extra={"page_count": page_count},
        )

        elements: list[TextElement | TableElement | ImageElement] = []
        section_stack: list[tuple[int, str]] = []

        for page_data in pages_data:
            page_number = page_data["page_number"]

            items: list[dict[str, Any]] = []
            for line in page_data["text_blocks"]:
                items.append({"kind": "text", "top": line["bbox"][1], "line": line})
            for table in page_data["tables"]:
                items.append(
                    {"kind": "table", "top": table["bbox"][1], "table": table}
                )
            for pdfplumber_image in page_data["images"]:
                items.append(
                    {
                        "kind": "image_meta",
                        "top": pdfplumber_image.get("top", float("inf")),
                        "image": pdfplumber_image,
                    }
                )

            try:
                pypdf_images = list(pypdf_reader.pages[page_number - 1].images)
            except Exception as exc:
                raise ValueError(
                    f"Failed to extract images from page {page_number} of {path}: {exc}"
                ) from exc

            image_meta_by_name = {
                item["image"]["name"]: item
                for item in items
                if item["kind"] == "image_meta"
            }
            items = [item for item in items if item["kind"] != "image_meta"]
            for pypdf_image in pypdf_images:
                base_name = Path(pypdf_image.name).stem
                matched = image_meta_by_name.get(base_name)
                top = matched["image"]["top"] if matched else float("inf")
                items.append(
                    {"kind": "image", "top": top, "pypdf_image": pypdf_image, "matched": matched}
                )

            items.sort(key=lambda item: item["top"])

            for item in items:
                if item["kind"] == "text":
                    line = item["line"]
                    is_heading = (
                        body_size is not None
                        and line["size"] > body_size * _HEADING_SIZE_RATIO
                    )
                    heading_level = None
                    if is_heading:
                        heading_level = 1
                        while section_stack and section_stack[-1][0] >= heading_level:
                            section_stack.pop()
                        section_stack.append((heading_level, line["content"]))

                    section_path = (
                        [t for _, t in section_stack] if section_stack else None
                    )

                    source = SourceReference(
                        document_id=document_id,
                        source_type="pdf",
                        location=PDFLocation(
                            page_number=page_number, bbox=line["bbox"]
                        ),
                    )
                    elements.append(
                        TextElement(
                            element_id=str(uuid.uuid4()),
                            element_index=len(elements),
                            source=source,
                            section_path=section_path,
                            content=line["content"],
                            heading_level=heading_level,
                        )
                    )

                elif item["kind"] == "table":
                    table = item["table"]
                    section_path = (
                        [t for _, t in section_stack] if section_stack else None
                    )
                    source = SourceReference(
                        document_id=document_id,
                        source_type="pdf",
                        location=PDFLocation(
                            page_number=page_number, bbox=table["bbox"]
                        ),
                    )
                    elements.append(
                        TableElement(
                            element_id=str(uuid.uuid4()),
                            element_index=len(elements),
                            source=source,
                            section_path=section_path,
                            columns=table["columns"],
                            rows=table["rows"],
                        )
                    )

                elif item["kind"] == "image":
                    pypdf_image = item["pypdf_image"]
                    matched = item["matched"]
                    bbox = None
                    label = None
                    if matched:
                        img_meta = matched["image"]
                        bbox = (
                            img_meta["x0"],
                            img_meta["top"],
                            img_meta["x1"],
                            img_meta["bottom"],
                        )
                        label = _nearest_caption(bbox, page_data["lines"])

                    section_path = (
                        [t for _, t in section_stack] if section_stack else None
                    )

                    image_id = str(uuid.uuid4())
                    suffix = Path(pypdf_image.name).suffix or ".bin"
                    asset_subdir = self.asset_dir / document_id
                    asset_subdir.mkdir(parents=True, exist_ok=True)
                    asset_path = asset_subdir / f"{image_id}{suffix}"
                    asset_path.write_bytes(pypdf_image.data)

                    source = SourceReference(
                        document_id=document_id,
                        source_type="pdf",
                        location=PDFLocation(page_number=page_number, bbox=bbox),
                    )
                    elements.append(
                        ImageElement(
                            element_id=str(uuid.uuid4()),
                            element_index=len(elements),
                            source=source,
                            section_path=section_path,
                            image_id=image_id,
                            image_uri=str(asset_path),
                            label=label,
                            description=None,
                        )
                    )

        return NormalizedDocument(
            document_id=document_id,
            metadata=metadata,
            elements=elements,
        )
