import csv
import hashlib
import uuid
from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.models.document import (
    CSVLocation,
    DocumentMetadata,
    NormalizedDocument,
    SourceReference,
    TableElement,
)


def _is_blank_row(row: list[str]) -> bool:
    return len(row) == 0 or all(cell.strip() == "" for cell in row)


class CSVLoader(DocumentLoader):
    """Loads a CSV file into a NormalizedDocument with a single TableElement.

    Row numbering convention: rows are 1-indexed against the physical rows
    of the source file, including the header. The header is row 1; data
    rows start at row 2. A row made up entirely of empty/whitespace cells
    is treated as blank and excluded from both the header search and the
    table body, but it still consumes a row number since it is a real
    physical row in the file.
    """

    def load(
        self,
        file_path: str | Path,
        uploaded_by: str | None = None,
    ) -> NormalizedDocument:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if not path.is_file():
            raise ValueError(f"CSV path is not a file: {path}")

        file_bytes = path.read_bytes()
        file_size = len(file_bytes)
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        document_id = str(uuid.uuid4())

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                all_rows = list(csv.reader(f))
        except csv.Error as exc:
            raise ValueError(f"Failed to parse CSV file {path}: {exc}") from exc

        metadata = DocumentMetadata(
            filename=path.name,
            file_type="csv",
            file_size=file_size,
            content_hash=content_hash,
            uploaded_by=uploaded_by,
        )

        elements: list[TableElement] = []

        numbered_rows = list(enumerate(all_rows, start=1))
        non_blank_rows = [(n, r) for n, r in numbered_rows if not _is_blank_row(r)]

        if non_blank_rows:
            _header_row_number, header = non_blank_rows[0]
            data_rows = non_blank_rows[1:]

            if data_rows:
                row_start = data_rows[0][0]
                row_end = data_rows[-1][0]
                rows: list[list[str]] = [row for _, row in data_rows]

                source = SourceReference(
                    document_id=document_id,
                    source_type="csv",
                    location=CSVLocation(row_start=row_start, row_end=row_end),
                )

                elements.append(
                    TableElement(
                        element_id=str(uuid.uuid4()),
                        element_index=0,
                        source=source,
                        columns=header,
                        rows=rows,
                    )
                )

        return NormalizedDocument(
            document_id=document_id,
            metadata=metadata,
            elements=elements,
        )
