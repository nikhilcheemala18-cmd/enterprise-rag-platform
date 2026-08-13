from datetime import datetime
from typing import Any, Annotated, Literal

from pydantic import BaseModel, Field, model_validator




class PDFLocation(BaseModel):
    page_number: int
    bbox: tuple[float, float, float, float] | None = None


class DocxLocation(BaseModel):
    paragraph_index: int | None = None
    table_index: int | None = None
    section_index: int | None = None


class ExcelLocation(BaseModel):
    sheet_name: str
    row_start: int
    row_end: int


class CSVLocation(BaseModel):
    row_start: int
    row_end: int



class SourceReference(BaseModel):
    document_id: str
    source_type: Literal["pdf", "docx", "excel", "csv"]
    location: PDFLocation | DocxLocation | ExcelLocation | CSVLocation | None = None

    @model_validator(mode="after")
    def check_location_matches_source_type(self) -> "SourceReference":
        expected = {
            "pdf": PDFLocation,
            "docx": DocxLocation,
            "excel": ExcelLocation,
            "csv": CSVLocation,
        }[self.source_type]

        if self.location is not None and not isinstance(
            self.location, expected
        ):
            raise ValueError(
                f"source_type='{self.source_type}' requires "
                f"{expected.__name__}, "
                f"got {type(self.location).__name__}"
            )

        return self



class BaseElement(BaseModel):
    element_id: str
    element_index: int
    source: SourceReference
    section_path: list[str] | None = None


class TextElement(BaseElement):
    element_type: Literal["text"] = "text"
    content: str
    heading_level: int | None = None


class TableElement(BaseElement):
    element_type: Literal["table"] = "table"
    columns: list[str]
    rows: list[list[Any]]
    raw_header_rows: list[list[Any]] | None = None


class ImageElement(BaseElement):
    element_type: Literal["image"] = "image"
    image_id: str
    image_uri: str
    label: str | None = None
    description: str | None = None


NormalizedElement = Annotated[
    TextElement | TableElement | ImageElement,
    Field(discriminator="element_type"),
]



class DocumentMetadata(BaseModel):
    filename: str
    file_type: Literal["pdf", "docx", "excel", "csv"]
    file_size: int
    content_hash: str

    uploaded_by: str | None = None
    created_at: datetime | None = None

    extra: dict[str, Any] = Field(default_factory=dict)




class NormalizedDocument(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    elements: list[NormalizedElement] = Field(default_factory=list)