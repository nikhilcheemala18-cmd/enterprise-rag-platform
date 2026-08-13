from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_element_ids: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
