from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    text_chunk_size: int = Field(default=500, gt=0)
    text_chunk_overlap: int = Field(default=75, ge=0)
    table_max_rows: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def check_overlap_less_than_chunk_size(self) -> "ChunkingConfig":
        if self.text_chunk_overlap >= self.text_chunk_size:
            raise ValueError(
                "text_chunk_overlap must be less than text_chunk_size "
                f"(got overlap={self.text_chunk_overlap}, size={self.text_chunk_size})"
            )
        return self
