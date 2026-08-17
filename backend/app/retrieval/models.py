from typing import Any

from pydantic import BaseModel, Field, model_validator


class RetrievalRequest(BaseModel):
    """A single retrieval request, independent of any storage/index
    implementation. `query` and/or `query_embedding` may be supplied --
    which one(s) are present determines whether HybridSearchService runs
    lexical-only, vector-only, or true hybrid fusion.

    document_id is an optional scoping filter at the interface level.
    There is no tenant_id here -- multi-tenant isolation has not been
    defined anywhere in this codebase yet and is not introduced by this
    task; retrieval must not be treated as tenant-safe until that exists.
    """

    query: str | None = Field(default=None, min_length=1)
    query_embedding: list[float] | None = Field(default=None, min_length=1)
    top_k: int = Field(default=10, gt=0)
    document_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_at_least_one_query_form(self) -> "RetrievalRequest":
        if self.query is None and self.query_embedding is None:
            raise ValueError(
                "RetrievalRequest requires at least one of query or query_embedding"
            )
        return self


class RetrievalResult(BaseModel):
    """A single ranked result, independent of any storage/index
    implementation. `score` means whatever the producing stage means by
    it: a lexical ts_rank, a vector similarity, or -- after
    reciprocal_rank_fusion -- the fused RRF score.
    """

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
