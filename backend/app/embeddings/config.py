from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Identifies which embedding model produced (or must produce) a set
    of vectors, and how wide they are.

    This is deliberately separate from app.indexing.config.DatabaseConfig
    -- that config describes the pgvector column's storage width (which
    may be left unset/dimension-agnostic), while this one is the
    authoritative description of a specific embedding model. Every
    EmbeddingProvider carries exactly one EmbeddingConfig and validates
    every vector it returns against `dimension`, so a provider/config
    mismatch fails immediately instead of silently mixing incompatible
    vectors into the same index.
    """

    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
