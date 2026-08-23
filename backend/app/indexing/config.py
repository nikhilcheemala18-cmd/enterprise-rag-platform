import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Loads backend/.env into the process environment (if the file exists;
# a no-op otherwise, e.g. in production where real env vars are injected
# by the platform). Does not override variables already set in the
# environment -- that's python-dotenv's default. Runs once, at import
# time of this module, which is always before any call to
# DatabaseConfig.from_env() below.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class DatabaseConfig(BaseModel):
    """Connection settings for the PostgreSQL/pgvector indexing backend.

    embedding_dimension is intentionally NOT part of the Chunk model --
    it belongs here, at the storage/index configuration layer, because
    the production embedding model (and therefore its vector width)
    hasn't been chosen yet. Leave it unset to keep the pgvector column
    dimension-agnostic until it is.
    """

    database_url: str = Field(min_length=1)
    embedding_dimension: int | None = Field(default=None, gt=0)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required but not set. "
                "Copy backend/.env.example to backend/.env and fill in a "
                "real PostgreSQL connection string."
            )

        raw_dimension = os.environ.get("EMBEDDING_DIMENSION")
        embedding_dimension = int(raw_dimension) if raw_dimension else None

        return cls(database_url=database_url, embedding_dimension=embedding_dimension)
