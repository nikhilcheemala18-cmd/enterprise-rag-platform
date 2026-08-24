import os

from app.embeddings.base import EmbeddingProvider
from app.embeddings.bge import BGEEmbeddingProvider
from app.embeddings.config import EmbeddingConfig
from app.embeddings.gemini import GeminiEmbeddingProvider, default_gemini_config
from app.embeddings.local import DeterministicTestEmbeddingProvider

DEFAULT_PROVIDER_NAME = "bge"
EMBEDDING_PROVIDER_ENV_VAR = "EMBEDDING_PROVIDER"


def build_embedding_provider() -> EmbeddingProvider:
    """Selects a concrete EmbeddingProvider based on the EMBEDDING_PROVIDER
    environment variable ("bge" | "gemini" | "deterministic").

    Defaults to "bge" -- a real local semantic model that needs no API
    key -- rather than silently defaulting to Gemini, per the requirement
    that Gemini must not become the sole hardcoded provider. Each branch
    only constructs an already-existing provider class with its own
    already-existing configuration/validation; no new embedding logic
    lives here.

    EMBEDDING_DIMENSION (optional, same variable DatabaseConfig already
    reads) is honored for "gemini" and "deterministic", whose dimension
    is actually configurable. It's not threaded into "bge", which only
    ever produces 768-dimensional vectors -- passing a mismatched
    dimension there would just be rejected by BGEEmbeddingProvider's own
    validation for no benefit.
    """
    provider_name = os.environ.get(EMBEDDING_PROVIDER_ENV_VAR, DEFAULT_PROVIDER_NAME).strip().lower()

    if provider_name == "bge":
        return BGEEmbeddingProvider()

    if provider_name == "gemini":
        dimension = _optional_dimension()
        config = default_gemini_config(dimension) if dimension else default_gemini_config()
        return GeminiEmbeddingProvider(config=config)

    if provider_name == "deterministic":
        config = EmbeddingConfig(
            provider_name="deterministic-test",
            model_name="deterministic-test-v1",
            dimension=_optional_dimension() or 768,
        )
        return DeterministicTestEmbeddingProvider(config)

    raise RuntimeError(
        f"Unknown {EMBEDDING_PROVIDER_ENV_VAR}={provider_name!r}; "
        "expected one of: bge, gemini, deterministic"
    )


def _optional_dimension() -> int | None:
    raw = os.environ.get("EMBEDDING_DIMENSION")
    return int(raw) if raw else None
