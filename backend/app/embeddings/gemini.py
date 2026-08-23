import os
from typing import Any

from google import genai
from google.genai import types

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig

GEMINI_MODEL_NAME = "gemini-embedding-2"
GEMINI_DEFAULT_DIMENSION = 1536
GEMINI_MIN_DIMENSION = 128
GEMINI_MAX_DIMENSION = 3072
GEMINI_MAX_INPUT_TOKENS = 8192
GEMINI_DOCUMENT_NO_TITLE = "none"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"


def default_gemini_config(dimension: int = GEMINI_DEFAULT_DIMENSION) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider_name="google", model_name=GEMINI_MODEL_NAME, dimension=dimension
    )


def _format_query_input(text: str) -> str:
    return f"task: search result | query: {text}"


def _format_document_input(text: str, title: str = GEMINI_DOCUMENT_NO_TITLE) -> str:
    return f"title: {title} | text: {text}"


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Hosted semantic embedding provider: Google's gemini-embedding-2 via
    the official `google-genai` SDK. Requires network access and a
    Gemini API key -- unlike BGEEmbeddingProvider, this is NOT local/
    offline, and every call is a real, billable (outside the free tier)
    API request. Never called from ordinary unit tests; see
    tests/test_gemini_embedding.py (fake client, no network) vs.
    tests/test_gemini_embedding_integration.py (real API, environment-
    gated on GEMINI_API_KEY being set).

    Facts below were verified against ai.google.dev/gemini-api/docs and
    cross-checked directly against the installed google-genai SDK source
    (EmbedContentConfig's field docstrings, not just the web docs) before
    implementing -- not assumed from memory:

    - Model: "gemini-embedding-2" is the current multimodal embedding
      model (as opposed to the older, text-only "gemini-embedding-001",
      which is a different, still-supported model with different
      behavior -- e.g. it *does* support passing a list to `contents`
      for one-embedding-per-item batching; see BATCHING below for why
      gemini-embedding-2 is handled differently).
    - Input limit: 8,192 tokens.
    - Output dimension: configurable via `output_dimensionality`,
      supported range 128-3072 (this class validates EmbeddingConfig.
      dimension falls in that range at construction time, before any API
      call). 1536 is this project's chosen default, per the task's
      instruction, not a hard requirement of the API itself.
    - Normalization: gemini-embedding-2 documents automatic
      renormalization of its output, including for truncated
      (non-default) dimensions such as 768/1536. No manual
      normalize()/unit-length step is performed by this provider --
      doing so would be redundant with what the API already guarantees.
    - Query vs. document: gemini-embedding-2 does NOT support the
      task_type parameter that gemini-embedding-001 uses. The docs are
      explicit: "You cannot use the task_type field for the
      gemini-embedding-2 model. Instead, include the task as an
      instruction in your prompt." So EmbedContentConfig.task_type is
      never set here (setting it for this model is documented as
      unsupported, not merely optional). Instead, the shared `is_query`
      flag from EmbeddingProvider selects a text-prefix format, per the
      documented convention:
        query:    "task: search result | query: {text}"
        document: "title: {title} | text: {text}"
      Chunk carries no title field, so documents always use the
      documented "no title" placeholder, GEMINI_DOCUMENT_NO_TITLE
      ("none") -- never a fabricated title.

    BATCHING: gemini-embedding-2 is documented to produce a single
    aggregated embedding when `contents` is given a list of multiple
    texts, rather than one embedding per item (this is a real behavior
    difference from gemini-embedding-001). Passing a list would silently
    violate this provider's "one vector per input text, in order"
    contract, so _embed_texts() issues one embed_content call per text
    instead of a single multi-item call. This is the safe, verifiably
    correct choice; it is not maximally throughput-optimal, and a
    dedicated multi-item batching path could be added later if/when this
    aggregation behavior is reconfirmed against a live account (see
    module docstring / completion report for this caveat).
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        api_key: str | None = None,
        client: Any = None,
    ):
        config = config or default_gemini_config()

        if config.model_name != GEMINI_MODEL_NAME:
            raise ValueError(
                f"GeminiEmbeddingProvider only supports model_name={GEMINI_MODEL_NAME!r}, "
                f"got {config.model_name!r}"
            )
        if not (GEMINI_MIN_DIMENSION <= config.dimension <= GEMINI_MAX_DIMENSION):
            raise ValueError(
                f"{GEMINI_MODEL_NAME} supports output dimensions between "
                f"{GEMINI_MIN_DIMENSION} and {GEMINI_MAX_DIMENSION}, "
                f"got {config.dimension}"
            )

        super().__init__(config)

        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or os.environ.get(GEMINI_API_KEY_ENV_VAR)
            if not resolved_key:
                raise RuntimeError(
                    f"{GEMINI_API_KEY_ENV_VAR} environment variable is required but "
                    "not set. Copy backend/.env.example to backend/.env and fill in "
                    "a real Gemini API key "
                    "(see https://ai.google.dev/gemini-api/docs/api-key)."
                )
            self._client = genai.Client(api_key=resolved_key)

    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        if not texts:
            return []

        # gemini-embedding-2 does not accept task_type -- only
        # output_dimensionality is set here. The query/document
        # distinction is carried by the formatted input text instead.
        request_config = types.EmbedContentConfig(
            output_dimensionality=self.config.dimension,
        )

        vectors: list[list[float]] = []
        for text in texts:
            formatted_input = (
                _format_query_input(text) if is_query else _format_document_input(text)
            )
            try:
                result = self._client.models.embed_content(
                    model=self.config.model_name,
                    contents=formatted_input,
                    config=request_config,
                )
            except Exception as exc:
                # Deliberately do not include str(exc) -- some SDK/HTTP
                # error messages can echo request details. Only the
                # exception type name is surfaced; the original exception
                # is still chained via `from exc` for anyone with
                # legitimate access to full tracebacks/logs.
                raise RuntimeError(
                    f"Gemini embedding request failed for model="
                    f"{self.config.model_name!r} ({type(exc).__name__})"
                ) from exc

            vectors.append([float(v) for v in result.embeddings[0].values])

        return vectors
