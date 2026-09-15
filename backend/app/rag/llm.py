import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"  # same variable app/embeddings/gemini.py uses
GEMINI_LLM_MODEL_ENV_VAR = "GEMINI_LLM_MODEL"
GEMINI_LLM_FALLBACK_MODEL_ENV_VAR = "GEMINI_LLM_FALLBACK_MODEL"
GEMINI_DEFAULT_LLM_MODEL_NAME = "gemini-3.7-flash"
GEMINI_DEFAULT_FALLBACK_LLM_MODEL_NAME = "gemini-3.5-flash"
GEMINI_DEFAULT_TEMPERATURE = 0.2
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 1024
GEMINI_DEFAULT_MAX_RETRIES = 2
GEMINI_DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
GEMINI_TRANSIENT_STATUS_CODES = {503}
GEMINI_TRANSIENT_STATUSES = {"UNAVAILABLE"}


class LLMConfig(BaseModel):
    """Identifies which model a text-generation call should use.

    Deliberately separate from app.embeddings.config.EmbeddingConfig --
    embedding and generation are different responsibilities with
    different models, even when (as here) both happen to be served by
    Gemini. There is no `dimension` field here; it doesn't apply to text
    generation.
    """

    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    fallback_model_name: str | None = Field(default=None, min_length=1)


def default_gemini_llm_config() -> LLMConfig:
    model_name = os.environ.get(GEMINI_LLM_MODEL_ENV_VAR, GEMINI_DEFAULT_LLM_MODEL_NAME)
    fallback_model_name = _resolve_fallback_model_name(
        os.environ.get(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR)
    )
    return LLMConfig(
        provider_name="google",
        model_name=model_name,
        fallback_model_name=fallback_model_name,
    )


def _resolve_fallback_model_name(value: str | None) -> str | None:
    if value is None or not value.strip():
        return GEMINI_DEFAULT_FALLBACK_LLM_MODEL_NAME
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    return value.strip()


class LLMProvider(ABC):
    """Converts a prompt into generated text. Not coupled to any specific
    vendor -- concrete subclasses (initially GeminiLLMProvider) own that.
    RAGService depends only on this abstraction.

    `system_instruction` is a separate parameter, not concatenated into
    `prompt`, specifically so a provider can pass it through its own
    dedicated system-instruction channel where one exists (Gemini's
    GenerateContentConfig.system_instruction) rather than as ordinary
    user-turn text -- the API itself then treats it with different
    priority than the (untrusted) prompt content, reinforcing the
    "retrieved context is data, not instructions" rule at the transport
    level, not just in prompt wording.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        raise NotImplementedError


class LLMError(RuntimeError):
    """Base class for failures raised by an LLMProvider."""


class LLMServiceUnavailableError(LLMError):
    """The upstream LLM service is temporarily unavailable."""


class GeminiLLMProvider(LLMProvider):
    """Hosted text-generation provider: Gemini via the official
    `google-genai` SDK's `client.models.generate_content`. Requires
    network access and a Gemini API key -- every call is a real,
    billable (outside the free tier) API request.

    Model choice verified directly against current official docs
    (ai.google.dev/gemini-api/docs/text-generation and .../models) at
    implementation time, not assumed from memory: "gemini-3.7-flash" is
    listed there as the current "New Stable" general-purpose model.
    Configurable via GEMINI_LLM_MODEL if a different model is wanted
    later, without any code change.

    Deliberately uses `client.models.generate_content` (the same `client.
    models.*` family already used by GeminiEmbeddingProvider for
    embed_content), not the newer, much larger "Interactions" API
    surface (client.interactions, with agents/tools/environments/MCP
    servers) -- that API is built for multi-turn agentic workflows and
    would be a disproportionate dependency for a single-shot "prompt in,
    text out" RAG call.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        api_key: str | None = None,
        client: Any = None,
        temperature: float = GEMINI_DEFAULT_TEMPERATURE,
        max_output_tokens: int = GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
        max_retries: int = GEMINI_DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = GEMINI_DEFAULT_RETRY_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        config = config or default_gemini_llm_config()
        super().__init__(config)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep

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

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        request_config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        try:
            return self._generate_with_retries(
                model_name=self.config.model_name,
                prompt=prompt,
                request_config=request_config,
            )
        except LLMServiceUnavailableError as primary_exc:
            fallback_model_name = self.config.fallback_model_name
            if not fallback_model_name or fallback_model_name == self.config.model_name:
                raise

            try:
                return self._generate_with_retries(
                    model_name=fallback_model_name,
                    prompt=prompt,
                    request_config=request_config,
                )
            except LLMServiceUnavailableError as fallback_exc:
                raise LLMServiceUnavailableError(
                    "Gemini generation service temporarily unavailable for both "
                    f"primary model={self.config.model_name!r} and fallback "
                    f"model={fallback_model_name!r} ({type(fallback_exc.__cause__).__name__})"
                ) from fallback_exc
            except LLMError:
                raise
            except Exception as fallback_exc:
                raise LLMError(
                    f"Gemini fallback generation request failed for model="
                    f"{fallback_model_name!r} ({type(fallback_exc).__name__})"
                ) from fallback_exc

    def _generate_with_retries(
        self,
        *,
        model_name: str,
        prompt: str,
        request_config: types.GenerateContentConfig,
    ) -> str:
        last_transient_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=request_config,
                )
                break
            except Exception as exc:
                if not _is_transient_gemini_error(exc):
                    # Deliberately no str(exc) -- see GeminiEmbeddingProvider for
                    # the same convention. Original exception still chained.
                    raise LLMError(
                        f"Gemini generation request failed for model="
                        f"{model_name!r} ({type(exc).__name__})"
                    ) from exc

                last_transient_error = exc
                if attempt >= self.max_retries:
                    raise LLMServiceUnavailableError(
                        f"Gemini generation service temporarily unavailable for model="
                        f"{model_name!r} ({type(exc).__name__})"
                    ) from exc

                self._sleep(self.retry_backoff_seconds * (2**attempt))
        else:
            # Unreachable, but keeps static analysis honest if the loop
            # structure changes later.
            raise LLMServiceUnavailableError(
                f"Gemini generation service temporarily unavailable for model="
                f"{model_name!r} ({type(last_transient_error).__name__})"
            ) from last_transient_error

        text = response.text
        if not text:
            raise LLMError(
                f"Gemini returned an empty response for model={model_name!r}"
            )
        return text


def _is_transient_gemini_error(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ClientError):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code in GEMINI_TRANSIENT_STATUS_CODES:
        return True

    response_json = getattr(exc, "response_json", None)
    if isinstance(response_json, dict):
        error = response_json.get("error")
        if isinstance(error, dict) and error.get("status") in GEMINI_TRANSIENT_STATUSES:
            return True

    return isinstance(exc, genai_errors.ServerError) and status_code is None
