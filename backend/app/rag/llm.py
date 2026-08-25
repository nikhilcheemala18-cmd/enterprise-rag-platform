import os
from abc import ABC, abstractmethod
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"  # same variable app/embeddings/gemini.py uses
GEMINI_LLM_MODEL_ENV_VAR = "GEMINI_LLM_MODEL"
GEMINI_DEFAULT_LLM_MODEL_NAME = "gemini-3.7-flash"
GEMINI_DEFAULT_TEMPERATURE = 0.2
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 1024


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


def default_gemini_llm_config() -> LLMConfig:
    model_name = os.environ.get(GEMINI_LLM_MODEL_ENV_VAR, GEMINI_DEFAULT_LLM_MODEL_NAME)
    return LLMConfig(provider_name="google", model_name=model_name)


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
    ):
        config = config or default_gemini_llm_config()
        super().__init__(config)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

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
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        try:
            response = self._client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=request_config,
            )
        except Exception as exc:
            # Deliberately no str(exc) -- see GeminiEmbeddingProvider for
            # the same convention. Original exception still chained.
            raise RuntimeError(
                f"Gemini generation request failed for model="
                f"{self.config.model_name!r} ({type(exc).__name__})"
            ) from exc

        text = response.text
        if not text:
            raise RuntimeError(
                f"Gemini returned an empty response for model={self.config.model_name!r}"
            )
        return text
