"""
Dependency-light unit tests for GeminiLLMProvider.

Every test here uses a FakeGeminiClient -- none of them make a real
network call or require GEMINI_API_KEY.
"""

import os
import unittest

from app.rag.llm import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_DEFAULT_FALLBACK_LLM_MODEL_NAME,
    GEMINI_DEFAULT_LLM_MODEL_NAME,
    GEMINI_LLM_FALLBACK_MODEL_ENV_VAR,
    GEMINI_LLM_MODEL_ENV_VAR,
    GeminiLLMProvider,
    LLMConfig,
    LLMError,
    LLMProvider,
    LLMServiceUnavailableError,
    default_gemini_llm_config,
)


class FakeGenerateContentResponse:
    def __init__(self, text: str | None):
        self.text = text


class FakeModels:
    def __init__(
        self,
        response_text: str = "generated answer",
        raise_error: Exception | None = None,
        side_effects: list[Exception | str | None] | None = None,
    ):
        self.response_text = response_text
        self.raise_error = raise_error
        self.side_effects = list(side_effects or [])
        self.calls: list[tuple] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config.system_instruction, config.temperature, config.max_output_tokens))
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return FakeGenerateContentResponse(effect if effect is not None else self.response_text)
        if self.raise_error is not None:
            raise self.raise_error
        return FakeGenerateContentResponse(self.response_text)


class FakeGeminiClient:
    def __init__(
        self,
        response_text: str = "generated answer",
        raise_error: Exception | None = None,
        side_effects: list[Exception | str | None] | None = None,
    ):
        self.models = FakeModels(
            response_text=response_text,
            raise_error=raise_error,
            side_effects=side_effects,
        )


class FakeGeminiError(Exception):
    def __init__(self, status_code: int | None = None, status: str | None = None):
        super().__init__("provider detail that should not leak")
        self.status_code = status_code
        self.response_json = {"error": {"status": status}} if status else None


class TestLLMConfigAndDefaults(unittest.TestCase):
    def test_default_model_name(self):
        self.assertEqual(GEMINI_DEFAULT_LLM_MODEL_NAME, "gemini-3.7-flash")

    def test_default_config_uses_default_model(self):
        config = default_gemini_llm_config()
        self.assertEqual(config.model_name, GEMINI_DEFAULT_LLM_MODEL_NAME)
        self.assertEqual(config.fallback_model_name, GEMINI_DEFAULT_FALLBACK_LLM_MODEL_NAME)
        self.assertEqual(config.provider_name, "google")

    def test_env_var_overrides_default_model(self):
        original = os.environ.get(GEMINI_LLM_MODEL_ENV_VAR)
        try:
            os.environ[GEMINI_LLM_MODEL_ENV_VAR] = "gemini-3.5-flash"
            config = default_gemini_llm_config()
            self.assertEqual(config.model_name, "gemini-3.5-flash")
        finally:
            if original is None:
                os.environ.pop(GEMINI_LLM_MODEL_ENV_VAR, None)
            else:
                os.environ[GEMINI_LLM_MODEL_ENV_VAR] = original

    def test_env_var_overrides_fallback_model(self):
        original = os.environ.get(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR)
        try:
            os.environ[GEMINI_LLM_FALLBACK_MODEL_ENV_VAR] = "gemini-flash-latest"
            config = default_gemini_llm_config()
            self.assertEqual(config.fallback_model_name, "gemini-flash-latest")
        finally:
            if original is None:
                os.environ.pop(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR, None)
            else:
                os.environ[GEMINI_LLM_FALLBACK_MODEL_ENV_VAR] = original

    def test_blank_fallback_env_uses_default_fallback(self):
        original = os.environ.get(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR)
        try:
            os.environ[GEMINI_LLM_FALLBACK_MODEL_ENV_VAR] = ""
            config = default_gemini_llm_config()
            self.assertEqual(config.fallback_model_name, GEMINI_DEFAULT_FALLBACK_LLM_MODEL_NAME)
        finally:
            if original is None:
                os.environ.pop(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR, None)
            else:
                os.environ[GEMINI_LLM_FALLBACK_MODEL_ENV_VAR] = original

    def test_fallback_can_be_disabled_by_env_var(self):
        original = os.environ.get(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR)
        try:
            os.environ[GEMINI_LLM_FALLBACK_MODEL_ENV_VAR] = "none"
            config = default_gemini_llm_config()
            self.assertIsNone(config.fallback_model_name)
        finally:
            if original is None:
                os.environ.pop(GEMINI_LLM_FALLBACK_MODEL_ENV_VAR, None)
            else:
                os.environ[GEMINI_LLM_FALLBACK_MODEL_ENV_VAR] = original


class TestGeminiLLMProviderConstruction(unittest.TestCase):
    def test_constructs_with_fake_client_no_network(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient())
        self.assertIsInstance(provider, LLMProvider)

    def test_default_config_used_when_none_provided(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient())
        self.assertEqual(provider.config.model_name, GEMINI_DEFAULT_LLM_MODEL_NAME)

    def test_explicit_config_is_used(self):
        config = LLMConfig(provider_name="google", model_name="gemini-3.5-flash")
        provider = GeminiLLMProvider(config=config, client=FakeGeminiClient())
        self.assertEqual(provider.config.model_name, "gemini-3.5-flash")

    def test_default_temperature_and_max_tokens(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient())
        self.assertEqual(provider.temperature, 0.2)
        self.assertEqual(provider.max_output_tokens, 1024)

    def test_custom_temperature_and_max_tokens(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient(), temperature=0.7, max_output_tokens=256)
        self.assertEqual(provider.temperature, 0.7)
        self.assertEqual(provider.max_output_tokens, 256)


class TestGeminiApiKeyValidation(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)
        os.environ.pop(GEMINI_API_KEY_ENV_VAR, None)
        os.environ.pop("GOOGLE_API_KEY", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_missing_api_key_raises_clear_error_without_network(self):
        with self.assertRaises(RuntimeError) as ctx:
            GeminiLLMProvider()
        self.assertIn(GEMINI_API_KEY_ENV_VAR, str(ctx.exception))

    def test_client_injection_bypasses_api_key_requirement(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient())
        self.assertIsNotNone(provider)


class TestGeminiLLMProviderGenerate(unittest.TestCase):
    def setUp(self):
        self.fake_client = FakeGeminiClient(response_text="The revenue grew by 18%.")
        self.provider = GeminiLLMProvider(client=self.fake_client)

    def test_generate_returns_response_text(self):
        result = self.provider.generate("What was revenue growth?")
        self.assertEqual(result, "The revenue grew by 18%.")
        self.assertEqual(len(self.fake_client.models.calls), 1)

    def test_generate_passes_prompt_as_contents(self):
        self.provider.generate("What was revenue growth?")
        _, contents, _, _, _ = self.fake_client.models.calls[0]
        self.assertEqual(contents, "What was revenue growth?")

    def test_generate_passes_system_instruction_separately(self):
        self.provider.generate("question", system_instruction="Follow the rules.")
        _, _, system_instruction, _, _ = self.fake_client.models.calls[0]
        self.assertEqual(system_instruction, "Follow the rules.")

    def test_generate_without_system_instruction_passes_none(self):
        self.provider.generate("question")
        _, _, system_instruction, _, _ = self.fake_client.models.calls[0]
        self.assertIsNone(system_instruction)

    def test_generate_uses_configured_model(self):
        self.provider.generate("question")
        model, _, _, _, _ = self.fake_client.models.calls[0]
        self.assertEqual(model, GEMINI_DEFAULT_LLM_MODEL_NAME)

    def test_generate_passes_temperature_and_max_tokens(self):
        self.provider.generate("question")
        _, _, _, temperature, max_tokens = self.fake_client.models.calls[0]
        self.assertEqual(temperature, 0.2)
        self.assertEqual(max_tokens, 1024)


class TestGeminiLLMProviderErrorHandling(unittest.TestCase):
    def test_api_error_is_wrapped_and_does_not_leak_the_api_key(self):
        secret_key = "AIzaSyFAKESECRETVALUEFORTESTINGONLY123"
        underlying_error = Exception(f"401 Unauthorized: bad key {secret_key}")
        provider = GeminiLLMProvider(client=FakeGeminiClient(raise_error=underlying_error))

        with self.assertRaises(LLMError) as ctx:
            provider.generate("question")

        self.assertNotIn(secret_key, str(ctx.exception))

    def test_underlying_exception_is_chained(self):
        original = Exception("boom")
        provider = GeminiLLMProvider(client=FakeGeminiClient(raise_error=original))
        with self.assertRaises(LLMError) as ctx:
            provider.generate("question")
        self.assertIs(ctx.exception.__cause__, original)

    def test_empty_response_text_raises_clear_error(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient(response_text=None))
        with self.assertRaises(LLMError):
            provider.generate("question")

    def test_transient_503_succeeds_after_retry(self):
        sleeps: list[float] = []
        client = FakeGeminiClient(
            side_effects=[FakeGeminiError(status_code=503, status="UNAVAILABLE"), "Recovered answer."]
        )
        provider = GeminiLLMProvider(
            client=client,
            max_retries=2,
            retry_backoff_seconds=0.25,
            sleep=sleeps.append,
        )

        result = provider.generate("question")

        self.assertEqual(result, "Recovered answer.")
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(sleeps, [0.25])

    def test_transient_503_exhaustion_raises_unavailable(self):
        sleeps: list[float] = []
        client = FakeGeminiClient(
            side_effects=[
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
            ]
        )
        provider = GeminiLLMProvider(
            config=LLMConfig(
                provider_name="google",
                model_name="gemini-3.7-flash",
                fallback_model_name=None,
            ),
            client=client,
            max_retries=2,
            retry_backoff_seconds=0.5,
            sleep=sleeps.append,
        )

        with self.assertRaises(LLMServiceUnavailableError):
            provider.generate("question")

        self.assertEqual(len(client.models.calls), 3)
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_primary_503_exhaustion_attempts_fallback_model(self):
        sleeps: list[float] = []
        client = FakeGeminiClient(
            side_effects=[
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                "Fallback answer.",
            ]
        )
        provider = GeminiLLMProvider(
            config=LLMConfig(
                provider_name="google",
                model_name="gemini-3.7-flash",
                fallback_model_name="gemini-3.5-flash",
            ),
            client=client,
            max_retries=2,
            retry_backoff_seconds=0.25,
            sleep=sleeps.append,
        )

        result = provider.generate("question")

        self.assertEqual(result, "Fallback answer.")
        self.assertEqual([call[0] for call in client.models.calls], [
            "gemini-3.7-flash",
            "gemini-3.7-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
        ])
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_permanent_provider_failure_is_not_retried(self):
        sleeps: list[float] = []
        client = FakeGeminiClient(raise_error=FakeGeminiError(status_code=401))
        provider = GeminiLLMProvider(
            config=LLMConfig(
                provider_name="google",
                model_name="gemini-3.7-flash",
                fallback_model_name="gemini-3.5-flash",
            ),
            client=client,
            max_retries=2,
            retry_backoff_seconds=0.5,
            sleep=sleeps.append,
        )

        with self.assertRaises(LLMError):
            provider.generate("question")

        self.assertEqual(len(client.models.calls), 1)
        self.assertEqual(sleeps, [])

    def test_primary_and_fallback_503_exhaustion_raises_clean_unavailable(self):
        secret = "AIzaSyFAKESECRETVALUEFORTESTINGONLY123"
        sleeps: list[float] = []
        client = FakeGeminiClient(
            side_effects=[
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
                FakeGeminiError(status_code=503, status="UNAVAILABLE"),
            ]
        )
        provider = GeminiLLMProvider(
            config=LLMConfig(
                provider_name="google",
                model_name="gemini-3.7-flash",
                fallback_model_name="gemini-3.5-flash",
            ),
            client=client,
            max_retries=2,
            retry_backoff_seconds=0.5,
            sleep=sleeps.append,
        )

        with self.assertRaises(LLMServiceUnavailableError) as ctx:
            provider.generate(f"question with secret {secret}")

        self.assertEqual([call[0] for call in client.models.calls], [
            "gemini-3.7-flash",
            "gemini-3.7-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash",
        ])
        self.assertEqual(sleeps, [0.5, 1.0, 0.5, 1.0])
        self.assertNotIn(secret, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
