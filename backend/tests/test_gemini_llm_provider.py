"""
Dependency-light unit tests for GeminiLLMProvider.

Every test here uses a FakeGeminiClient -- none of them make a real
network call or require GEMINI_API_KEY.
"""

import os
import unittest

from app.rag.llm import (
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_DEFAULT_LLM_MODEL_NAME,
    GEMINI_LLM_MODEL_ENV_VAR,
    GeminiLLMProvider,
    LLMConfig,
    LLMProvider,
    default_gemini_llm_config,
)


class FakeGenerateContentResponse:
    def __init__(self, text: str | None):
        self.text = text


class FakeModels:
    def __init__(self, response_text: str = "generated answer", raise_error: Exception | None = None):
        self.response_text = response_text
        self.raise_error = raise_error
        self.calls: list[tuple] = []

    def generate_content(self, *, model, contents, config):
        if self.raise_error is not None:
            raise self.raise_error
        self.calls.append((model, contents, config.system_instruction, config.temperature, config.max_output_tokens))
        return FakeGenerateContentResponse(self.response_text)


class FakeGeminiClient:
    def __init__(self, response_text: str = "generated answer", raise_error: Exception | None = None):
        self.models = FakeModels(response_text=response_text, raise_error=raise_error)


class TestLLMConfigAndDefaults(unittest.TestCase):
    def test_default_model_name(self):
        self.assertEqual(GEMINI_DEFAULT_LLM_MODEL_NAME, "gemini-3.7-flash")

    def test_default_config_uses_default_model(self):
        config = default_gemini_llm_config()
        self.assertEqual(config.model_name, GEMINI_DEFAULT_LLM_MODEL_NAME)
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

        with self.assertRaises(RuntimeError) as ctx:
            provider.generate("question")

        self.assertNotIn(secret_key, str(ctx.exception))

    def test_underlying_exception_is_chained(self):
        original = Exception("boom")
        provider = GeminiLLMProvider(client=FakeGeminiClient(raise_error=original))
        with self.assertRaises(RuntimeError) as ctx:
            provider.generate("question")
        self.assertIs(ctx.exception.__cause__, original)

    def test_empty_response_text_raises_clear_error(self):
        provider = GeminiLLMProvider(client=FakeGeminiClient(response_text=None))
        with self.assertRaises(RuntimeError):
            provider.generate("question")


if __name__ == "__main__":
    unittest.main()
