from typing import Any

from sfcr.extract.extractor import LLMClient  # interface
from sfcr.extract.extractor import MockLLM  # your existing rule-based mock
from sfcr.extract.ollama_client import OllamaLLM


def create_llm_client(provider: str, **kwargs: Any) -> LLMClient:
    if provider == "ollama":
        return OllamaLLM(**kwargs)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM provider: {provider}. Use 'ollama' or 'mock'.")
