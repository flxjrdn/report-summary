from typing import Any

from sfcr.llm.llm_text_client import LLMTextClient
from sfcr.llm.mock_text_client import MockTextClient
from sfcr.llm.ollama_text_client import OllamaTextClient


def create_llm_text_client(provider: str, **kwargs: Any) -> LLMTextClient:
    if provider == "ollama":
        return OllamaTextClient(**kwargs)
    elif provider == "mock":
        return MockTextClient()
    raise ValueError(f"Unknown LLM provider: {provider}. Use 'ollama' or 'mock'.")
