from dataclasses import dataclass
from typing import Any, Dict, Optional

from sfcr.llm.llm_text_client import LLMTextClient


@dataclass
class MockTextClient(LLMTextClient):
    def generate_raw(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        return "this is a mock text"
