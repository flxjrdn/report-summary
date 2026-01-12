from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sfcr.extract.schema import ResponseLLM
from sfcr.llm.llm_text_client import LLMResponseMeta, LLMTextClient


@dataclass
class MockTextClient(LLMTextClient):
    model: str = "mock"

    def generate_raw(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Mark as completed; never truncated.
        self.last_meta = LLMResponseMeta(
            status="completed",
            incomplete_reason=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )

        # Return a VALID ResponseLLM JSON string.
        # "not_found" is safest for end-to-end runs.
        obj = ResponseLLM(
            status="not_found",
            value_unscaled=None,
            scale=None,
            unit=None,
            source_text=None,
        )
        return json.dumps(obj.model_dump(exclude_none=False), ensure_ascii=False)
