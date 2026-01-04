from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sfcr.extract.schema import ResponseLLM


@dataclass
class MockTextClient:
    model: str = "mock"

    def generate_raw(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Return a VALID ResponseLLM JSON string.
        # "not_found" is safest for end-to-end runs.
        obj = ResponseLLM(
            status="not_found",
            value_unscaled=0.0,
            scale=None,
            unit=None,
            source_text=None,
        )
        return json.dumps(obj.model_dump(exclude_none=False), ensure_ascii=False)