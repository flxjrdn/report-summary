from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LLMResponseMeta:
    """
    Optional metadata about the most recent generation call.

    Clients that cannot provide this can leave it as default/None.
    """

    status: Optional[str] = None  # e.g. "completed" | "incomplete"
    incomplete_reason: Optional[str] = None  # e.g. "max_output_tokens"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMTextClient:
    """
    Interface: implement generate_raw(prompt, ...) -> str
    """

    # Last-call metadata (best-effort)
    last_meta: LLMResponseMeta = LLMResponseMeta()

    def generate_raw(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    def was_truncated(self) -> bool:
        """
        Best-effort indicator that the last call likely truncated output
        due to token limits. Safe default: False.
        """
        return self.last_meta.status == "incomplete" and (
            self.last_meta.incomplete_reason or ""
        ) in {"max_output_tokens", "max_context_length"}
