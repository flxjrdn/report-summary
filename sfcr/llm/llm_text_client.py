from typing import Any, Dict, Optional  # ---------- LLM interface ----------


class LLMTextClient:
    """
    Interface: implement .generate_raw(field, section_text, pages) -> str
    """

    def generate_raw(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError
