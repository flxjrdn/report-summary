from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from sfcr.llm.llm_text_client import LLMTextClient

# Note: do NOT import openai at module import time if you want tests to run without it.
# We import lazily in _client().


def _stable_json(obj: Any) -> str:
    """Deterministic JSON for hashing/caching."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def openai_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a JSON Schema to satisfy OpenAI structured outputs constraints:
    - For every object schema: additionalProperties=false
    - required must exist and include every property key
    """
    schema = json.loads(json.dumps(schema))  # deep copy

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            # Recurse first
            for k, v in list(node.items()):
                node[k] = _walk(v)

            # Enforce on objects
            if node.get("type") == "object" and isinstance(
                node.get("properties"), dict
            ):
                props = node["properties"]
                node["additionalProperties"] = False
                node["required"] = list(props.keys())

            return node

        if isinstance(node, list):
            return [_walk(x) for x in node]

        return node

    return _walk(schema)


def _response_to_text(resp: Any) -> str:
    # 1) Fast path (SDK provides this on many versions)
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2) Fallback: walk resp.output items
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks: list[str] = []
        for item in out:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                # Many SDK versions use .type == "output_text" and .text
                ctype = getattr(c, "type", None)
                if ctype in ("output_text", "text"):
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t:
                        chunks.append(t)
        if chunks:
            return "".join(chunks).strip()

    return ""


@dataclass
class OpenAITextClient(LLMTextClient):
    """
    Minimal text-generation client using the OpenAI Responses API.

      - generate_raw(prompt, ...) -> str  (raw model output)
    """

    model: str = "gpt-5-nano"
    temperature: float = 0.0
    max_output_tokens = 1200
    timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    enable_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache") / "sfcr" / "openai")
    debug_raw: bool = False
    debug_dir: Path = field(default_factory=lambda: Path("artifacts") / "openai-debug")

    def _client(self):
        """
        Lazily construct OpenAI client.
        Requires `pip install openai` and env var OPENAI_API_KEY if api_key not provided.
        """
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install openai"
            ) from e

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing OpenAI API key. Set environment variable OPENAI_API_KEY"
            )

        kwargs: Dict[str, Any] = {"api_key": key, "timeout": self.timeout_s}
        return OpenAI(**kwargs)

    def _cache_path(self, payload: Dict[str, Any]) -> Path:
        h = _sha256_text(_stable_json(payload))
        return self.cache_dir / self.model / f"{h}.txt"

    def generate_raw(
        self,
        prompt: str,
        *,
        strict_schema: bool = True,
        json_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a raw text response.

        Parameters
        ----------
        prompt:
          - str: user message
          - or list of {"role": "...", "content": "..."} items (system/user/assistant)
            (matches your pipeline style if you already build message lists)
        json_schema:
          if provided, enables Structured Outputs using Responses `text.format` with json_schema.
          This is the recommended way to enforce schema adherence.  [oai_citation:1‡OpenAI Plattform](https://platform.openai.com/docs/guides/structured-outputs)
        """
        # Normalize input into Responses-style `input`
        input_items = [{"role": "user", "content": prompt}]

        # Build Responses payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": "minimal"},
        }

        # GPT-5 models do not support temperature
        is_gpt5 = self.model.startswith("gpt-5")
        if not is_gpt5:
            payload["temperature"] = self.temperature

        if json_schema is not None:
            schema_strict = openai_strict_json_schema(json_schema)
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "json_schema",
                    "strict": bool(strict_schema),
                    "schema": schema_strict,
                }
            }
        else:
            payload["text"] = {"format": {"type": "text", "name": "text"}}

        # Cache
        if self.enable_cache:
            path = self._cache_path(payload)
            if path.exists():
                return path.read_text(encoding="utf-8")

        # Call API with basic retry
        client = self._client()
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.responses.create(**payload)
                text = _response_to_text(resp)

                if not isinstance(text, str):
                    raise RuntimeError(f"Unexpected response text type: {type(text)}")

                text = text.strip()

                if self.enable_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")

                if self.debug_raw:
                    self.debug_dir.mkdir(parents=True, exist_ok=True)
                    dbg = {
                        "payload": payload,
                        "raw_text": text,
                        "model": self.model,
                    }
                    dbg_path = (
                        self.debug_dir
                        / f"{int(time.time())}_{_sha256_text(_stable_json(payload))[:12]}.json"
                    )
                    dbg_path.write_text(
                        json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

                return text

            except Exception as e:  # pragma: no cover (network)
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (attempt + 1))

        raise RuntimeError(
            f"OpenAI request failed after retries: {last_err}"
        ) from last_err


if __name__ == "__main__":
    textclient = OpenAITextClient(model="gpt-5-nano")
    out = textclient.generate_raw("Write a one-sentence bedtime story about a unicorn.")
    print(out)
