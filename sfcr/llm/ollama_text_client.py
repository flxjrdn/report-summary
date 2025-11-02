from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from sfcr.llm.llm_text_client import LLMTextClient


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class OllamaTextClient(LLMTextClient):
    model: str = "mistral"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    num_ctx: int = 8192
    num_predict: int = 400
    timeout_s: float = 30.0
    max_retries: int = 1
    backoff_base_s: float = 1.2
    backoff_jitter_s: float = 0.4
    enable_cache: bool = False
    cache_dir: Path = field(default_factory=lambda: Path(".cache/ollama_text"))
    debug_dir: Path = field(default_factory=lambda: Path(".cache/ollama_text_debug"))
    debug_raw: bool = False  # set True to persist raw responses

    def __post_init__(self):
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_cache_key(
        self,
        prompt: str,
        fmt: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
    ) -> str:
        key_src = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "format": fmt or None,
                "options": options or None,
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return _sha1(key_src)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.txt"

    def generate_raw(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,  # passed to Ollama as "format"
        options: Optional[Dict[str, Any]] = None,  # merged into "options"
    ) -> str:
        """
        Return Ollama's .json()['response'] (a STRING). No parsing here.
        Raise for HTTP errors; retry with simple backoff on exceptions.
        """
        opts = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if options:
            opts.update(options)

        if self.enable_cache:
            key = self._make_cache_key(prompt, json_schema, opts)
            p = self._cache_path(key)
            if p.exists():
                return p.read_text(encoding="utf-8")

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": opts,
                }
                if json_schema is not None:
                    payload["format"] = json_schema
                r = requests.post(
                    f"{self.host}/api/generate", json=payload, timeout=self.timeout_s
                )
                r.raise_for_status()
                data = r.json()
                resp = data.get("response", "")
                if self.enable_cache:
                    key = self._make_cache_key(prompt, json_schema, opts)
                    self._cache_path(key).write_text(resp, encoding="utf-8")
                if self.debug_raw:
                    self._save_debug(prompt, resp, None)
                return resp
            except Exception as e:
                last_err = e
                if self.debug_raw:
                    self._save_debug(prompt, None, e)
                if attempt >= self.max_retries:
                    raise
                sleep_s = (self.backoff_base_s**attempt) + random.uniform(
                    0, self.backoff_jitter_s
                )
                time.sleep(min(8.0, sleep_s))

        # Should not reach here; re-raise last error defensively
        if last_err:
            raise last_err
        raise RuntimeError("Unknown failure in OllamaTextClient.generate_raw")

    def _save_debug(self, prompt: str, raw: Optional[str], exc: Optional[Exception]):
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            rec = {
                "prompt": prompt[-4000:],
                "raw": raw,
                "error": repr(exc) if exc else None,
            }
            (self.debug_dir / f"ollama-{ts}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
