from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from sfcr.extract.extractor import FieldDef, LLMClient
from sfcr.extract.schema import Evidence, ExtractionLLM


# ---- small utils -------------------------------------------------------------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _window_text(field: FieldDef, section_text: str, token_budget: int) -> str:
    if _approx_token_count(section_text) <= token_budget:
        return section_text
    lower = section_text.lower()
    hits: List[Tuple[int, int]] = []
    for kw in (field.keywords or [])[:8]:
        i = lower.find(kw.lower())
        if i >= 0:
            hits.append((max(0, i - 600), min(len(section_text), i + 800)))
        if len(hits) >= 3:
            break
    if not hits:
        char_budget = token_budget * 4
        head = section_text[: int(char_budget * 0.85)]
        tail = section_text[-int(char_budget * 0.15) :]
        return head + ("\n...\n" + tail if tail else "")
    windows: List[Tuple[int, int]] = []
    for s, e in sorted(hits):
        if not windows or s > windows[-1][1] + 50:
            windows.append((s, e))
        else:
            windows[-1] = (windows[-1][0], max(windows[-1][1], e))
    pieces: list[str] = []
    char_budget = token_budget * 4
    for s, e in windows:
        if sum(len(p) for p in pieces) >= char_budget:
            break
        pieces.append(section_text[s:e])
    text = "\n...\n".join(pieces)
    if _approx_token_count(text) > token_budget:
        text = text[: token_budget * 4]
    return text


# -----------------------------------------------------------------------------

_JSON_SCHEMA_HINT = """
Return ONLY a single JSON object with the following keys:
- field_id (string)  -> must equal the requested field id
- status (string)    -> one of: "ok", "not_found", "ambiguous"
- value (number|null)          # numeric magnitude as printed (UNSCALED)
- unit ("EUR"|"%"|null)        # "EUR" for amounts; "%" for percentages
- scale (number|null)          # 1|1000|1e6|1e9 if TEUR/Mio/Mrd was indicated; else null/1
- currency ("EUR"|null)        # "EUR" if unit is EUR; else null
- period_end (YYYY-MM-DD|null)
- evidence (array of {page:int, ref?:string, snippet_hash?:string}) with at least one item
- source_text (string|null)    # short verbatim snippet (<=200 chars) around the number
- scale_source ("row"|"column"|"caption"|"nearby"|"model_guess"|null)
- notes (string|null)

Rules:
- Parse numbers in German locale (thousands '.' and decimal ',').
- Keep "value" UNscaled; put the magnitude in "scale".
- If you cannot find a unique value, set status "not_found" or "ambiguous" and leave numeric fields null.
- Output ONLY JSON, with no extra text.
"""


@dataclass
class OllamaLLM(LLMClient):
    """
    Local Ollama backend for the SFCR extractor.
    Works with models like 'mistral', 'llama3', 'qwen2', etc.
    """

    model: str = "mistral"
    temperature: float = 0.0
    host: str = "http://127.0.0.1:11434"
    input_token_limit: int = 1200
    output_max_tokens: int = 200  # Ollama: num_predict
    timeout_s: float = 30.0
    max_retries: int = 2
    backoff_base_s: float = 1.2
    backoff_jitter_s: float = 0.4
    cache_dir: Path = field(default_factory=lambda: Path(".cache/ollama"))
    enable_cache: bool = True

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self, field: FieldDef, section_text: str, page_start: int, page_end: int
    ) -> ExtractionLLM:
        bounded = _window_text(field, section_text, self.input_token_limit)
        prompt = self._build_prompt(field, bounded, page_start, page_end)

        cache_key = self._make_cache_key(field, prompt)
        if self.enable_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                try:
                    out = ExtractionLLM(**cached)
                    if not out.evidence:
                        out.evidence = [Evidence(page=page_start, ref=None)]
                    if out.field_id != field.id:
                        out.field_id = field.id
                    if out.unit == "EUR" and out.currency is None:
                        out.currency = "EUR"
                    return out
                except Exception:
                    pass

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                data = {
                    "model": self.model,
                    "format": "json",
                    "prompt": prompt,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.output_max_tokens,
                    },
                }
                r = requests.post(
                    f"{self.host}/api/generate", json=data, timeout=self.timeout_s
                )
                r.raise_for_status()
                resp = r.json()
                raw = resp.get("response", "")
                data = self._parse_json_strict(raw, field_id=field.id)
                out = ExtractionLLM(**data)
                if not out.evidence:
                    out.evidence = [Evidence(page=page_start, ref=None)]
                if out.field_id != field.id:
                    out.field_id = field.id
                if out.unit == "EUR" and out.currency is None:
                    out.currency = "EUR"

                if self.enable_cache:
                    self._write_cache(cache_key, out)
                return out

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = (self.backoff_base_s**attempt) + random.uniform(
                    0, self.backoff_jitter_s
                )
                time.sleep(min(8.0, sleep_s))

        return ExtractionLLM(
            field_id=field.id,
            status="ambiguous",
            value=None,
            unit=field.unit if field.unit in ("EUR", "%") else None,
            scale=None,
            currency="EUR" if field.unit == "EUR" else None,
            period_end=None,
            evidence=[Evidence(page=page_start, ref=None)],
            source_text=None,
            scale_source=None,
            notes=f"ollama_error:{type(last_err).__name__ if last_err else 'unknown'}",
        )

    # --- helpers --------------------------------------------------------------
    def _build_prompt(
        self, field: FieldDef, text: str, page_start: int, page_end: int
    ) -> str:
        kw = ", ".join((field.keywords or [])[:8])
        return f"""
You are an information extraction engine for Solvency II SFCR sections.

Task:
Extract exactly one value for the field below from the provided section snippet.
If the value is not uniquely and explicitly present, set status "not_found" or "ambiguous" and leave numeric fields null.

Field:
- field_id: {field.id}
- expected_unit: {field.unit}
- page_range: {page_start}-{page_end}
- helpful_keywords: {kw or "-"}

{_JSON_SCHEMA_HINT}

Section text (German, pages {page_start}-{page_end}):
---
{text}
---
""".strip()

    def _parse_json_strict(self, raw: str, *, field_id: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return self._ensure_defaults(data, field_id)
        except Exception:
            pass
        fenced = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.DOTALL)
        m = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    return self._ensure_defaults(data, field_id)
            except Exception:
                pass
        return {
            "field_id": field_id,
            "status": "ambiguous",
            "value": None,
            "unit": None,
            "scale": None,
            "currency": None,
            "period_end": None,
            "evidence": [],
            "source_text": None,
            "scale_source": None,
            "notes": "json_parse_failed",
        }

    def _ensure_defaults(self, data: Dict[str, Any], field_id: str) -> Dict[str, Any]:
        d = dict(data)
        d.setdefault("field_id", field_id)
        d.setdefault("status", "ambiguous")
        d.setdefault("value", None)
        d.setdefault("unit", None)
        d.setdefault("scale", None)
        d.setdefault("currency", None)
        d.setdefault("period_end", None)
        d.setdefault("evidence", [])
        d.setdefault("source_text", None)
        d.setdefault("scale_source", None)
        d.setdefault("notes", None)
        return d

    def _make_cache_key(self, field: FieldDef, prompt: str) -> str:
        key_src = json.dumps(
            {
                "model": self.model,
                "temp": self.temperature,
                "field": field.id,
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return _sha1(key_src)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, key: str, out: ExtractionLLM) -> None:
        p = self._cache_path(key)
        try:
            p.write_text(
                json.dumps(out.model_dump(exclude_none=True), ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
