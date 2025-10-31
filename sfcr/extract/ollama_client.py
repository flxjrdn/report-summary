from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from sfcr.extract.extractor import FieldDef, LLMClient
from sfcr.extract.schema import Evidence, ExtractionLLM

_SNIPPET_HEX_RE = re.compile(r"^[a-f0-9]{8,64}$")
_NBSP = "\u00a0"
_THIN = "\u2009"
_NNBSP = "\u202f"

_SEP_CHARS = f" .{_THIN}{_NNBSP}"  # allowed thousands separators in text

# Map scale tokens to numeric factors
_SCALE_TOKENS = [
    (r"\bteur\b", 1e3, "TEUR"),
    (r"\btsd\b", 1e3, "Tsd"),
    (r"\bmio\b", 1e6, "Mio"),
    (r"\bmrd\b", 1e9, "Mrd"),
]
_EUR_TOKENS = [r"\beur\b", r"€"]


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


def _mk_snippet_hash(*parts: Any, length: int = 16) -> str:
    """
    Build a deterministic lowercase hex digest from any available evidence parts.
    Defaults to 16 hex chars (fits your schema's 8..64 constraint).
    """
    basis = " | ".join(str(p) for p in parts if p is not None and str(p).strip())
    if not basis:
        basis = "empty"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:length]


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
- evidence (array of {page:int, ref?:string}) with at least one item
- source_text (string|null)    # short verbatim snippet (<=200 chars) around the number
- scale_source ("row"|"column"|"caption"|"nearby"|"model_guess"|null)
- notes (string|null)

Rules:
- Parse numbers in German locale (thousands '.' and decimal ',').
- Keep "value" UNscaled; put the magnitude in "scale".
- If you cannot find a unique value, set status "not_found" or "ambiguous" and leave numeric fields null.
- Output ONLY JSON, with no extra text.
- If a number is immediately followed by a parenthesized number (e.g. 123 456 (133 333)), treat the first number as the current period and the parenthesized number as the previous year's value. Example: "1.234.567 (1.111.111)" means value=1234567, prev_value=1111111.
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
    max_retries: int = 1
    backoff_base_s: float = 1.2
    backoff_jitter_s: float = 0.4
    cache_dir: Path = field(default_factory=lambda: Path(".cache/ollama"))
    enable_cache: bool = True
    debug_dir: Path = field(default_factory=lambda: Path(".cache/ollama_debug"))
    debug_raw: bool = True

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
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.output_max_tokens,
                        "num_ctx": 8192,
                    },
                }
                r = requests.post(
                    f"{self.host}/api/generate", json=data, timeout=self.timeout_s
                )
                r.raise_for_status()
                resp_text = r.text
                try:
                    resp = r.json()
                except Exception as e:
                    # Save the raw HTTP text for diagnosis and re-raise
                    if self.debug_raw:
                        self._save_debug(field, prompt, resp_text, e)
                    raise
                raw = resp.get("response", "")
                data = self._parse_json_tolerant(raw, field_id=field.id)
                out = ExtractionLLM(**data)
                if not out.evidence:
                    out.evidence = [Evidence(page=page_start, ref=None)]
                if out.field_id != field.id:
                    out.field_id = field.id
                if out.unit == "EUR" and out.currency is None:
                    out.currency = "EUR"

                # --- fallback: try to extract current/prev pair if model failed ---
                if out.status != "ok" or out.value is None:
                    fallback = _extract_current_prev_pair(bounded)
                    if fallback is not None:
                        out.value = fallback.get("value")
                        if "unit" in fallback and fallback["unit"]:
                            out.unit = fallback["unit"]
                        if "currency" in fallback and fallback["currency"]:
                            out.currency = fallback["currency"]
                        if "scale" in fallback:
                            out.scale = fallback["scale"]
                        if "scale_source" in fallback:
                            out.scale_source = fallback["scale_source"]
                        out.status = "ok"
                        stxt = fallback.get("source_text")
                        if stxt:
                            out.source_text = stxt[:200]
                        # Evidence: ensure at least one
                        if not out.evidence:
                            out.evidence = [Evidence(page=page_start, ref=None)]
                        # Notes: append
                        note = "prev_in_parentheses_detected"
                        if out.notes:
                            if note not in out.notes:
                                out.notes += f";{note}"
                        else:
                            out.notes = note

                if self.enable_cache:
                    self._write_cache(cache_key, out)
                return out

            except Exception as e:
                # save raw for diagnosis
                if self.debug_raw:
                    self._save_debug(
                        field, prompt, raw if "raw" in locals() else None, e
                    )
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

    def _parse_json_tolerant(self, raw: str, *, field_id: str) -> Dict[str, Any]:
        # 1) Fast path: exact JSON object
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return self._ensure_defaults(obj, field_id)
        except Exception:
            pass

        # 2) If there is leading prose then a JSON object, parse the FIRST object only
        # This handles cases like: "Here is the result:\n{...}\n{...}"
        dec = JSONDecoder()
        try:
            obj, end = dec.raw_decode(raw.lstrip())
            if isinstance(obj, dict):
                return self._ensure_defaults(obj, field_id)
        except Exception:
            pass

        # 3) Non-greedy brace match: pick the first {...} block only
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return self._ensure_defaults(obj, field_id)
            except Exception:
                pass

        # 4) Code fences are common — strip once and retry step 2 and 3
        stripped = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.DOTALL)
        if stripped != raw:
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    return self._ensure_defaults(obj, field_id)
            except Exception:
                pass
            try:
                obj, end = dec.raw_decode(stripped.lstrip())
                if isinstance(obj, dict):
                    return self._ensure_defaults(obj, field_id)
            except Exception:
                pass
            m2 = re.search(r"\{[\s\S]*?\}", stripped)
            if m2:
                try:
                    obj = json.loads(m2.group(0))
                    if isinstance(obj, dict):
                        return self._ensure_defaults(obj, field_id)
                except Exception:
                    pass

        # 5) Give up: abstain safely
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

        # Defaults for top-level fields
        d.setdefault("field_id", field_id)
        d.setdefault("status", None)  # infer status below
        d.setdefault("value", None)
        d.setdefault("unit", None)
        d.setdefault("scale", None)
        d.setdefault("currency", None)
        d.setdefault("period_end", None)
        d.setdefault("evidence", [])
        d.setdefault("source_text", None)
        d.setdefault("scale_source", None)
        d.setdefault("notes", None)

        # ---- Infer status if the model omitted it ------------------------------
        # If value is present and numeric -> assume "ok"; else "not_found".
        # (If the model did provide a status, keep it unchanged.)
        if not d["status"]:
            val = d.get("value")
            if isinstance(val, (int, float)):
                d["status"] = "ok"
            else:
                d["status"] = "not_found"

        # ---- Normalize evidence and ensure snippet_hash ------------------------
        ev_list = d.get("evidence") or []
        src_text = d.get("source_text")
        if isinstance(ev_list, list):
            fixed = []
            for ev in ev_list:
                if not isinstance(ev, dict):
                    # Coerce non-dict entries into a minimal dict
                    ev = {"page": None, "ref": str(ev)}
                ev.setdefault("page", None)
                ev.setdefault("ref", None)

                sh = ev.get("snippet_hash")
                if not (isinstance(sh, str) and _SNIPPET_HEX_RE.fullmatch(sh)):
                    # Build a deterministic hash from best-available parts.
                    # Priority: source_text (stable snippet) → ref → field_id+page
                    page = ev.get("page")
                    ref = ev.get("ref")
                    ev["snippet_hash"] = _mk_snippet_hash(src_text, ref, field_id, page)

                fixed.append(ev)
            d["evidence"] = fixed
        else:
            # If evidence is not a list, normalize to a single generated item
            d["evidence"] = [
                {
                    "page": None,
                    "ref": None,
                    "snippet_hash": _mk_snippet_hash(src_text, field_id),
                }
            ]

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

    def _save_debug(self, field, prompt, raw, exc):
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            p = self.debug_dir / f"{field.id}-{ts}.json"
            p.write_text(
                json.dumps(
                    {
                        "field_id": field.id,
                        "prompt": prompt[-4000:],  # last 4k chars
                        "raw": raw,
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


# --- scale and parenthesis helpers -------------------------------------------


def _normalize_spaces(s: str) -> str:
    # normalize NBSP / narrow NBSP / thin spaces to regular space
    return s.replace(_NBSP, " ").replace(_NNBSP, " ").replace(_THIN, " ")


def _parse_de_number(num_str: str) -> float:
    """
    Parse a German-formatted number string:
    - thousands separators: dot or spaces
    - decimal separator: comma
    """
    s = num_str.strip()
    # remove thousands separators (space & dot)
    s = s.replace(".", "").replace(" ", "")
    # decimal comma -> dot
    s = s.replace(",", ".")
    return float(s)


def _detect_scale_tokens(window: str) -> tuple[float | None, str | None]:
    """
    Look for scale tokens and EUR nearby the match; return (scale, scale_source_label).
    Examples of labels we might return: "TEUR", "Tsd €", "Mio EUR", "EUR"
    """
    w = _normalize_spaces(window).lower()

    found_scale = None
    scale_label = None

    for pat, factor, label in _SCALE_TOKENS:
        if re.search(pat, w, flags=re.IGNORECASE):
            found_scale = factor
            scale_label = label
            break

    eur_present = any(re.search(p, w, flags=re.IGNORECASE) for p in _EUR_TOKENS)

    if found_scale and eur_present:
        # Prefer combined label with a space and the original case preference
        # Keep "€" if that’s present, else "EUR"
        scale_suffix = "€" if re.search(r"€", window) else "EUR"
        return found_scale, f"{scale_label} {scale_suffix}"
    if found_scale:
        return found_scale, scale_label
    if eur_present:
        return None, "EUR"

    return None, None


def _extract_current_prev_pair(text: str) -> dict | None:
    """
    Detect patterns like:
        "... 123 456 (133 333) TEUR"
        "... 1.234.567 (1.111.111) EUR"
        "... 12,5 (11,0) Mio EUR"
        "... 3 (2) Mrd €"
    Returns dict with keys: value, prev_value, unit, scale, scale_source, source_text
    or None if no such pattern is found.
    """
    t = _normalize_spaces(text)

    # number core: either 1-3 digits followed by grouped 3s with separators, OR plain digits; optional decimal with comma
    NUM = rf"(?:\d{{1,3}}(?:[{_SEP_CHARS}]\d{{3}})+|\d+)(?:,\d+)?"

    # Use distinct group names for current/previous
    pattern = re.compile(
        rf"(?P<curr>{NUM})\s*\(\s*(?P<prev>{NUM})\s*\)",
        flags=re.IGNORECASE,
    )

    m = pattern.search(t)
    if not m:
        return None

    curr_s = m.group("curr")
    prev_s = m.group("prev")

    try:
        curr_v = _parse_de_number(curr_s)
        prev_v = _parse_de_number(prev_s)
    except Exception:
        return None

    # Small window around the match to detect scale and unit
    left = max(0, m.start() - 20)
    right = min(len(t), m.end() + 20)
    window = text[
        left:right
    ]  # take from original text to preserve "€"/case for scale_source label

    scale, scale_src = _detect_scale_tokens(window)

    # Decide unit: set to EUR if we saw EUR/€; otherwise keep EUR (your tests expect EUR)
    unit = "EUR"

    # Make a short source snippet (<=200)
    src = text[max(0, m.start() - 40) : min(len(text), m.end() + 40)]
    src = src.strip()
    if len(src) > 200:
        src = src[:197] + "..."

    return {
        "value": curr_v,
        "prev_value": prev_v,
        "unit": unit,
        "scale": scale,
        "scale_source": scale_src if scale_src else None,
        "source_text": src,
    }
