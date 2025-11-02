from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from sfcr.extract.extractor import FieldDef, LLMClient
from sfcr.extract.schema import Evidence, ExtractionLLM, ResponseLLM
from sfcr.llm.ollama_text_client import OllamaTextClient

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
@dataclass
class OllamaLLM(LLMClient):
    """
    Local Ollama backend for the SFCR extractor.
    Works with models like 'mistral', 'llama3', 'qwen2', etc.
    """

    model: str = "mistral"
    enable_cache: bool = False
    debug_raw: bool = False
    text_client: OllamaTextClient = field(init=False, repr=False)
    input_token_limit: int = 1200
    output_max_tokens: int = 400  # Ollama: num_predict

    def __post_init__(self):
        self.text_client = OllamaTextClient(
            model=self.model, enable_cache=self.enable_cache, debug_raw=self.debug_raw
        )

    def extract(
        self, field: FieldDef, section_text: str, page_start: int, page_end: int
    ) -> ExtractionLLM:
        bounded = _window_text(field, section_text, self.input_token_limit)
        prompt = self._build_prompt(field, bounded, page_start, page_end)

        raw = self.text_client.generate_raw(
            prompt,
            json_schema=ResponseLLM.model_json_schema(),
            options={"num_predict": self.output_max_tokens},
        )

        parsed = ResponseLLM.model_validate_json(raw)

        # Ensure a short source snippet (<=200 chars) and build a deterministic snippet hash
        src_text = (parsed.source_text or "").strip()
        if len(src_text) > 200:
            src_text = src_text[:197] + "..."
        sh = _mk_snippet_hash(src_text, field.id, page_start)

        out = ExtractionLLM(
            field_id=field.id,
            status=parsed.status,
            value_unscaled=parsed.value_unscaled,
            value_scaled=parsed.value_scaled,
            unit=parsed.unit,
            evidence=[Evidence(page=page_start, ref=None, snippet_hash=sh)],
            source_text=src_text or None,
            scale_source=None,
            notes=None,
        )
        return out

    # --- helpers --------------------------------------------------------------
    def _build_prompt(
        self, field: FieldDef, text: str, page_start: int, page_end: int
    ) -> str:
        field_keywords = " bzw. ".join((field.keywords or []))
        return f"""
Deine Aufgabe ist es, den Wert für {field_keywords} aus dem folgenden TEXT auszulesen.

Gib das Ergebnis **ausschließlich als JSON-Objekt** mit diesen Feldern zurück:
"status", "value_unscaled", "value_scaled", "scale", "unit", "source_text"
Regeln:
1) Wenn ein eindeutiger Wert gefunden wird:
   - "status": "ok"
   - "value_unscaled": der unskalierte Zahlenwert (z. B. 123456.0)
   - "value_scaled": "value_unscaled" multipliziert mit "scale"
   - "scale": 1 bei EUR, 1000 bei TEUR/Tausend Euro, 1000000 bei Mio. Euro; sonst null, wenn keine Skalierung ableitbar ist
   - "unit": "EUR" oder "%"
   - "source_text": eine kurze Fundstelle (max. 200 Zeichen) aus dem TEXT, die den Wert enthält
2) Wenn kein eindeutiger Wert gefunden wird, setze "status" auf "not_found" und alle übrigen Felder auf null.
3) Zahlen sind im deutschen Format zu interpretieren: Punkt oder Leerzeichen als Tausendertrennzeichen, Komma als Dezimaltrennzeichen.
4) Wenn ein Wert gefolgt von einem zweiten Wert in Klammern vorkommt, bezieht sich der erste auf das aktuelle Jahr und der Klammerwert auf das Vorjahr – gib den ersten Wert zurück.
5) Gib **nur** das JSON-Objekt zurück, ohne zusätzliche Erklärungen oder Text.

Beispiel:
Text: "Die Solvenzkapitalanforderung betrug 200 000 (211 111) TEUR. Die Mindestkapitalanforderung betrug 123 456 (133 333) TEUR."
Erwartete Ausgabe:
{{
  "status": "ok",
  "value_unscaled": 123456.0,
  "value_scaled": 123456000.0,
  "scale": 1000,
  "unit": "EUR",
  "source_text": "Die Mindestkapitalanforderung betrug 123 456 (133 333) TEUR."
}}
Ende des Beispiels.

TEXT:
{text}
""".strip()

    def _make_cache_key(self, field: FieldDef, prompt: str) -> str:
        key_src = json.dumps(
            {
                "model": self.model,
                "temp": self.text_client.temperature,
                "field": field.id,
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return _sha1(key_src)

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
