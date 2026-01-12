from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import yaml

from sfcr.extract.schema import Evidence, ExtractionLLM, ResponseLLM, VerifiedExtraction
from sfcr.extract.verify import verify_extraction
from sfcr.ingest.schema import IngestionResult
from sfcr.llm.llm_text_client import LLMTextClient
from sfcr.utils.textnorm import normalize_hyphenation

# ---------- field taxonomy ----------


@dataclass
class FieldDef:
    id: str
    subsection_hint: str  # "A".."E" (usually), could be "D"/"E"
    unit: str  # "EUR" or "%"
    typical_scale: float | None
    keywords: List[str]
    notes: str | None


def load_fields(path: Path) -> List[FieldDef]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: List[FieldDef] = []
    for r in rows:
        subsection_hint = r.get("subsection_hint")
        if subsection_hint is None:
            subsection_hint = r.get("section_hint", "E")
        out.append(
            FieldDef(
                id=r["id"],
                subsection_hint=subsection_hint,
                unit=r["unit"],
                typical_scale=r.get("typical_scale"),
                keywords=r.get("keywords", []),
                notes=r.get("notes", ""),
            )
        )
    return out


def _subsection_span_for(
    sub_id: str, ingestion: IngestionResult
) -> Optional[Tuple[int, int]]:
    for s in ingestion.subsections:
        if sub_id == getattr(s, "code", None):
            return s.start_page, s.end_page
    return None


def _letter_from_sub_hint(sub_hint: str) -> str:
    if sub_hint and len(sub_hint) > 0:
        first_char = sub_hint[0].upper()
        if first_char in ("A", "B", "C", "D", "E"):
            return first_char
    if sub_hint.startswith("S."):
        return "E"
    return "E"


# ---------- simple text utilities ----------


def extract_text_pages(pdf_path: Path, start: int, end: int) -> Tuple[str, List[str]]:
    """
    Return (joined_text, page_texts[]), inclusive 1-based pages.
    """
    doc = fitz.open(pdf_path)
    try:
        start0 = max(1, start) - 1
        end0 = min(end, doc.page_count) - 1
        pages = []
        for i in range(start0, end0 + 1):
            p = doc.load_page(i)
            pages.append(p.get_text("text"))
        return ("\n".join(pages), pages)
    finally:
        doc.close()


# TODO this needs some work
def harvest_scale_tokens(page_texts: List[str]) -> List[Tuple[str, str]]:
    """
    Very small heuristic: look for scale phrases commonly near tables/captions.
    Returned as [(token_text, source_tag), ...] in precedence order.
    """
    tokens: List[Tuple[str, str]] = []
    # naive scan of first 3 lines of first few pages for captions
    for idx, t in enumerate(page_texts[:3]):
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for ln in lines[:5]:
            if any(
                key in ln
                for key in (
                    "Angaben in",
                    "in TEUR",
                    "in Mio",
                    "EUR",
                    "Euro",
                    "TEUR",
                    "Mio",
                    "Mrd",
                )
            ):
                tokens.append((ln, "caption"))
    # column/row hints (very rough)
    for t in page_texts[:2]:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for ln in lines[:30]:
            if any(k in ln for k in ("EUR", "TEUR", "Mio", "Mrd", "in EUR", "in TEUR")):
                tokens.append((ln, "column"))
    # nearby fallback
    if not tokens and page_texts:
        tokens.append((page_texts[0][:200], "nearby"))
    return tokens


def _mk_snippet_hash(*parts: object, length: int = 16) -> str:
    basis = " | ".join(str(p) for p in parts if p is not None and str(p).strip())
    if not basis:
        basis = "empty"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:length]


@dataclass
class LLMExtractor:
    """
    Orchestrates extraction for ANY provider/model by calling a text client.

    - builds prompt
    - calls text_client.generate_raw(...)
    - parses ResponseLLM JSON
    - returns ExtractionLLM with deterministic evidence snippet_hash
    """

    text_client: LLMTextClient
    output_max_tokens: int = 400  # used by Ollama; OpenAI ignores in options

    def extract(
        self, field: FieldDef, section_text: str, page_start: int, page_end: int
    ) -> ExtractionLLM:
        prompt = self._build_prompt(field, section_text, page_start, page_end)

        raw = self.text_client.generate_raw(
            prompt=prompt,
            json_schema=ResponseLLM.model_json_schema(),
            options={"num_predict": self.output_max_tokens},
        )

        raw = raw.strip()
        if not raw:
            return ExtractionLLM(
                field_id=field.id, status="not_found", evidence=[], source_text=None
            )

        parsed = ResponseLLM.model_validate_json(raw)

        src_text = (parsed.source_text or "").strip()
        if len(src_text) > 200:
            src_text = src_text[:197] + "..."

        sh = _mk_snippet_hash(src_text, field.id, page_start)

        return ExtractionLLM(
            field_id=field.id,
            status=parsed.status,
            value_unscaled=parsed.value_unscaled,
            unit=parsed.unit,
            scale=parsed.scale,
            evidence=[Evidence(page=page_start, ref=None, snippet_hash=sh)],
            source_text=src_text or None,
            scale_source=None,  # TODO remove scale_source
            notes=None,  # TODO remove notes
        )

    def _build_prompt(
        self, field: FieldDef, text: str, page_start: int, page_end: int
    ) -> str:
        field_keywords = " bzw. ".join((field.keywords or []))
        return f"""
You are an information extraction engine for German SFCR sections.

Task:
Extract exactly one value for the field below from the provided section text.
If the value is not uniquely and explicitly present, set status to "not_found" or "ambiguous"
and set all numeric fields to null.

Field:
- field_id: {field.id}
- expected_unit: {field.unit}
- page_range: {page_start}-{page_end}
- helpful_keywords: {field_keywords}
- notes: {field.notes}

Return ONLY one JSON object with EXACTLY these keys (always present):
status, value_unscaled, scale, unit, source_text, scale_source, notes

Definitions:
- value_unscaled: the number as printed, WITHOUT applying scale
- scale: 1 | 1000 | 1000000 | 1000000000 | null
- unit: "EUR" for monetary amounts, "%" for percentages, else null
- source_text: verbatim excerpt (<=200 chars) that includes the value AND the nearby label/keyword
- scale_source: one of "row","column","caption","nearby","model_guess", or null

Rules:
1) Number parsing (German locale):
   - Thousands separators may be ".", spaces, NBSP.
   - Decimal separator may be ",".
2) Previous year in parentheses:
   - If you see "X (Y)" where X and Y are numbers next to each other,
     treat X as current year and Y as previous year. Return X.
3) Disambiguation:
   - Prefer a number that appears on the SAME LINE (or same table row) as one of the helpful_keywords.
   - If multiple candidates remain, set status="ambiguous".
4) Scale detection (set scale, do NOT multiply into value_unscaled):
   - If the relevant row/column/caption/nearby text indicates:
     * "EUR" or "€" -> scale = 1 (unless TEUR/Tsd/Mio/Mrd is stated)
     * "TEUR", "Tsd", "Tsd €", "Tausend" -> scale = 1000
     * "Mio", "Million" -> scale = 1000000
     * "Mrd", "Milliarde" -> scale = 1000000000
   - scale_source should reflect where you found the scale token (row/column/caption/nearby).
   - If no scale info is present, set scale=null and scale_source=null.
5) If status != "ok":
   - value_unscaled=null, scale=null, unit=null, scale_source=null
   - source_text may be null
6) Treat hyphenated line breaks as merged words (e.g. "Mindestkapitalanfor-\\nderung" -> "Mindestkapitalanforderung").
7) Output ONLY JSON. No prose, no markdown.

Section text:
---
{text}
---
""".strip()


# ---------- Orchestration ----------


def _section_span_for(
    letter: str, ingestion: IngestionResult
) -> Optional[Tuple[int, int]]:
    for s in ingestion.sections:
        if s.section == letter:
            return (s.start_page, s.end_page)
    return None


def extract_for_document(
    doc_id: str,
    pdf_path: Path,
    ingestion_json: Path,
    fields_yaml: Path,
    extractor: LLMExtractor,
) -> List[VerifiedExtraction]:
    """f
    Run extraction + verification for a single document.
    """

    ingestion = IngestionResult(
        **json.loads(ingestion_json.read_text(encoding="utf-8"))
    )
    field_defs = load_fields(fields_yaml)

    results: List[VerifiedExtraction] = []

    for f in field_defs:
        span = _subsection_span_for(f.subsection_hint, ingestion)
        if not span:
            letter = _letter_from_sub_hint(f.subsection_hint)
            span = _section_span_for(letter, ingestion)
        if not span:
            # no section → not found
            results.append(
                VerifiedExtraction(
                    doc_id=doc_id,
                    field_id=f.id,
                    status="not_found",
                    verified=False,
                    value_canonical=None,
                    unit=f.unit if f.unit in ("EUR", "%") else None,
                    confidence=0.0,
                    evidence=[],
                    source_text=None,
                    scale_applied=None,
                    scale_source=None,
                    verifier_notes="no_section",
                )
            )
            continue

        print(f.id)  # TODO delete
        start, end = span
        section_text, page_texts = extract_text_pages(pdf_path, start, end)
        section_text, page_texts = (
            normalize_hyphenation(section_text),
            [normalize_hyphenation(page_text) for page_text in page_texts],
        )
        # LLM pass
        llm_out = extractor.extract(f, section_text, start, end)
        print(llm_out.value_unscaled)  # TODO delete
        print(llm_out.scale)  # TODO delete
        # context scale tokens (caption > column > nearby)
        tokens = harvest_scale_tokens(page_texts)
        # deterministic verification
        ver = verify_extraction(
            doc_id=doc_id,
            extr=llm_out,
            typical_scale=f.typical_scale,
            context_scale_tokens=tokens,
            ratio_check=None,  # you can wire SII ratio check at the caller once you have related fields
        )
        results.append(ver)

    return results


def write_jsonl(rows: List[VerifiedExtraction], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(r.model_dump(exclude_none=True), ensure_ascii=False) + "\n"
            )
