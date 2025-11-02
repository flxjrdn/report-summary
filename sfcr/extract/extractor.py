from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import yaml

from sfcr.extract.schema import Evidence, ExtractionLLM, VerifiedExtraction
from sfcr.extract.verify import verify_extraction
from sfcr.ingest.schema import IngestionResult
from sfcr.utils.textnorm import normalize_hyphenation

# ---------- field taxonomy ----------


@dataclass
class FieldDef:
    id: str
    subsection_hint: str  # "A".."E" (usually), could be "D"/"E"
    unit: str  # "EUR" or "%"
    typical_scale: float | None
    keywords: List[str]


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


# ---------- LLM interface ----------


class LLMClient:
    """
    Interface: implement .extract(field, section_text, pages) -> ExtractionLLM
    Must return strict JSON-compatible fields per ExtractionLLM schema.
    """

    def extract(
        self, field: FieldDef, section_text: str, page_start: int, page_end: int
    ) -> ExtractionLLM:
        raise NotImplementedError


class MockLLM(LLMClient):
    """
    Rule-based placeholder so you can run end-to-end now.
    Looks for the first number after a keyword; guesses scale from keywords.
    """

    import re as _re

    def extract(
        self, field: FieldDef, section_text: str, page_start: int, page_end: int
    ) -> ExtractionLLM:
        text = section_text
        # cheap locate: find window around first keyword
        pos = min(
            (
                text.lower().find(k.lower())
                for k in field.keywords
                if text.lower().find(k.lower()) >= 0
            ),
            default=-1,
        )
        window = text[max(0, pos - 200) : pos + 400] if pos >= 0 else text[:600]
        # very naive number regex (DE): 123.456,78 or 123.456 or 123,4%
        m = self._re.search(
            r"(\(?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?%?)|(\(?\d+(?:,\d+)?\)?%?)", window
        )
        if not m:
            return ExtractionLLM(
                field_id=field.id, status="not_found", evidence=[], source_text=None
            )
        snip = m.group(0)
        # guess unit & scale source
        unit = field.unit
        guess_scale = None
        scale_source = "model_guess"
        if any(x in window for x in ("TEUR", "Tsd", "Tausend")):
            guess_scale = 1e3
            scale_source = "nearby"
        elif any(x in window for x in ("Mio", "Million")):
            guess_scale = 1e6
            scale_source = "nearby"
        elif any(x in window for x in ("Mrd", "Milliarde")):
            guess_scale = 1e9
            scale_source = "nearby"

        return ExtractionLLM(
            field_id=field.id,
            status="ok",
            value_unscaled=None,
            value_scaled=None,
            unit=unit,
            scale=guess_scale,
            evidence=[Evidence(page=page_start, ref=None)],
            source_text=snip,
            scale_source=scale_source,
            notes=None,
        )


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
    llm: Optional[LLMClient] = None,
) -> List[VerifiedExtraction]:
    """f
    Run extraction + verification for a single document.
    """
    if llm is None:
        llm = MockLLM()

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
        print(f"start: {start}, end: {end}")  # TODO delete
        section_text, page_texts = extract_text_pages(pdf_path, start, end)
        section_text, page_texts = (
            normalize_hyphenation(section_text),
            [normalize_hyphenation(page_text) for page_text in page_texts],
        )
        # LLM pass
        llm_out = llm.extract(f, section_text, start, end)
        print(llm_out.value_scaled)  # TODO delete
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
