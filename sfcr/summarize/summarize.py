from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF

from sfcr.llm.llm_text_client import LLMTextClient
from sfcr.llm.llm_text_client_factory import create_llm_text_client
from sfcr.utils.textnorm import normalize_hyphenation


@dataclass
class Section:
    section_id: str
    # title: str #  TODO include title
    start_page: int  # 1-based inclusive
    end_page: int  # 1-based inclusive


def _read_ingestion_sections(ingest_json: Path) -> List[Section]:
    """
    Expect your ingestion artifact to contain a list of sections with:
      section_id, title, start_page, end_page (and possibly subsections)
    """
    data = json.loads(ingest_json.read_text(encoding="utf-8"))
    sections: List[Section] = []
    for sec in data.get("sections", []):
        sections.append(
            Section(
                section_id=sec["section"],
                # title=sec.get("title", sec["section_id"]), # TODO include title
                start_page=int(sec["start_page"]),
                end_page=int(sec["end_page"]),
            )
        )
    # Keep A..E in order; POST at the end if present
    order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    sections.sort(key=lambda s: (order.get(s.section_id, 99), s.start_page))
    print(f"Reading ingestion sections complete... found {len(sections)} sections")
    return sections


def _extract_text_for_pages(pdf: Path, start_page: int, end_page: int) -> str:
    """Concatenate text for [start_page..end_page] (1-based, inclusive)."""
    doc = fitz.open(pdf)
    try:
        parts: List[str] = []
        for i in range(start_page - 1, end_page):
            p = doc.load_page(i)
            parts.append(p.get_text("text"))
        raw = "\n".join(parts)
        return normalize_hyphenation(raw)
    finally:
        doc.close()


def _chunk_text(s: str, max_chars: int = 12000, overlap: int = 800) -> List[str]:
    """
    Simple character-based chunking (LLM-agnostic). Keeps overlaps so we don't lose context
    around boundaries. Tweak sizes to match your local context window.
    """
    s = s.strip()
    if len(s) <= max_chars:
        return [s]
    chunks = []
    i = 0
    while i < len(s):
        j = min(len(s), i + max_chars)
        chunk = s[i:j]
        chunks.append(chunk)
        if j == len(s):
            break
        i = max(0, j - overlap)
    return chunks


_SUMMARY_SYSTEM_INSTR = (
    "Fasse den SFCR-Abschnitt in 3–6 sachlichen Stichpunkten zusammen. "
    "Nenne wesentliche Zahlen mit Einheiten. Keine erfundenen Zahlen. "
    "Keine wörtlichen Kopien."
)


def _section_prompt(section: Section) -> str:
    return (
        f"{_SUMMARY_SYSTEM_INSTR}\n\n"
        # f"Abschnitt: {section.section_id} — {section.title}\n" # TODO include title
        f"Abschnitt: {section.section_id}\n"
        f"Anweisungen:\n"
        f"- Fasse nur den gegebenen Kontext zusammen.\n"
        f"- Bevorzuge kurze bullet points (• ...).\n"
        f"- Wenn die Eingabe leer ist oder keine relevanten Informationen enthält, antworte mit 'Keine relevanten Inhalte gefunden.'\n"
        f"\n--- BEGINN ABSCHNITT TEXT ---\n"
        f"{{chunk}}\n"
        f"--- ENDE ABSCHNITT TEXT ---\n"
    )


def _synthesis_prompt() -> str:
    return (
        "Du erhältst mehrere Teilzusammenfassungen desselben SFCR-Abschnitts."
        "Fasse sie zu 3–6 Stichpunkten zusammen, entferne Duplikate und übernimm jeweils die präzisesten Zahlenangaben."
        "Füge keinerlei Informationen hinzu, die nicht in den Teilsummaries enthalten sind."
        "Teilzusammenfassungen:"
        "{bullets}"
    )


def _call_llm_generate(llm: LLMTextClient, prompt: str) -> str:
    out = llm.generate_raw(prompt)
    return (out or "").strip()


def summarize_section(
    llm: LLMTextClient,
    pdf: Path,
    section: Section,
    max_chars_per_chunk: int = 12000,
    overlap: int = 800,
) -> str:
    """
    Tries to summarize the whole section in one go. If the result is truncated,
    we split the section into several chunks, summarize them individually and merge them.
    """
    print(f"summarizing section {section.section_id} ...")
    text = _extract_text_for_pages(pdf, section.start_page, section.end_page)
    if not text.strip():
        return "Keine relevanten Inhalte gefunden."

    # --- 1) try single-pass ---------------------------------------------------
    prompt = _section_prompt(section).replace("{chunk}", text)

    # keep output short to save money
    resp = llm.generate_raw(prompt, options={"max_output_tokens": 700})
    resp = (resp or "").strip()

    # If OpenAI and truncated -> fallback to chunking
    was_truncated = False
    if hasattr(llm, "was_truncated") and callable(getattr(llm, "was_truncated")):
        try:
            was_truncated = bool(llm.was_truncated())
        except Exception:
            was_truncated = False

    if resp and not was_truncated:
        return resp

    # --- 2) fallback: chunk + synthesize -------------------------------------
    print(
        "Summary result was truncated, so we chunk the section and summarize individual chunks, before merging them back together."
    )
    chunks = _chunk_text(text, max_chars=max_chars_per_chunk, overlap=overlap)

    partials: List[str] = []
    for ch in chunks:
        p = _section_prompt(section).replace("{chunk}", ch)
        partials.append((_call_llm_generate(llm, p) or "").strip())

    if len(partials) == 1:
        return partials[0] or "Keine relevanten Inhalte gefunden."

    joined = "\n\n---\n\n".join(partials)
    synth = _synthesis_prompt().replace("{bullets}", joined)
    final = _call_llm_generate(llm, synth).strip()
    return final or joined


def write_summaries_jsonl(
    out_path: Path,
    doc_id: str,
    sections: Iterable[Section],
    pdf: Path,
    llm: LLMTextClient,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sec in sections:
            summary = summarize_section(llm, pdf, sec)
            rec = {
                "doc_id": doc_id,
                "section_id": sec.section_id,
                # "title": sec.title, # TODO include title
                "start_page": sec.start_page,
                "end_page": sec.end_page,
                "summary": summary,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_summarize(
    *,
    doc_id: str,
    pdf_path: Path,
    ingest_json: Path,
    out_jsonl: Path,
    provider: str = "ollama",
    model: str = "mistral",  # or "llama3.1:8b-instruct"
) -> Path:
    llm = create_llm_text_client(provider=provider, model=model)
    sections = _read_ingestion_sections(ingest_json)
    write_summaries_jsonl(out_jsonl, doc_id, sections, pdf_path, llm)
    return out_jsonl


if __name__ == "__main__":
    run_summarize(
        doc_id="sikv_2023",
        pdf_path=Path(
            "/Users/felixjordan/Documents/code/report-summary/data/sfcrs/sikv_2023.pdf"
        ),
        ingest_json=Path(
            "/Users/felixjordan/Documents/code/report-summary/artifacts/ingest/sikv_2023.ingest.json"
        ),
        out_jsonl=Path(
            "/Users/felixjordan/Documents/code/report-summary/artifacts/summaries/sikv_2023.summaries.jsonl"
        ),
        provider="openai",
        model="gpt-5-mini",
    )
