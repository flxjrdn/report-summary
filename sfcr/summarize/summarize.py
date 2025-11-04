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
    "You summarize Solvency II SFCR sections for actuaries. "
    "Write 3–6 concise bullet points, factual and neutral. "
    "Include material quantitative values (with units) and any notable changes vs prior year, "
    "but do not invent numbers. If the section is largely qualitative, focus on key themes. "
    "Avoid boilerplate; avoid copying sentences verbatim."
)


def _section_prompt(section: Section) -> str:
    return (
        f"{_SUMMARY_SYSTEM_INSTR}\n\n"
        # f"Section: {section.section_id} — {section.title}\n" # TODO include title
        f"Section: {section.section_id}\n"
        f"Instructions:\n"
        f"- Summarize only the content provided.\n"
        f"- Prefer short bullet points (• ...).\n"
        f"- If the input is empty or not informative, respond with 'No material content found.'\n"
        f"\n--- BEGIN SECTION TEXT ---\n"
        f"{{chunk}}\n"
        f"--- END SECTION TEXT ---\n"
    )


def _synthesis_prompt() -> str:
    return (
        "You are given multiple partial summaries of the same SFCR section. "
        "Merge them into 3–6 bullets, removing duplication, keeping the most precise numbers. "
        "Do not add information not present in the partial summaries.\n\n"
        "Partial summaries:\n"
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
    """Chunk → summarize each chunk → synthesize (if needed). Returns final bullet list as text."""
    text = _extract_text_for_pages(pdf, section.start_page, section.end_page)
    if not text.strip():
        return "No material content found."

    chunks = _chunk_text(text, max_chars=max_chars_per_chunk, overlap=overlap)

    partials: List[str] = []
    for ch in chunks:
        prompt = _section_prompt(section).replace("{chunk}", ch)
        resp = _call_llm_generate(llm, prompt)
        partials.append(resp)

    if len(partials) == 1:
        return partials[0]

    joined = "\n\n---\n\n".join(partials)
    synth = _synthesis_prompt().replace("{bullets}", joined)
    final = _call_llm_generate(llm, synth)
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
