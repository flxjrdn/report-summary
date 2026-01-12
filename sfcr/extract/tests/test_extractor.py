from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from sfcr.extract.extractor import LLMExtractor, extract_for_document
from sfcr.extract.schema import Evidence, ExtractionLLM, VerifiedExtraction
from sfcr.llm.llm_text_client import LLMTextClient

# ---------------------------
# Test helpers
# ---------------------------


class _StubTextClient(LLMTextClient):
    """
    Simple stub: returns the provided `raw` string verbatim.
    Also records last prompt so tests can assert it.
    """

    def __init__(self, raw: str):
        self._raw = raw
        self.last_prompt: Optional[str] = None
        self.last_json_schema: Optional[dict[str, Any]] = None
        self.last_options: Optional[dict[str, Any]] = None

    def generate_raw(
        self,
        prompt: str,
        *,
        strict_schema: bool = True,
        json_schema: Optional[dict[str, Any]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_json_schema = json_schema
        self.last_options = options
        return self._raw


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- change this helper -------------------------------------------------


def _ingestion_payload(
    *,
    sections: list[dict[str, Any]],
    subsections: list[dict[str, Any]],
    page_count: int = 3,
):
    # Minimal shape required by IngestionResult(**json.loads(...))
    return {
        "doc_id": "d",
        "pdf_sha256": None,
        "page_count": page_count,
        "sections": sections,
        "subsections": subsections,
        "coverage_ratio": 1.0,
        "issues": [],
    }


def _fields_yaml_one(
    *,
    field_id: str,
    subsection_hint: str,
    unit: str = "EUR",
    typical_scale: float | None = None,
):
    # load_fields expects list[dict]
    rows = [
        {
            "id": field_id,
            "subsection_hint": subsection_hint,
            "unit": unit,
            "typical_scale": typical_scale,
            "keywords": ["TestKeyword"],
            "notes": "",
        }
    ]
    return yaml_dump(rows)


def yaml_dump(obj: Any) -> str:
    # Avoid importing yaml in tests if you prefer; but extractor.py uses yaml anyway.
    import yaml  # type: ignore

    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)


# ---------------------------
# Tests: LLMExtractor.extract
# ---------------------------


def test_llm_extractor_empty_raw_returns_not_found():
    client = _StubTextClient(raw="   \n")
    ex = LLMExtractor(text_client=client)

    field = type(
        "F",
        (),
        {
            "id": "x",
            "unit": "EUR",
            "keywords": ["K"],
            "notes": "",
            "subsection_hint": "E",
        },
    )
    out = ex.extract(field, "some text", 10, 12)

    assert out.field_id == "x"
    assert out.status == "not_found"
    assert out.evidence == []
    assert out.source_text is None


def test_llm_extractor_parses_response_and_creates_snippet_hash_hex():
    raw = json.dumps(
        {
            "status": "ok",
            "value_unscaled": 123456.0,
            "scale": 1000.0,
            "unit": "EUR",
            "source_text": "Mindestkapitalanforderung 123 456 (133 333) TEUR",
            "scale_source": "row",
            "notes": None,
        },
        ensure_ascii=False,
    )
    client = _StubTextClient(raw=raw)
    ex = LLMExtractor(text_client=client)

    field = type(
        "F",
        (),
        {
            "id": "mcr_total",
            "unit": "EUR",
            "keywords": ["Mindestkapitalanforderung"],
            "notes": "",
            "subsection_hint": "E.2",
        },
    )
    out = ex.extract(field, "section text", 5, 7)

    assert out.status == "ok"
    assert out.value_unscaled == 123456.0
    assert out.scale == 1000.0
    assert out.unit == "EUR"
    assert out.evidence and isinstance(out.evidence[0], Evidence)
    assert out.evidence[0].page == 5
    assert out.evidence[0].snippet_hash is not None
    assert re.fullmatch(
        r"[a-f0-9]{16}", out.evidence[0].snippet_hash
    )  # extractor uses length=16


def test_llm_extractor_truncates_source_text_to_200_chars():
    long_src = "X" * 250
    raw = json.dumps(
        {
            "status": "ok",
            "value_unscaled": 1.0,
            "scale": None,
            "unit": "EUR",
            "source_text": long_src,
            "scale_source": None,
            "notes": None,
        }
    )
    client = _StubTextClient(raw=raw)
    ex = LLMExtractor(text_client=client)

    field = type(
        "F",
        (),
        {
            "id": "foo",
            "unit": "EUR",
            "keywords": ["K"],
            "notes": "",
            "subsection_hint": "E",
        },
    )
    out = ex.extract(field, "t", 1, 1)

    assert out.source_text is not None
    assert len(out.source_text) <= 200
    assert out.source_text.endswith("...")


# ---------------------------
# Tests: extract_for_document orchestration
# ---------------------------


def test_extract_for_document_prefers_subsection_span_over_section(
    monkeypatch, tmp_path: Path
):
    """
    If subsection_hint exists in ingestion.subsections, extract_for_document must use that span.
    We assert this by checking the page_start/page_end passed into extract_text_pages via monkeypatch.
    """
    # ingestion: subsection E.2 spans 2..2, section E spans 1..3
    ingestion = _ingestion_payload(
        sections=[{"section": "E", "start_page": 1, "end_page": 3}],
        subsections=[
            {
                "section": "E",
                "code": "E.2",
                "title": "Sub",
                "start_page": 2,
                "end_page": 2,
            }
        ],
    )
    ingest_path = tmp_path / "doc.ingest.json"
    _write_json(ingest_path, ingestion)

    fields_path = tmp_path / "fields.yaml"
    _write_text(
        fields_path,
        _fields_yaml_one(
            field_id="scr_total",
            subsection_hint="E.2",
            unit="EUR",
            typical_scale=1000.0,
        ),
    )

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")  # won't be opened due to monkeypatch

    called = {}

    def fake_extract_text_pages(_pdf_path: Path, start: int, end: int):
        called["start"] = start
        called["end"] = end
        # return "section_text", page_texts list size = end-start+1
        return "Some content", ["Angaben in TEUR\nSome content"]

    monkeypatch.setattr(
        "sfcr.extract.extractor.extract_text_pages", fake_extract_text_pages
    )
    monkeypatch.setattr("sfcr.extract.extractor.normalize_hyphenation", lambda s: s)

    # extractor stub returns ok, with evidence page == page_start
    raw = json.dumps(
        {
            "status": "ok",
            "value_unscaled": 10.0,
            "scale": 1000.0,
            "unit": "EUR",
            "source_text": "SCR 10 TEUR",
            "scale_source": "row",
            "notes": None,
        }
    )
    extractor = LLMExtractor(text_client=_StubTextClient(raw))

    res = extract_for_document(
        doc_id="d",
        pdf_path=pdf_path,
        ingestion_json=ingest_path,
        fields_yaml=fields_path,
        extractor=extractor,
    )

    assert called["start"] == 2
    assert called["end"] == 2
    assert len(res) == 1
    assert isinstance(res[0], VerifiedExtraction)
    assert res[0].field_id == "scr_total"


def test_extract_for_document_falls_back_to_section_when_subsection_missing(
    monkeypatch, tmp_path: Path
):
    """
    If subsection span isn't found, extractor should infer section from first letter and use section span.
    """
    ingestion = _ingestion_payload(
        page_count=9,
        sections=[{"section": "E", "start_page": 7, "end_page": 9}],
        subsections=[],
    )
    ingest_path = tmp_path / "doc.ingest.json"
    _write_json(ingest_path, ingestion)

    fields_path = tmp_path / "fields.yaml"
    # subsection_hint is E.2, but ingestion has no subsections, so fall back to E section span
    _write_text(
        fields_path,
        _fields_yaml_one(
            field_id="mcr_total",
            subsection_hint="E.2",
            unit="EUR",
            typical_scale=1000.0,
        ),
    )

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    called = {}

    def fake_extract_text_pages(_pdf_path: Path, start: int, end: int):
        called["start"] = start
        called["end"] = end
        return "Some content", ["Some content"] * (end - start + 1)

    monkeypatch.setattr(
        "sfcr.extract.extractor.extract_text_pages", fake_extract_text_pages
    )
    monkeypatch.setattr("sfcr.extract.extractor.normalize_hyphenation", lambda s: s)

    raw = json.dumps(
        {
            "status": "ok",
            "value_unscaled": 1.0,
            "scale": None,
            "unit": "EUR",
            "source_text": "MCR 1 EUR",
            "scale_source": "row",
            "notes": None,
        }
    )
    extractor = LLMExtractor(text_client=_StubTextClient(raw))

    res = extract_for_document(
        doc_id="d",
        pdf_path=pdf_path,
        ingestion_json=ingest_path,
        fields_yaml=fields_path,
        extractor=extractor,
    )

    assert called["start"] == 7
    assert called["end"] == 9
    assert len(res) == 1


def test_extract_for_document_no_section_returns_not_found_verified_false(
    tmp_path: Path,
):
    """
    If neither subsection nor section span exists, result should be status=not_found and verifier_notes=no_section.
    """
    ingestion = _ingestion_payload(sections=[], subsections=[])
    ingest_path = tmp_path / "doc.ingest.json"
    _write_json(ingest_path, ingestion)

    fields_path = tmp_path / "fields.yaml"
    _write_text(
        fields_path,
        _fields_yaml_one(
            field_id="scr_total",
            subsection_hint="E.2",
            unit="EUR",
            typical_scale=1000.0,
        ),
    )

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    # extractor won't be used, but must be provided
    extractor = LLMExtractor(text_client=_StubTextClient(raw="{}"))

    res = extract_for_document(
        doc_id="d",
        pdf_path=pdf_path,
        ingestion_json=ingest_path,
        fields_yaml=fields_path,
        extractor=extractor,
    )

    assert len(res) == 1
    r = res[0]
    assert r.status == "not_found"
    assert r.verified is False
    assert r.verifier_notes == "no_section"
    assert r.value_canonical is None


def test_extract_for_document_uses_evidence_page_to_choose_page_text_for_scale(
    monkeypatch, tmp_path: Path
):
    """
    extract_for_document chooses page_text_for_scale based on llm_out.evidence[0].page.
    We simulate span start=10 end=12 with three pages, and set evidence page=12 -> should pass page_texts[2].
    """
    ingestion = _ingestion_payload(
        page_count=12,
        sections=[{"section": "E", "start_page": 10, "end_page": 12}],
        subsections=[
            {
                "section": "E",
                "code": "E.2",
                "title": "Sub",
                "start_page": 10,
                "end_page": 12,
            }
        ],
    )
    ingest_path = tmp_path / "doc.ingest.json"
    _write_json(ingest_path, ingestion)

    fields_path = tmp_path / "fields.yaml"
    _write_text(
        fields_path,
        _fields_yaml_one(
            field_id="scr_total",
            subsection_hint="E.2",
            unit="EUR",
            typical_scale=1000.0,
        ),
    )

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    page_texts = ["p10 caption: Angaben in TEUR", "p11 ...", "p12 Einheit: Mio EUR"]

    def fake_extract_text_pages(_pdf_path: Path, start: int, end: int):
        return "Some content", page_texts

    monkeypatch.setattr(
        "sfcr.extract.extractor.extract_text_pages", fake_extract_text_pages
    )
    monkeypatch.setattr("sfcr.extract.extractor.normalize_hyphenation", lambda s: s)

    captured = {}

    def fake_verify_extraction(
        *, doc_id, extr, typical_scale, page_text_for_scale=None, ratio_check=None
    ):
        captured["page_text_for_scale"] = page_text_for_scale
        # return a minimal VerifiedExtraction that satisfies your model
        return VerifiedExtraction(
            doc_id=doc_id,
            field_id=extr.field_id,
            status=extr.status,
            verified=False,
            value_canonical=None,
            unit=extr.unit,
            confidence=0.0,
            evidence=extr.evidence,
            source_text=extr.source_text,
            scale_applied=None,
            scale_source=None,
            verifier_notes="stub",
        )

    monkeypatch.setattr(
        "sfcr.extract.extractor.verify_extraction", fake_verify_extraction
    )

    # Make the LLM output have evidence[0].page == 12 (span end)
    # We can't directly force evidence in LLMExtractor.extract (it uses page_start),
    # so monkeypatch extractor.extract to return a custom ExtractionLLM.
    def fake_extractor_extract(_self, field, section_text, page_start, page_end):
        return ExtractionLLM(
            field_id=field.id,
            status="ok",
            value_unscaled=12.0,
            unit="EUR",
            scale=None,
            evidence=[Evidence(page=12, ref=None, snippet_hash="deadbeefdeadbeef")],
            source_text="SCR 12",
            scale_source=None,
            notes=None,
        )

    extractor = LLMExtractor(text_client=_StubTextClient(raw="{}"))
    monkeypatch.setattr(LLMExtractor, "extract", fake_extractor_extract)

    _ = extract_for_document(
        doc_id="d",
        pdf_path=pdf_path,
        ingestion_json=ingest_path,
        fields_yaml=fields_path,
        extractor=extractor,
    )

    assert captured["page_text_for_scale"] == page_texts[2]
