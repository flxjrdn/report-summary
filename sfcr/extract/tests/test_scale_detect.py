from sfcr.extract.scale_detect import (
    NBSP,
    NNBSP,
    THIN,
    _candidate_caption_lines,
    infer_scale,
    infer_scale_from_page_caption,
    infer_scale_from_source_text,
    infer_scale_near_source,
)


def test_infer_scale_from_source_text_teur_with_eur():
    hit = infer_scale_from_source_text(
        "Mindestkapitalanforderung 123 456 (133 333) TEUR"
    )
    assert hit is not None
    assert hit.scale == 1000.0
    assert hit.unit == "EUR"
    assert hit.label in ("TEUR EUR", "TEUR €", "TEUR")  # depends if EUR/€ is present
    assert "TEUR" in (hit.evidence_line or "")


def test_infer_scale_from_source_text_tsd_euro_symbol():
    hit = infer_scale_from_source_text("Angaben: 1 234 Tsd €")
    assert hit is not None
    assert hit.scale == 1000.0
    assert hit.unit == "EUR"
    assert hit.label == "Tsd €"


def test_infer_scale_from_source_text_mio_eur():
    hit = infer_scale_from_source_text("Kapitalanlagen 12,5 Mio EUR")
    assert hit is not None
    assert hit.scale == 1_000_000.0
    assert hit.unit == "EUR"
    assert hit.label == "Mio EUR"


def test_infer_scale_from_source_text_mrd_with_soft_hyphen_and_nbsp():
    # Soft hyphen should be removed; NBSP becomes normal space
    text = f"Eigenmittel 3{NBSP}Mrd\u00ad EUR"
    hit = infer_scale_from_source_text(text)
    assert hit is not None
    assert hit.scale == 1_000_000_000.0
    assert hit.unit == "EUR"
    assert hit.label == "Mrd EUR"


def test_infer_scale_from_source_text_eur_only_returns_scale_1():
    hit = infer_scale_from_source_text("Solvabilitätskapitalanforderung 1 312 850 EUR")
    assert hit is not None
    assert hit.scale == 1.0
    assert hit.unit == "EUR"
    assert hit.label == "EUR"


def test_infer_scale_from_source_text_percent():
    hit = infer_scale_from_source_text("Solvabilitätsquote 391% (389%)")
    assert hit is not None
    assert hit.scale is None
    assert hit.unit == "%"
    assert hit.label == "%"


def test_infer_scale_from_source_text_none_when_no_unit_or_scale():
    assert infer_scale_from_source_text("irgendein Text ohne Einheit") is None


def test_candidate_caption_lines_picks_cue_lines_and_scale_tokens():
    page = "\n".join(
        [
            "Irgendwas oben",
            "Angaben in TEUR",  # cue
            "noch eine zeile",
            "Beträge in Mio EUR",  # cue + token
            "random",
            "Marktrisiko 123 456",  # should not be selected
        ]
    )
    lines = _candidate_caption_lines(page)
    assert "Angaben in TEUR" in lines
    assert "Beträge in Mio EUR" in lines
    assert "Marktrisiko 123 456" not in lines


def test_infer_scale_from_page_caption_prefers_explicit_scale_not_plain_eur():
    page = "\n".join(
        [
            "Überschrift",
            "Angaben in TEUR",
            "Tabelle 1",
            "Marktrisiko 1 000",
        ]
    )
    hit = infer_scale_from_page_caption(page)
    assert hit is not None
    assert hit.scale == 1000.0
    assert hit.unit == "EUR"
    assert "TEUR" in (hit.label or "")


def test_infer_scale_from_page_caption_returns_none_for_plain_eur_only():
    page = "\n".join(
        [
            "Überschrift",
            "Alle Beträge in EUR",
            "Marktrisiko 1 000",
        ]
    )
    hit = infer_scale_from_page_caption(page)
    # function spec: caption should only return explicit TEUR/Tsd/Mio/Mrd
    assert hit is None


def test_infer_scale_near_source_finds_scale_token_nearby():
    source = "Solvabilitätskapitalanforderung 1 312 850"
    page = "\n".join(
        [
            "bla bla",
            "Angaben in TEUR",  # near source within window
            "noch mehr bla",
            source,
            "Diversifikationseffekt -885 282",
        ]
    )
    hit = infer_scale_near_source(page, source, window_chars=500)
    assert hit is not None
    assert hit.scale == 1000.0
    assert hit.unit == "EUR"
    assert "TEUR" in (hit.label or "")


def test_infer_scale_near_source_eur_only_returns_scale_1():
    source = "Solvabilitätskapitalanforderung 1 312 850"
    page = "\n".join(
        [
            "bla bla",
            source,
            "Einheit: EUR",
        ]
    )
    hit = infer_scale_near_source(page, source, window_chars=500)
    assert hit is not None
    assert hit.scale == 1.0
    assert hit.unit == "EUR"
    assert hit.label == "EUR"


def test_infer_scale_near_source_percent_returns_unit_percent():
    source = "Solvabilitätsquote 391"
    page = "\n".join(
        [
            source,
            "Solvabilitätsquote 391%",
        ]
    )
    hit = infer_scale_near_source(page, source, window_chars=500)
    assert hit is not None
    assert hit.scale is None
    assert hit.unit == "%"
    assert hit.label == "%"


def test_infer_scale_near_source_requires_locatable_source_text():
    # Source not found and fingerprint too short -> None
    hit = infer_scale_near_source("some page", "short", window_chars=200)
    assert hit is None


def test_infer_scale_precedence_row_beats_nearby_and_caption():
    # Row has EUR only -> scale=1 from row should win even if page says TEUR
    source = "Solvabilitätskapitalanforderung 1 312 850 EUR"
    page = "\n".join(
        [
            "Angaben in TEUR",
            "irgendwas",
            source,
        ]
    )
    hit = infer_scale(source_text=source, page_text=page, typical_scale=1000.0)
    assert hit.scale == 1.0
    assert hit.unit == "EUR"
    assert hit.label == "EUR"


def test_infer_scale_precedence_nearby_beats_caption():
    # No row signal, but nearby has Mio, caption has TEUR -> nearby should win
    source = "Eigenmittel insgesamt 12,5"
    page = "\n".join(
        [
            "Angaben in TEUR",  # caption signal
            "bla bla bla bla bla bla bla bla bla bla",
            source,
            "Einheit: Mio EUR",  # nearby signal
        ]
    )
    hit = infer_scale(source_text=source, page_text=page, typical_scale=1000.0)
    assert hit.scale == 1_000_000.0
    assert hit.unit == "EUR"
    assert hit.label == "Mio EUR"


def test_infer_scale_caption_used_when_no_row_and_no_nearby():
    source = "Eigenmittel insgesamt 12,5"
    page = "\n".join(
        [
            "Angaben in Mio EUR",
            "bla bla",
            "irgendwo anders",
            "nicht der source text exakt",
        ]
    )
    hit = infer_scale(source_text=source, page_text=page, typical_scale=1000.0)
    assert hit.scale == 1_000_000.0
    assert hit.unit == "EUR"
    assert "Mio" in (hit.label or "")


def test_infer_scale_default_used_when_no_signals():
    hit = infer_scale(source_text=None, page_text=None, typical_scale=1000.0)
    assert hit.scale == 1000.0
    assert hit.unit == "EUR"
    assert hit.label == "default"


def test_infer_scale_unknown_when_no_signals_and_no_default():
    hit = infer_scale(source_text=None, page_text=None, typical_scale=None)
    assert hit.scale is None
    assert hit.unit is None
    assert hit.label is None


def test_whitespace_variants_in_scale_token_detection():
    # Make sure NBSP/NNBSP/THIN are normalized and still match tokens
    source = f"Angaben in{NBSP}TEUR"
    hit = infer_scale_from_source_text(source)
    assert hit is not None
    assert hit.scale == 1000.0

    source2 = f"Angaben in{NNBSP}Mio{THIN}EUR"
    hit2 = infer_scale_from_source_text(source2)
    assert hit2 is not None
    assert hit2.scale == 1_000_000.0
    assert hit2.unit == "EUR"
