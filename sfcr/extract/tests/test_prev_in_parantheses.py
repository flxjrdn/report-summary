# Pytest suite for the deterministic fallback that parses patterns like:
#   "<current> (<previous>) [TEUR|Tsd|Mio|Mrd] [EUR|€]"
# and returns a dict with keys: value, prev_value, unit, scale, scale_source, source_text.

import pytest

from sfcr.extract.ollama_client import _extract_current_prev_pair


@pytest.mark.parametrize(
    "text, exp_value, exp_prev, exp_scale, exp_scale_src, exp_unit",
    [
        # Space-separated thousands + TEUR
        (
            "die Mindestkapitalanforderung beträgt 123 456 (133 333) TEUR",
            123_456,
            133_333,
            1e3,
            "TEUR",
            "EUR",
        ),
        # Dot-separated thousands + EUR word
        (
            "Solvabilitätskapitalanforderung 1.234.567 (1.111.111) EUR",
            1_234_567,
            1_111_111,
            None,
            "EUR",
            "EUR",
        ),
        # NBSP + thin space + Tsd €
        (
            "... beträgt 1\u00a0234\u202f567 (1\u00a0111\u202f000) Tsd €",
            1_234_567,
            1_111_000,
            1e3,
            "Tsd €",
            "EUR",
        ),
        # Millions with comma decimals + EUR
        ("Der Betrag beträgt 12,5 (11,0) Mio EUR", 12.5, 11.0, 1e6, "Mio EUR", "EUR"),
        # Billions (Mrd) + Euro sign
        ("Die Position beträgt 3 (2) Mrd €", 3.0, 2.0, 1e9, "Mrd €", "EUR"),
    ],
)
def test_extract_current_prev_pair_positive(
    text, exp_value, exp_prev, exp_scale, exp_scale_src, exp_unit
):
    result = _extract_current_prev_pair(text)
    assert result is not None, "Expected a parsed result, got None"

    # required keys
    for key in ("value", "prev_value", "unit", "scale", "scale_source", "source_text"):
        assert key in result, f"Missing key in result: {key}"

    # numerics (allow float tolerance)
    if isinstance(exp_value, float):
        assert result["value"] == pytest.approx(exp_value)
    else:
        assert result["value"] == exp_value

    if isinstance(exp_prev, float):
        assert result["prev_value"] == pytest.approx(exp_prev)
    else:
        assert result["prev_value"] == exp_prev

    # unit / scale meta
    assert result["unit"] == exp_unit
    assert result["scale"] == exp_scale
    assert result["scale_source"] == exp_scale_src

    # sanity on source_text (should echo the matched fragment)
    assert isinstance(result["source_text"], str) and len(result["source_text"]) > 0
    # At minimum, current/prev numerals should appear in the snippet
    # (spaces/dots may be normalized; so just check a subset)
    assert "(" in result["source_text"] and ")" in result["source_text"]


def test_extract_current_prev_pair_no_match_returns_none():
    text = "Dies ist ein Text ohne passende Zahlenklammern – nichts in Klammern."
    assert _extract_current_prev_pair(text) is None
