import pytest

from sfcr.extract.verify import (
    ParsedNumber,
    apply_scale,
    cross_checks,
    infer_scale,
    parse_number_de,
)


def test_parse_basic_de():
    p = parse_number_de("Solvenzkapitalanforderung (SCR) in TEUR: 123.456")
    assert p and p.value == 123_456 and not p.is_percent


@pytest.mark.parametrize(
    "text,expected",
    [
        # dot-separated thousands
        ("1.234", ParsedNumber(value=1234.0, is_percent=False, is_negative=False)),
        (
            "12.345.678",
            ParsedNumber(value=12345678.0, is_percent=False, is_negative=False),
        ),
        # space-separated thousands (regular space)
        ("1 234", ParsedNumber(value=1234.0, is_percent=False, is_negative=False)),
        (
            "12 345 678",
            ParsedNumber(value=12345678.0, is_percent=False, is_negative=False),
        ),
        # space-separated thousands (NBSP)
        ("1\u00a0234", ParsedNumber(value=1234.0, is_percent=False, is_negative=False)),
        (
            "12\u00a0345\u00a0678",
            ParsedNumber(value=12345678.0, is_percent=False, is_negative=False),
        ),
        # space-separated thousands (narrow no-break space U+202F)
        ("1\u202f234", ParsedNumber(value=1234.0, is_percent=False, is_negative=False)),
        (
            "12\u202f345\u202f678",
            ParsedNumber(value=12345678.0, is_percent=False, is_negative=False),
        ),
        # space-separated thousands (thin space U+2009)
        ("1\u2009234", ParsedNumber(value=1234.0, is_percent=False, is_negative=False)),
        (
            "12\u2009345\u2009678",
            ParsedNumber(value=12345678.0, is_percent=False, is_negative=False),
        ),
        # decimals with comma
        ("1.234,56", ParsedNumber(value=1234.56, is_percent=False, is_negative=False)),
        (
            "12 345,678",
            ParsedNumber(value=12345.678, is_percent=False, is_negative=False),
        ),
        # percent values (no space)
        ("12,5%", ParsedNumber(value=12.5, is_percent=True, is_negative=False)),
        ("1.234%", ParsedNumber(value=1234.0, is_percent=True, is_negative=False)),
        # percent values (with space)
        ("12,5 %", ParsedNumber(value=12.5, is_percent=True, is_negative=False)),
        ("1.234 %", ParsedNumber(value=1234.0, is_percent=True, is_negative=False)),
        # negatives in parentheses (no decimals, no percent)
        ("(1.234)", ParsedNumber(value=-1234.0, is_percent=False, is_negative=True)),
        # negatives in parentheses (with decimals)
        (
            "(1.234,56)",
            ParsedNumber(value=-1234.56, is_percent=False, is_negative=True),
        ),
        # negatives in parentheses (with percent)
        ("(12,5%)", ParsedNumber(value=-12.5, is_percent=True, is_negative=True)),
        ("(12,5 %)", ParsedNumber(value=-12.5, is_percent=True, is_negative=True)),
        # large numbers with multiple groups
        (
            "1.234.567,89",
            ParsedNumber(value=1234567.89, is_percent=False, is_negative=False),
        ),
        (
            "(1.234.567,89%)",
            ParsedNumber(value=-1234567.89, is_percent=True, is_negative=True),
        ),
        # non-matches
        ("no number here", None),
        (
            "abc123",
            ParsedNumber(value=123.0, is_percent=False, is_negative=False),
        ),  # parse_number_de parses first number
        ("(abc)", None),
        ("%", None),
        ("(%)", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_number_de(text, expected):
    result = parse_number_de(text)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert abs(result.value - expected.value) < 1e-6
        assert result.is_percent == expected.is_percent
        assert result.is_negative == expected.is_negative


def test_parse_decimal_percent():
    p = parse_number_de("Solvenzquote: 231,4%")
    assert p and p.is_percent and abs(p.value - 231.4) < 1e-9


def test_parse_negative_parentheses():
    p = parse_number_de("(1.234,50)")
    assert p and p.value == -1234.50


def test_infer_scale():
    scale, src = infer_scale(
        [("Angaben in TEUR", "caption"), ("Betrag (EUR)", "column")]
    )
    assert scale == 1e3 and src == "caption"


def test_apply_scale():
    assert apply_scale(123.0, 1e3) == 123_000.0
    assert apply_scale(123.0, None) == 123.0


def test_cross_checks():
    values = {
        "eof_t1": 900_000.0,
        "eof_t2": 100_000.0,
        "eof_total": 1_000_000.0,
        "scr_total": 500_000.0,
        "sii_ratio_pct": 200.0,
        "mcr_total": 200_000.0,
    }
    checks = cross_checks(values)
    assert checks["sum_eof"][0] is True
    assert checks["sii_ratio"][0] is True
    assert checks["mcr_le_scr"][0] is True
