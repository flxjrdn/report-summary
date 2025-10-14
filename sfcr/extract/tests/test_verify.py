from sfcr.extract.verify import apply_scale, cross_checks, infer_scale, parse_number_de


def test_parse_basic_de():
    p = parse_number_de("Solvenzkapitalanforderung (SCR) in TEUR: 123.456")
    assert p and p.value == 123_456 and not p.is_percent


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
