from sfcr.utils.textnorm import normalize_hyphenation


def test_dehyphenation_example():
    raw = "Mindestkapitalanfor-\nderung beträgt 123 456 TEUR"
    out = normalize_hyphenation(raw)
    assert "Mindestkapitalanforderung" in out


def test_soft_hyphen_removed():
    raw = "Solvenzkapitalan\u00adforderung"
    out = normalize_hyphenation(raw)
    assert out == "Solvenzkapitalanforderung"


def test_hyphen_kept_inside_line():
    raw = "Risikomanagement-System und -Prozesse"
    out = normalize_hyphenation(raw)
    assert out == raw  # should not touch real hyphens


def test_midword_newline_no_hyphen():
    raw = "versicherungstech-\n nische Rückstellungen"
    out = normalize_hyphenation(raw)
    assert "versicherungstechnische" in out


def test_spaces_and_nbsp():
    raw = "Mindest\u00a0kapitalanfor- \n   derung"
    out = normalize_hyphenation(raw)
    assert "Mindestkapitalanforderung" in out
