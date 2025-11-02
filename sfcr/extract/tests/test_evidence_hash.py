from sfcr.extract.ollama_client import _SNIPPET_HEX_RE, _mk_snippet_hash


def hex_ok(s: str) -> bool:
    return bool(_SNIPPET_HEX_RE.fullmatch(s))


def test__mk_snippet_hash_deterministic_and_changes():
    h1 = _mk_snippet_hash("some snippet", "Table 17.01", "scr_total", 68)
    h2 = _mk_snippet_hash("some snippet", "Table 17.01", "scr_total", 68)
    h3 = _mk_snippet_hash("some snippet", "Table 17.01", "scr_total", 69)

    assert hex_ok(h1)
    assert len(h1) == 16
    assert h1 == h2, "same inputs must produce same hash"
    assert h1 != h3, "changing page should change the hash"
