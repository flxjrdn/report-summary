from sfcr.extract.ollama_client import _SNIPPET_HEX_RE, OllamaLLM, _mk_snippet_hash


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


def test_ensure_defaults_generates_snippet_hash_when_missing():
    llm = OllamaLLM(enable_cache=False, debug_raw=False)

    payload = {
        "field_id": "scr_total",
        "value": 1312850,  # numeric → status should infer to "ok"
        "unit": "EUR",
        "currency": "EUR",
        "evidence": [{"page": 68, "ref": "Solvabilitätskapitalanforderung"}],
        "source_text": "Solvabilitätskapitalanforderung 1 312 850 Tsd €",
        # snippet_hash intentionally missing
    }

    out = llm._ensure_defaults(payload, field_id="scr_total")
    assert out["status"] == "ok"
    assert isinstance(out["evidence"], list) and len(out["evidence"]) == 1
    sh = out["evidence"][0]["snippet_hash"]
    assert isinstance(sh, str) and hex_ok(sh), "should auto-generate a hex snippet_hash"


def test_ensure_defaults_replaces_invalid_snippet_hash_and_infers_not_found():
    llm = OllamaLLM(enable_cache=False, debug_raw=False)

    payload = {
        "field_id": "scr_total",
        "status": None,  # not provided by the model
        "value": None,  # → should infer "not_found"
        "unit": None,
        "evidence": [
            {
                "page": 68,
                "ref": "Solvabilitätskapitalanforderung",
                "snippet_hash": "Solvabilitatskapitalanforderung_2024",  # invalid (non-hex)
            }
        ],
        "source_text": None,
    }

    out = llm._ensure_defaults(payload, field_id="scr_total")
    assert out["status"] == "not_found"
    ev = out["evidence"][0]
    assert "snippet_hash" in ev and hex_ok(ev["snippet_hash"])
    assert ev["snippet_hash"] != "Solvabilitatskapitalanforderung_2024"


def test_ensure_defaults_normalizes_non_list_evidence():
    llm = OllamaLLM(enable_cache=False, debug_raw=False)

    payload = {
        "field_id": "tech_provisions_total",
        "value": None,
        "evidence": "free-text-evidence",  # invalid shape → should normalize to list[dict]
        "source_text": "versicherungstechnische Rückstellungen ...",
    }

    out = llm._ensure_defaults(payload, field_id="tech_provisions_total")
    assert isinstance(out["evidence"], list) and len(out["evidence"]) == 1
    ev = out["evidence"][0]
    assert "page" in ev and "ref" in ev and "snippet_hash" in ev
    assert hex_ok(ev["snippet_hash"])
