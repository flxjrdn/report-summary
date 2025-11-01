from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .schema import ExtractionLLM, VerifiedExtraction

DE_NBSP = "\u00a0"
LEADER_CHARS = r"\.\u2026\u00B7\u2219\u22EF\u2024\u2027\uf020·•⋯∙"

# Core numeric regex (German thousands '.' and decimal ',')
NUM_CORE = re.compile(
    r"""
    (?P<neg>\()?\s*
    (?P<int>(?:\d{1,3}(?:[ .\u202F\u2009]\d{3})+|\d+))
    (?P<dec>,\d+)?\s*
    (?P<pct>%){0,1}
    \)?                      # allow closing ')' for negative
    """,
    re.VERBOSE,
)

# Simple scale phrases (order matters: more specific first)
SCALE_PATTERNS = [
    (re.compile(r"\bMio\b|\bMillion(en)?\b", re.I), 1e6),
    (re.compile(r"\bMrd\b|\bMilliarde(n)?\b", re.I), 1e9),
    (re.compile(r"\bTEUR\b|\bTsd\b|\bTausend\b", re.I), 1e3),
    (re.compile(r"\bEUR\b|\bEuro\b", re.I), 1.0),
]


@dataclass
class ParsedNumber:
    value: float
    is_percent: bool
    is_negative: bool


def _to_float_de(intpart: str, decpart: Optional[str]) -> float:
    # remove thousands separators and swap decimal comma
    i = (
        intpart.replace(".", "")
        .replace(" ", "")
        .replace(DE_NBSP, "")
        .replace("\u202f", "")
        .replace("\u2009", "")
    )
    d = decpart.replace(",", ".") if decpart else ""
    s = f"{i}{d}"
    return float(s)


def parse_number_de(text: str) -> Optional[ParsedNumber]:
    """
    Parse the first German-formatted number in text.
    Supports negatives in parentheses and %.
    """
    if not text:
        return None
    # Normalize spaces: replace NBSP, narrow no-break space, and thin space with normal space
    text = text.replace(DE_NBSP, " ").replace("\u202f", " ").replace("\u2009", " ")
    # Collapse multiple spaces to a single space
    text = re.sub(r"\s+", " ", text)
    m = NUM_CORE.search(text)
    if not m:
        return None
    val = _to_float_de(m.group("int"), m.group("dec"))
    neg = bool(m.group("neg"))
    pct = bool(m.group("pct"))
    if neg:
        val = -val
    return ParsedNumber(value=val, is_percent=pct, is_negative=neg)


def infer_scale(context_tokens: Iterable[Tuple[str, str]]) -> Tuple[float, str]:
    """
    Inspect nearby tokens (caption/header/row text) to infer scale.
    Returns (scale, source).
    Precedence: caption > column > row > nearby; pass tokens in that order.
    """
    for token, source in context_tokens:
        if not token:
            continue
        for pat, factor in SCALE_PATTERNS:
            if pat.search(token):
                # special-case: "in TEUR" / "Angaben in TEUR" etc.
                if re.search(r"in\s+(TEUR|Tsd|Tausend)", token, re.I):
                    return (1e3, source)
                if re.search(r"in\s+Mio", token, re.I):
                    return (1e6, source)
                return (factor, source)
    # unknown → leave undecided; caller can fall back to typical scale
    return (None, "unknown")  # type: ignore


def _coerce_scale(x: Any) -> Optional[float]:
    """
    Accepts float/int/str like '1e3', '1000', or textual shorthands; returns float or None.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        # common shorthands if they sneak in from prompts/yaml
        if s in {"teur", "tsd", "tausend", "1k"}:
            return 1e3
        if s in {"mio", "million", "1m"}:
            return 1e6
        if s in {"mrd", "milliarde", "1b"}:
            return 1e9
        try:
            return float(s)  # handles "1000", "1e3", "1e6"
        except ValueError:
            return None
    return None


def apply_scale(value: float, scale: Optional[float]) -> float:
    sc = _coerce_scale(scale)
    if sc is None:
        sc = 1.0
    return value * sc


# ---------------- Verification ----------------


def verify_extraction(
    doc_id: str,
    extr: ExtractionLLM,
    *,
    typical_scale: Optional[float],
    context_scale_tokens: Iterable[Tuple[str, str]],  # list of (text, source_tag)
    ratio_check: Optional[
        Tuple[float, float]
    ] = None,  # (expected, tolerance_abs) OR provide via cross_checks()
) -> VerifiedExtraction:
    """
    Deterministically re-parse source_text and apply scale.
    """
    # Default not verified
    base = VerifiedExtraction(
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
    )

    # Status gates
    if extr.status != "ok" or extr.source_text is None:
        return base

    # 1) Deterministic parse of source_text
    p = parse_number_de(extr.source_text)
    if p is None:
        base.verifier_notes = "parse_failed"
        return base

    # 2) Unit check
    if p.is_percent and extr.unit != "%":
        base.verifier_notes = "unit_mismatch_percent"
        return base
    if (not p.is_percent) and extr.unit == "%":
        base.verifier_notes = "unit_mismatch_number_vs_percent"
        return base

    # 3) Scale resolution
    scale_inferred, scale_src = infer_scale(context_scale_tokens)
    scale_final = extr.scale or scale_inferred or typical_scale or 1.0

    # 4) Canonicalize
    value_canon = apply_scale(p.value, scale_final)

    # 5) Confidence
    conf = 0.35  # base if parse ok
    if scale_inferred:
        conf += 0.2
    if extr.scale and extr.scale == scale_inferred:
        conf += 0.1
    if extr.evidence:
        conf += 0.1
    conf = min(1.0, conf)

    # 6) Optional ratio check (caller supplies)
    notes = []
    if ratio_check and extr.unit == "%":
        expected, tol = ratio_check
        if abs(value_canon - expected) <= tol:
            conf = min(1.0, conf + 0.15)
        else:
            notes.append(f"ratio_mismatch expected={expected} got={value_canon}")

    return VerifiedExtraction(
        **{
            **base.model_dump(),
            "verified": True,
            "value_canonical": value_canon,
            "confidence": conf,
            "scale_applied": float(scale_final),
            "scale_source": scale_src
            if scale_inferred
            else ("provided" if extr.scale else "typical"),
            "verifier_notes": ";".join(notes) if notes else None,
        }
    )


# ---------------- Cross-checks ----------------


def cross_checks(values: Dict[str, float]) -> Dict[str, Tuple[bool, str]]:
    """
    Simple internal arithmetic checks.
    values: canonical EUR/% map like {"eof_total": ..., "eof_t1": ..., "scr_total": ..., "sii_ratio_pct": ...}
    Returns: check_name -> (pass?, detail)
    """
    out: Dict[str, Tuple[bool, str]] = {}

    def ok(name: str, cond: bool, msg: str):
        out[name] = (cond, msg)

    # A) EOF_total ≈ T1 + T2 (within ± max(1 unit of lowest scale, 0.1% of total))
    if all(k in values for k in ("eof_total", "eof_t1", "eof_t2")):
        total = values["eof_total"]
        sum_t = values["eof_t1"] + values["eof_t2"]
        tol = max(500.0, 0.001 * abs(total))  # 500 EUR if TEUR-typical; tune if needed
        ok(
            "sum_eof",
            abs(total - sum_t) <= tol,
            f"tot={total:.2f} sum={sum_t:.2f} tol={tol:.2f}",
        )

    # B) SII ratio ≈ 100 * EOF_total / SCR
    if (
        all(k in values for k in ("eof_total", "scr_total", "sii_ratio_pct"))
        and values["scr_total"]
    ):
        expected = 100.0 * values["eof_total"] / values["scr_total"]
        got = values["sii_ratio_pct"]
        tol_pp = 0.2  # percentage points
        ok(
            "sii_ratio",
            abs(got - expected) <= tol_pp,
            f"exp={expected:.2f} got={got:.2f} tol={tol_pp:.2f}pp",
        )

    # C) MCR/SCR sanity (0 < MCR <= SCR)
    if all(k in values for k in ("mcr_total", "scr_total")):
        ok(
            "mcr_le_scr",
            values["mcr_total"] <= values["scr_total"],
            "MCR must not exceed SCR",
        )

    return out
