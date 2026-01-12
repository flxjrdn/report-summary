from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from sfcr.extract.schema import ScaleSource

NBSP = "\u00a0"
NNBSP = "\u202f"
THIN = "\u2009"


def _norm_ws(s: str) -> str:
    return (
        s.replace(NBSP, " ")
        .replace(NNBSP, " ")
        .replace(THIN, " ")
        .replace("\u00ad", "")  # soft hyphen
    )


# --- tokens ------------------------------------------------------------------

# Prefer explicit scale tokens over plain "EUR"
_SCALE_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"\bteur\b", re.IGNORECASE), 1_000.0, "TEUR"),
    (re.compile(r"\btsd\.?\b", re.IGNORECASE), 1_000.0, "Tsd"),
    (re.compile(r"\btausend\b", re.IGNORECASE), 1_000.0, "Tausend"),
    (re.compile(r"\bmio\.?\b|\bmillion(en)?\b", re.IGNORECASE), 1_000_000.0, "Mio"),
    (
        re.compile(r"\bmrd\.?\b|\bmilliard(en)?\b", re.IGNORECASE),
        1_000_000_000.0,
        "Mrd",
    ),
]

_EUR_PAT = re.compile(r"\bEUR\b|€", re.IGNORECASE)
_PCT_PAT = re.compile(r"%")

# Caption-ish cues that often carry the scale for a whole table
_CAPTION_CUES = (
    "angaben in",
    "in teur",
    "in tsd",
    "in mio",
    "in mrd",
    "beträge in",
    "beträge",
)


@dataclass(frozen=True)
class ScaleHit:
    scale: Optional[float]  # 1000/1e6/1e9 or None
    unit: Optional[str]  # "EUR" | "%" | None
    source: Optional[ScaleSource]  # "row"|"nearby"|"caption"|"default"|None
    label: Optional[str]  # e.g. "TEUR", "Tsd €", ...
    evidence_line: Optional[str]  # the line where we found it (debugging)


def _detect_unit(text: str) -> Optional[str]:
    t = _norm_ws(text)
    if _PCT_PAT.search(t):
        return "%"
    if _EUR_PAT.search(t):
        return "EUR"
    return None


def _detect_scale_token(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Return (scale, label) if TEUR/Tsd/Mio/Mrd present; else (None, None).
    """
    t = _norm_ws(text)
    for pat, factor, label in _SCALE_PATTERNS:
        if pat.search(t):
            # Add "EUR/€" to label if present
            if _EUR_PAT.search(t):
                suffix = "€" if "€" in t else "EUR"
                return factor, f"{label} {suffix}"
            return factor, label
    return None, None


def _detect_scale_token_closest(
    text: str, *, anchor: int
) -> Tuple[Optional[float], Optional[str]]:
    """
    Find the scale token (TEUR/Tsd/Mio/Mrd) closest to `anchor` in `text`.
    Returns (scale, label).
    """
    t = _norm_ws(text)

    best = None  # (distance, factor, label, match_start)
    for pat, factor, label in _SCALE_PATTERNS:
        for m in pat.finditer(t):
            dist = abs(m.start() - anchor)
            if best is None or dist < best[0]:
                best = (dist, factor, label, m.start())

    if best is None:
        return None, None

    _, factor, base_label, _ = best

    # Add "EUR/€" to label if present anywhere in the same text window
    if _EUR_PAT.search(t):
        suffix = "€" if "€" in t else "EUR"
        return factor, f"{base_label} {suffix}"
    return factor, base_label


def infer_scale_from_source_text(source_text: Optional[str]) -> Optional[ScaleHit]:
    """
    Strongest signal: token in same snippet/row as the extracted value.
    """
    if not source_text:
        return None
    unit = _detect_unit(source_text)
    scale, label = _detect_scale_token(source_text)
    if scale is not None:
        return ScaleHit(
            scale=scale,
            unit=unit or "EUR",
            source="row",
            label=label,
            evidence_line=source_text,
        )
    # If we only saw EUR and nothing else, scale is 1 (but we often treat as scale=None and apply 1 later)
    if unit == "EUR":
        return ScaleHit(
            scale=1.0, unit="EUR", source="row", label="EUR", evidence_line=source_text
        )
    if unit == "%":
        return ScaleHit(
            scale=None, unit="%", source="row", label="%", evidence_line=source_text
        )
    return None


def _candidate_caption_lines(page_text: str, *, max_lines: int = 80) -> list[str]:
    """
    Extract likely caption/header lines from a page:
    - take first N lines (tables often start near top)
    - keep lines that contain caption cues or scale tokens
    """
    t = _norm_ws(page_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    lines = lines[:max_lines]
    out: list[str] = []
    for ln in lines:
        low = ln.lower()
        if any(cue in low for cue in _CAPTION_CUES):
            out.append(ln)
            continue
        # also keep explicit scale tokens
        sc, _ = _detect_scale_token(ln)
        if sc is not None:
            out.append(ln)
            continue
    return out


def infer_scale_from_page_caption(page_text: str) -> Optional[ScaleHit]:
    """
    Weaker signal: scale token in caption/header zone.
    Only returns TEUR/Tsd/Mio/Mrd (not plain EUR).
    """
    for ln in _candidate_caption_lines(page_text):
        sc, label = _detect_scale_token(ln)
        if sc is None:
            continue
        unit = _detect_unit(ln) or "EUR"
        return ScaleHit(
            scale=sc, unit=unit, source="caption", label=label, evidence_line=ln
        )
    return None


def infer_scale_near_source(
    page_text: str,
    source_text: Optional[str],
    *,
    window_chars: int = 800,
) -> Optional[ScaleHit]:
    """
    Medium signal: find a best-effort match of the source_text in the page and scan nearby for tokens.
    If we can’t locate source_text reliably, return None (don’t guess globally).
    """
    if not source_text:
        return None

    page = _norm_ws(page_text)
    src = _norm_ws(source_text).strip()
    if not src:
        return None

    # Try to find a substring; if not found, try shortened fingerprint (first ~60 chars)
    idx = page.find(src)
    if idx < 0:
        fp = src[:60]
        if len(fp) < 20:
            return None
        idx = page.find(fp)
        if idx < 0:
            return None

    left = max(0, idx - window_chars)
    right = min(len(page), idx + len(src) + window_chars)
    near = page[left:right]

    # Prefer explicit scale tokens near the source
    anchor = idx - left  # source position within 'near'

    sc, label = _detect_scale_token_closest(near, anchor=anchor)
    if sc is not None:
        unit = _detect_unit(near) or "EUR"
        lines = [ln.strip() for ln in near.splitlines() if ln.strip()]
        ev = lines[0] if lines else None
        return ScaleHit(
            scale=sc, unit=unit, source="nearby", label=label, evidence_line=ev
        )

    # If we only see EUR near source, treat as scale=1
    if _EUR_PAT.search(near):
        unit = "EUR"
        return ScaleHit(
            scale=1.0, unit=unit, source="nearby", label="EUR", evidence_line=None
        )

    if _PCT_PAT.search(near):
        return ScaleHit(
            scale=None, unit="%", source="nearby", label="%", evidence_line=None
        )

    return None


def infer_scale(
    *,
    source_text: Optional[str],
    page_text: Optional[str],
    typical_scale: Optional[float],
) -> ScaleHit:
    """
    Combine signals in precedence order:
      1) row/source_text
      2) near source on same page
      3) caption/header on same page
      4) default typical_scale
      5) unknown
    """
    # 1) row
    hit = infer_scale_from_source_text(source_text)
    if hit is not None:
        return hit

    # 2) nearby & 3) caption
    if page_text:
        hit = infer_scale_near_source(page_text, source_text)
        if hit is not None:
            return hit

        hit = infer_scale_from_page_caption(page_text)
        if hit is not None:
            return hit

    # 4) default
    if typical_scale is not None:
        # If typical_scale is 1, you may prefer to return scale=1 as default; up to you.
        return ScaleHit(
            scale=float(typical_scale),
            unit="EUR",
            source="default",
            label="default",
            evidence_line=None,
        )

    # 5) unknown
    return ScaleHit(scale=None, unit=None, source=None, label=None, evidence_line=None)
