from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .scale_detect import infer_scale
from .schema import ExtractionLLM, ScaleSource, VerifiedExtraction

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


ALLOWED_SCALES = {None, 1.0, 1000.0, 1_000_000.0, 1_000_000_000.0}

# Allowed thousands separators inside numbers
_SEP_CHARS = r"[ .\u00a0\u202f\u2009]"

# German number core (no sign here; parentheses handled outside)
_NUM_DE = rf"(?:\d{{1,3}}(?:{_SEP_CHARS}\d{{3}})+|\d+)"

# Optional decimal part
_NUM_DE_FULL = rf"(?:{_NUM_DE}(?:,\d+)?)"

_CURRENT_PREV_RE = re.compile(
    rf"(?P<curr>{_NUM_DE_FULL})\s*\(\s*(?P<prev>{_NUM_DE_FULL})\s*\)",
    re.VERBOSE,
)


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


def extract_numbers_de(text: str) -> List[float]:
    """
    Extract ALL German-formatted numbers from text.

    Returns a list of floats (unscaled), in textual order.
    Percent signs are ignored here; unit checks happen elsewhere.
    """
    if not text:
        return []

    # Normalize spaces exactly like parse_number_de
    t = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    t = re.sub(r"\s+", " ", t)

    values: List[float] = []

    # NUM_CORE already exists in your module; reuse it
    for m in NUM_CORE.finditer(t):
        try:
            val = _to_float_de(m.group("int"), m.group("dec"))
            if m.group("neg"):
                val = -val
            values.append(val)
        except Exception:
            continue

    return values


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


def _parse_de_number(num_str: str) -> float:
    # same logic you already use elsewhere
    s = num_str.strip()
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "").replace(" ", "")
    s = s.replace(",", ".")
    return float(s)


def extract_current_prev_pair(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    t = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    t = re.sub(r"\s+", " ", t)

    m = _CURRENT_PREV_RE.search(t)
    if not m:
        return None

    try:
        curr = _parse_de_number(m.group("curr"))
        prev = _parse_de_number(m.group("prev"))
        return curr, prev
    except Exception:
        return None


# ---------------- Verification ----------------
def verify_extraction(
    *,
    doc_id: str,
    extr: ExtractionLLM,
    typical_scale: Optional[float],
    page_text_for_scale: str | None = None,
    ratio_check: Optional[Tuple[float, float]] = None,
) -> VerifiedExtraction:
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
        verifier_notes=None,
    )

    # Gate
    if extr.status != "ok" or extr.value_unscaled is None:
        base.verifier_notes = "no_value_or_not_ok"
        return base

    notes: list[str] = []

    # --- Unit sanity check (weakly) -----------------------------------------
    # Only a heuristic: we look at source_text if present
    if extr.source_text:
        st = extr.source_text
        has_pct = "%" in st
        has_eur = ("EUR" in st) or ("€" in st)

        if has_pct and extr.unit != "%":
            notes.append("unit_mismatch_snippet_percent")
        if has_eur and extr.unit == "%":
            notes.append("unit_mismatch_snippet_eur")

    # --- Scale resolution ----------------------------------------------------
    hit = infer_scale(
        source_text=extr.source_text,
        page_text=page_text_for_scale,
        typical_scale=typical_scale,
    )

    # Prefer model-provided scale only if allowed; otherwise ignore it
    model_scale = extr.scale if extr.scale in ALLOWED_SCALES else None

    scale_final: float | None = None
    scale_source: ScaleSource | None = None

    # 1) model scale wins (if allowed)
    if model_scale is not None:
        scale_final = float(model_scale)
        scale_source = extr.scale_source  # may be None, that's okay

    # 2) otherwise use inferred scale
    if scale_final is None and hit.scale is not None:
        scale_final = float(hit.scale)
        scale_source = hit.source

    # 3) EUR amounts: missing scale => assume 1.0, record as default
    if scale_final is None and extr.unit == "EUR":
        scale_final = 1.0
        scale_source = scale_source or "default"

    # --- Canonical value from LLM value_unscaled ----------------------------
    value_canon: float | None = None

    if extr.value_unscaled is not None:
        if extr.unit == "%":
            # For percent, scale is irrelevant: canonical == unscaled
            value_canon = float(extr.value_unscaled)
        else:
            # For EUR (and other numeric amounts), apply scale if known
            value_canon = (
                float(extr.value_unscaled) * float(scale_final)
                if scale_final is not None
                else None
            )

    if value_canon is not None:
        value_canon = round(value_canon, 2)

    # --- Determine unit -----------------------------------------------------
    unit = extr.unit if value_canon is not None else None

    # --- Value consistency check against snippet (non-destructive) ----------
    # This should NOT pick a different value; only checks consistency.
    if extr.source_text:
        pair = extract_current_prev_pair(
            extr.source_text
        )  # returns (curr, prev) or None
        candidates = []
        if pair:
            curr, prev = pair
            candidates.extend([curr, prev])
        candidates.extend(extract_numbers_de(extr.source_text))  # list[float]

        # Check if model value is among candidates (tolerance helps decimals)
        ok_match = any(
            abs(float(extr.value_unscaled) - c) <= 1e-6 * max(1.0, abs(c))
            for c in candidates
        )

        if not ok_match:
            # Not fatal (LLM might have normalized formatting), but should reduce confidence
            notes.append("value_not_found_in_source_text")

        # If we detect X(Y) and model picked Y (prev) instead of X, flag it
        if pair and abs(float(extr.value_unscaled) - pair[1]) <= 1e-6 * max(
            1.0, abs(pair[1])
        ):
            notes.append("looks_like_prev_year_value")

    # --- Confidence ---------------------------------------------------------
    # Start low; add points for strong evidence.
    conf = 0.25

    # snippet present is good
    if extr.source_text:
        conf += 0.15

    # evidence page present is good
    if extr.evidence:
        conf += 0.10

    # scale quality
    if scale_source in ("row",):
        conf += 0.30
    elif scale_source in ("nearby",):
        conf += 0.20
    elif scale_source in ("caption",):
        conf += 0.15
    elif scale_source in ("default",):
        conf += 0.05

    # penalize issues
    if any(n.startswith("unit_mismatch") for n in notes):
        conf -= 0.25
    if "value_not_found_in_source_text" in notes:
        conf -= 0.35
    if "looks_like_prev_year_value" in notes:
        conf -= 0.25

    conf = round(conf, 2)
    conf = max(0.0, min(1.0, conf))

    # Optional ratio check for percent fields
    if ratio_check and extr.unit == "%":
        expected, tol = ratio_check
        if value_canon is not None and abs(value_canon - expected) <= tol:
            conf = min(1.0, conf + 0.15)
        else:
            notes.append(f"ratio_mismatch expected={expected} got={value_canon}")

    # Decide verified: in your system "verified" really means "passed basic checks"
    blocking = any(
        n.startswith(block_text)
        for block_text in (
            "value_not_found_in_source_text",
            "looks_like_prev_year_value",
            "unit_mismatch",
        )
        for n in notes
    )
    verified = conf >= 0.50 and not blocking and extr.unit is not None

    if extr.unit == "%":
        scale_final = None
        scale_source = None

    return VerifiedExtraction(
        **{
            **base.model_dump(),
            "verified": verified,
            "value_canonical": value_canon,
            "unit": unit,
            "confidence": conf,
            "scale_applied": float(scale_final) if scale_final is not None else None,
            "scale_source": scale_source,
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
