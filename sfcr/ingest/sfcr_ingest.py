"""
sfcr_ingest.py

Ingestion pipeline for German SFCR Solo PDFs:
- Detects section boundaries A..E
- Detects subsections A.1, B.2, ...
- Produces structured outputs with confidence & provenance

Author: you
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

# -----------------------------
# Utilities & configuration
# -----------------------------

NBSP = "\u00a0"
SOFT_HYPHEN = "\u00ad"

SECTION_PATTERNS: Dict[str, re.Pattern] = {
    # A–E (German + fallback English variants you will actually encounter)
    "A": re.compile(
        r"^A\.\s*(Geschäftsmodell(?: und Leistung)?|Geschäftstätigkeit(?: und Geschäftsergebnis)?|Geschäftsmodell und Leistung)",
        re.I,
    ),
    "B": re.compile(r"^B\.\s*(Governance[- ]?System|(System der )?Governance)", re.I),
    "C": re.compile(r"^C\.\s*(Risikoprofil)", re.I),
    "D": re.compile(
        r"^D\.\s*(Bewertung f[üu]r Solven[z]?-?zwecke|Bewertung für Solvabilitätszwecke)",
        re.I,
    ),
    "E": re.compile(r"^E\.\s*(Kapitalmanagement)", re.I),
}

SUBSECTION_PATTERN = re.compile(r"^([A-E])\.(\d{1,2})(?:\.(\d{1,2}))?\s+(.+)$", re.I)

TOC_LINE_PATTERN = re.compile(r"^([A-E])\.\s+(.+?)\s+\.{3,}\s+(\d{1,3})$")

DOT_RUN_RE = re.compile(r"^\.*$")  # spans that are only dots
NUM_RE = re.compile(r"^\d{1,4}$")
LETTER_RE = re.compile(r"^[A-E]$", re.I)

LEADER_CHARS = r"\.\u2026\u00B7\u2219\u22EF\u2024\u2027\uf020·•⋯∙"  # ., …, ·, etc.
RIGHT_TOKEN_RE = re.compile(
    rf"""^(?P<title>.*?)
         [\s{LEADER_CHARS}]*      # optional leaders / spaces
         (?P<page>\d{{1,4}})\s*$  # trailing page number
    """,
    re.VERBOSE,
)
LEFT_TOPLEVEL_RE = re.compile(r"^([A-E])\.$", re.I)  # e.g., "A."
LEFT_SUBSECTION_RE = re.compile(r"^([A-E])\.\d", re.I)  # e.g., "A.1", "B.12"
LEADER_CHARS = (
    r"\.\u2026\u00B7\u2219\u22EF\u2024\u2027\uf020·•⋯∙"  # dot leader variants
)
LEFT_LETTER_ONLY_RE = re.compile(r"^([A-E])$", re.I)  # "A"
LEFT_TEIL_RE = re.compile(
    r"^(?:Teil|Abschnitt)\s*([A-E])\.?$", re.I
)  # "Teil A", "Abschnitt B."

HEADER_FOOTER_FREQ_THRESHOLD = 0.6  # if a line appears on >60% pages at very top/bottom -> treat as running header/footer


def normalize_text(s: str) -> str:
    """Basic normalization for heading detection."""
    if not s:
        return s
    s = s.replace(NBSP, " ").replace(SOFT_HYPHEN, "")
    # strip double spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def roughly_top_left(
    bbox: Tuple[float, float, float, float], page_rect: fitz.Rect
) -> float:
    """Heuristic score for being top-left on page (0..0.4 approx)."""
    x0, y0, x1, y1 = bbox
    w, h = page_rect.width, page_rect.height
    score = 0.0
    if y0 < 0.25 * h:
        score += 0.2
    if x0 < 0.35 * w:
        score += 0.2
    return score


def whitespace_gap_above(
    page: fitz.Page, bbox: Tuple[float, float, float, float]
) -> float:
    """
    Crude whitespace signal: larger gap above top of bbox => more likely to be a section heading.
    Returns 0..0.2.
    """
    x0, y0, x1, y1 = bbox
    # crude: distance to nearest span above within 40px vertical radius
    # We use page text (raw) to estimate; keep it cheap
    return 0.1 if y0 < 120 else 0.0  # simple but effective; tune later


def _is_leader_token(txt: str) -> bool:
    if not txt:
        return False
    # pure leaders or a single dot are considered filler
    return all(ch in LEADER_CHARS + " " for ch in txt) or txt == "."


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class HeadingHit:
    letter: str
    page: int
    score: float
    text: str
    bbox: Tuple[float, float, float, float]
    source: str  # 'bookmark' | 'toc' | 'regex'
    details: Dict[str, Any]


@dataclass
class SectionSpan:
    section: str  # 'A'..'E'
    start_page: int
    end_page: int
    confidence: float
    detectors: Dict[
        str, bool
    ]  # {'bookmark': True/False, 'toc': True/False, 'regex': True/False}


@dataclass
class SubsectionSpan:
    section: str  # 'A'
    code: str  # 'A.1' or 'B.2.1'
    title: str
    start_page: int
    end_page: int
    confidence: float


@dataclass
class IngestionResult:
    doc_id: str
    pdf_sha256: Optional[str]
    page_count: int
    sections: List[SectionSpan]
    subsections: List[SubsectionSpan]
    coverage_ratio: float
    issues: List[str]


@dataclass
class TocItem:
    page: int
    title: str
    left_marker: str  # e.g., "A.", "A.1", "Teil A", "" (empty if none)


# -----------------------------
# Core classes
# -----------------------------


class PDFLoader:
    """Light wrapper around PyMuPDF with helpers."""

    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)

    def page_count(self) -> int:
        return len(self.doc)

    def get_page(self, i: int) -> fitz.Page:
        return self.doc[i]

    def get_toc(self) -> List[Tuple[int, str, int]]:
        # [(level, title, page)]
        try:
            return self.doc.get_toc(simple=True) or []
        except Exception:
            return []

    def text_blocks(self, page_index: int) -> List[Dict[str, Any]]:
        page = self.get_page(page_index)
        return page.get_text("dict")["blocks"]

    def rect(self, page_index: int) -> fitz.Rect:
        return self.get_page(page_index).rect


class BookmarkDetector:
    """Extract bookmarks pointing to A..E-like titles as soft anchors."""

    def __init__(self):
        pass

    @staticmethod
    def _classify_letter(title: str) -> Optional[str]:
        title_norm = normalize_text(title)
        for letter, pat in SECTION_PATTERNS.items():
            if pat.search(title_norm):
                return letter
        # Also accept raw "A. " starts to be generous:
        m = re.match(r"^([A-E])\.", title_norm)
        if m:
            return m.group(1).upper()
        return None

    def detect(self, loader: PDFLoader) -> List[HeadingHit]:
        hits: List[HeadingHit] = []
        for level, title, page in loader.get_toc():
            letter = self._classify_letter(title)
            if letter:
                hits.append(
                    HeadingHit(
                        letter=letter,
                        page=max(1, page),  # PyMuPDF TOC pages are 1-based already
                        score=0.55,  # modest prior
                        text=title,
                        bbox=(0, 0, 0, 0),
                        source="bookmark",
                        details={"level": level},
                    )
                )
        return hits


class ToCDetector:
    """
    Geometry-aware ToC detector:
    - Groups spans on the same baseline (y) within a tolerance.
    - Reconstructs logical lines even if 'A', title, dot-leader, and page are separate blocks.
    - Extracts (letter, title, page) robustly.
    """

    def __init__(
        self,
        max_pages_scan: int = 6,
        y_tolerance: float = 3.0,
        min_tokens_per_line: int = 2,
    ):
        self.max_pages_scan = max_pages_scan
        self.y_tolerance = y_tolerance
        self.min_tokens = min_tokens_per_line

    def _iter_spans(self, loader, page_index: int):
        page = loader.get_page(page_index)
        pd = page.get_text("dict")
        for blk in pd.get("blocks", []):
            for line in blk.get("lines", []):
                for sp in line.get("spans", []):
                    txt = normalize_text(sp.get("text", ""))
                    if not txt:
                        continue
                    x0, y0, x1, y1 = sp.get("bbox", (0, 0, 0, 0))
                    yield {
                        "text": txt,
                        "bbox": (x0, y0, x1, y1),
                        "x": x0,
                        "y": y0,  # use top as baseline proxy (good enough for ToC)
                        "size": sp.get("size", 0.0),
                        "flags": sp.get("flags", 0),
                    }

    def _group_by_baseline(
        self, spans: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group spans by similar y (baseline) within tolerance.
        Returns list of lines, each a list of spans sorted left->right.
        """
        if not spans:
            return []
        # sort by y then x
        spans = sorted(spans, key=lambda s: (round(s["y"]), s["x"]))
        lines: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        last_y: Optional[float] = None

        for sp in spans:
            if last_y is None or abs(sp["y"] - last_y) <= self.y_tolerance:
                current.append(sp)
                last_y = sp["y"] if last_y is None else (last_y + sp["y"]) / 2.0
            else:
                current.sort(key=lambda s: s["x"])
                if current:
                    lines.append(current)
                current = [sp]
                last_y = sp["y"]
        if current:
            current.sort(key=lambda s: s["x"])
            lines.append(current)
        return lines

    def _extract_line_triplet(
        self, line_spans: List[Dict[str, Any]], page_width: float
    ) -> Optional[Tuple[str, str, int]]:
        """
        Return (section_letter, title, page) for one baseline line.
        Supports:
          - 2 spans: ["A.", "Risikoprofil……39"]
          - multi spans: ["A", ".", "Risikoprofil", "……", "39"]
          - "Teil A" / "Abschnitt B" as left markers
          - merged rightmost token with trailing page number
        Skips subsections like "A.1".
        """
        if not line_spans:
            return None

        # 1) Identify left marker (A..E), accepting several forms; reject subsections
        letter_idx = None
        letter_val = None

        # try strict "A." first
        for i, sp in enumerate(line_spans):
            t = sp["text"]
            m = LEFT_TOPLEVEL_RE.match(t)
            if m:
                letter_idx = i
                letter_val = m.group(1).upper()
                break
            if LEFT_SUBSECTION_RE.match(t):  # ignore "A.1" lines for top-level
                return None

        # try "Teil A" / "Abschnitt B"
        if letter_idx is None:
            for i, sp in enumerate(line_spans):
                t = sp["text"]
                m = LEFT_TEIL_RE.match(t)
                if m:
                    letter_idx = i
                    letter_val = m.group(1).upper()
                    break

        # try "A" optionally followed by "." in next token
        if letter_idx is None:
            for i in range(len(line_spans)):
                t = line_spans[i]["text"]
                m = LEFT_LETTER_ONLY_RE.match(t)
                if m:
                    if i + 1 < len(line_spans) and line_spans[i + 1]["text"] == ".":
                        letter_idx = i + 1  # treat the '.' as the last marker token
                    else:
                        letter_idx = i
                    letter_val = m.group(1).upper()
                    break

        if letter_idx is None or letter_val is None:
            return None

        # 2) Identify page token:
        #    a) Rightmost pure number, else
        #    b) Trailing digits inside the rightmost token (merged "Title……39")
        page_idx = None
        for i in range(len(line_spans) - 1, -1, -1):
            t = line_spans[i]["text"]
            if NUM_RE.match(t or ""):
                page_idx = i
                break

        merged_right = None
        if page_idx is None:
            # try parsing trailing digits from the last token
            tlast = line_spans[-1]["text"]
            m = RIGHT_TOKEN_RE.match(tlast)
            if m:
                merged_right = m
                page_idx = len(line_spans) - 1
            else:
                return None

        # ensure marker is to the left of page index
        if letter_idx >= page_idx:
            return None

        # 3) Build title from tokens between (letter_idx) and (page_idx)
        title_tokens: List[str] = []
        for j in range(letter_idx + 1, page_idx):
            txt = line_spans[j]["text"]
            if _is_leader_token(txt):
                continue
            title_tokens.append(txt)

        title = " ".join(title_tokens).strip()

        # If title empty (e.g., all leaders), and we parsed a merged right token, use its prefix
        if not title and merged_right is not None:
            title = merged_right.group("title").strip()

        # Final cleaning
        title = re.sub(r"\s+", " ", title)
        if not title:
            return None

        # 4) Page number
        if merged_right is not None:
            try:
                page_num = int(merged_right.group("page"))
            except ValueError:
                return None
        else:
            try:
                page_num = int(line_spans[page_idx]["text"])
            except ValueError:
                # last resort: also try merged parsing on that token
                m = RIGHT_TOKEN_RE.match(line_spans[page_idx]["text"])
                if not m:
                    return None
                try:
                    page_num = int(m.group("page"))
                except ValueError:
                    return None

        return (letter_val, title, page_num)

    def _extract_toc_item_from_line(
        self, line_spans: List[Dict[str, Any]]
    ) -> Optional[TocItem]:
        """
        Generic ToC line parser:
        - Accepts left marker in many forms ("A.", "A", "Teil A", "A.1", or none)
        - Extracts trailing page from the rightmost token or merged leaders
        - Returns TocItem(page, title, left_marker)
        """
        if not line_spans:
            return None

        # Try to find a rightmost page number
        page_idx = None
        for i in range(len(line_spans) - 1, -1, -1):
            t = line_spans[i]["text"]
            if NUM_RE.match(t or ""):
                page_idx = i
                break

        merged_right = None
        if page_idx is None:
            # Try merged "Title……39"
            tlast = line_spans[-1]["text"]
            m = RIGHT_TOKEN_RE.match(tlast)
            if not m:
                return None
            merged_right = m
            page_idx = len(line_spans) - 1
            title_right = m.group("title").strip()
            try:
                page_num = int(m.group("page"))
            except ValueError:
                return None
        else:
            # Page in its own token
            try:
                page_num = int(line_spans[page_idx]["text"])
            except ValueError:
                m = RIGHT_TOKEN_RE.match(line_spans[page_idx]["text"])
                if not m:
                    return None
                title_right = m.group("title").strip()
                try:
                    page_num = int(m.group("page"))
                except ValueError:
                    return None
                merged_right = m

        # Left marker (if any)
        left_marker = ""
        letter_idx = None
        for i, sp in enumerate(line_spans[:page_idx]):
            t = sp["text"]
            if LEFT_SUBSECTION_RE.match(t):
                left_marker = t  # e.g., "E.1"
                letter_idx = i
                break
            m = LEFT_TOPLEVEL_RE.match(t)
            if m:
                left_marker = m.group(1).upper() + "."
                letter_idx = i
                break
            m2 = LEFT_TEIL_RE.match(t)
            if m2:
                left_marker = f"Teil {m2.group(1).upper()}"
                letter_idx = i
                break
            m3 = LEFT_LETTER_ONLY_RE.match(t)
            if m3:
                # If next token is ".", treat "A" + "." as "A."
                if i + 1 < page_idx and line_spans[i + 1]["text"] == ".":
                    left_marker = m3.group(1).upper() + "."
                    letter_idx = i + 1
                else:
                    left_marker = m3.group(1).upper()
                    letter_idx = i
                break

        # Build title from tokens between left marker and page
        start_j = (letter_idx + 1) if letter_idx is not None else 0
        title_tokens = []
        for j in range(start_j, page_idx):
            txt = line_spans[j]["text"]
            if txt and all(ch in LEADER_CHARS + " " for ch in txt):
                continue
            if txt == ".":
                continue
            title_tokens.append(txt)

        title = " ".join(title_tokens).strip()
        if not title and merged_right:
            title = title_right  # prefix before trailing page inside merged right token
        title = re.sub(r"\s+", " ", title).strip()

        if not title:
            return None

        return TocItem(page=page_num, title=title, left_marker=left_marker)

    def detect_items(self, loader) -> List[TocItem]:
        """Return generic ToC items from first N pages."""
        items: List[TocItem] = []
        n_pages = loader.page_count()
        limit = min(n_pages, self.max_pages_scan)
        for pi in range(limit):
            spans = list(self._iter_spans(loader, pi))
            lines = self._group_by_baseline(spans)
            for line in lines:
                item = self._extract_toc_item_from_line(line)
                if item:
                    items.append(item)
        # De-duplicate (some PDFs repeat ToC across a spread)
        seen = set()
        uniq: List[TocItem] = []
        for it in items:
            key = (it.page, it.title.lower(), it.left_marker.lower())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)
        return uniq

    def detect(self, loader) -> List[HeadingHit]:
        hits: List[HeadingHit] = []
        n_pages = loader.page_count()
        limit = min(n_pages, self.max_pages_scan)

        for pi in range(limit):
            page = loader.get_page(pi)
            page_width = page.rect.width

            spans = list(self._iter_spans(loader, pi))
            lines = self._group_by_baseline(spans)

            for line in lines:
                triplet = self._extract_line_triplet(line, page_width)
                if not triplet:
                    continue
                letter, title, page_num = triplet
                # Soft score; actual on-page heading detection will verify/boost later
                hits.append(
                    HeadingHit(
                        letter=letter,
                        page=page_num,
                        score=0.55,
                        text=title,
                        bbox=(0, 0, 0, 0),
                        source="toc",
                        details={"toc_page": pi + 1, "reconstructed": True},
                    )
                )
        return hits


class RegexHeadingDetector:
    """Scan each page for A..E headings and score them with typography & position."""

    def __init__(self):
        pass

    def _candidate_spans(
        self, page_dict: Dict[str, Any], page_rect: fitz.Rect
    ) -> List[Tuple[str, float, bool, Tuple[float, float, float, float], str]]:
        cands = []
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw = span.get("text", "")
                    text = normalize_text(raw)
                    if not text:
                        continue
                    # strict: only consider short-ish spans
                    if len(text) > 120:
                        continue
                    # match any SECTION pattern
                    matched_letter = None
                    for letter, pat in SECTION_PATTERNS.items():
                        if pat.search(text):
                            matched_letter = letter
                            break
                    if not matched_letter:
                        continue
                    size = float(span.get("size", 0.0))
                    bold = bool(span.get("flags", 0) & 2)
                    bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                    # scoring
                    score = min(1.0, (size / 18.0))  # normalize ~14-20pt into 0..1
                    if bold:
                        score += 0.15
                    score += roughly_top_left(bbox, page_rect)
                    score += whitespace_gap_above(None, bbox)  # cheap cue
                    cands.append((matched_letter, score, bold, bbox, text))
        return cands

    def detect(self, loader: PDFLoader) -> List[HeadingHit]:
        hits: List[HeadingHit] = []
        for i in range(loader.page_count()):
            page = loader.get_page(i)
            page_dict = page.get_text("dict")
            page_rect = page.rect
            cands = self._candidate_spans(page_dict, page_rect)
            if not cands:
                continue
            # best per letter per page
            best_by_letter: Dict[str, Tuple] = {}
            for letter, score, bold, bbox, text in cands:
                prev = best_by_letter.get(letter)
                if (prev is None) or (score > prev[1]):
                    best_by_letter[letter] = (letter, score, bold, bbox, text)
            for letter, score, bold, bbox, text in best_by_letter.values():
                hits.append(
                    HeadingHit(
                        letter=letter,
                        page=i + 1,
                        score=min(0.95, score),
                        text=text,
                        bbox=bbox,
                        source="regex",
                        details={"bold": bold, "size_score": min(1.0, score)},
                    )
                )
        return hits


class SectionFuser:
    """
    Fuse bookmarks/ToC/regex signals into ordered A..E sections,
    enforce monotonic page order, and compute confidence.
    """

    def __init__(self, required_letters=("A", "B", "C", "D", "E")):
        self.required = list(required_letters)

    def _choose_start_pages(
        self, hits: List[HeadingHit]
    ) -> Dict[str, List[HeadingHit]]:
        per_letter: Dict[str, List[HeadingHit]] = {k: [] for k in self.required}
        for h in hits:
            if h.letter in per_letter:
                per_letter[h.letter].append(h)
        # keep top few per letter to allow order constraints later
        for k in per_letter.keys():
            per_letter[k].sort(key=lambda x: (-x.score, x.page))
            per_letter[k] = per_letter[k][:5]
        return per_letter

    def fuse(
        self, loader: PDFLoader, hits: List[HeadingHit]
    ) -> Tuple[List[SectionSpan], List[str]]:
        issues: List[str] = []
        per_letter = self._choose_start_pages(hits)

        # Greedy pass enforcing A->E and non-decreasing pages
        chosen: Dict[str, HeadingHit] = {}
        last_page = 1
        for letter in self.required:
            candidates = [h for h in per_letter[letter] if h.page >= last_page]
            if not candidates and per_letter[letter]:
                # allow slight backtrack within 1 page if strong evidence
                candidates = [h for h in per_letter[letter] if h.page >= last_page - 1]
            if not candidates:
                issues.append(f"Missing section {letter}")
                continue
            best = max(candidates, key=lambda h: h.score)
            # Demote if going backwards significantly
            if best.page < last_page:
                best = HeadingHit(
                    best.letter,
                    last_page,
                    best.score * 0.7,
                    best.text,
                    best.bbox,
                    best.source,
                    best.details,
                )
                issues.append(f"Adjusted {letter} start to maintain order")
            chosen[letter] = best
            last_page = max(last_page, best.page)

        # Create page spans (start_i .. start_{i+1}-1)
        spans: List[SectionSpan] = []
        pages_total = loader.page_count()
        for i, letter in enumerate(self.required):
            if letter not in chosen:
                continue
            this_hit = chosen[letter]
            start_page = this_hit.page
            # next letters' start
            next_page = pages_total + 1
            for j in range(i + 1, len(self.required)):
                nxt = self.required[j]
                if nxt in chosen:
                    next_page = chosen[nxt].page
                    break
            end_page = max(start_page, next_page - 1)
            # combine detector presence per letter
            detector_presence = {"bookmark": False, "toc": False, "regex": False}
            for h in per_letter[letter]:
                detector_presence[h.source] = True or detector_presence[h.source]
            # confidence: base on chosen hit score + how many detectors fired
            detectors_count = sum(1 for v in detector_presence.values() if v)
            conf = min(1.0, this_hit.score + 0.1 * (detectors_count - 1))
            spans.append(
                SectionSpan(
                    section=letter,
                    start_page=start_page,
                    end_page=end_page,
                    confidence=conf,
                    detectors=detector_presence,
                )
            )

        # coverage sanity check
        covered_pages = 0
        merged = []
        for sp in spans:
            covered_pages += sp.end_page - sp.start_page + 1
            merged.append((sp.start_page, sp.end_page))
        coverage_ratio = covered_pages / max(1, pages_total)
        if coverage_ratio < 0.5:
            issues.append(f"Low coverage: {coverage_ratio:.2f}")

        return spans, issues


class SubsectionDetector:
    """Detect A.1 / B.2 style subsections within each section's page range."""

    def __init__(self):
        pass

    def detect(
        self, loader: PDFLoader, section_spans: List[SectionSpan]
    ) -> List[SubsectionSpan]:
        subs: List[SubsectionSpan] = []
        for sp in section_spans:
            # Collect subsection hits as tuples: (code, title, page, confidence)
            hits: List[Tuple[str, str, int, float]] = []
            for p in range(sp.start_page - 1, sp.end_page):
                page = loader.get_page(p)
                page_dict = page.get_text("dict")
                page_rect = page.rect
                for blk in page_dict.get("blocks", []):
                    for line in blk.get("lines", []):
                        txt = normalize_text(
                            "".join([s["text"] for s in line.get("spans", [])])
                        )
                        if not txt:
                            continue
                        m = SUBSECTION_PATTERN.match(txt)
                        if not m:
                            continue
                        letter, n1, n2, title = m.groups()
                        letter = letter.upper()
                        if letter != sp.section:
                            continue
                        code = f"{letter}.{n1}" + (f".{n2}" if n2 else "")
                        conf = 0.6 + roughly_top_left(
                            line["spans"][0]["bbox"], page_rect
                        )
                        hits.append((code, title, p + 1, min(0.95, conf)))
            # Deduplicate by (code, page)
            seen = set()
            deduped_hits: List[Tuple[str, str, int, float]] = []
            for h in hits:
                key = (h[0], h[2])
                if key in seen:
                    continue
                seen.add(key)
                deduped_hits.append(h)
            # Sort by page, then code for stability
            deduped_hits.sort(key=lambda x: (x[2], x[0]))
            # Materialize continuous spans
            n = len(deduped_hits)
            for i, (code, title, page, confidence) in enumerate(deduped_hits):
                # Start page for this subsection
                start_page = max(sp.start_page, page)
                # End page: if not last, up to next subsection's page; else to section end
                if i + 1 < n:
                    next_page = deduped_hits[i + 1][2]
                    end_page = min(sp.end_page, next_page)
                else:
                    end_page = sp.end_page
                # Allow end_page equal to start_page
                if start_page > end_page:
                    continue
                subs.append(
                    SubsectionSpan(
                        section=sp.section,
                        code=code,
                        title=title,
                        start_page=start_page,
                        end_page=end_page,
                        confidence=confidence,
                    )
                )
        return subs


# -----------------------------
# Orchestrator (Ingestor)
# -----------------------------


class SFCRIngestor:
    """
    High-level ingestion orchestrator.
    Usage:
        ing = SFCRIngestor(doc_id="de_foo_2023", pdf_path="foo.pdf")
        result = ing.run()
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    """

    def __init__(self, doc_id: str, pdf_path: str):
        self.doc_id = doc_id
        self.loader = PDFLoader(pdf_path)
        self.bookmarks = BookmarkDetector()
        self.toc = ToCDetector(max_pages_scan=6)
        self.regex = RegexHeadingDetector()
        self.fuser = SectionFuser()
        self.subs = SubsectionDetector()

    def _possible_header_footer_lines(self) -> Tuple[set, set]:
        """
        Identify frequent header/footer lines to ignore if you later expand heuristics.
        Currently not used to filter headings (kept simple), but returned for diagnostics.
        """
        top_lines: Dict[str, int] = {}
        bot_lines: Dict[str, int] = {}
        n = self.loader.page_count()
        for i in range(n):
            page = self.loader.get_page(i)
            blocks = page.get_text("blocks")  # plain text blocks
            if not blocks:
                continue
            # top-most and bottom-most line heuristics
            tops = [b for b in blocks if b[1] < 60]  # y0
            bots = [
                b for b in blocks if (page.rect.height - b[3]) < 60
            ]  # page height - y1
            if tops:
                tline = normalize_text(tops[0][4])
                if tline:
                    top_lines[tline] = top_lines.get(tline, 0) + 1
            if bots:
                bline = normalize_text(bots[-1][4])
                if bline:
                    bot_lines[bline] = bot_lines.get(bline, 0) + 1
        tops_keep = {
            k
            for k, v in top_lines.items()
            if v / max(1, n) > HEADER_FOOTER_FREQ_THRESHOLD
        }
        bots_keep = {
            k
            for k, v in bot_lines.items()
            if v / max(1, n) > HEADER_FOOTER_FREQ_THRESHOLD
        }
        return tops_keep, bots_keep

    def run(self) -> IngestionResult:
        # 1) Signals
        hits: List[HeadingHit] = []
        hits += self.bookmarks.detect(self.loader)
        hits += self.toc.detect(self.loader)
        hits += self.regex.detect(self.loader)

        # 2) Fuse & enforce order
        sections, issues = self.fuser.fuse(self.loader, hits)

        # 3) Subsections
        subsections = self.subs.detect(self.loader, sections)

        # Find E's start page (or the latest section’s end if E is missing)
        last_ae_start = 0
        last_ae_section = None
        for sp in sections:
            if sp.section in ("A", "B", "C", "D", "E"):
                last_ae_start = max(last_ae_start, sp.start_page)
                last_ae_section = sp

        # Pull generic ToC items
        toc_items = self.toc.detect_items(self.loader)

        def is_e_subsection(left_marker: str) -> bool:
            # things like "E.1", "E.2.3", or "Abschnitt E" (treat as under E)
            if not left_marker:
                return False
            if re.match(r"^E\.\d", left_marker, re.I):
                return True
            if re.match(r"^Abschnitt\s*E\b", left_marker, re.I):
                return True
            if re.match(r"^Teil\s*E\b", left_marker, re.I):
                return True
            # Pure "E." is the section itself (already handled), not a subsection
            return False

        # Choose the earliest ToC item whose page is strictly after E's start,
        # and which is NOT an E-subsection. This is our "post" start candidate.
        post_candidate = None
        for it in sorted(toc_items, key=lambda x: x.page):
            if it.page <= max(1, last_ae_start):
                continue
            if is_e_subsection(it.left_marker):
                continue
            post_candidate = it
            break

        if post_candidate:
            # Append synthetic trailing section Z (generic name for display)
            detectors = {
                "post_toc": True,
                "post_regex": False,
                "post_bookmark": False,
                "bookmark": False,
                "toc": False,
                "regex": False,
            }
            sections.append(
                SectionSpan(
                    section="Z",
                    start_page=post_candidate.page,
                    end_page=self.loader.page_count(),
                    confidence=0.8,  # ToC-derived; you can tune
                    detectors=detectors,
                )
            )
            if last_ae_section:
                last_ae_section.end_page = post_candidate.page - 1

        # 4) Coverage metric
        if sections:
            covered = sum(sp.end_page - sp.start_page + 1 for sp in sections)
            coverage_ratio = covered / self.loader.page_count()
        else:
            coverage_ratio = 0.0
            issues.append("No sections detected")

        # 5) Package result
        return IngestionResult(
            doc_id=self.doc_id,
            pdf_sha256=None,  # compute separately if you want: hashlib.sha256(open(...,'rb').read()).hexdigest()
            page_count=self.loader.page_count(),
            sections=sections,
            subsections=subsections,
            coverage_ratio=coverage_ratio,
            issues=issues,
        )
