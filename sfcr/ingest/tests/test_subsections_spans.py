# Import from your ingestion module
from sfcr.ingest.sfcr_ingest import SectionSpan, SubsectionDetector

# --- Tiny fakes to avoid real PDFs -------------------------------------------


class _FakeRect:
    def __init__(self, w=1000, h=1000):
        self.width = w
        self.height = h


class _FakePage:
    def __init__(self, lines):
        # lines: list[str] for this page
        self._lines = lines
        self.rect = _FakeRect()

    def get_text(self, mode):
        assert mode == "dict"
        # Build a minimal structure: blocks -> lines -> spans[text, bbox]
        blocks = []
        if self._lines:
            line_objs = []
            for t in self._lines:
                line_objs.append(
                    {
                        "spans": [
                            {
                                "text": t,
                                "bbox": (50.0, 50.0, 500.0, 80.0),
                                "size": 16.0,
                                "flags": 0,
                            }
                        ]
                    }
                )
            blocks.append({"lines": line_objs})
        return {"blocks": blocks}


class _FakeLoader:
    def __init__(self, pages_text):
        """
        pages_text: dict[int->list[str]] mapping 0-based page index to list of lines
        """
        self._pages_text = pages_text
        self._max_index = max(pages_text.keys()) if pages_text else -1

    def page_count(self):
        return self._max_index + 1

    def get_page(self, i: int):
        lines = self._pages_text.get(i, [])
        return _FakePage(lines)


# --- Tests -------------------------------------------------------------------


def test_subsections_within_bounds_and_continuous():
    """
    Section A spans pages 5..12 (1-based).
    We place 'A.1' on page 5 (index 4) and 'A.2' on page 9 (index 8).
    Expect two SubsectionSpan objects:
      - A.1: start=5, end=9
      - A.2: start=9, end=12
    Continuous: end of first equals start of next.
    """
    pages = {
        4: ["A.1 Unterabschnitt Eins"],  # page 5
        8: ["A.2 Unterabschnitt Zwei"],  # page 9
        # noise to ensure dedup is stable
        10: ["B.1 Should Be Ignored"],  # wrong letter in A-section
    }
    loader = _FakeLoader(pages)
    detector = SubsectionDetector()

    section = SectionSpan(
        section="A",
        start_page=5,
        end_page=12,
    )

    subs = detector.detect(loader, [section])
    # Filter to section A (defensive)
    subs = [s for s in subs if s.section == "A"]

    # Expect exactly 2
    assert len(subs) == 2, f"got {[(s.code, s.start_page, s.end_page) for s in subs]}"

    # Sorted by start_page for stable assertions
    subs.sort(key=lambda s: (s.start_page, s.code))
    s1, s2 = subs

    assert s1.code == "A.1"
    assert s1.start_page == 5
    assert s1.end_page == 9

    assert s2.code == "A.2"
    assert s2.start_page == 9
    assert s2.end_page == 12

    # Bounds within section
    for s in subs:
        assert section.start_page <= s.start_page <= section.end_page
        assert section.start_page <= s.end_page <= section.end_page
        assert s.start_page <= s.end_page

    # Continuous (allowed to touch): end of first equals start of second
    assert s1.end_page == s2.start_page


def test_subsections_clipped_to_section_and_ignore_mismatched_letters():
    """
    Section A spans pages 4..6 (1-based).
    We place:
      - 'A.1' on page 3 (index 2) -> OUTSIDE the section start -> should not be seen
      - 'A.2' on page 5 (index 4) -> inside -> first/only hit
      - 'B.1' on page 5 -> wrong letter -> ignored
    Expect a single SubsectionSpan:
      - A.2: start=max(4,5)=5, end=6 (section end, since no next hit)
    """
    pages = {
        2: ["A.1 Vor Abschnitt (außerhalb)"],  # page 3 (outside)
        4: ["A.2 Innerhalb", "B.1 Falscher Buchstabe"],  # page 5
    }
    loader = _FakeLoader(pages)
    detector = SubsectionDetector()

    section = SectionSpan(
        section="A",
        start_page=4,
        end_page=6,
    )

    subs = detector.detect(loader, [section])
    subs = [s for s in subs if s.section == "A"]

    assert len(subs) == 1
    s = subs[0]
    assert s.code == "A.2"
    assert s.start_page == 5
    assert s.end_page == 6

    # Bounds & monotonicity
    assert section.start_page <= s.start_page <= section.end_page
    assert section.start_page <= s.end_page <= section.end_page
    assert s.start_page <= s.end_page
