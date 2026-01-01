# sfcr/ingest/tests/test_subsection_detector_toc.py

from sfcr.ingest.sfcr_ingest import SectionSpan, SubsectionDetector, TocItem


def test_subsections_within_bounds_and_continuous():
    """
    Section A spans pages 5..12 (1-based).
    ToC contains:
      - A.1 on page 5
      - A.2 on page 9
    Expect two SubsectionSpan objects:
      - A.1: start=5, end=9
      - A.2: start=9, end=12
    Continuous: end of first equals start of next.
    """
    detector = SubsectionDetector()

    section = SectionSpan(section="A", start_page=5, end_page=12)

    toc_items = [
        TocItem(page=5, title="Unterabschnitt Eins", left_marker="A.1"),
        TocItem(page=9, title="Unterabschnitt Zwei", left_marker="A.2"),
        # noise: wrong letter should be ignored
        TocItem(page=9, title="Should Be Ignored", left_marker="B.1"),
    ]

    subs = detector.detect(
        toc_items=toc_items,
        section_spans=[section],
    )
    subs = [s for s in subs if s.section == "A"]

    assert len(subs) == 2, f"got {[(s.code, s.start_page, s.end_page) for s in subs]}"

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


def test_subsections_ignore_outside_bounds_and_mismatched_letters():
    """
    Section A spans pages 4..6 (1-based).
    ToC contains:
      - A.1 on page 3 -> OUTSIDE section -> ignored
      - A.2 on page 5 -> inside -> first/only hit
      - B.1 on page 5 -> wrong letter -> ignored
    Expect a single SubsectionSpan:
      - A.2: start=5, end=6
    """
    detector = SubsectionDetector()

    section = SectionSpan(section="A", start_page=4, end_page=6)

    toc_items = [
        TocItem(page=3, title="Vor Abschnitt (außerhalb)", left_marker="A.1"),
        TocItem(page=5, title="Innerhalb", left_marker="A.2"),
        TocItem(page=5, title="Falscher Buchstabe", left_marker="B.1"),
    ]

    subs = detector.detect(
        toc_items=toc_items,
        section_spans=[section],
    )
    subs = [s for s in subs if s.section == "A"]

    assert len(subs) == 1
    s = subs[0]
    assert s.code == "A.2"
    assert s.start_page == 5
    assert s.end_page == 6

    assert section.start_page <= s.start_page <= section.end_page
    assert section.start_page <= s.end_page <= section.end_page
    assert s.start_page <= s.end_page