from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from sfcr.db import (
    db_path_default,
    get_extractions_for_doc,
    get_summaries_for_doc,
    init_db,
    list_documents,
    load_catalog,
    load_extractions_from_dir,
    load_summaries_from_dir,
)


# Optional: render PDF page images
def render_pdf_page(pdf_path: Path, page: int, zoom: float = 1.5) -> Optional[bytes]:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        if page < 1 or page > doc.page_count:
            return None
        pix = doc.load_page(page - 1).get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), alpha=False
        )
        return pix.tobytes("png")
    except Exception:
        return None


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        raise RuntimeError("This version of Streamlit has no rerun method.")


def main():
    st.set_page_config(page_title="SFCR Extractor Viewer", layout="wide")
    st.title("SFCR Viewer")

    db_path = db_path_default()

    # Documents sidebar
    docs = list_documents(db_path)
    display_names = [d["display_name"] for d in docs]
    st.sidebar.header("Dokumente")
    if not docs:
        st.sidebar.warning(
            "Keine Dokumente in DB gefunden. Klicke 'Init DB', dann 'Load from JSONL'."
        )

    display_name = st.sidebar.selectbox("Dokument auswählen", display_names, index=0)
    doc_id = None
    pdf_path = None
    for d in docs:
        if d["display_name"] == display_name:
            doc_id = d["doc_id"]
            pdf_path = d["pdf_path"]
            break

    # Filters
    st.sidebar.header("Filter")
    show_only_failed = st.sidebar.checkbox("Zeige nur unverifizierte Werte")
    show_source = st.sidebar.checkbox("Zeige Text-Quelle")

    # colA, colB, colC = st.columns([2, 2, 1])

    # with colA:
    if st.sidebar.button(
        "1) Init DB", help="Create SQLite with tables/views if missing"
    ):
        p = init_db(db_path)
        st.success(f"DB initialized at {p}")

    # with colB:
    if st.sidebar.button(
        "2) Load from JSONL", help="Load all *.extractions.jsonl into DB"
    ):
        n_docs = load_catalog()
        _, _ = load_extractions_from_dir()
        _, _ = load_summaries_from_dir()
        st.success(f"Loaded {n_docs} docs")

    # with colC:
    if st.sidebar.button("↻ Refresh"):
        safe_rerun()

    if not docs:
        st.stop()

    st.header(f"{display_name}")

    # Summaries
    summaries = get_summaries_for_doc(doc_id)

    st.subheader("Zusammenfassungen")
    if not summaries:
        st.info(
            "Keine Zusammenfassungen für das Dokument gefunden. Erzeuge neue Zusammenfassungen und lade sie in die DB."
        )
    else:
        # Build stable tab order; fall back to title
        labels = [
            f"{s['section_id']} — {s.get('title') or ''}".strip(" —") for s in summaries
        ]
        tabs = st.tabs(labels)
        for tab, s in zip(tabs, summaries):
            with tab:
                meta = (
                    f"Pages {s.get('start_page')}–{s.get('end_page')}"
                    if s.get("start_page") and s.get("end_page")
                    else ""
                )
                if meta:
                    st.caption(meta)
                # The summaries are multi-line text; render as Markdown
                st.markdown(s.get("summary") or "_(empty)_")

    # Table
    rows = get_extractions_for_doc(doc_id, db_path)
    if show_only_failed:
        rows = [r for r in rows if not r.get("verified")]

    st.subheader("Werte")
    if not rows:
        st.info(
            "Für das Dokument wurden keine Werte in der DB gefunden. Extrahiere zunächst die Werte und lade sie dann in die DB."
        )
        st.stop()

    # Summary chips
    total = len(rows)
    verified = sum(1 for r in rows if r.get("verified"))
    st.write(f"Verified: **{verified}/{total}**")

    # Render table with expanders
    for r in rows:
        ok = "✅" if r.get("verified") else "❌"
        field = r["field_id"]
        val = r.get("value_canonical")
        unit = r.get("unit") or ""
        conf = r.get("confidence")
        page = r.get("page")
        status = r.get("status") or ""
        issues = r.get("issues") or ""
        header = f"{ok} **{field}** — {val if val is not None else '—'} {unit}  ·  p.{page or '—'}  ·  conf={conf or 0:.2f}  ·  {status}"
        with st.expander(header, expanded=False):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.write(
                    f"Scale applied: `{r.get('scale_applied')}` ({r.get('scale_source') or '—'})"
                )
                if show_source and r.get("source_text"):
                    st.code(r["source_text"])
                if issues:
                    st.warning(issues)
            with c2:
                if pdf_path and page:
                    img = render_pdf_page(Path(pdf_path), int(page))
                    if img:
                        st.image(
                            img,
                            caption=f"{Path(pdf_path).name} — page {page}",
                            width="stretch",
                        )
                    else:
                        st.info(
                            "Page preview unavailable (no PDF path or PyMuPDF missing)."
                        )
                else:
                    st.info("No page evidence recorded.")


if __name__ == "__main__":
    main()
