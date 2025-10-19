from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from sfcr.config import get_settings
from sfcr.db import (
    db_path_default,
    get_extractions_for_doc,
    init_db,
    list_documents,
    load_extractions_from_dir,
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


def main():
    st.set_page_config(page_title="SFCR Extractor Viewer", layout="wide")
    st.title("SFCR Extractor — Verified Values")

    cfg = get_settings()
    db_path = db_path_default()
    colA, colB, colC = st.columns([2, 2, 1])

    with colA:
        st.caption(f"Output dir: `{cfg.output_dir}`")
        if st.button("1) Init DB", help="Create SQLite with tables/views if missing"):
            p = init_db(db_path)
            st.success(f"DB initialized at {p}")

    with colB:
        if st.button("2) Load from JSONL", help="Load all *.extractions.jsonl into DB"):
            n_docs, n_rows = load_extractions_from_dir(cfg.output_dir, db_path)
            st.success(f"Loaded {n_rows} rows from {n_docs} docs")

    with colC:
        if st.button("↻ Refresh"):
            st.experimental_rerun()

    # Documents sidebar
    docs = list_documents(db_path)
    doc_ids = [d["doc_id"] for d in docs]
    st.sidebar.header("Documents")
    if not docs:
        st.sidebar.warning(
            "No documents in DB yet. Click 'Init DB' then 'Load from JSONL'."
        )
        st.stop()

    doc_id = st.sidebar.selectbox("Choose document", doc_ids, index=0)
    pdf_path = None
    for d in docs:
        if d["doc_id"] == doc_id:
            pdf_path = d["pdf_path"]
            break

    # Filters
    st.sidebar.header("Filters")
    show_only_failed = st.sidebar.checkbox("Only unverified / failed")
    show_source = st.sidebar.checkbox("Show source text")

    # Table
    rows = get_extractions_for_doc(doc_id, db_path)
    if show_only_failed:
        rows = [r for r in rows if not r.get("verified")]

    st.subheader(f"Results for `{doc_id}`")
    if not rows:
        st.info("No extractions for this document.")
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
