from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sfcr.config import get_settings

# ---------- connection / schema ----------


def db_path_default() -> Path:
    cfg = get_settings()
    return Path(cfg.output_dir) / "sfcr.sqlite"


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    db = db_path or db_path_default()
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: Optional[Path] = None) -> Path:
    con = connect(db_path)
    cur = con.cursor()
    # documents: one row per PDF
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
      doc_id     TEXT PRIMARY KEY,
      pdf_path   TEXT NOT NULL,
      sha256     TEXT,
      page_count INTEGER,
      updated_at TEXT DEFAULT (datetime('now'))
    );
    """)
    # extractions: one row per (doc_id, field_id)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS extractions (
      doc_id          TEXT NOT NULL,
      field_id        TEXT NOT NULL,
      value_canonical REAL,
      unit            TEXT,
      verified        INTEGER NOT NULL DEFAULT 0,
      confidence      REAL,
      page            INTEGER,
      status          TEXT,
      issues          TEXT,
      source_text     TEXT,
      scale_applied   REAL,
      scale_source    TEXT,
      updated_at      TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (doc_id, field_id),
      FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );
    """)
    # summaries: one row per (doc_id, section_id)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
      doc_id      TEXT NOT NULL,
      section_id  TEXT NOT NULL,
      title       TEXT,
      start_page  INTEGER,
      end_page    INTEGER,
      summary     TEXT NOT NULL,
      updated_at  TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (doc_id, section_id),
      FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );
    """)
    # convenience view
    cur.execute("DROP VIEW IF EXISTS current_verified;")
    cur.execute("""
    CREATE VIEW current_verified AS
      SELECT * FROM extractions WHERE verified = 1;
    """)
    con.commit()
    con.close()
    return db_path or db_path_default()


# ---------- utilities ----------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_pdf_for_doc(doc_id: str) -> Optional[Path]:
    """Best-effort: look in SFCR_DATA for <doc_id>.pdf"""
    cfg = get_settings()
    cand = Path(cfg.data_dir) / f"{doc_id}.pdf"
    return cand if cand.exists() else None


# ---------- load from *.extractions.jsonl ----------


def load_extractions_from_dir(
    out_dir: Optional[Path] = None, db_path: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Scan <output_dir> for *.extractions.jsonl and upsert into SQLite.
    Returns: (n_docs_updated, n_rows_upserted)
    """
    cfg = get_settings()
    root = out_dir or Path(cfg.output_dir_extract)
    con = connect(db_path)
    cur = con.cursor()

    n_docs, n_rows = 0, 0
    for jpath in sorted(root.glob("*.extractions.jsonl")):
        doc_id = jpath.stem.replace(".extractions", "")
        # ensure documents row
        pdf = _find_pdf_for_doc(doc_id)
        sha, pages = None, None
        if pdf and pdf.exists():
            try:
                import fitz

                pages = fitz.open(pdf).page_count
            except Exception:
                pages = None
            try:
                sha = _sha256_file(pdf)
            except Exception:
                sha = None
        cur.execute(
            "INSERT INTO documents(doc_id, pdf_path, sha256, page_count) VALUES(?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET pdf_path=excluded.pdf_path, sha256=COALESCE(excluded.sha256, documents.sha256), page_count=COALESCE(excluded.page_count, documents.page_count), updated_at=datetime('now')",
            (doc_id, str(pdf or ""), sha, pages),
        )
        n_docs += 1

        # upsert each extraction line
        with jpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                j = json.loads(line)
                ev = j.get("evidence") or []
                page = None
                if isinstance(ev, list) and ev:
                    page = ev[0].get("page")
                issues = j.get("verifier_notes")
                cur.execute(
                    """
                    INSERT INTO extractions
                      (doc_id, field_id, value_canonical, unit, verified, confidence, page, status, issues, source_text, scale_applied, scale_source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(doc_id, field_id) DO UPDATE SET
                      value_canonical=excluded.value_canonical,
                      unit=excluded.unit,
                      verified=excluded.verified,
                      confidence=excluded.confidence,
                      page=excluded.page,
                      status=excluded.status,
                      issues=excluded.issues,
                      source_text=excluded.source_text,
                      scale_applied=excluded.scale_applied,
                      scale_source=excluded.scale_source,
                      updated_at=datetime('now');
                    """,
                    (
                        j["doc_id"],
                        j["field_id"],
                        j.get("value_canonical"),
                        j.get("unit"),
                        1 if j.get("verified") else 0,
                        j.get("confidence"),
                        page,
                        j.get("status"),
                        issues,
                        j.get("source_text"),
                        j.get("scale_applied"),
                        j.get("scale_source"),
                    ),
                )
                n_rows += 1

    con.commit()
    con.close()
    return n_docs, n_rows


def load_summaries_from_dir(
    out_dir: Optional[Path] = None, db_path: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Scan <output_dir_extract>/summaries for *.summaries.jsonl and upsert into SQLite.
    Returns: n_rows_upserted
    """
    cfg = get_settings()
    # default directory is the `summaries` subdir under the extract output root
    root = out_dir or Path(cfg.output_dir_summaries)
    con = connect(db_path)
    cur = con.cursor()

    n_section_summaries = 0
    n_docs = 0
    for jpath in sorted(root.glob("*.summaries.jsonl")):
        with jpath.open("r", encoding="utf-8") as fh:
            n_docs += 1
            for line in fh:
                if not line.strip():
                    continue
                j = json.loads(line)
                # ensure documents row exists/updated similarly to extractions
                doc_id = j["doc_id"]
                pdf = _find_pdf_for_doc(doc_id)
                sha, pages = None, None
                if pdf and pdf.exists():
                    try:
                        import fitz

                        pages = fitz.open(pdf).page_count
                    except Exception:
                        pages = None
                    try:
                        sha = _sha256_file(pdf)
                    except Exception:
                        sha = None
                cur.execute(
                    "INSERT INTO documents(doc_id, pdf_path, sha256, page_count) VALUES(?,?,?,?) "
                    "ON CONFLICT(doc_id) DO UPDATE SET pdf_path=excluded.pdf_path, sha256=COALESCE(excluded.sha256, documents.sha256), page_count=COALESCE(excluded.page_count, documents.page_count), updated_at=datetime('now')",
                    (doc_id, str(pdf or ""), sha, pages),
                )

                # upsert summary row per (doc_id, section_id)
                cur.execute(
                    """
                    INSERT INTO summaries
                      (doc_id, section_id, title, start_page, end_page, summary, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(doc_id, section_id) DO UPDATE SET
                      title=excluded.title,
                      start_page=excluded.start_page,
                      end_page=excluded.end_page,
                      summary=excluded.summary,
                      updated_at=datetime('now');
                    """,
                    (
                        doc_id,
                        j.get("section_id"),
                        j.get("title"),
                        j.get("start_page"),
                        j.get("end_page"),
                        j.get("summary") or "",
                    ),
                )
                n_section_summaries += 1

    con.commit()
    con.close()
    return n_docs, n_section_summaries


# ---------- queries for UI ----------


def list_documents(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    con = connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT doc_id, pdf_path, page_count FROM documents ORDER BY doc_id"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_extractions_for_doc(
    doc_id: str, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT field_id, value_canonical, unit, verified, confidence, page, status, issues, source_text, scale_applied, scale_source
        FROM extractions WHERE doc_id = ? ORDER BY field_id
    """,
        (doc_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_summaries_for_doc(
    doc_id: str, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT section_id, title, start_page, end_page, summary
        FROM summaries
        WHERE doc_id = ?
        ORDER BY CASE section_id WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4 WHEN 'E' THEN 5 ELSE 99 END, start_page
        """,
        (doc_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
