from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import typer
from rich import print

from sfcr.db import init_db as db_init
from sfcr.db import load_catalog, load_extractions_from_dir, load_summaries_from_dir
from sfcr.eval.eval import evaluate, format_report, load_gold, load_preds
from sfcr.eval.goldgen import generate_gold
from sfcr.extract.batch import extract_directory
from sfcr.extract.extractor import LLMExtractor, extract_for_document, write_jsonl
from sfcr.llm.llm_text_client_factory import create_llm_text_client
from sfcr.summarize.summarize import run_summarize

# If not installed in editable mode, add repo root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfcr.config import get_settings
from sfcr.ingest.schema import IngestionResult
from sfcr.ingest.sfcr_ingest import SFCRIngestor

app = typer.Typer(add_completion=False, help="SFCR demo pipeline (lean CLI)")


def _sha256(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@app.command()
def ingest(
    doc_id: str = typer.Option(
        "", help="Document ID referring to a pdf file to be ingested"
    ),
    outdir: Path = typer.Option(
        None, "--outdir", help="Output dir; defaults to SFCR_OUTPUT or repo default"
    ),
):
    if doc_id is None:
        raise ValueError(f"{doc_id} is required when calling ingest for a single file")

    cfg = get_settings()
    pdf_path = cfg.pdfs_dir / f"{doc_id}.pdf"
    if not pdf_path.is_file():
        raise ValueError(
            f"the specified doc_id has to refer to an existing PDF: {pdf_path}, but this file does not exist"
        )
    ingest_dir(src=pdf_path, outdir=outdir)


@app.command()
def ingest_dir(
    src: Path = typer.Argument(
        None, help="PDF path or directory; defaults to SFCR_DATA or repo default"
    ),
    outdir: Path = typer.Option(
        None, "--outdir", help="Output dir; defaults to SFCR_OUTPUT or repo default"
    ),
):
    """
    Ingest a directory of PDFs.
    Precedence: CLI args > env (SFCR_DATA/SFCR_OUTPUT) > repo defaults.
    """
    cfg = get_settings()
    # Resolve effective paths
    effective_src: Path = src or cfg.pdfs_dir
    effective_out: Path = outdir or cfg.output_dir_ingest

    files = (
        [effective_src]
        if effective_src.is_file()
        else sorted(effective_src.glob("**/*.pdf"))
    )
    if not files:
        typer.secho(f"No PDFs found in {effective_src}", fg="red")
        raise typer.Exit(1)

    for p in files:
        doc_id = p.stem
        ing = SFCRIngestor(doc_id=doc_id, pdf_path=str(p))
        res = ing.run()

        payload = {
            "schema_version": "1.0.0",
            "doc_id": doc_id,
            "pdf_sha256": _sha256(p),
            "page_count": ing.loader.page_count(),
            "sections": [s.__dict__ for s in res.sections],
            "subsections": [s.__dict__ for s in res.subsections],
            "coverage_ratio": res.coverage_ratio,
            "issues": res.issues,
        }
        # Validate against frozen contract & dump deterministically
        ir = IngestionResult(**payload)
        payload_dict = ir.model_dump(exclude_none=True)
        json_text = json.dumps(
            payload_dict,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        out_path = effective_out / f"{doc_id}.ingest.json"
        out_path.write_text(
            json_text,
            encoding="utf-8",
        )
        print(f"[green]✓[/green] {p.name} → {out_path}")


@app.command()
def extract(
    pdf: Path = typer.Argument(
        None, help="PDF path; defaults to first PDF under pdfs_dir"
    ),
    ingest_json: Path = typer.Option(None, help="Path to *.ingest.json for this PDF"),
    fields: Path = typer.Option(
        None, help="Path to fields.yaml (default: sfcr/extract/fields.yaml)"
    ),
    out: Path = typer.Option(
        None,
        help="Output JSONL (default: <output_dir_extract>/<doc_id>.extractions.jsonl)",
    ),
    provider: str = typer.Option("mock", help="LLM provider: ollama | openai | mock"),
    model: str = typer.Option(
        "", help="Model name for provider (e.g., 'mistral' for ollama)"
    ),
):
    cfg = get_settings()
    if pdf is None:
        pdfs = sorted(Path(cfg.pdfs_dir).glob("*.pdf"))
        if not pdfs:
            typer.secho("No PDFs found; specify --pdf", fg="red")
            raise typer.Exit(1)
        pdf = pdfs[0]
    doc_id = pdf.stem

    if ingest_json is None:
        cand = Path(cfg.output_dir_ingest) / f"{doc_id}.ingest.json"
        if not cand.exists():
            typer.secho(f"Missing ingestion JSON: {cand}", fg="red")
            raise typer.Exit(1)
        ingest_json = cand

    if fields is None:
        fields = Path("sfcr/extract/fields.yaml")
        if not fields.exists():
            typer.secho(f"fields.yaml not found: {fields}", fg="red")
            raise typer.Exit(1)

    if out is None:
        out = Path(cfg.output_dir_extract) / f"{doc_id}.extractions.jsonl"

    llm = create_llm_text_client(provider, model=model)
    extractor = LLMExtractor(text_client=llm)
    rows = extract_for_document(doc_id, pdf, ingest_json, fields, extractor=extractor)
    write_jsonl(rows, out)
    print(f"[green]✓[/green] wrote {out}")


@app.command("extract-dir")
def extract_dir(
    src: Path = typer.Argument(None, help="Directory of PDFs; defaults to SFCR_DATA"),
    fields: Path = typer.Option(
        Path("sfcr/extract/fields.yaml"), help="Path to fields.yaml"
    ),
    provider: str = typer.Option("mock", help="LLM provider: openai | ollama | mock"),
    model: str = typer.Option("", help="Model for provider (e.g., 'mistral')"),
    pattern: str = typer.Option("*.pdf", help="Glob for PDFs under src"),
    resume: bool = typer.Option(
        True, help="Skip PDFs with existing .extractions.jsonl"
    ),
    limit: int = typer.Option(-1, help="Process at most N PDFs; -1 = no limit"),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable progress bar output"
    ),
):
    cfg = get_settings()
    src_dir = src or Path(cfg.pdfs_dir)
    if not src_dir.exists():
        typer.secho(f"Source dir not found: {src_dir}", fg="red")
        raise typer.Exit(1)
    if not fields.exists():
        typer.secho(f"fields.yaml not found: {fields}", fg="red")
        raise typer.Exit(1)

    llm = create_llm_text_client(provider, model=model)
    extractor = LLMExtractor(text_client=llm)

    processed, skipped = extract_directory(
        src_dir=src_dir,
        fields_yaml=fields,
        pattern=pattern,
        extractor=extractor,
        resume=resume,
        limit=None if limit is None or limit < 0 else int(limit),
        show_progress=not no_progress,
    )
    print(f"\n=== Batch done ===\nProcessed: {processed}  Skipped: {skipped}")


@app.command()
def eval(
    gold_csv: Path = typer.Argument(
        Path("data/gold/gold.csv"), help="Gold CSV (doc_id,field_id,unit,value)"
    ),
    preds_dir: Path = typer.Option(
        None, help="Dir containing *.extractions.jsonl (defaults to output_dir_extract)"
    ),
    report_out: Path = typer.Option(None, help="Optional path to write a text report"),
):
    """
    Evaluate verified extractions against a small gold set.
    """
    cfg = get_settings()
    preds_root = preds_dir or cfg.output_dir_extract

    gold = load_gold(gold_csv)
    preds = load_preds(preds_root)
    res, errors = evaluate(gold, preds)

    text = format_report(res)
    print(text)
    if errors:
        print("\n--- Issues (first 50) ---")
        for e in errors[:50]:
            print(e)

    if report_out:
        report_out.write_text(
            text + ("\n\n" + "\n".join(errors) if errors else ""), encoding="utf-8"
        )
        print(f"[green]✓[/green] wrote {report_out}")


@app.command()
def gold(
    out: Path = typer.Option(
        None, help="Path to gold.csv (default: data/gold/gold.csv)"
    ),
    include_unverified: bool = typer.Option(
        False, "--include-unverified", help="Also add unverified rows"
    ),
    no_backup: bool = typer.Option(
        False, "--no-backup", help="Do not write gold.csv.bak before merging"
    ),
):
    """
    Merge current extractions into gold.csv (non-destructive).
    Existing entries are preserved; new (doc_id, field_id) pairs are appended.
    """
    path = generate_gold(
        out_path=out, only_verified=not include_unverified, backup=not no_backup
    )
    print(f"[green]✓[/green] gold written to {path}")


@app.command()
def summarize(
    pdf: Path = typer.Option(
        None,
        help="Path to PDF; defaults to <pdfs_dir>/<doc_id>.pdf (or recursive search)",
    ),
    ingest_json: Path = typer.Option(
        None,
        help="Path to *.ingest.json; defaults to <output_dir_ingest>/<doc_id>.ingest.json",
    ),
    out: Path = typer.Option(
        None,
        help="Output JSONL; defaults to <output_dir_extract>/summaries/<doc_id>.summaries.jsonl",
    ),
    provider: str = typer.Option("mock", help="LLM provider (e.g., 'ollama')"),
    model: str = typer.Option("mock", help="Model name for provider (e.g., 'mistral')"),
):
    """
    Create a concise, section-level summary (A–E) JSONL for DOC_ID.
    """
    cfg = get_settings()

    if pdf is None:
        pdfs = sorted(Path(cfg.pdfs_dir).glob("*.pdf"))
        if not pdfs:
            typer.secho("No PDFs found; specify --pdf", fg="red")
            raise typer.Exit(1)
        pdf = pdfs[0]
    doc_id = pdf.stem

    if ingest_json is None:
        cand = Path(cfg.output_dir_ingest) / f"{doc_id}.ingest.json"
        if not cand.exists():
            typer.secho(f"Missing ingestion JSON: {cand}", fg="red")
            raise typer.Exit(1)
        ingest_json = cand

    if out is None:
        out = Path(cfg.output_dir_summaries) / f"{doc_id}.summaries.jsonl"

    out.parent.mkdir(parents=True, exist_ok=True)

    # Run summarization
    path = run_summarize(
        doc_id=doc_id,
        pdf_path=pdf,
        ingest_json=ingest_json,
        out_jsonl=out,
        provider=provider,
        model=model,
    )
    print(f"[green]✓[/green] wrote {path}")


@app.command("summarize-dir")
def summarize_dir(
    src: Path = typer.Argument(
        None, help="Directory containing *.ingest.json; defaults to SFCR_OUTPUT/ingest"
    ),
    provider: str = typer.Option(
        "mock", help="LLM provider (e.g., 'ollama' or 'mock')"
    ),
    model: str = typer.Option("mock", help="Model name for provider (e.g., 'mistral')"),
    pattern: str = typer.Option(
        "*.ingest.json", help="Glob pattern for ingestion JSONs"
    ),
    skip_existing: bool = typer.Option(True, help="Skip docs with existing summaries"),
    limit: int = typer.Option(-1, help="Process at most N docs; -1 = no limit"),
):
    cfg = get_settings()
    target = src or Path(cfg.output_dir_ingest)
    files = sorted(target.glob(pattern))
    if not files:
        typer.secho(f"No ingestion JSON files found in {target}", fg="red")
        raise typer.Exit(1)
    processed = 0
    skipped = 0
    for f in files:
        doc_id = f.name.replace(".ingest.json", "")
        out_path = Path(cfg.output_dir_summaries) / f"{doc_id}.summaries.jsonl"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        pdf = Path(cfg.pdfs_dir) / f"{doc_id}.pdf"
        if not pdf.exists():
            typer.secho(f"Warning: PDF not found for {doc_id}, skipping", fg="yellow")
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        run_summarize(
            doc_id=doc_id,
            pdf_path=pdf,
            ingest_json=f,
            out_jsonl=out_path,
            provider=provider,
            model=model,
        )
        processed += 1
        if limit >= 0 and processed >= limit:
            break
    print(
        f"[green]✓[/green] Summarize-dir done. Processed: {processed}  Skipped: {skipped}"
    )


@app.command("db-init")
def db_init_cmd():
    p = db_init()
    print(f"[green]✓[/green] DB ready at {p}")


@app.command("db-load")
def db_load_cmd():
    n_docs = load_catalog()
    print(f"[green]✓[/green] loaded {n_docs} documents")

    n_docs_extr, n_values_extr = load_extractions_from_dir()
    print(f"[green]✓[/green] loaded {n_values_extr} values from {n_docs_extr} docs")

    n_docs_summaries, n_section_summaries = load_summaries_from_dir()
    print(
        f"[green]✓[/green] loaded {n_section_summaries} section summaries from {n_docs_summaries} docs"
    )


@app.command("ui")
def ui_cmd():
    """
    Launch the Streamlit viewer.
    """
    import subprocess
    import sys

    app = "tools/ui_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", app], check=False)


if __name__ == "__main__":
    app()
