from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import typer
from rich import print

from sfcr.eval.eval import evaluate, format_report, load_gold, load_preds
from sfcr.eval.goldgen import generate_gold
from sfcr.extract.extractor import extract_for_document, write_jsonl

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
    src: Path = typer.Argument(
        None, help="PDF path or directory; defaults to SFCR_DATA or repo default"
    ),
    outdir: Path = typer.Option(
        None, "--outdir", help="Output dir; defaults to SFCR_OUTPUT or repo default"
    ),
    doc_id: str = typer.Option(None, help="Optional override for single-file ingest"),
):
    """
    Ingest a PDF or a directory of PDFs.
    Precedence: CLI args > env (SFCR_DATA/SFCR_OUTPUT) > repo defaults.
    """
    cfg = get_settings()
    # Resolve effective paths
    effective_src: Path | None = src or cfg.data_dir
    effective_out: Path | None = outdir or cfg.output_dir

    if effective_src is None:
        raise ValueError("No source path for pdf files specified")
    if effective_out is None:
        raise ValueError("No output path for ingestion results specified")

    files = (
        [effective_src]
        if effective_src.is_file()
        else sorted(effective_src.glob("**/*.pdf"))
    )
    if not files:
        typer.secho(f"No PDFs found in {effective_src}", fg="red")
        raise typer.Exit(1)

    for p in files:
        did = doc_id or p.stem
        ing = SFCRIngestor(doc_id=did, pdf_path=str(p))
        res = ing.run()

        payload = {
            "schema_version": "1.0.0",
            "doc_id": did,
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
        out_path = effective_out / f"{did}.ingest.json"
        out_path.write_text(
            json_text,
            encoding="utf-8",
        )
        print(f"[green]✓[/green] {p.name} → {out_path}")


@app.command("validate")
def validate_file(json_path: Path):
    """Validate a single *.ingest.json against the schema."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    IngestionResult(**data)
    print("[green]OK[/green]")


@app.command("validate-dir")
def validate_dir(
    dirpath: Path = typer.Argument(
        None, help="Directory containing .ingest.json; defaults to SFCR_OUTPUT"
    ),
):
    """Validate all *.ingest.json in a directory (defaults to cfg.output_dir)."""
    cfg = get_settings()
    target = dirpath or cfg.output_dir

    if target is None:
        raise ValueError("No target directory to validate was specified")

    files = sorted(target.glob("*.ingest.json"))
    if not files:
        typer.secho(f"No .ingest.json files found in {target}", fg="yellow")
        raise typer.Exit(1)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        IngestionResult(**data)
        print(f"[green]OK[/green] {f.name}")


@app.command()
def extract(
    pdf: Path = typer.Argument(
        None, help="PDF path; defaults to first PDF under data_dir"
    ),
    ingest_json: Path = typer.Option(None, help="Path to *.ingest.json for this PDF"),
    fields: Path = typer.Option(
        None, help="Path to fields.yaml (defaults to sfcr/extract/fields.yaml)"
    ),
    out: Path = typer.Option(
        None, help="Output JSONL (defaults to <output_dir>/<doc_id>.extractions.jsonl)"
    ),
):
    """
    Run extraction+verification for one document (uses ingestion spans).
    """
    cfg = get_settings()
    # resolve paths
    if pdf is None:
        # pick first PDF in data_dir
        pdfs = sorted(Path(cfg.data_dir).glob("*.pdf"))
        if not pdfs:
            typer.secho("No PDFs found; specify --pdf", fg="red")
            raise typer.Exit(1)
        pdf = pdfs[0]
    doc_id = pdf.stem

    if ingest_json is None:
        cand = Path(cfg.output_dir) / f"{doc_id}.ingest.json"
        if not cand.exists():
            typer.secho(f"Missing ingestion JSON: {cand}", fg="red")
            raise typer.Exit(1)
        ingest_json = cand

    if fields is None:
        fields = Path("sfcr/extract/fields.yaml")
        if not fields.exists():
            typer.secho(f"fields.yaml not found at {fields}", fg="red")
            raise typer.Exit(1)

    if out is None:
        out = Path(cfg.output_dir) / f"{doc_id}.extractions.jsonl"

    rows = extract_for_document(
        doc_id, pdf, ingest_json, fields, llm=None
    )  # use MockLLM by default
    write_jsonl(rows, out)
    print(f"[green]✓[/green] wrote {out}")


@app.command()
def eval(
    gold_csv: Path = typer.Argument(
        Path("data/gold/gold.csv"), help="Gold CSV (doc_id,field_id,unit,value)"
    ),
    preds_dir: Path = typer.Option(
        None, help="Dir containing *.extractions.jsonl (defaults to output_dir)"
    ),
    report_out: Path = typer.Option(None, help="Optional path to write a text report"),
):
    """
    Evaluate verified extractions against a small gold set.
    """
    cfg = get_settings()
    preds_root = preds_dir or cfg.output_dir
    if preds_root is None:
        raise ValueError("No predictions directory specified")

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


if __name__ == "__main__":
    app()
