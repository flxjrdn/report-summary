from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer
from rich import print

from ingest.schema import IngestionResult  # pydantic v2 schema
from ingest.sfcr_ingest import SFCRIngestor  # your module

app = typer.Typer(add_completion=False, help="SFCR demo pipeline (lean CLI)")


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@app.command()
def ingest(
    pdf: Path = typer.Argument(
        ..., exists=True, readable=True, help="PDF path or directory"
    ),
    outdir: Path = typer.Option(
        Path("artifacts/ingest"), help="Where to write .ingest.json"
    ),
    doc_id: str = typer.Option(
        None, help="Override doc_id (defaults to filename stem)"
    ),
):
    """
    Run ingestion on a PDF or all PDFs in a directory.
    Produces validated *.ingest.json files (schema v1.0.0).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    files = [pdf] if pdf.is_file() else sorted(p for p in pdf.glob("**/*.pdf"))
    if not files:
        typer.secho("No PDFs found.", fg="red")
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
        # validate contract
        ir = IngestionResult(**payload)
        out_path = outdir / f"{did}.ingest.json"
        out_path.write_text(
            json.dumps(
                ir.model_dump(
                    exclude_none=True, sort_keys=True, ensure_ascii=False, indent=2
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[green]✓[/green] {p.name} → {out_path}")


@app.command()
def validate(json_path: Path):
    """
    Validate an existing *.ingest.json against the schema (exit 1 on failure).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    IngestionResult(**data)
    print("[green]OK[/green]")


if __name__ == "__main__":
    app()
