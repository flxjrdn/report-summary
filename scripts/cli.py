from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import typer
from rich import print

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


if __name__ == "__main__":
    app()
