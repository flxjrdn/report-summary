from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from sfcr.config import get_settings
from sfcr.extract.extractor import extract_for_document, write_jsonl


def iter_pdfs(root: Path, pattern: str = "*.pdf") -> Iterable[Path]:
    return sorted(root.glob(pattern))


def find_ingest_json(stem: str, out_dir: Path) -> Path:
    return out_dir / f"{stem}.ingest.json"


def _get_spent_usd(llm) -> float:
    return float(getattr(llm, "spent_usd", 0.0))  # Ollama client has none → 0.0


def extract_directory(
    src_dir: Path,
    fields_yaml: Path,
    *,
    pattern: str = "*.pdf",
    out_dir: Optional[Path] = None,
    llm=None,  # prebuilt client (Ollama or Mock)
    resume: bool = True,
    limit: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[int, int, float]:
    cfg = get_settings()
    out_dir = out_dir or cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = list(iter_pdfs(src_dir, pattern))
    if limit is not None and limit >= 0:
        all_pdfs = all_pdfs[:limit]

    processed = 0
    skipped = 0

    def process(pdf: Path) -> bool:
        nonlocal processed, skipped
        doc_id = pdf.stem
        ingest_json = find_ingest_json(doc_id, out_dir)
        if not ingest_json.exists():
            skipped += 1
            print(
                f"[yellow]skip[/yellow] no ingestion JSON for {pdf.name} (expected {ingest_json.name})"
            )
            return True

        out_path = out_dir / f"{doc_id}.extractions.jsonl"
        if resume and out_path.exists():
            skipped += 1
            print(f"[yellow]skip[/yellow] already exists: {out_path.name}")
            return True

        rows = extract_for_document(
            doc_id=doc_id,
            pdf_path=pdf,
            ingestion_json=ingest_json,
            fields_yaml=fields_yaml,
            llm=llm,
        )
        write_jsonl(rows, out_path)
        processed += 1
        print(
            f"[green]✓[/green] {pdf.name} → {out_path.name}  (spent≈${_get_spent_usd(llm):.4f})"
        )
        return True

    if show_progress:
        with Progress(
            TextColumn("[bold]Extract[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("docs", total=len(all_pdfs))
            for pdf in all_pdfs:
                cont = process(pdf)
                progress.update(task, advance=1)
                if not cont:
                    break
    else:
        for pdf in all_pdfs:
            if not process(pdf):
                break

    return processed, skipped, _get_spent_usd(llm)
