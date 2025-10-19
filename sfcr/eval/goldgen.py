from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sfcr.config import get_settings


def collect_extractions(dir_path: Path) -> List[dict]:
    """
    Read all *.extractions.jsonl in dir_path and return normalized rows:
      {doc_id, field_id, unit, value, verified, status}
    """
    items: List[dict] = []
    for f in sorted(dir_path.glob("*.extractions.jsonl")):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                j = json.loads(line)
                items.append(
                    {
                        "doc_id": j.get("doc_id"),
                        "field_id": j.get("field_id"),
                        "unit": j.get("unit"),
                        "value": j.get("value_canonical"),
                        "verified": bool(j.get("verified", False)),
                        "status": j.get("status", ""),
                    }
                )
    return items


def _load_existing_gold(gold_path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Load existing gold.csv (if any) and return a dict keyed by (doc_id, field_id).
    Keeps the whole row, so we can write it back unchanged.
    """
    existing: Dict[Tuple[str, str], dict] = {}
    if gold_path.exists():
        with gold_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row.get("doc_id") or not row.get("field_id"):
                    continue
                key = (row["doc_id"].strip(), row["field_id"].strip())
                # Preserve exactly what was in the file
                existing[key] = {
                    "doc_id": row["doc_id"].strip(),
                    "field_id": row["field_id"].strip(),
                    "unit": (row.get("unit") or "").strip(),
                    "value": (row.get("value") or "").strip(),
                }
    return existing


def merge_into_gold(
    extractions: List[dict],
    gold_path: Path,
    *,
    only_verified: bool = True,
    backup: bool = True,
) -> Tuple[int, int]:
    """
    Merge new (doc_id, field_id) pairs into gold.csv.
    - Does NOT overwrite existing gold rows.
    - If only_verified=True, only include verified extractions.
    - Creates parent dir; optional .bak backup.

    Returns: (num_existing, num_added)
    """
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_gold(gold_path)
    n_existing = len(existing)

    # Optional backup
    if backup and gold_path.exists():
        shutil.copyfile(gold_path, gold_path.with_suffix(gold_path.suffix + ".bak"))

    # Build new rows to add (skip any keys already present)
    to_add: Dict[Tuple[str, str], dict] = {}
    for j in extractions:
        if only_verified and not j["verified"]:
            continue
        key = (j["doc_id"], j["field_id"])
        if key in existing or key in to_add:
            continue
        to_add[key] = {
            "doc_id": j["doc_id"],
            "field_id": j["field_id"],
            "unit": j.get("unit") or "",
            "value": "",  # leave value blank, to ensure it is verified manually
        }

    # Merge: existing first (unchanged), then new rows in stable order
    fieldnames = ["doc_id", "field_id", "unit", "value"]
    with gold_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Write existing exactly as-is
        for row in existing.values():
            w.writerow(row)
        # Append new rows sorted for readability (by doc_id, field_id)
        for (_, _), row in sorted(
            to_add.items(), key=lambda kv: (kv[1]["doc_id"], kv[1]["field_id"])
        ):
            w.writerow(row)

    return n_existing, len(to_add)


def generate_gold(
    out_path: Optional[Path] = None,
    only_verified: bool = True,
    backup: bool = True,
) -> Path:
    """
    Convenience wrapper:
      - Reads extractions from Settings.output_dir
      - Merges into data/gold/gold.csv (or provided out_path)
    """
    cfg = get_settings()
    extractions_root = cfg.output_dir  # where *.extractions.jsonl live
    gold_path = out_path or (cfg.project_root / "data" / "gold" / "gold.csv")

    items = collect_extractions(extractions_root)
    _, added = merge_into_gold(
        items, gold_path, only_verified=only_verified, backup=backup
    )
    print(f"Gold updated: {gold_path} (+{added} new rows)")
    return gold_path
