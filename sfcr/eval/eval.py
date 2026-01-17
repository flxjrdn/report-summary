from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GoldRow:
    doc_id: str
    field_id: str
    unit: str
    value: float


@dataclass
class PredRow:
    doc_id: str
    field_id: str
    unit: Optional[str]
    value_canonical: Optional[float]
    verified: bool
    status: str


def load_gold(csv_path: Path) -> Dict[Tuple[str, str], GoldRow]:
    gold: Dict[Tuple[str, str], GoldRow] = {}
    print(f"Reading {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if not row.get("doc_id"):
                continue
            key = (row["doc_id"].strip(), row["field_id"].strip())
            gold[key] = GoldRow(
                doc_id=row["doc_id"].strip(),
                field_id=row["field_id"].strip(),
                unit=row["unit"].strip(),
                value=float(row["value"]),
            )
    return gold


def load_preds(jsonl_dir: Path) -> Dict[Tuple[str, str], PredRow]:
    """
    Reads all *.extractions.jsonl in a directory.
    Keeps the latest occurrence per (doc_id, field_id) if duplicates exist.
    """
    preds: Dict[Tuple[str, str], PredRow] = {}
    for p in sorted(jsonl_dir.glob("*.extractions.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                key = (j["doc_id"], j["field_id"])
                preds[key] = PredRow(
                    doc_id=j["doc_id"],
                    field_id=j["field_id"],
                    unit=j.get("unit"),
                    value_canonical=j.get("value_canonical"),
                    verified=bool(j.get("verified", False)),
                    status=j.get("status", "ok"),
                )
    return preds


# --- Matching & metrics -------------------------------------------------------


def within_tolerance(gold: GoldRow, pred: PredRow) -> bool:
    if pred.value_canonical is None or pred.unit is None:
        return False
    if pred.unit != gold.unit:
        return False
    # Tolerances: EUR -> max(500, 0.001 * |gold|); % -> 0.2 pp
    if gold.unit == "EUR":
        tol = max(500.0, 0.001 * abs(gold.value))
    elif gold.unit == "%":
        tol = 0.2
    else:
        tol = 0.0
    return abs(pred.value_canonical - gold.value) <= tol


@dataclass
class EvalResult:
    n_gold: int
    n_verified: int
    n_correct_verified: int
    n_within_tol: int
    n_missing: int
    n_wrong_unit: int
    n_unverified_but_ok: int
    # rates
    accuracy: float
    precision_verified: float
    recall_verified: float
    verified_coverage: float
    abstention_rate: float


def evaluate(
    gold: Dict[Tuple[str, str], GoldRow], preds: Dict[Tuple[str, str], PredRow]
) -> Tuple[EvalResult, List[str]]:
    n_gold = len(gold)
    n_verified = 0
    n_correct_verified = 0
    n_within_tol = 0
    n_missing = 0
    n_wrong_unit = 0
    n_unverified_but_ok = 0
    errors: List[str] = []

    for key, g in gold.items():
        p = preds.get(key)
        if not p:
            n_missing += 1
            errors.append(f"MISSING {g.doc_id}/{g.field_id}")
            continue

        ok = within_tolerance(g, p)
        if ok:
            n_within_tol += 1

        if p.verified:
            n_verified += 1
            if ok:
                n_correct_verified += 1
            else:
                errors.append(
                    f"WRONG {g.doc_id}/{g.field_id} pred={p.value_canonical} {p.unit} gold={g.value} {g.unit}"
                )
        else:
            if ok:
                n_unverified_but_ok += 1
            # unit mismatch is a helpful counter
            if p.unit is not None and p.unit != g.unit:
                n_wrong_unit += 1

    # rates
    accuracy = (n_within_tol / n_gold) if n_gold else 0.0
    precision_verified = (n_correct_verified / n_verified) if n_verified else 0.0
    recall_verified = (n_correct_verified / n_gold) if n_gold else 0.0
    verified_coverage = (n_verified / n_gold) if n_gold else 0.0
    abstention_rate = 1.0 - verified_coverage

    res = EvalResult(
        n_gold=n_gold,
        n_verified=n_verified,
        n_correct_verified=n_correct_verified,
        n_within_tol=n_within_tol,
        n_missing=n_missing,
        n_wrong_unit=n_wrong_unit,
        n_unverified_but_ok=n_unverified_but_ok,
        accuracy=accuracy,
        precision_verified=precision_verified,
        recall_verified=recall_verified,
        verified_coverage=verified_coverage,
        abstention_rate=abstention_rate,
    )
    return res, errors


def format_report(res: EvalResult) -> str:
    lines = []

    def pct(x: float) -> str:
        return f"{100*x:.1f}%"

    lines.append("=== Extraction Evaluation ===")
    lines.append(f"Gold items            : {res.n_gold}")
    lines.append(f"Verified predictions  : {res.n_verified}")
    lines.append(
        f"Within tolerance (all): {res.n_within_tol} / {res.n_gold}  (accuracy={pct(res.accuracy)})"
    )
    lines.append(
        f"Correct among verified: {res.n_correct_verified} / {res.n_verified}  (precision={pct(res.precision_verified)})"
    )
    lines.append(f"Recall (verified corr): {pct(res.recall_verified)}")
    lines.append(
        f"Verified coverage     : {pct(res.verified_coverage)}   (abstention={pct(res.abstention_rate)})"
    )
    lines.append(f"Unverified but OK     : {res.n_unverified_but_ok}")
    lines.append(f"Wrong-unit count      : {res.n_wrong_unit}")
    return "\n".join(lines)
