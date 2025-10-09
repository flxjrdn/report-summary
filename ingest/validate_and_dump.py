import json
from pathlib import Path

from .schema import IngestionResult


def dump_ingestion_result(result_dict: dict, out_path: Path) -> None:
    # validate
    ir = IngestionResult(**result_dict)
    # deterministic JSON
    out_path.write_text(
        json.dumps(ir.model_dump(by_alias=False, exclude_none=True), indent=2),
        encoding="utf-8",
    )
