import glob
import json
import pathlib

from sfcr.ingest.schema import IngestionResult


def test_examples_validate():
    for p in glob.glob("examples/*.ingest.json"):
        data = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        IngestionResult(**data)  # raises if invalid


def test_basic_invariants():
    # minimal synthetic example
    data = {
        "schema_version": "1.0.0",
        "doc_id": "foo",
        "pdf_sha256": "0" * 64,
        "page_count": 50,
        "sections": [
            {
                "section": "A",
                "start_page": 3,
                "end_page": 8,
            },
            {
                "section": "B",
                "start_page": 9,
                "end_page": 15,
            },
            {
                "section": "Z",
                "start_page": 46,
                "end_page": 50,
            },
        ],
        "subsections": [],
        "coverage_ratio": 0.7,
        "issues": [],
    }
    IngestionResult(**data)
