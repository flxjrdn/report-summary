import json
import os
import pathlib

from definitions import PATH_SCHEMA
from ingest.schema import export_json_schema

out = pathlib.Path(os.path.join(PATH_SCHEMA, "ingestion.schema.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(export_json_schema(), indent=2), encoding="utf-8")
print(f"Wrote {out}")
