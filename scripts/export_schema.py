import json

from sfcr.config import get_settings
from sfcr.ingest.schema import export_json_schema

out = get_settings().schema_file
if out is None:
    raise ValueError("No schema file specified")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(export_json_schema(), indent=2), encoding="utf-8")
print(f"Wrote {out}")
