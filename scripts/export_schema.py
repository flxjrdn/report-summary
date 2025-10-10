import json

from sfcr.config import config
from sfcr.ingest.schema import export_json_schema

out = config.schema_file
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(export_json_schema(), indent=2), encoding="utf-8")
print(f"Wrote {out}")
