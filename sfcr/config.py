from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "artifacts" / "ingest"
    schema_file: Path = project_root / "schema" / "ingestion.schema.json"

    class Config:
        env_prefix = "SFCR_"  # allows SFCR_OUTPUT_DIR override


config = Settings()
