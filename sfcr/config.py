from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for paths & files.
    Environment overrides use the SFCR_ prefix, e.g.:
      SFCR_DATA=/path/to/pdfs
      SFCR_OUTPUT=/tmp/out
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # we provide explicit aliases below
        extra="ignore",
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_dir: Path | None = Field(
        default=None,
        validation_alias="SFCR_DATA",
    )
    output_dir: Path | None = Field(
        default=None,
        validation_alias="SFCR_OUTPUT",
    )
    schema_file: Path | None = None

    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        # Accept None / empty string and resolve later
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v  # let pydantic try

    def model_post_init(self, __context) -> None:
        # Fill defaults if env not provided
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "samples"
        if self.output_dir is None:
            self.output_dir = self.project_root / "artifacts" / "ingest"
        if self.schema_file is None:
            self.schema_file = self.project_root / "schema" / "ingestion.schema.json"
        # Ensure the output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Lazy singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
