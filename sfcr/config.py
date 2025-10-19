from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for paths & files.
    Environment overrides (examples):
      SFCR_DATA=/path/to/pdfs
      SFCR_OUTPUT=/tmp/out
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # we provide explicit env names per field below
        extra="ignore",
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    data_dir: Path = Field(default=Path("data") / "samples", env="SFCR_DATA")
    output_dir: Path = Field(default=Path("artifacts") / "ingest", env="SFCR_OUTPUT")
    schema_file: Path = Field(default=Path("schema") / "ingestion.schema.json")

    @field_validator("data_dir", "output_dir", "schema_file", mode="after")
    def _expanduser(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("data_dir", "output_dir", "schema_file", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        # Accept strings from env and coerce; allow Path passthrough.
        if v is None:
            return v  # will not occur for non-optional fields unless user passes None explicitly
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return Path(s).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    def model_post_init(self, __context) -> None:
        # Resolve relative paths against project_root
        if not self.data_dir.is_absolute():
            self.data_dir = (self.project_root / self.data_dir).resolve()
        if not self.output_dir.is_absolute():
            self.output_dir = (self.project_root / self.output_dir).resolve()
        if not self.schema_file.is_absolute():
            self.schema_file = (self.project_root / self.schema_file).resolve()

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Lazy singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
