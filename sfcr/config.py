from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for paths & files.
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # we provide explicit env names per field below
        extra="ignore",
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    data_dir: Path = Field(default=Path("data"))
    pdfs_dir: Path = Field(default=Path("data") / "sfcrs", env="SFCR_DATA")
    output_dir: Path = Field(default=Path("artifacts"), env="SFCR_OUTPUT")

    @field_validator("data_dir", "pdfs_dir", "output_dir", mode="after")
    def _expanduser(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("data_dir", "pdfs_dir", "output_dir", mode="before")
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

    @computed_field(return_type=Path)
    def output_dir_ingest(self) -> Path:
        dir_ingest = self.output_dir / "ingest"
        dir_ingest.mkdir(parents=True, exist_ok=True)
        return dir_ingest

    @computed_field(return_type=Path)
    def output_dir_extract(self) -> Path:
        dir_extract = self.output_dir / "extract"
        dir_extract.mkdir(parents=True, exist_ok=True)
        return dir_extract

    @computed_field(return_type=Path)
    def output_dir_summaries(self) -> Path:
        dir_summaries = self.output_dir / "summaries"
        dir_summaries.mkdir(parents=True, exist_ok=True)
        return dir_summaries

    def model_post_init(self, __context) -> None:
        # Resolve relative paths against project_root
        if not self.data_dir:
            self.data_dir = (self.project_root / self.data_dir).resolve()
        if not self.pdfs_dir.is_absolute():
            self.pdfs_dir = (self.project_root / self.pdfs_dir).resolve()
        if not self.output_dir.is_absolute():
            self.output_dir = (self.project_root / self.output_dir).resolve()

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_ingest.mkdir(parents=True, exist_ok=True)
        self.output_dir_extract.mkdir(parents=True, exist_ok=True)
        self.output_dir_summaries.mkdir(parents=True, exist_ok=True)


# Lazy singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
