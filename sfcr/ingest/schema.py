from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -----------------------------
# Type aliases
# -----------------------------
SectionLetter = Literal["A", "B", "C", "D", "E", "Z"]  # Z = synthetic post-E section

# -----------------------------
# Core schema
# -----------------------------
class SectionSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: SectionLetter
    # 1-based inclusive indices
    start_page: int = Field(
        ..., ge=1, description="1-based inclusive page index where section starts"
    )
    end_page: int = Field(
        ..., ge=1, description="1-based inclusive page index where section ends"
    )

    @model_validator(mode="after")
    def _page_order(self) -> "SectionSpan":
        if self.start_page > self.end_page:
            raise ValueError("start_page must be ≤ end_page")
        return self


class SubsectionSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # only A–E have subsections
    section: Literal["A", "B", "C", "D", "E"]
    code: str = Field(..., pattern=r"^[A-E]\.\d{1,2}(\.\d{1,2})?$")
    title: str = Field(..., min_length=1, max_length=300)
    start_page: int = Field(
        ..., ge=1, description="1-based inclusive page index where subsection starts"
    )
    end_page: int = Field(
        ..., ge=1, description="1-based inclusive page index where subsection ends"
    )

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", (v or "")).strip()
        if not v:
            raise ValueError("empty title")
        return v


class IngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field("1.0.0", description="SemVer of the ingestion schema")
    doc_id: str = Field(..., min_length=1, max_length=200)
    pdf_sha256: Optional[str] = Field(None, pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(..., ge=1)

    sections: List[SectionSpan]
    subsections: List[SubsectionSpan] = Field(default_factory=list)

    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_sections(self) -> "IngestionResult":
        secs = self.sections or []
        letters = [s.section for s in secs]

        # ordered & non-overlapping within [1, page_count]
        last_end = 0
        for s in secs:
            if s.start_page < 1 or s.end_page > self.page_count:
                raise ValueError("section page out of bounds")
            if s.start_page <= last_end:
                raise ValueError(
                    "sections must be strictly ordered and non-overlapping"
                )
            last_end = s.end_page

        # at most one Z; if present, last & ends at page_count
        if letters.count("Z") > 1:
            raise ValueError("only one post section (Z) allowed")
        if "Z" in letters:
            z = next(s for s in secs if s.section == "Z")
            if secs and z is not secs[-1]:
                raise ValueError("post section (Z) must be last")
            if z.end_page != self.page_count:
                raise ValueError("post section (Z) must end at page_count")

        return self


# -----------------------------
# JSON Schema export
# -----------------------------
def export_json_schema() -> dict:
    """Export the JSON Schema for this contract (Pydantic v2)."""
    return IngestionResult.model_json_schema()
