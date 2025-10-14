from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["ok", "not_found", "ambiguous"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(..., ge=1)
    ref: Optional[str] = Field(default=None, description="e.g., table[r3,c2] or bbox")
    snippet_hash: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{8,64}$")


class ExtractionLLM(BaseModel):
    """
    The model MUST return this (pre-verification).
    """

    model_config = ConfigDict(extra="forbid")

    field_id: str
    status: Status
    value: Optional[float] = None  # value *before* applying 'scale'
    unit: Optional[Literal["EUR", "%"]] = None
    scale: Optional[float] = Field(default=None, description="1|1e3|1e6|…")
    currency: Optional[Literal["EUR"]] = "EUR"
    period_end: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    evidence: List[Evidence] = Field(default_factory=list)
    source_text: Optional[str] = Field(default=None, max_length=200)
    scale_source: Optional[
        Literal["row", "column", "caption", "nearby", "model_guess"]
    ] = None
    notes: Optional[str] = None

    @field_validator("source_text")
    @classmethod
    def _strip_source(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = re.sub(r"\s+", " ", v).strip()
        return v or None


class VerifiedExtraction(BaseModel):
    """
    Post-verification, canonicalized to EUR or % with provenance.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    field_id: str
    status: Status
    verified: bool
    value_canonical: Optional[float] = None  # EUR or %; final canonical
    unit: Optional[Literal["EUR", "%"]] = None
    currency: Optional[Literal["EUR"]] = None
    period_end: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    source_text: Optional[str] = None
    scale_applied: Optional[float] = None
    scale_source: Optional[str] = None
    verifier_notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "VerifiedExtraction":
        if self.verified and (self.value_canonical is None or self.unit is None):
            raise ValueError("verified=True requires value_canonical and unit")
        return self
