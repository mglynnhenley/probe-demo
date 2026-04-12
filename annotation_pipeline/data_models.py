"""Pydantic models for policy-violation annotation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PolicyViolationSpan(BaseModel):
    """A substring of the completion that explicitly violates the policy."""

    model_config = ConfigDict(extra="ignore")

    span: str
    verification_note: str = ""
    index: Optional[int] = None  # character offset into completion (UTF-8 code points, Python str index)


class GenerationRecord(BaseModel):
    """One line of generations JSONL (input) or annotated JSONL (output)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    source_dataset: str = ""
    model: str
    completion: str
    policy: str = Field(description="Natural language policy; violations are relative to this text.")
    annotations: Optional[list[PolicyViolationSpan]] = None
    annotator_model: Optional[str] = None
