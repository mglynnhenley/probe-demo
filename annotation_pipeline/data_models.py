"""Pydantic models for policy-violation and hallucination annotation."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PolicyViolationSpan(BaseModel):
    """A substring of the completion that explicitly violates the policy."""

    model_config = ConfigDict(extra="ignore")

    span: str
    verification_note: str = ""
    index: Optional[int] = None  # character offset into completion (UTF-8 code points, Python str index)


class AnnotatedSpan(BaseModel):
    """A fact-checked entity span within a completion."""

    model_config = ConfigDict(extra="ignore")

    span: str
    label: Optional[Literal["Supported", "Not Supported", "Insufficient Information"]] = None
    verification_note: str = ""
    index: Optional[int] = None

    @field_validator("label", mode="before")
    @classmethod
    def normalise_label(cls, v):
        if v is None:
            return None
        mapping = {
            "supported": "Supported",
            "not supported": "Not Supported",
            "insufficient information": "Insufficient Information",
        }
        return mapping.get(str(v).strip().lower(), None)


class GenerationRecord(BaseModel):
    """One line of generations JSONL (input) or annotated JSONL (output)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    source_dataset: str = ""
    model: str
    completion: str
    policy: str = Field(default="", description="Natural language policy; violations are relative to this text.")
    answer: Optional[str] = Field(default=None, description="Ground-truth answer for factual QA datasets.")
    annotations: Optional[list[Union[AnnotatedSpan, PolicyViolationSpan]]] = None
    annotator_model: Optional[str] = None
