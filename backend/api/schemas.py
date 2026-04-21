"""OpenAI-compatible request/response schemas with score extensions."""

from __future__ import annotations

from typing import Literal, Optional
import time

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9

    # Score extensions — ignored when the service has no probes loaded
    include_scores: bool = True
    # Path to a ValueHeadProbe checkpoint (.pt). Falls back to PROBE_PATH env var.
    probe_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[Literal["stop", "length"]] = "stop"


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage
    # Extension: per-probe, per-token scores aligned with generated tokens.
    # scores["hallucination"][i] is the score for the i-th generated token.
    # None when include_scores=False or no probes are loaded.
    scores: Optional[dict[str, list[float]]] = None


# ---------------------------------------------------------------------------
# Streaming response chunks
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    # Extension: one score per active probe for the token in this chunk.
    # scores["hallucination"] is the score for the current token.
    # None on role/finish chunks or when include_scores=False.
    scores: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# Models list (OpenAI /v1/models compatibility)
# ---------------------------------------------------------------------------

class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "organization"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
