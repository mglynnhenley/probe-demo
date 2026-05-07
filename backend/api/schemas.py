"""OpenAI-compatible request/response schemas with score extensions."""

from __future__ import annotations

from typing import Any, Literal, Optional
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
    # Core
    model: str
    messages: list[ChatMessage]
    stream: bool = False

    # Sampling — all map directly to vLLM SamplingParams
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # OpenAI o-series alias for max_tokens
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: Optional[list[str] | str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None

    # Logprobs — OpenAI uses bool + top_logprobs int; vLLM uses a single int count
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None  # number of top tokens to return logprobs for

    # Token bias — OpenAI uses str keys (stringified token ids); vLLM uses int keys
    logit_bias: Optional[dict[str, float]] = None

    # Response format — json_object maps to vLLM guided_decoding; json_schema is best-effort
    response_format: Optional[dict[str, Any]] = None

    # Fields that are accepted but have no vLLM equivalent (safe to ignore)
    user: Optional[str] = None
    store: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None
    stream_options: Optional[dict[str, Any]] = None

    # Probe extensions (passed via extra_body in the OpenAI SDK)
    include_scores: bool = True
    probe_path: Optional[str] = None
    # Closed-source routing: if set (or if model != local model name), tokens are
    # sourced from this model and forced through the local vLLM model for probing.
    closed_source_model: Optional[str] = None
    block_size: int = 1


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
