"""Thin typed wrapper around the OpenAI SDK for the probe-demo backend.

The OpenAI SDK preserves unknown response fields in .model_extra, but they are
untyped. This module subclasses the SDK's Pydantic models to declare `scores`
as a real field, and provides a ProbeClient whose chat.completions.create()
returns those extended types automatically.

Usage (drop-in for openai.OpenAI):

    from probe_sdk import ProbeClient

    client = ProbeClient("http://localhost:8000")

    # Non-streaming — response.scores is typed
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.scores)   # {"hallucination": [0.03, 0.15, ...]}

    # Streaming — each chunk.scores is typed
    for chunk in client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    ):
        print(chunk.scores)  # {"hallucination": 0.03} or None
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, overload

import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import ConfigDict


# ---------------------------------------------------------------------------
# Extended response types
# ---------------------------------------------------------------------------

class ChatCompletionWithScores(ChatCompletion):
    """ChatCompletion extended with a typed `scores` field."""
    model_config = ConfigDict(extra="allow")
    scores: dict[str, list[float]] | None = None

    @classmethod
    def from_response(cls, r: ChatCompletion) -> ChatCompletionWithScores:
        return cls.model_validate({**r.model_dump(), **(r.model_extra or {})})


class ChatCompletionChunkWithScores(ChatCompletionChunk):
    """ChatCompletionChunk extended with a typed `scores` field."""
    model_config = ConfigDict(extra="allow")
    scores: dict[str, float] | None = None

    @classmethod
    def from_chunk(cls, c: ChatCompletionChunk) -> ChatCompletionChunkWithScores:
        return cls.model_validate({**c.model_dump(), **(c.model_extra or {})})


# ---------------------------------------------------------------------------
# Thin client wrapper
# ---------------------------------------------------------------------------

class ProbeCompletions:
    def __init__(self, completions: openai.resources.chat.Completions) -> None:
        self._completions = completions

    @overload
    def create(self, *, stream: None | bool = False, **kwargs: Any) -> ChatCompletionWithScores: ...
    @overload
    def create(self, *, stream: bool, **kwargs: Any) -> ChatCompletionWithScores | Iterator[ChatCompletionChunkWithScores]: ...

    def create(
        self,
        *,
        stream: bool = False,
        include_scores: bool = True,
        probe_path: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletionWithScores | Iterator[ChatCompletionChunkWithScores]:
        extra = {"include_scores": include_scores}
        if probe_path is not None:
            extra["probe_path"] = probe_path

        if stream:
            return self._stream(extra_body=extra, **kwargs)

        response = self._completions.create(stream=False, extra_body=extra, **kwargs)
        return ChatCompletionWithScores.from_response(response)

    def _stream(self, **kwargs: Any) -> Iterator[ChatCompletionChunkWithScores]:
        for chunk in self._completions.create(stream=True, **kwargs):
            yield ChatCompletionChunkWithScores.from_chunk(chunk)


class ProbeChat:
    def __init__(self, chat: openai.resources.Chat) -> None:
        self.completions = ProbeCompletions(chat.completions)


class ProbeClient:
    """OpenAI-compatible client for the probe-demo backend.

    Identical interface to openai.OpenAI, but chat.completions.create()
    returns ChatCompletionWithScores / ChatCompletionChunkWithScores so
    that .scores is a typed attribute rather than buried in .model_extra.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        resolved_url = base_url or os.environ.get("PROBE_BACKEND_URL", "http://localhost:8000")
        self._client = openai.OpenAI(
            base_url=f"{resolved_url}/v1",
            api_key=api_key or os.environ.get("PROBE_BACKEND_API_KEY", "not-required"),
        )
        self.chat = ProbeChat(self._client.chat)
        self.models = self._client.models
