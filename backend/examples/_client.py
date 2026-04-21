"""Shared client factory and score field helpers for all examples."""

from __future__ import annotations

import os

import openai


def chunk_scores(chunk: openai.types.chat.ChatCompletionChunk) -> dict[str, float] | None:
    """Extract per-probe scores from a streaming chunk.

    Returns e.g. {"hallucination": 0.03} or None on role/finish chunks.
    """
    return (chunk.model_extra or {}).get("scores")


def response_scores(response: openai.types.chat.ChatCompletion) -> dict[str, list[float]] | None:
    """Extract per-probe token score arrays from a non-streaming response.

    Returns e.g. {"hallucination": [0.03, 0.15, ...]} aligned with completion tokens.
    """
    return (response.model_extra or {}).get("scores")


def get_client() -> openai.OpenAI:
    base_url = os.environ.get("PROBE_BACKEND_URL", "http://localhost:8000")
    return openai.OpenAI(base_url=f"{base_url}/v1", api_key="not-required")


def get_base_url() -> str:
    return os.environ.get("PROBE_BACKEND_URL", "http://localhost:8000")
