#!/usr/bin/env python
"""OpenAI-compatible FastAPI backend with per-token probe scores."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Generator, Optional

import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionUsage,
    DeltaMessage,
    ModelCard,
    ModelList,
    ProbeScore,
)
from service import get_service, lifespan

uvloop.install()

_thread_pool = concurrent.futures.ThreadPoolExecutor()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _chunk_to_sse(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


async def _stream_sync_gen(
    gen,
    fastapi_request: Request | None = None,
    on_disconnect=None,
) -> AsyncGenerator[str, None]:
    """Run a blocking sync generator on a thread pool, yielding SSE strings."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def _run():
        try:
            for item in gen:
                loop.call_soon_threadsafe(q.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _DONE)

    _thread_pool.submit(_run)

    try:
        while True:
            if fastapi_request is not None:
                get_task = asyncio.ensure_future(q.get())
                try:
                    done, _ = await asyncio.wait({get_task}, timeout=2.0)
                except asyncio.CancelledError:
                    get_task.cancel()
                    raise
                if not done:
                    get_task.cancel()
                    try:
                        await get_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    if await fastapi_request.is_disconnected():
                        if on_disconnect:
                            on_disconnect()
                        return
                    continue
                item = get_task.result()
            else:
                item = await q.get()

            if item is _DONE:
                return
            yield item
    except (asyncio.CancelledError, GeneratorExit):
        if on_disconnect:
            on_disconnect()
        raise


# ---------------------------------------------------------------------------
# Generation → SSE chunks
# ---------------------------------------------------------------------------

def _build_chunks(
    request: ChatCompletionRequest,
    cancel_event: threading.Event,
    completion_id: str,
    created: int,
) -> Generator[str, None, None]:
    """Sync generator: yield SSE strings from the service's streaming output."""
    svc = get_service()

    # Role chunk
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=svc.model_name,
        choices=[ChatCompletionChunkChoice(
            delta=DeltaMessage(role="assistant"),
        )],
    )
    yield _chunk_to_sse(first_chunk)

    messages = [m.model_dump() for m in request.messages]

    for delta, token_index, per_layer in svc.generate_streaming(
        messages=messages,
        probe_path=request.probe_path,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        include_probe_scores=request.include_probe_scores,
        cancel_event=cancel_event,
    ):
        if delta is None:
            # Completion sentinel — emit [DONE]
            break

        probe_scores: list[ProbeScore] | None = None
        if per_layer:
            probe_scores = [
                ProbeScore(layer=layer, score=score, token_index=token_index)
                for layer, score in per_layer
            ]

        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=svc.model_name,
            choices=[ChatCompletionChunkChoice(
                delta=DeltaMessage(content=delta),
            )],
            probe_scores=probe_scores,
        )
        yield _chunk_to_sse(chunk)

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=svc.model_name,
        choices=[ChatCompletionChunkChoice(
            delta=DeltaMessage(),
            finish_reason="stop",
        )],
    )
    yield _chunk_to_sse(final_chunk)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Probe-Demo Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> ModelList:
    svc = get_service()
    return ModelList(data=[ModelCard(id=svc.model_name)])


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    fastapi_request: Request,
) -> Any:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        cancel_event = threading.Event()
        gen = _build_chunks(request, cancel_event, completion_id, created)
        return StreamingResponse(
            _stream_sync_gen(gen, fastapi_request=fastapi_request, on_disconnect=cancel_event.set),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming: collect all tokens then return
    svc = get_service()
    messages = [m.model_dump() for m in request.messages]
    all_text: list[str] = []
    all_probe_probs: list[float] = []
    all_probe_scores: list[list[ProbeScore]] = []

    for delta, token_index, per_layer in svc.generate_streaming(
        messages=messages,
        probe_path=request.probe_path,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        include_probe_scores=request.include_probe_scores,
    ):
        if delta is None:
            break
        all_text.append(delta)
        if per_layer:
            scores = [
                ProbeScore(layer=layer, score=score, token_index=token_index or 0)
                for layer, score in per_layer
            ]
            # Aggregate to a single prob (mean across layers) for the flat list
            all_probe_probs.append(sum(s.score for s in scores) / len(scores))
            all_probe_scores.append(scores)

    generated_text = "".join(all_text)

    prompt_tokens = len(svc.tokenizer.encode("".join(m.content for m in request.messages)))
    completion_tokens = len(svc.tokenizer.encode(generated_text, add_special_tokens=False))

    return ChatCompletion(
        id=completion_id,
        created=created,
        model=svc.model_name,
        choices=[ChatCompletionChoice(
            message=ChatCompletionMessage(content=generated_text),
        )],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        probe_probs=all_probe_probs if request.include_probe_scores else None,
        probe_scores=all_probe_scores if request.include_probe_scores else None,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
