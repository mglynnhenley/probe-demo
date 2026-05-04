#!/usr/bin/env python
"""OpenAI-compatible FastAPI backend with per-token probe scores."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Generator

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
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def _run():
        try:
            for item in gen:
                loop.call_soon_threadsafe(q.put_nowait, item)
        except Exception as exc:
            import traceback
            print(f"[STREAM] generator error: {exc}\n{traceback.format_exc()}", flush=True)
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

def _resolve_closed_source_model(request: ChatCompletionRequest, svc) -> str | None:
    """Return the closed-source model to use, or None for local-only generation.

    Closed-source routing is triggered when:
      - request.closed_source_model is explicitly set, OR
      - request.model does not match the local model name (and isn't 'default')
    """
    if request.closed_source_model:
        return request.closed_source_model
    local_names = {svc.model_name.lower(), "default"}
    if request.model.lower() not in local_names:
        return request.model
    return None


def _build_chunks(
    request: ChatCompletionRequest,
    cancel_event: threading.Event,
    completion_id: str,
    created: int,
) -> Generator[str, None, None]:
    """Sync generator: yield SSE strings from the service's streaming output."""
    svc = get_service()
    cs_model = _resolve_closed_source_model(request, svc)

    # Role chunk
    display_model = cs_model or svc.model_name
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=display_model,
        choices=[ChatCompletionChunkChoice(
            delta=DeltaMessage(role="assistant"),
        )],
    )
    yield _chunk_to_sse(first_chunk)

    messages = [m.model_dump() for m in request.messages]
    probes = svc.get_probes(request.probe_path) if request.include_scores else []

    if cs_model:
        gen = svc.generate_closed_source_streaming(
            messages=messages,
            closed_source_model=cs_model,
            probe_path=request.probe_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            include_probe_scores=request.include_scores,
            cancel_event=cancel_event,
            block_size=request.block_size,
        )
    else:
        gen = svc.generate_streaming(
            messages=messages,
            probe_path=request.probe_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            include_probe_scores=request.include_scores,
            cancel_event=cancel_event,
        )

    for delta, token_index, per_layer in gen:
        if delta is None:
            break

        scores: dict[str, float] | None = None
        if per_layer:
            scores = {name: score for (name, _), (_, score) in zip(probes, per_layer)}

        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=display_model,
            choices=[ChatCompletionChunkChoice(
                delta=DeltaMessage(content=delta),
            )],
            scores=scores,
        )
        yield _chunk_to_sse(chunk)

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=display_model,
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


@app.get("/v1/models/{model_id:path}")
def retrieve_model(model_id: str) -> ModelCard:
    # Return a card for any model ID — closed-source models are valid routing targets.
    return ModelCard(id=model_id)


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
    cs_model = _resolve_closed_source_model(request, svc)
    display_model = cs_model or svc.model_name
    messages = [m.model_dump() for m in request.messages]
    probes = svc.get_probes(request.probe_path) if request.include_scores else []
    all_text: list[str] = []
    scores_by_probe: dict[str, list[float]] = {name: [] for name, _ in probes}

    if cs_model:
        gen = svc.generate_closed_source_streaming(
            messages=messages,
            closed_source_model=cs_model,
            probe_path=request.probe_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            include_probe_scores=request.include_scores,
            block_size=request.block_size,
        )
    else:
        gen = svc.generate_streaming(
            messages=messages,
            probe_path=request.probe_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            include_probe_scores=request.include_scores,
        )

    for delta, token_index, per_layer in gen:
        if delta is None:
            break
        all_text.append(delta)
        if per_layer:
            for (name, _), (_, score) in zip(probes, per_layer):
                scores_by_probe[name].append(score)

    generated_text = "".join(all_text)

    prompt_tokens = len(svc.tokenizer.encode("".join(m.content for m in request.messages)))
    completion_tokens = len(svc.tokenizer.encode(generated_text, add_special_tokens=False))

    scores: dict[str, list[float]] | None = None
    if request.include_scores and scores_by_probe:
        scores = scores_by_probe

    return ChatCompletion(
        id=completion_id,
        created=created,
        model=display_model,
        choices=[ChatCompletionChoice(
            message=ChatCompletionMessage(content=generated_text),
        )],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        scores=scores,
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
