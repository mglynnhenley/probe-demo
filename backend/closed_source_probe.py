"""
Closed-source model probe streaming for probe-demo.

Streams tokens from a closed-source model via OpenRouter (OpenAI-compatible SSE)
or the native Anthropic API, and runs each token through the local open-source
vLLM model as a forced decode step to obtain hidden-state probe values.

Architecture
------------
  1. Build the full chat-template prompt from the same messages used for both models.
  2. Stream tokens from OpenRouter/Anthropic into an asyncio.Queue concurrently.
  3. Tokens are accumulated into blocks of `block_size`. Once a block is full
     (or the stream ends), all tokens in the block are probed concurrently via
     asyncio.gather:
       - Token i uses prompt = current_prompt_ids + block[:i] as its prefix
       - KV cache (enable_prefix_caching=True) serves the shared prefix
       - Each request computes one new decode token (max_tokens=2)
       - allowed_token_ids=[token_id] forces the specific token
       - read_step_async(req_id, 1) retrieves the decode-step hidden state
  4. All tokens in a block are emitted once the block is fully probed.
  5. block_size=1 gives token-by-token streaming with minimal latency.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator

import vllm
from vllm_probe_plugin import _result_bus
from service import apply_chat_template_str, encode_text


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

_ANTHROPIC_DIRECT_MODELS: frozenset[str] = frozenset({
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
})


def is_anthropic_model(model: str) -> bool:
    bare = model[len("anthropic/"):] if model.startswith("anthropic/") else model
    return bare in _ANTHROPIC_DIRECT_MODELS


def _anthropic_model_id(model: str) -> str:
    if model.startswith("anthropic/"):
        return model[len("anthropic/"):]
    return model


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _stream_anthropic(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    max_tokens: int | None,
    temperature: float,
    queue: asyncio.Queue,
) -> None:
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)

        system: str | anthropic.NotGiven = anthropic.NOT_GIVEN
        chat_messages: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        kwargs: dict = dict(
            model=_anthropic_model_id(model),
            messages=chat_messages,
            max_tokens=max_tokens or 4096,
            temperature=temperature,
        )
        if system is not anthropic.NOT_GIVEN:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    await queue.put(text)
        await queue.put(None)
    except Exception as e:
        await queue.put(e)


async def _stream_openrouter(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int | None,
    temperature: float,
    queue: asyncio.Queue,
) -> None:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                await queue.put(delta)
        await queue.put(None)
    except Exception as e:
        await queue.put(e)


# ---------------------------------------------------------------------------
# Single forced-token probe step
# ---------------------------------------------------------------------------

async def _probe_forced_token(
    token_id: int,
    current_prompt_ids: list[int],
    llm,
    run_probe_step_fn,
) -> float:
    req_id = f"cs-{uuid.uuid4().hex[:12]}"
    sp = vllm.SamplingParams(
        max_tokens=2,
        temperature=0.0,
        allowed_token_ids=[token_id],
    )

    async def _drain():
        async for _ in llm.generate(
            prompt={"prompt_token_ids": list(current_prompt_ids)},
            sampling_params=sp,
            request_id=req_id,
        ):
            pass

    gen_task = asyncio.create_task(_drain())
    internal_req_id, step_hidden = await _result_bus.read_step_async(req_id, 1, timeout=5.0)

    if not step_hidden:
        bus_keys = _result_bus.known_request_ids()
        finished = [rid for rid in bus_keys if _result_bus.is_request_finished(rid)]
        print(
            f"[PROBE_STEP] WARNING: no hidden state after 5s "
            f"req_id={req_id} bus_keys={bus_keys} finished={finished}",
            flush=True,
        )

    await gen_task
    probe_prob = run_probe_step_fn(step_hidden)
    _result_bus.clear(internal_req_id)
    _result_bus.clear_prefill(internal_req_id)
    return probe_prob


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_closed_source_with_probe_streaming(
    messages: list[dict[str, str]],
    llm,
    tokenizer,
    run_probe_step_fn,
    closed_source_model: str,
    openrouter_api_key: str,
    openrouter_base_url: str,
    anthropic_api_key: str = "",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    block_size: int = 1,
    token_sink: Any = None,
) -> AsyncGenerator[dict, None]:
    """Async generator yielding probe_batch and done dicts.

    Token events are delivered immediately via token_sink (bypassing the
    generator) so probe-task congestion cannot delay token delivery.

    Yields:
      {"probe_batch": True, "probes": [{"index": i, "probe_prob": p}, ...]}
      {"done": True, "generated_token_ids", "generated_tokens",
       "probe_probs", "generated_text"}
      {"error": str}
    """
    prompt_str = apply_chat_template_str(tokenizer, messages)
    prompt_token_ids: list[int] = encode_text(tokenizer, prompt_str)

    use_anthropic = is_anthropic_model(closed_source_model) and bool(anthropic_api_key)
    provider = "Anthropic" if use_anthropic else "OpenRouter"
    print(
        f"[CS_PROBE] Starting {provider} stream: model={closed_source_model} "
        f"prompt_tokens={len(prompt_token_ids)} max_tokens={max_tokens}",
        flush=True,
    )

    queue: asyncio.Queue = asyncio.Queue()
    if use_anthropic:
        or_task = asyncio.create_task(
            _stream_anthropic(
                messages=messages, model=closed_source_model,
                api_key=anthropic_api_key, max_tokens=max_tokens,
                temperature=temperature, queue=queue,
            )
        )
    else:
        or_task = asyncio.create_task(
            _stream_openrouter(
                messages=messages, model=closed_source_model,
                api_key=openrouter_api_key, base_url=openrouter_base_url,
                max_tokens=max_tokens, temperature=temperature, queue=queue,
            )
        )

    all_token_ids: list[int] = []
    all_tokens: list[str] = []
    all_probs: list[float] = []
    current_prompt_ids: list[int] = list(prompt_token_ids)
    text_so_far = ""
    next_token_index = 0  # monotonic — advances on every sink, immune to gather races

    stream_done = False
    t_start = time.monotonic()
    t_first_cs: float | None = None
    t_last_cs: float | None = None
    t_first_probe: float | None = None
    _block_timings: list[tuple[int, float]] = []

    yield_queue: asyncio.Queue = asyncio.Queue()

    async def _fill_block(target_size: int) -> tuple[list[int], list[str]]:
        nonlocal stream_done, text_so_far, t_first_cs, t_last_cs, next_token_index
        ids: list[int] = []
        texts: list[str] = []
        while len(ids) < target_size and not stream_done:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Timeout waiting for closed-source model token")
            if item is None:
                stream_done = True
                break
            if isinstance(item, Exception):
                raise item
            chunk_ids = encode_text(tokenizer, item)
            now = time.monotonic()
            if t_first_cs is None and chunk_ids:
                t_first_cs = now
            if chunk_ids:
                t_last_cs = now
            for i, tid in enumerate(chunk_ids):
                tok_str = item if i == 0 else ""
                text_so_far += tok_str
                global_idx = next_token_index
                next_token_index += 1
                tok_event = {
                    "token_id": tid,
                    "token": tok_str,
                    "probe_prob": None,
                    "index": global_idx,
                    "text_so_far": text_so_far,
                }
                if token_sink is not None:
                    token_sink(tok_event)
                else:
                    yield_queue.put_nowait(tok_event)
                    await asyncio.sleep(0)
                ids.append(tid)
                texts.append(tok_str)
        return ids, texts

    async def _run_pipeline() -> None:
        nonlocal t_first_probe
        try:
            block, block_texts = await _fill_block(1)
            if not block:
                yield_queue.put_nowait(None)
                return

            while block:
                block_start_index = len(all_token_ids)
                prompt_prefix = list(current_prompt_ids)
                t0 = time.monotonic()

                probe_coro = asyncio.gather(*[
                    _probe_forced_token(
                        token_id=block[i],
                        current_prompt_ids=prompt_prefix + list(block[:i]),
                        llm=llm,
                        run_probe_step_fn=run_probe_step_fn,
                    )
                    for i in range(len(block))
                ])
                probe_probs, (next_block, next_texts) = await asyncio.gather(
                    probe_coro,
                    _fill_block(block_size),
                )

                t_block_ms = (time.monotonic() - t0) * 1000
                _block_timings.append((len(block), t_block_ms))
                print(
                    f"[CS_BLOCK] n={len(block)} wall={t_block_ms:.1f}ms "
                    f"per_token={t_block_ms/len(block):.1f}ms "
                    f"tokens_so_far={block_start_index + len(block)}",
                    flush=True,
                )
                if t_first_probe is None:
                    t_first_probe = time.monotonic()

                probe_entries = []
                for i, (tid, tok, prob) in enumerate(zip(block, block_texts, probe_probs)):
                    all_token_ids.append(tid)
                    all_tokens.append(tok)
                    all_probs.append(prob)
                    current_prompt_ids.append(tid)
                    probe_entries.append({"index": block_start_index + i, "probe_prob": prob})
                yield_queue.put_nowait({"probe_batch": True, "probes": probe_entries})

                block, block_texts = next_block, next_texts

            yield_queue.put_nowait(None)

        except Exception as exc:
            import traceback
            print(f"[CS_PIPELINE] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
            yield_queue.put_nowait(exc)

    try:
        pipeline_task = asyncio.create_task(_run_pipeline())
        while True:
            item = await yield_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        await pipeline_task
    finally:
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
        or_task.cancel()
        try:
            await or_task
        except asyncio.CancelledError:
            pass

    t_end = time.monotonic()
    n_tokens = len(all_token_ids)
    total_elapsed = t_end - t_start

    cs_stream_dur = (
        (t_last_cs - t_first_cs)
        if t_first_cs and t_last_cs and t_last_cs > t_first_cs
        else None
    )
    cs_tps = n_tokens / cs_stream_dur if cs_stream_dur and n_tokens else 0.0
    probe_dur = (t_end - t_first_probe) if t_first_probe else None
    probe_tps = n_tokens / probe_dur if probe_dur and n_tokens else 0.0
    avg_probe_ms = (probe_dur / n_tokens * 1000) if probe_dur and n_tokens else 0.0

    print(
        f"[CS_PROBE_STATS] tokens={n_tokens} total={total_elapsed:.2f}s "
        f"cs_tps={cs_tps:.1f} probe_tps={probe_tps:.1f} avg_probe_ms={avg_probe_ms:.1f}",
        flush=True,
    )

    yield {
        "done": True,
        "generated_token_ids": all_token_ids,
        "generated_tokens": all_tokens,
        "probe_probs": all_probs,
        "generated_text": text_so_far,
    }
