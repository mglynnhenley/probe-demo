"""
Analyze endpoint: score pre-existing text token-by-token via a prefill pass.

No text is generated. The full prompt (chat prefix + assistant text) is submitted
as a single vLLM request; hidden states are collected from the prefill pass via
extract_prefill_hidden_states_async, then each token is scored through every probe
using run_probe_step — the same function used by generate_streaming and
generate_closed_source_streaming — before yielding results.

Architecture
------------
  1. Build the chat-template prefix (system + user turn) and tokenise the
     assistant text separately, so we can slice prefix tokens out later.
  2. Submit prompt_ids = prefix_ids + assistant_ids to vLLM with max_tokens=16
     (enough decode budget for the scheduler to flush all prefill chunks).
  3. Concurrently collect hidden states from extract_prefill_hidden_states_async.
  4. Slice off the prefix rows, keeping only the assistant-token hidden states.
  5. Score each token with run_probe_step (handles MLP, CovSeq, MultiLayerCovSeq
     uniformly via a rolling buffer — same as the decode path).
  6. Yield (token_text, token_index, [(layer_idx, score), ...]) in batch_size
     chunks, then a (None, None, None) sentinel.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from typing import Any, AsyncGenerator

import vllm
from vllm_probe_plugin import _result_bus, extract_prefill_hidden_states_async

from prefill_utils import collect_assistant_hidden_states


async def analyze_streaming(
    text: str,
    llm,
    tokenizer,
    probes: list[tuple[str, Any]],
    chat_template_kwargs: dict[str, Any],
    user_prompt: str | None = None,
    batch_size: int = 64,
) -> AsyncGenerator[tuple[str | None, int | None, list[tuple[int, float]] | None], None]:
    """Async generator: prefill *text* and stream per-token probe scores.

    Uses run_probe_step from service for all probe architectures (MLP,
    CovSeq, MultiLayerCovSeq), maintaining per-probe rolling buffers across
    tokens exactly as the decode path does.

    Args:
        text:                 The assistant text to score.
        llm:                  vLLM AsyncLLM instance.
        tokenizer:            HuggingFace tokenizer matching the model.
        probes:               List of (name, ValueHeadProbe) pairs to run.
        chat_template_kwargs: Extra kwargs forwarded to apply_chat_template.
        user_prompt:          User turn prepended as context for the chat prefix.
                              Defaults to a generic prompt if not provided.
        batch_size:           Tokens to yield per batch (controls streaming
                              granularity; does not affect probe computation).

    Yields:
        (token_text, token_index, [(layer_idx, score), ...]) for each token,
        then (None, None, None) as the termination sentinel.
    """
    from service import apply_chat_template_ids, encode_text, run_probe_step

    req_id = f"analyze-{uuid.uuid4().hex[:12]}"
    print(f"[ANALYZE] request_id={req_id} text_len={len(text)}", flush=True)

    user_content = (
        user_prompt.strip()
        if user_prompt and user_prompt.strip()
        else "The user asked a question. The assistant responded with the following:"
    )
    prefix_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    prefix_ids: list[int] = apply_chat_template_ids(tokenizer, prefix_messages, chat_template_kwargs)
    assistant_ids: list[int] = encode_text(tokenizer, text)
    if not assistant_ids:
        raise ValueError("text tokenises to zero tokens")

    n = len(assistant_ids)
    n_prefix = len(prefix_ids)
    prompt_ids = prefix_ids + assistant_ids

    token_texts: list[str] = [
        tokenizer.decode([tid], skip_special_tokens=False,
                         clean_up_tokenization_spaces=False)
        for tid in assistant_ids
    ]

    # max_tokens=16 gives the scheduler enough decode budget to flush all
    # prefill chunks before it terminates; only the prefill states are used.
    sp = vllm.SamplingParams(max_tokens=16, temperature=0.0)

    all_prefill: dict[int, list] = {}

    async def _drain():
        async for _ in llm.generate(
            prompt={"prompt_token_ids": prompt_ids},
            sampling_params=sp,
            request_id=req_id,
        ):
            pass

    gen_task = asyncio.create_task(_drain())

    try:
        async for chunk in extract_prefill_hidden_states_async(req_id):
            for layer_id, tensor in chunk.items():
                all_prefill.setdefault(layer_id, []).append(tensor)
    finally:
        await gen_task
        _result_bus.clear_prefill(req_id)
        _result_bus.clear(req_id)

    hidden = collect_assistant_hidden_states(
        all_prefill, req_id=req_id, n=n, n_prefix=n_prefix
    )

    # Score every token through every probe using run_probe_step — the same
    # function used on the decode path.  Rolling buffers (None = uninitialised)
    # are maintained per probe so CovSeq / MultiLayerCovSeq windows accumulate
    # correctly across tokens.
    probe_scores: dict[str, list[float]] = {pname: [] for pname, _ in probes}
    buffers: dict[str, deque | dict | None] = {pname: None for pname, _ in probes}

    from models import MultiLayerCovSeqModel

    for i in range(n):
        for probe_name, probe in probes:
            if isinstance(probe.model, MultiLayerCovSeqModel):
                layer_ids = probe.cfg.model.layer_indices
                hs_dict = {lid: hidden[lid][i:i+1] for lid in layer_ids if lid in hidden}
                if not hs_dict:
                    probe_scores[probe_name].append(0.0)
                    continue
                score, buffers[probe_name] = run_probe_step(probe, hs_dict, buffers[probe_name])
            else:
                hs = hidden.get(probe.cfg.layer_idx)
                if hs is None:
                    probe_scores[probe_name].append(0.0)
                    continue
                score, buffers[probe_name] = run_probe_step(probe, hs[i:i+1], buffers[probe_name])
            probe_scores[probe_name].append(score)

    # Yield in batches so the caller can stream results progressively.
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            per_layer = [
                (probe.cfg.layer_idx, probe_scores[pname][i])
                for pname, probe in probes
            ]
            yield token_texts[i], i, per_layer

    yield None, None, None  # sentinel
