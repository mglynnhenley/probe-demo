"""Helpers for turning raw prefill hidden-state chunks into per-layer tensors.

The connector emits hidden states as a stream of per-layer chunks, each shaped
[chunk_tokens, hidden_size]. Callers that want the full per-token tensor for a
prompt need to concatenate chunks and slice off the prefix. With prefix caching
enabled, vLLM may skip prefill for cached prefix tokens, so the concatenated
row count varies between the full prompt length (cold cache) and just the new
tokens (full prefix hit) — the assistant/target rows are always the trailing
ones either way.
"""
from __future__ import annotations

import torch


def collect_assistant_hidden_states(
    all_prefill: dict[int, list[torch.Tensor]],
    req_id: str,
    n: int,
    n_prefix: int,
) -> dict[int, torch.Tensor]:
    """Concat per-layer prefill chunks, validate, and return the trailing n rows.

    Args:
        all_prefill: {layer_id: [chunk_tensor, ...]} accumulated from
            extract_prefill_hidden_states_async. Each chunk is expected to be
            shape [chunk_tokens, hidden_size].
        req_id:    Request id, used only for log/error context.
        n:         Number of assistant tokens to score.
        n_prefix:  Number of prefix (system+user) tokens — used only for
                   logging the expected row-count range.

    Returns:
        {layer_id: tensor[n, hidden_size]} containing only the assistant rows.

    Raises:
        RuntimeError: when no chunks arrived, when chunks are not 2-D, when the
            hidden-size dimension is zero, or when fewer than n rows arrived.
    """
    if not all_prefill:
        raise RuntimeError(
            f"[ANALYZE] req={req_id}: no prefill hidden states received "
            f"(expected {n_prefix + n} prompt tokens, {n} assistant tokens). "
            f"Likely causes: connector did not run, request_id matching failed, "
            f"or the engine never reached the prefill phase."
        )

    hidden: dict[int, torch.Tensor] = {}
    for layer_id, chunks in all_prefill.items():
        chunk_shapes = [tuple(c.shape) for c in chunks]
        full = torch.cat(chunks, dim=0)
        print(
            f"[ANALYZE] req={req_id} layer={layer_id} "
            f"chunks={len(chunks)} chunk_shapes={chunk_shapes} "
            f"concat_shape={tuple(full.shape)} dtype={full.dtype} "
            f"expected_rows∈[{n}, {n_prefix + n}] (prefix-cache dependent)",
            flush=True,
        )
        if full.dim() != 2:
            raise RuntimeError(
                f"[ANALYZE] req={req_id} layer={layer_id}: expected 2-D "
                f"[tokens, hidden_size] tensor, got shape {tuple(full.shape)}"
            )
        rows, hidden_size = full.shape
        if hidden_size == 0:
            raise RuntimeError(
                f"[ANALYZE] req={req_id} layer={layer_id}: hidden_size dim is 0 "
                f"(rows={rows}). The connector emitted zero-feature tensors — "
                f"check that aux_hidden_states layers are registered correctly "
                f"and that the runner's hidden_size matches the model config."
            )
        if rows < n:
            raise RuntimeError(
                f"[ANALYZE] req={req_id} layer={layer_id}: only {rows} rows "
                f"received but need at least {n} (assistant tokens). "
                f"Either the connector dropped tensors, or the request was "
                f"truncated mid-prefill."
            )
        hidden[layer_id] = full[-n:]
    return hidden
