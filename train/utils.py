#!/usr/bin/env python3
"""GPU / dtype helpers and shared probe tokenization for vLLM probe training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def apply_chat_template_to_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[dict[str, str]],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    **template_kwargs: Any,
) -> str:
    """Render ``messages`` with ``apply_chat_template(..., tokenize=False)``.

    Single place for v4 (returns ``str``) vs v5+ (may return a dict with ``text``).
    ``chat_template_kwargs`` (e.g. ``enable_thinking`` for Gemma) is merged with
    ``template_kwargs`` (e.g. ``add_generation_prompt=True``).
    """
    merged = {**(chat_template_kwargs or {}), **template_kwargs}
    out = tokenizer.apply_chat_template(messages, tokenize=False, **merged)
    if isinstance(out, str):
        return out
    if isinstance(out, dict) and isinstance(out.get("text"), str):
        return out["text"]
    if hasattr(out, "get"):
        text = out.get("text")
        if isinstance(text, str):
            return text
    raise TypeError(
        f"apply_chat_template returned {type(out)!r}; expected str (tokenize=False). "
        "If you use transformers v5, ensure tokenize=False in kwargs."
    )


def encode_probe_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[dict[str, str]],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    return_offsets_mapping: bool = False,
    **template_kwargs: Any,
) -> Any:
    """Template → string → encode with ``add_special_tokens=False``.

    This is the only encoding path used for vLLM ``prompt_token_ids`` and label
    offsets in :class:`~trainer.ProbeTrainer`; use it for dataset filtering too.
    """
    text = apply_chat_template_to_text(
        tokenizer, messages, chat_template_kwargs=chat_template_kwargs, **template_kwargs
    )
    enc_kw: dict[str, Any] = {
        "add_special_tokens": False,
        "return_offsets_mapping": return_offsets_mapping,
    }
    if not return_offsets_mapping:
        enc_kw["return_attention_mask"] = False
    return tokenizer(text, **enc_kw)


def probe_prefill_token_count(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    """Token length of the full templated chat: **user message + assistant completion** (not prompt-only).

    Matches vLLM prefill for probe training (same as :func:`encode_probe_chat` with both turns).
    """
    enc = encode_probe_chat(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        chat_template_kwargs=chat_template_kwargs,
        return_offsets_mapping=False,
    )
    return len(enc["input_ids"])


def probe_completion_char_start(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    """Character offset in rendered chat text where the assistant completion begins.

    Some chat templates (Gemma 4, for example) render
    ``[user] + add_generation_prompt=True`` differently from the prefix of a full
    ``[user, assistant]`` conversation, so using the former can misalign labels.
    We therefore locate the **actual completion text** inside the fully rendered chat.
    """
    full_text = apply_chat_template_to_text(
        tokenizer,
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        chat_template_kwargs=chat_template_kwargs,
    )
    start = full_text.rfind(completion)
    if start < 0:
        raise ValueError(
            "Could not locate assistant completion inside rendered chat template; "
            "check tokenizer.apply_chat_template behavior for this model."
        )
    return start


def map_char_labels_to_completion_token_labels(
    annotations: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    comp_char_start: int,
) -> Tuple[torch.Tensor, int]:
    """Map per-completion-character labels (0 / 1 / -100) to completion **token** labels.

    Same convention as :class:`~trainer.ProbeTrainer` loss: one label per completion
    token; ``1`` if any covered character is violation, else ``0`` if any non-padding
    non-violation, else ``-100`` (ignored by loss).
    """
    comp_idx = next(
        (idx for idx, (_, end) in enumerate(offset_mapping) if end > comp_char_start),
        len(offset_mapping),
    )
    comp_offsets = offset_mapping[comp_idx:]

    annotations_token = torch.full((len(comp_offsets),), -100.0, dtype=torch.float32)
    for token_index, (start, end) in enumerate(comp_offsets):
        rel_start = max(0, start - comp_char_start)
        rel_end = min(len(annotations), end - comp_char_start)
        annotations_range = annotations[rel_start:rel_end]
        if len(annotations_range) == 0:
            continue
        if (annotations_range == 1).any():
            annotations_token[token_index] = 1.0
        elif (annotations_range == 0).any():
            annotations_token[token_index] = 0.0
        elif (annotations_range == -100).all():
            annotations_token[token_index] = -100.0

    return annotations_token, comp_idx


def probe_completion_token_labels(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    annotations_val: Union[Sequence[float], torch.Tensor],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Token-level violation labels for the assistant completion (single example).

    Uses :func:`encode_probe_chat` with offsets and :func:`map_char_labels_to_completion_token_labels`
    so training, ``pos_weight``, and metrics use one convention.
    """
    enc = encode_probe_chat(
        tokenizer,
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        chat_template_kwargs=chat_template_kwargs,
        return_offsets_mapping=True,
    )
    if isinstance(annotations_val, torch.Tensor):
        ann = annotations_val.float().flatten()
    else:
        ann = torch.tensor(annotations_val, dtype=torch.float32)
    comp_start = probe_completion_char_start(
        tokenizer, prompt, completion, chat_template_kwargs=chat_template_kwargs
    )
    labels, _comp_idx = map_char_labels_to_completion_token_labels(
        ann, enc["offset_mapping"], comp_start
    )
    return labels


def get_compute_capability(device_idx: int = 0) -> int:
    """Return major compute capability (e.g. 8 for Ampere+).

    Used for chunked-prefill / CUDA-graph heuristics (cc >= 8 vs older GPUs).
    Returns ``0`` if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability(device_idx)[0]


def get_dtype(device_idx: int = 0) -> torch.dtype:
    """Pick a dtype from GPU capability: Ampere+ → bfloat16, else float16."""
    if get_compute_capability(device_idx) >= 8:
        return torch.bfloat16
    return torch.float16


def resolve_gpu_counts(tp_required: int) -> tuple[int, int]:
    """Return ``(tensor_parallel_size, data_parallel_size)`` from available GPUs.

    ``tp_required`` is the tensor-parallel degree the model needs (from config).
    Uses at most ``min(available, tp_required)`` for TP; remaining GPUs become DP
    (replicas). Falls back to ``(1, 1)`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 1, 1

    n_available = torch.cuda.device_count()
    tp = min(n_available, tp_required)
    dp = max(1, n_available // tp)
    return tp, dp


def effective_probe_max_model_len(llm: object, config_max_model_len: int) -> int:
    """Smaller of config cap and vLLM’s resolved :attr:`ModelConfig.max_model_len`.

    vLLM may clamp ``max_model_len`` to the checkpoint’s context limit (e.g. 2048) even
    when the YAML asks for 4096. Dataset truncation must use this value or runs fail
    validation with “maximum context length is …”.
    """
    resolved = int(llm.llm_engine.model_config.max_model_len)
    return min(int(config_max_model_len), resolved)


def optimal_batch_size(llm: object, max_model_len: int, safety_factor: float = 0.8) -> int:
    """Largest batch size that fits vLLM’s KV cache at ``max_model_len`` tokens per sequence."""
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    blocks_per_seq = (max_model_len + block_size - 1) // block_size
    max_seqs = int(num_gpu_blocks * safety_factor) // blocks_per_seq
    return max(1, max_seqs)


def save_training_config_json(cfg: object, path: Path | str) -> None:
    """Serialize a training config dataclass to JSON."""
    import json
    from dataclasses import asdict

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def _json_sanitize(obj: object) -> object:
    """Make Trainer log values JSON-serializable (numpy scalars, tensors, etc.)."""
    import numpy as np

    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return float(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def save_training_metrics_json(trainer: object, path: Path | str) -> None:
    """Write ``Trainer.state.log_history`` and a short run summary to ``training_metrics.json``.

    :class:`~trainer.ProbeTrainer` coalesces logging so each global step is typically a
    single history row (loss + probe metrics; eval/runtime merged into that step).

    Serializes in memory first, then writes atomically.
    """
    import json
    import os
    import tempfile

    path = Path(path)
    state = trainer.state
    summary = {
        "global_step": int(state.global_step),
        "epoch": float(state.epoch) if state.epoch is not None else None,
        "best_metric": getattr(state, "best_metric", None),
        "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
    }
    history = [_json_sanitize(entry) for entry in state.log_history]
    payload = {"summary": summary, "log_history": history}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    """
    counting the total and trainable parameters in a model

    Args:
        model (nn.Module): The model whose parameters are being counted

    Returns:
        Tuple[int, int]: The total parameters, followed by the trainable parameters
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return total_params, trainable_params
