#!/usr/bin/env python3
"""GPU / dtype helpers for vLLM probe training."""

from __future__ import annotations

from pathlib import Path

import torch


def get_compute_capability(device_idx: int = 0) -> int:
    """Return major compute capability (e.g. 8 for Ampere+).

    Used for chunked-prefill / CUDA-graph heuristics (cc >= 8 vs older GPUs).
    Returns ``0`` if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability(device_idx)[0]


def get_dtype(device_idx: int = 0) -> str:
    """Pick a load dtype from GPU capability: Ampere+ → ``bfloat16``, else ``float16``."""
    if get_compute_capability(device_idx) >= 8:
        return "bfloat16"
    return "float16"


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
