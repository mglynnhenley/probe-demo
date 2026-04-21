#!/usr/bin/env python3
"""
Save prefill hidden states for every example in an annotated JSONL dataset.

Each example is encoded as a full user+assistant chat (prompt + completion) and
passed through vLLM as a prefill-only call (max_tokens=1).  The resulting hidden
states for the configured layer are saved to ``<output_dir>/<hash>.npz`` where
``<hash>`` is the first 16 hex characters of SHA-256(prompt + "\\0" + completion).

The hash-based naming makes the script safe to stop and restart: any file that
already exists is skipped, so no generation is duplicated.

Usage
-----
    python generate.py configs/probe/default_config.yaml -o activations/

The data path and other vLLM settings come from the config YAML (same file used
by ``train.py``).  Use ``--data-path`` to override ``annotations_jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import torch
import vllm
import vllm_probe_plugin
from tqdm import tqdm
from vllm.lora.request import LoRARequest
from vllm_probe_plugin import extract_prefill_hidden_states

from config import TrainingConfig
from utils import (
    effective_probe_max_model_len,
    encode_probe_chat,
    get_compute_capability,
    get_dtype,
    optimal_batch_size,
    resolve_gpu_counts,
)
from vllm_probe_plugin import configure_llm


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" in row and "prompt" not in row:
                row["prompt"] = row.pop("question")
            if "prompt" not in row or "completion" not in row:
                raise ValueError(f"Row missing prompt/completion: {line[:200]!r}")
            rows.append(row)
    return rows


def _split_rows(
    rows: List[Dict[str, Any]],
    split: str,
    val_fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Mirror the train/eval split used by the trainer (HF train_test_split)."""
    from datasets import Dataset

    dataset = Dataset.from_list(rows)
    splits = dataset.train_test_split(test_size=val_fraction, seed=seed)
    if split == "train":
        return list(splits["train"])
    if split == "eval":
        return list(splits["test"])
    return rows  # "all"


def _content_hash(prompt: str, completion: str) -> str:
    """16-char hex prefix of SHA-256(prompt + NUL + completion)."""
    digest = hashlib.sha256(
        f"{prompt}\x00{completion}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Activation generation
# ---------------------------------------------------------------------------

def _generate_batch(
    llm: vllm.LLM,
    token_id_lists: List[List[int]],
    layer_idx: int,
    lora_request: Optional[LoRARequest],
) -> List[torch.Tensor]:
    """Run vLLM prefill and return per-sequence hidden states for ``layer_idx``."""
    sampling_params = vllm.SamplingParams(n=1, temperature=0.0, max_tokens=1)
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": ids} for ids in token_id_lists],
        sampling_params=sampling_params,
        use_tqdm=False,
        lora_request=lora_request,
    )
    result: List[torch.Tensor] = []
    for output in outputs:
        per_req = extract_prefill_hidden_states(output)
        result.append(per_req[layer_idx])
    return result


def _save_npz_atomic(path: Path, hidden_states: np.ndarray, n_completion_tokens: int, layer_idx: int) -> None:
    """Write .npz atomically via a temp file so partial writes are never left on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp.npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                hidden_states=hidden_states,
                n_completion_tokens=np.array(n_completion_tokens, dtype=np.int64),
                layer_idx=np.array(layer_idx, dtype=np.int64),
            )
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory to write .npz activation files.",
)
@click.option(
    "--data-path", "-d",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override annotations_jsonl from config.",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=None,
    help="Number of sequences per vLLM call (default: KV-budget optimal).",
)
@click.option(
    "--split",
    type=click.Choice(["all", "train", "eval"]),
    default="all",
    show_default=True,
    help="Process only the train split, eval split, or all examples.",
)
def main(
    config_path: Path,
    output_dir: Path,
    data_path: Optional[Path],
    batch_size: Optional[int],
    split: str,
) -> None:
    cfg = TrainingConfig.from_yaml(config_path)
    if cfg.dtype in (None, "auto"):
        cfg.dtype = get_dtype() if torch.cuda.is_available() else None

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    data_source = data_path or Path(cfg.annotations_jsonl)
    rows = _load_rows(data_source)
    print(f"Loaded {len(rows)} rows from {data_source}")

    if split != "all":
        rows = _split_rows(rows, split, cfg.val_fraction, cfg.seed)
        print(f"Using '{split}' split: {len(rows)} rows")

    # ── Skip already-saved examples ────────────────────────────────────────
    todo: List[Tuple[Dict[str, Any], str, Path]] = []
    for row in rows:
        h = _content_hash(row["prompt"], row["completion"])
        out_path = output_dir / f"{h}.npz"
        if not out_path.exists():
            todo.append((row, h, out_path))

    n_done = len(rows) - len(todo)
    print(f"{n_done} already saved, {len(todo)} to generate")
    if not todo:
        print("Nothing to do.")
        return

    # ── Initialise vLLM ────────────────────────────────────────────────────
    vllm_probe_plugin.register()

    tp, dp = resolve_gpu_counts(cfg.tensor_parallel_size)
    major_cc = get_compute_capability() if torch.cuda.is_available() else 0
    chunked_prefill = (
        cfg.enable_chunked_prefill
        if cfg.enable_chunked_prefill is not None
        else major_cc >= 8
    )

    print(
        f"Loading vLLM: model={cfg.model_name}, layer={cfg.layer_idx}, "
        f"tp={tp}, dp={dp}, chunked_prefill={chunked_prefill}"
    )
    llm = configure_llm(
        model=cfg.model_name,
        layers=[cfg.layer_idx],
        dtype=cfg.dtype,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=cfg.trust_remote_code,
        enable_chunked_prefill=chunked_prefill,
        enable_lora=cfg.lora_present,
        cudagraph_capture_sizes=[8, 16, 32, 64, 128], # minimise startup time
    )

    lora_request: Optional[LoRARequest] = None
    if cfg.lora_present:
        lora_request = LoRARequest(
            lora_name="lora_probe",
            lora_int_id=1,
            lora_path=cfg.lora_path,
        )
        print(f"LoRA enabled: {cfg.lora_path}")

    effective_mml = effective_probe_max_model_len(llm, cfg.max_model_len)
    tokenizer = llm.get_tokenizer()
    max_prefill_tokens = effective_mml - 1  # reserve 1 for the dummy decode token

    bs = batch_size or optimal_batch_size(llm, effective_mml)
    print(f"Batch size: {bs} sequences per vLLM call")

    # ── Generate ───────────────────────────────────────────────────────────
    n_saved = 0
    n_skipped = 0

    for batch_start in tqdm(range(0, len(todo), bs), unit="batch", desc="generating", dynamic_ncols=True):
        chunk = todo[batch_start : batch_start + bs]

        token_id_lists: List[List[int]] = []
        n_completion_tokens_list: List[int] = []
        valid_chunk: List[Tuple[Dict[str, Any], str, Path]] = []

        for row, h, out_path in chunk:
            prompt, completion = row["prompt"], row["completion"]

            enc = encode_probe_chat(
                tokenizer,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                chat_template_kwargs=cfg.chat_template_kwargs,
            )
            n_tokens = len(enc["input_ids"])

            if n_tokens > max_prefill_tokens:
                tqdm.write(
                    f"[skip] {h}: {n_tokens} tokens > limit {max_prefill_tokens}"
                )
                n_skipped += 1
                continue

            # Count completion tokens so callers can slice hidden_states[-n:] to get
            # only the assistant portion without re-tokenising.
            enc_prompt_only = encode_probe_chat(
                tokenizer,
                [{"role": "user", "content": prompt}],
                chat_template_kwargs=cfg.chat_template_kwargs,
                add_generation_prompt=True,
            )
            n_completion_tokens = n_tokens - len(enc_prompt_only["input_ids"])

            token_id_lists.append(enc["input_ids"])
            n_completion_tokens_list.append(max(0, n_completion_tokens))
            valid_chunk.append((row, h, out_path))

        if not token_id_lists:
            continue

        hidden_states_list = _generate_batch(llm, token_id_lists, cfg.layer_idx, lora_request)

        for (row, h, out_path), hs, n_comp in zip(
            valid_chunk, hidden_states_list, n_completion_tokens_list
        ):
            arr = hs.cpu().to(torch.float32).numpy()
            _save_npz_atomic(out_path, arr, n_comp, cfg.layer_idx)
            n_saved += 1

    print(
        f"Done. Saved {n_saved} new files to {output_dir}/ "
        f"({n_skipped} skipped: too long)."
    )
    print(
        "Each .npz contains:\n"
        "  hidden_states       — float32 [seq_len, hidden_size] (full prefill)\n"
        "  n_completion_tokens — int64 scalar; use hidden_states[-n:] for assistant tokens\n"
        "  layer_idx           — int64 scalar"
    )


if __name__ == "__main__":
    main()
