#!/usr/bin/env python
"""
utility functions for loading and processing activation data
"""

from pathlib import Path
import hashlib
import sys
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm

_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from data import _load_annotation_rows  # noqa: E402
from utils import probe_completion_token_labels  # noqa: E402


def prepare_records(
    data: List[Dict[str, Any]],
    model_name: str,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Align completion hidden states with per-token labels for each loaded record.

    Each returned dict has:
      - hidden_states: np.ndarray (n_completion_tokens, hidden_size)
      - token_labels:  np.ndarray (n_completion_tokens,) — 0.0 / 1.0, -100 for ignored tokens
      - id, prompt, completion, layer_idx
    """
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = chat_template_kwargs or {}

    records = []
    n_skipped = 0
    for entry in tqdm(data, desc="Aligning labels", dynamic_ncols=True):
        prompt = entry.get("prompt", "")
        completion = entry.get("completion", "")
        annotations_val = entry.get("annotations_val", [])
        n_completion_tokens = int(entry["n_completion_tokens"])
        hidden_states = entry["hidden_states"]

        completion_hs = (
            hidden_states[-n_completion_tokens:] if n_completion_tokens > 0 else hidden_states[:0]
        )
        token_labels = probe_completion_token_labels(
            tokenizer, prompt, completion, annotations_val,
            chat_template_kwargs=kwargs,
        ).numpy()

        n = min(len(completion_hs), len(token_labels))
        if n == 0:
            n_skipped += 1
            continue

        records.append({
            "id": entry.get("id"),
            "hidden_states": completion_hs[:n],
            "token_labels": token_labels[:n],
            "prompt": prompt,
            "completion": completion,
            "layer_idx": int(entry["layer_idx"]),
        })

    if n_skipped:
        print(f"Skipped {n_skipped} record(s) with no completion tokens.")
    return records


def load_activations(
    activations_dir: Path,
    annotations_jsonl: Optional[Path] = None,
    num_to_load: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load activations from .npz files, optionally joined with annotation records.

    When annotations_jsonl is provided each returned dict merges the npz arrays
    (hidden_states, n_completion_tokens, layer_idx) with the corresponding annotation
    fields (id, prompt, completion, annotations_val, policy, model, ...).
    .npz files with no matching annotation record are skipped with a warning.
    """
    activations_dir = Path(activations_dir)
    file_paths = sorted(activations_dir.glob("*.npz"))
    if num_to_load is not None:
        file_paths = file_paths[:num_to_load]

    print(f"Found {len(file_paths)} files in {activations_dir}")

    ann_by_hash: Dict[str, Dict[str, Any]] = {}
    if annotations_jsonl is not None:
        rows = _load_annotation_rows(Path(annotations_jsonl))
        for row in rows:
            prompt = row.get("prompt", "")
            completion = row.get("completion", "")
            h = hashlib.sha256(f"{prompt}\x00{completion}".encode()).hexdigest()[:16]
            ann_by_hash[h] = row
        print(f"Loaded {len(ann_by_hash)} annotation records from {annotations_jsonl}")

    data_loaded: List[Dict[str, Any]] = []
    n_missing = 0
    for file_path in tqdm(file_paths, desc="Loading activations", dynamic_ncols=True):
        npz = np.load(file_path, allow_pickle=True)
        entry: Dict[str, Any] = {key: npz[key] for key in npz.files}

        if ann_by_hash:
            ann = ann_by_hash.get(file_path.stem)
            if ann is None:
                n_missing += 1
                continue
            entry.update(ann)

        data_loaded.append(entry)

    if n_missing:
        print(f"Warning: {n_missing} .npz file(s) had no matching annotation record and were skipped.")

    return data_loaded


if __name__ == "__main__":
    """this section is only to be used for testing. this script should ordinarily be imported"""
    activations_dir = Path("/pool/lmawby/data/probe_ann_activations/")
    annotations_jsonl = Path("data/annotated.jsonl")
    data = load_activations(activations_dir, annotations_jsonl)
    print(f"Loaded {len(data)} records")
    if data:
        print("Keys in first record:", list(data[0].keys()))
