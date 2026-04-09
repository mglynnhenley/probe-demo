#!/usr/bin/env python3
"""
Load annotated JSONL for training: per-character labels default to 0 (no violation);
explicit policy-violation spans are 1. Batched tensors pad with -100 so loss ignores padding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)


def _map_violation_spans_to_val(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace `annotations` with `annotations_val`: one float per completion character.

    - 0.0 = not labelled as a violation (default for every character).
    - 1.0 = character falls inside an annotated violation span (uses `index` + `span` length).
    """
    for index, line in enumerate(data):
        completion = line["completion"]
        annotations_list = line.get("annotations") or []
        annotations_vals = [0.0] * len(completion)

        for span in annotations_list:
            if not isinstance(span, dict):
                continue
            start = span.get("index")
            if start is None:
                log.debug("Skipping span without index: %r", span)
                continue
            start = int(start)
            span_text = span.get("span") or ""
            end = start + len(span_text)
            if start < 0 or start >= len(completion):
                log.warning("Span index out of range (id=%r): start=%s", line.get("id"), start)
                continue
            end = min(end, len(completion))
            if end <= start:
                continue
            annotations_vals[start:end] = [1.0] * (end - start)

        line.pop("annotations", None)
        line["annotations_val"] = annotations_vals

    return data


def _load_annotations(path: Path, test_size: float = 0.1) -> DatasetDict:
    """Load JSONL into a DatasetDict with train/test split.

    Expected JSONL fields include ``question``, ``completion``, and optional ``annotations``
    (list of dicts with ``span``, ``index``, ``verification_note``). Other fields are kept.
    """
    field_rename = {"question": "prompt"}

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for old, new in field_rename.items():
                if old in row:
                    row[new] = row.pop(old)
            if "prompt" not in row or "completion" not in row:
                raise ValueError(f"Missing question/prompt or completion: {line[:200]!r}")
            rows.append(row)

    rows = _map_violation_spans_to_val(rows)
    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=test_size)
    return DatasetDict(train=split["train"], test=split["test"])


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch prompts/completions and pad per-character labels with -100 (ignored by loss)."""
    return {
        "prompt": [item["prompt"] for item in batch],
        "completion": [item["completion"] for item in batch],
        "annotations_val": pad_sequence(
            [torch.tensor(item["annotations_val"], dtype=torch.float32) for item in batch],
            batch_first=True,
            padding_value=-100.0,
        ),
    }


def count_violation_and_nonviolation_tokens(dataset: Dataset) -> tuple[int, int]:
    """Return ``(n_violation, n_nonviolation)`` over all character labels (0/1 only in stored rows)."""
    n_violation = n_nonviolation = 0
    for example in dataset:
        for val in example["annotations_val"]:
            if val == 1.0:
                n_violation += 1
            elif val == 0.0:
                n_nonviolation += 1
    return n_violation, n_nonviolation


def summarize_token_class_balance(dataset: Dataset) -> dict[str, float | int]:
    """Aggregate token counts for the rare-positive (violation) setup.

    Policy-violation data is typically **heavily imbalanced**: label ``1`` only on short
    spans, label ``0`` on almost all completion characters. Use this to interpret
    ``pos_weight`` and evaluation metrics (e.g. majority-class baseline accuracy).
    """
    n_v, n_n = count_violation_and_nonviolation_tokens(dataset)
    total = n_v + n_n
    frac_pos = (n_v / total) if total else 0.0
    neg_per_pos = (n_n / n_v) if n_v else float("inf")
    return {
        "n_violation_tokens": n_v,
        "n_nonviolation_tokens": n_n,
        "fraction_violation_tokens": frac_pos,
        "neg_tokens_per_violation_token": neg_per_pos,
    }


def compute_pos_weight(
    dataset: Dataset,
    max_pos_weight: float | None = None,
) -> float:
    """``n_neg / n_pos`` for :class:`~torch.nn.BCEWithLogitsLoss` (positive class = violation = 1).

    Upweights gradient on rare **violation** tokens so the loss is not dominated by the
    overwhelming number of **non-violation** (0) tokens. Ignores padded positions (-100)
    in batched tensors (not present in per-example ``annotations_val`` lists).

    If ``max_pos_weight`` is set, the value is capped for numerical stability when
    violations are extremely rare.
    """
    n_pos = n_neg = 0
    for example in dataset:
        for val in example["annotations_val"]:
            if val == 1.0:
                n_pos += 1
            elif val == 0.0:
                n_neg += 1
    if n_pos == 0:
        raise ValueError(
            "No violation tokens (label 1) found in dataset — cannot compute pos_weight."
        )
    w = n_neg / n_pos
    if max_pos_weight is not None:
        w = min(w, max_pos_weight)
    return w


def truncate_dataset(dataset_dict: DatasetDict, tokenizer, max_model_len: int) -> DatasetDict:
    """Drop examples whose chat-template token length is >= ``max_model_len``."""
    def is_within_limit(example: Dict[str, Any]) -> bool:
        tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ],
            tokenize=True,
        )
        return len(tokens) < max_model_len

    n_before = {split: len(dataset_dict[split]) for split in dataset_dict}
    out = DatasetDict(
        {split: dataset_dict[split].filter(is_within_limit) for split in dataset_dict}
    )
    for split in out:
        removed = n_before[split] - len(out[split])
        if removed:
            print(f"[truncate_dataset] {split}: removed {removed} examples exceeding {max_model_len} tokens")
    return out


def build_annotations_dataloader(path: Path, test_size: float = 0.1) -> DatasetDict:
    """Build a ``DatasetDict`` from an annotated JSONL path (train/test split, no padding here)."""
    return _load_annotations(path, test_size=test_size)
