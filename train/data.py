#!/usr/bin/env python3
"""
Load annotated JSONL for training: per-character labels default to 0 (no violation);
explicit policy-violation spans are 1. Batched tensors pad with -100 so loss ignores padding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from models import ProbeModelConfig
from utils import encode_probe_chat, probe_completion_token_labels, probe_prefill_token_count

log = logging.getLogger(__name__)


@dataclass
class PreparedProbeBatch:
    """Encoded chat inputs plus per-completion-token labels for one trainer batch."""

    token_id_lists: List[List[int]]
    completion_token_labels: List[torch.Tensor]


@dataclass
class ProbeFeatureBatch:
    """Model-ready features grouped into a uniform batch shape."""

    features: torch.Tensor
    labels: torch.Tensor


_HALLUCINATION_LABELS = {"Not Supported", "Insufficient Information"}


def _map_violation_spans_to_val(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace `annotations` with `annotations_val`: one float per completion character.

    - 0.0 = not a violation (default).
    - 1.0 = character falls inside a violation span.

    For policy-violation annotations (no ``label`` field), every span is treated as a
    violation. For hallucination annotations (``label`` field present), only spans with
    label ``"Not Supported"`` or ``"Insufficient Information"`` are treated as violations;
    ``"Supported"`` spans are skipped.
    """
    for index, line in enumerate(data):
        completion = line["completion"]
        annotations_list = line.get("annotations") or []
        annotations_vals = [0.0] * len(completion)

        for span in annotations_list:
            if not isinstance(span, dict):
                continue
            label = span.get("label")
            if label is not None and label not in _HALLUCINATION_LABELS:
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


def _load_annotation_rows(path: Path) -> List[Dict[str, Any]]:
    """Load and normalize annotation rows from JSONL, preserving original order."""
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

    return _map_violation_spans_to_val(rows)


def _load_annotations(path: Path, test_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """Load JSONL into a DatasetDict with train/eval split (HuggingFace still uses ``test_size`` for the held-out fraction).

    Expected JSONL fields include ``question``, ``completion``, and optional ``annotations``
    (list of dicts with ``span``, ``index``, ``verification_note``). Other fields are kept.
    """
    rows = _load_annotation_rows(path)
    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(train=split["train"], eval=split["test"])


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


def prepare_probe_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    completions: List[str],
    annotations: torch.Tensor,
    *,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> PreparedProbeBatch:
    """Tokenize a trainer batch into vLLM inputs and completion-token labels."""
    token_id_lists: List[List[int]] = []
    completion_token_labels: List[torch.Tensor] = []

    for prompt, completion, annotation in zip(prompts, completions, annotations):
        encoding = encode_probe_chat(
            tokenizer,
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            chat_template_kwargs=chat_template_kwargs,
            return_offsets_mapping=False,
        )
        token_labels = probe_completion_token_labels(
            tokenizer,
            prompt,
            completion,
            annotation,
            chat_template_kwargs=chat_template_kwargs,
        )
        completion_token_labels.append(token_labels)
        token_id_lists.append(encoding["input_ids"])

    return PreparedProbeBatch(
        token_id_lists=token_id_lists,
        completion_token_labels=completion_token_labels,
    )


def _covseq_windows(
    hidden_states_list: List[torch.Tensor],
    completion_token_labels: List[torch.Tensor],
    window_size: int,
) -> Dict[int, tuple]:
    """Build CovSeq sliding-window buckets for one layer.

    Returns a dict mapping seq_len -> (stacked_windows, stacked_labels).
    """
    # Buckets hold chunks to be cat'd; avoids O(N) Python list decomposition per sequence.
    buckets: dict[int, List[torch.Tensor]] = {}
    label_buckets: dict[int, List[torch.Tensor]] = {}

    for hidden_states, token_labels in zip(hidden_states_list, completion_token_labels):
        n_completion_tokens = int(token_labels.shape[0])
        hs = hidden_states[-n_completion_tokens:] if n_completion_tokens else hidden_states[:0]
        n_tokens = min(hs.shape[0], token_labels.shape[0])
        if n_tokens == 0:
            continue

        hs = hs[:n_tokens].contiguous()
        token_labels = token_labels[:n_tokens].float()  # labels stay float32 for BCE

        # Prefix windows (seq_len < window_size): unavoidably variable-length, one per token.
        prefix_limit = min(window_size - 1, n_tokens)
        for seq_len in range(1, prefix_limit + 1):
            label = token_labels[seq_len - 1]
            if label == -100:
                continue
            buckets.setdefault(seq_len, []).append(hs[:seq_len].unsqueeze(0))
            label_buckets.setdefault(seq_len, []).append(label.unsqueeze(0))

        if n_tokens >= window_size:
            # stride_tricks gives a view — index directly into hs without materialising indices.
            full_windows = hs.unfold(0, window_size, 1)  # [n_windows, hidden_size, window_size]
            full_windows = full_windows.permute(0, 2, 1)  # [n_windows, window_size, hidden_size]
            full_labels = token_labels[window_size - 1:]
            keep = full_labels != -100
            if keep.any():
                buckets.setdefault(window_size, []).append(full_windows[keep])
                label_buckets.setdefault(window_size, []).append(full_labels[keep])

    return {
        seq_len: (
            torch.cat(buckets[seq_len], dim=0),
            torch.cat(label_buckets[seq_len], dim=0),
        )
        for seq_len in sorted(buckets)
    }


def build_probe_feature_batches(
    hidden_states_list,
    completion_token_labels: List[torch.Tensor],
    model_cfg: ProbeModelConfig,
) -> List[ProbeFeatureBatch]:
    """Build model-ready batches from completion hidden states and token labels.

    For ``covseq``, full-length windows are built with
    :func:`numpy.lib.stride_tricks.sliding_window_view`; the shorter prefix windows
    that occur near the start of each completion are added separately so the effective
    sequence length is truncated instead of padded.

    For ``multi_layer_covseq``, ``hidden_states_list`` is
    ``List[List[Tensor]]`` (one inner list per layer, each containing one tensor per
    sequence). Each :class:`ProbeFeatureBatch` has ``features`` as a
    ``List[Tensor]`` — one ``[batch, seq_len, hidden_size]`` tensor per layer —
    and ``labels`` as the shared label tensor.
    """
    if model_cfg.model_type == "multi_layer_covseq":
        # hidden_states_list: List[List[Tensor]], outer = layers, inner = sequences
        n_layers = len(hidden_states_list)
        if n_layers == 0:
            return []
        n_seqs = len(hidden_states_list[0])
        if any(len(hs) != n_seqs for hs in hidden_states_list):
            raise ValueError("All layers must have the same number of sequences")
        if n_seqs != len(completion_token_labels):
            raise ValueError(
                "hidden_states_list and completion_token_labels must have the same length"
            )

        window_size = model_cfg.covseq.window_size
        # Build windows for each layer; labels are shared so we take them from layer 0
        layer_windows = [
            _covseq_windows(hidden_states_list[li], completion_token_labels, window_size)
            for li in range(n_layers)
        ]
        # Intersect seq_lens present in all layers
        common_seq_lens = sorted(
            set(layer_windows[0].keys()).intersection(*[set(lw.keys()) for lw in layer_windows[1:]])
        )
        if not common_seq_lens:
            return []

        feature_batches: List[ProbeFeatureBatch] = []
        for seq_len in common_seq_lens:
            # features: list of [batch, seq_len, hidden_size] tensors, one per layer
            layer_tensors = [layer_windows[li][seq_len][0] for li in range(n_layers)]
            labels = layer_windows[0][seq_len][1]
            feature_batches.append(ProbeFeatureBatch(features=layer_tensors, labels=labels))
        return feature_batches

    if len(hidden_states_list) != len(completion_token_labels):
        raise ValueError(
            "hidden_states_list and completion_token_labels must have the same length"
        )

    if model_cfg.model_type == "mlp":
        features: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for hidden_states, token_labels in zip(hidden_states_list, completion_token_labels):
            n_completion_tokens = int(token_labels.shape[0])
            completion_hidden_states = hidden_states[-n_completion_tokens:] if n_completion_tokens else hidden_states[:0]
            n_tokens = min(completion_hidden_states.shape[0], token_labels.shape[0])
            if n_tokens == 0:
                continue
            token_labels = token_labels[:n_tokens]
            keep = token_labels != -100
            if not keep.any():
                continue
            features.append(completion_hidden_states[:n_tokens][keep])
            labels.append(token_labels[keep].float())  # labels stay float32 for BCE
        if not features:
            return []
        return [ProbeFeatureBatch(features=torch.cat(features, dim=0), labels=torch.cat(labels, dim=0))]

    if model_cfg.model_type != "covseq":
        raise ValueError(f"Unknown probe model type: {model_cfg.probe_model_type!r}")

    windows = _covseq_windows(hidden_states_list, completion_token_labels, model_cfg.covseq.window_size)
    return [
        ProbeFeatureBatch(features=feats, labels=labs)
        for feats, labs in windows.values()
    ]


def count_violation_and_nonviolation_completion_tokens(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Return ``(n_violation, n_nonviolation)`` over **completion token** labels (see :func:`probe_completion_token_labels`)."""
    n_violation = n_nonviolation = 0
    for example in tqdm(dataset, desc="Class balance", unit="rec", leave=False):
        tok = probe_completion_token_labels(
            tokenizer,
            example["prompt"],
            example["completion"],
            example["annotations_val"],
            chat_template_kwargs=chat_template_kwargs,
        )
        for val in tok.tolist():
            if val == 1.0:
                n_violation += 1
            elif val == 0.0:
                n_nonviolation += 1
    return n_violation, n_nonviolation


def summarize_token_class_balance(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Aggregate completion-token counts for the rare-positive (violation) setup.

    Uses the same tokenization and char→token rules as training (:func:`probe_completion_token_labels`).
    """
    n_v, n_n = count_violation_and_nonviolation_completion_tokens(
        dataset, tokenizer, chat_template_kwargs=chat_template_kwargs
    )
    total = n_v + n_n
    frac_pos = (n_v / total) if total else 0.0
    neg_per_pos = (n_n / n_v) if n_v else float("inf")
    return {
        "n_violation_tokens": n_v,
        "n_nonviolation_tokens": n_n,
        "fraction_violation_tokens": frac_pos,
        "neg_tokens_per_violation_token": neg_per_pos,
    }


def completion_token_pos_neg_counts(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Count positive (violation) and negative completion tokens for one row."""
    tok = probe_completion_token_labels(
        tokenizer,
        example["prompt"],
        example["completion"],
        example["annotations_val"],
        chat_template_kwargs=chat_template_kwargs,
    )
    n_pos = n_neg = 0
    for val in tok.tolist():
        if val == 1.0:
            n_pos += 1
        elif val == 0.0:
            n_neg += 1
    return n_pos, n_neg


def row_sampler_weight_from_token_counts(
    n_pos: int,
    n_neg: int,
    *,
    w_min: float = 1.0,
    w_max: float = 10.0,
) -> float:
    """Sampler weight linear in the **fraction** of violation (positive) completion tokens.

    ``frac_pos = n_pos / (n_pos + n_neg)`` (0 if no labelled completion tokens). Weight is
    ``w_min + (w_max - w_min) * frac_pos``, so rows with more positives **relative to row size**
    get larger weights, rows with no positives get ``w_min`` (every row stays in play). Remaining
    token-level imbalance is corrected by ``pos_weight`` on the loss, not by this mapping alone.

    Requires ``0 < w_min <= w_max``. Set ``w_min == w_max`` for uniform row sampling.
    """
    if w_min <= 0 or w_max < w_min:
        raise ValueError(f"w_min and w_max must satisfy 0 < w_min <= w_max; got {w_min=}, {w_max=}")
    total = n_pos + n_neg
    if total == 0:
        return w_min
    frac_pos = n_pos / total
    return w_min + (w_max - w_min) * frac_pos


def row_sampler_weight_from_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None = None,
    *,
    w_min: float = 1.0,
    w_max: float = 10.0,
) -> float:
    """Row weight for :class:`~torch.utils.data.WeightedRandomSampler`.

    Uses **completion token** counts only — the same labels the probe trains on
    (:func:`completion_token_pos_neg_counts` / :func:`utils.probe_completion_token_labels`),
    not raw character counts (tokens can merge/split vs characters).

    Weight is :func:`row_sampler_weight_from_token_counts` applied to ``(n_pos, n_neg)``.
    """
    n_pos, n_neg = completion_token_pos_neg_counts(example, tokenizer, chat_template_kwargs)
    return row_sampler_weight_from_token_counts(n_pos, n_neg, w_min=w_min, w_max=w_max)


def train_sampler_weights_and_effective_pos_weight(
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None = None,
    *,
    w_min: float = 1.0,
    w_max: float = 10.0,
) -> tuple[List[float], float, float]:
    """Build per-row sampler weights and BCE ``pos_weight`` consistent with weighted row sampling.

    Returns ``(weights, effective_pos_weight, raw_pos_weight)`` in a single pass.

    **Sampler.** Row ``i`` is drawn with probability proportional to ``w_i`` from
    :func:`row_sampler_weight_from_token_counts` (linear in violation **fraction** per row,
    between ``w_min`` and ``w_max``).

    **Effective token imbalance.** Under independent row draws with
    ``P(i) ∝ w_i``, the expected counts of neg vs pos **tokens** contributed per draw scale with
    ``Σ w_i n_{neg,i}`` and ``Σ w_i n_{pos,i}`` (same ``n`` as in the loss). The ratio
    ``(Σ w_i n_{neg,i}) / (Σ w_i n_{pos,i})`` is the imbalance of the positive class relative
    to the negative class in that **weighted token stream**, and matches what
    :class:`~torch.nn.BCEWithLogitsLoss` ``pos_weight`` is meant to correct when set to
    ``n_neg/n_pos`` for the training distribution.

    **Why effective can differ from raw neg/pos:** mixing ``w_i`` with per-row token counts
    changes the weighted token stream; compare logs to ``raw_pos_weight`` for the unweighted ratio.
    """
    weights: List[float] = []
    sw_pos = sw_neg = 0.0
    raw_pos = raw_neg = 0
    for i in tqdm(range(len(train_dataset)), desc="Sampler weights", unit="rec", leave=False):
        ex = train_dataset[i]
        n_pos, n_neg = completion_token_pos_neg_counts(ex, tokenizer, chat_template_kwargs)
        w = row_sampler_weight_from_token_counts(n_pos, n_neg, w_min=w_min, w_max=w_max)
        weights.append(w)
        sw_pos += w * n_pos
        sw_neg += w * n_neg
        raw_pos += n_pos
        raw_neg += n_neg
    if sw_pos <= 0:
        raise ValueError(
            "Weighted positive token mass is zero — check train data and sampler weights."
        )
    if raw_pos <= 0:
        raise ValueError(
            "No violation completion tokens (label 1) found in dataset — cannot compute pos_weight."
        )
    return weights, sw_neg / sw_pos, raw_neg / raw_pos


def debug_print_weighted_sampling_stats(
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    weights: List[float],
    effective_pw: float,
    *,
    train_batch_size: int,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> None:
    """Log sampler weights and effective loss ratio; use to sanity-check ``WeightedRandomSampler`` + ``pos_weight``.

    Under i.i.d. draws with ``P(row i) ∝ w_i``:
    - Expected pos (violation) completion tokens per **single** draw = ``(Σ w_i n_{pos,i}) / (Σ w_i)``.
    - A microbatch has **zero** violation completion tokens iff every drawn row has ``n_pos=0``;
      approx probability ``≈ p_0^B`` where ``p_0 = (Σ_{n_pos=0} w_i) / (Σ w_i)`` and ``B`` = batch size.
    """
    n = len(train_dataset)
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != dataset length {n}")

    sum_w = sum(weights)
    sw_pos = sw_neg = 0.0
    sum_w_pos0 = 0.0  # mass on rows with no violation completion tokens
    n_pos0 = n_neg0 = n_both = 0

    for i in range(n):
        ex = train_dataset[i]
        np_i, nn_i = completion_token_pos_neg_counts(ex, tokenizer, chat_template_kwargs)
        w = weights[i]
        sw_pos += w * np_i
        sw_neg += w * nn_i
        if np_i == 0:
            n_pos0 += 1
            sum_w_pos0 += w
        elif nn_i == 0:
            n_neg0 += 1
        else:
            n_both += 1

    eff_check = sw_neg / sw_pos
    if abs(eff_check - effective_pw) > 1e-3 * max(1.0, effective_pw):
        print(
            f"[debug_sampling] WARNING: recomputed (Σw·n_neg)/(Σw·n_pos)={eff_check:.6f} "
            f"!= effective_pw={effective_pw:.6f}"
        )

    ws = sorted(weights)
    median_w = ws[n // 2] if n else 0.0
    mean_w = sum_w / n if n else 0.0
    n_w0 = sum(1 for w in weights if w == 0.0)

    e_pos_per_draw = sw_pos / sum_w if sum_w else 0.0
    e_neg_per_draw = sw_neg / sum_w if sum_w else 0.0
    p0 = sum_w_pos0 / sum_w if sum_w else 1.0
    B = max(1, train_batch_size)
    p_batch_no_pos = p0**B

    print("[debug_sampling] per-row weight w: min={:.6g}, max={:.6g}, mean={:.6g}, median={:.6g}".format(
        ws[0] if n else 0.0, ws[-1] if n else 0.0, mean_w, median_w
    ))
    print(
        f"[debug_sampling] sum(w)={sum_w:.6g}, zero_w_rows={n_w0}, "
        f"rows_n_pos=0_frac_pos=0(w=w_min)={n_pos0}"
    )
    print(
        f"[debug_sampling] rows: n_pos_completion=0 → {n_pos0}; "
        f"n_neg_completion=0 → {n_neg0}; both>0 → {n_both}"
    )
    print(
        f"[debug_sampling] Σ(w·n_pos)={sw_pos:.6g}, Σ(w·n_neg)={sw_neg:.6g}, "
        f"(Σw·n_neg)/(Σw·n_pos)={eff_check:.6g} (BCE pos_weight should match)"
    )
    e_ratio = e_neg_per_draw / e_pos_per_draw if e_pos_per_draw else float("inf")
    print(
        f"[debug_sampling] under P(i)∝w: E[pos tok/draw]={e_pos_per_draw:.6g}, "
        f"E[neg tok/draw]={e_neg_per_draw:.6g}, E[neg]/E[pos]={e_ratio:.6g}"
    )
    print(
        f"[debug_sampling] P(one draw has zero pos completion tok) = "
        f"(Σ w on n_pos=0 rows)/sum(w) = {p0:.6f}"
    )
    print(
        f"[debug_sampling] i.i.d. batch size B={B}: approx P(batch Σ pos tok = 0) ≈ "
        f"p0^B = {p_batch_no_pos:.6g}  (many tqdm steps with n_v+_tok=0 can be expected if this is not tiny)"
    )


def compute_pos_weight(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None = None,
    max_pos_weight: float | None = None,
) -> float:
    """``n_neg / n_pos`` over the dataset **without** row sampling weights (for logging).

    Counts **completion tokenizer tokens** with labels 0/1 via :func:`probe_completion_token_labels`
    (same convention as :class:`~trainer.ProbeTrainer`). Positions with label ``-100`` are excluded.

    For the loss when using :class:`~torch.utils.data.WeightedRandomSampler`, prefer
    :func:`train_sampler_weights_and_effective_pos_weight` instead.

    If ``max_pos_weight`` is set, the value is capped for numerical stability when
    violations are extremely rare.
    """
    n_pos = n_neg = 0
    for example in tqdm(dataset, desc="Pos weight", unit="rec", leave=False):
        np_i, nn_i = completion_token_pos_neg_counts(example, tokenizer, chat_template_kwargs)
        n_pos += np_i
        n_neg += nn_i
    if n_pos == 0:
        raise ValueError(
            "No violation completion tokens (label 1) found in dataset — cannot compute pos_weight."
        )
    w = n_neg / n_pos
    if max_pos_weight is not None:
        w = min(w, max_pos_weight)
    return w


def truncate_dataset(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    max_model_len: int,
    chat_template_kwargs: Dict[str, Any] | None = None,
    decode_reserve_tokens: int = 1,
) -> DatasetDict:
    """Drop examples that cannot fit in vLLM's context at prefill + decode.

    Keeps rows where **prefill** token count ≤ ``max_model_len - decode_reserve_tokens``.
    Prefill count is the full templated chat (user + assistant), same as
    :func:`utils.probe_prefill_token_count` / training. ``decode_reserve_tokens`` should match
    ``SamplingParams.max_tokens`` in :class:`~trainer.ProbeTrainer` (default 1).
    """
    max_prefill_tokens = max_model_len - decode_reserve_tokens
    if max_prefill_tokens < 1:
        raise ValueError(
            f"max_model_len={max_model_len} too small for decode_reserve_tokens={decode_reserve_tokens}"
        )

    def is_within_limit(example: Dict[str, Any]) -> bool:
        n = probe_prefill_token_count(
            tokenizer,
            example["prompt"],
            example["completion"],
            chat_template_kwargs=chat_template_kwargs,
        )
        return n <= max_prefill_tokens

    n_before = {split: len(dataset_dict[split]) for split in dataset_dict}
    out = DatasetDict(
        {split: dataset_dict[split].filter(is_within_limit) for split in dataset_dict}
    )
    for split in out:
        removed = n_before[split] - len(out[split])
        if removed:
            print(
                f"[truncate_dataset] {split}: removed {removed} examples with "
                f">{max_prefill_tokens} prefill tokens (user+assistant chat; "
                f"limit max_model_len={max_model_len}, reserve_decode={decode_reserve_tokens})"
            )
    return out


def build_annotations_dataset_dict(path: Path, test_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """Build a ``DatasetDict`` from an annotated JSONL path (``train`` / ``eval`` split; no padding here)."""
    return _load_annotations(path, test_size=test_size, seed=seed)


def build_annotations_dataset(path: Path) -> Dataset:
    """Build a single unsplit ``Dataset`` from an annotated JSONL path."""
    return Dataset.from_list(_load_annotation_rows(path))
