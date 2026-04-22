#!/usr/bin/env python3
"""Score a trained probe on annotated data and plot ROC from saved token logits.

Uses HFHiddenStateExtractor (not vLLM) so this runs on Mac MPS, CPU, or CUDA
with the same code path.

Example:
  .venv/bin/python scripts/evaluate_probe_roc.py configs/probe/qwen_small_mac.yaml
  .venv/bin/python scripts/evaluate_probe_roc.py configs/probe/qwen_small_mac.yaml --split eval
"""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import click
import numpy as np
import torch
from tqdm.auto import tqdm

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
for path in (ROOT, TRAIN_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_dtype(name: str | None) -> torch.dtype:
    if name is None or name == "auto":
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            return torch.bfloat16
        return torch.float32
    return _DTYPE_MAP[name]


def iter_dataset_batches(dataset: Dataset, batch_size: int) -> Iterable[tuple[List[int], List[Dict[str, Any]]]]:
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(len(dataset), start + batch_size)))
        yield indices, [dataset[i] for i in indices]


def build_feature_batch_metadata(
    hidden_states_list: List[torch.Tensor],
    completion_token_labels: List[torch.Tensor],
    model_cfg: Any,
) -> List[dict[str, torch.Tensor]]:
    """Benchmark-only metadata aligned with ``build_probe_feature_batches`` output."""
    if len(hidden_states_list) != len(completion_token_labels):
        raise ValueError(
            "hidden_states_list and completion_token_labels must have the same length"
        )

    if model_cfg.model_type == "mlp":
        example_indices: List[torch.Tensor] = []
        token_indices: List[torch.Tensor] = []
        window_lengths: List[torch.Tensor] = []
        for example_idx, (hidden_states, token_labels) in enumerate(
            zip(hidden_states_list, completion_token_labels)
        ):
            n_completion_tokens = int(token_labels.shape[0])
            completion_hidden_states = (
                hidden_states[-n_completion_tokens:] if n_completion_tokens else hidden_states[:0]
            )
            n_tokens = min(completion_hidden_states.shape[0], token_labels.shape[0])
            if n_tokens == 0:
                continue
            token_labels = token_labels[:n_tokens]
            keep = token_labels != -100
            if not keep.any():
                continue
            kept_token_indices = keep.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
            n_kept = int(kept_token_indices.shape[0])
            example_indices.append(torch.full((n_kept,), example_idx, dtype=torch.long))
            token_indices.append(kept_token_indices)
            window_lengths.append(torch.ones((n_kept,), dtype=torch.long))
        if not example_indices:
            return []
        return [
            {
                "example_indices": torch.cat(example_indices, dim=0),
                "token_indices": torch.cat(token_indices, dim=0),
                "window_lengths": torch.cat(window_lengths, dim=0),
            }
        ]

    if model_cfg.model_type != "covseq":
        raise ValueError(f"Unknown probe model type: {model_cfg.probe_model_type!r}")

    window_size = model_cfg.covseq.window_size
    example_index_buckets: dict[int, List[torch.Tensor]] = {}
    token_index_buckets: dict[int, List[torch.Tensor]] = {}
    window_length_buckets: dict[int, List[torch.Tensor]] = {}

    for example_idx, (hidden_states, token_labels) in enumerate(
        zip(hidden_states_list, completion_token_labels)
    ):
        n_completion_tokens = int(token_labels.shape[0])
        completion_hidden_states = (
            hidden_states[-n_completion_tokens:] if n_completion_tokens else hidden_states[:0]
        )
        n_tokens = min(completion_hidden_states.shape[0], token_labels.shape[0])
        if n_tokens == 0:
            continue

        token_labels = token_labels[:n_tokens].float()
        prefix_limit = min(window_size - 1, n_tokens)
        for seq_len in range(1, prefix_limit + 1):
            label = token_labels[seq_len - 1]
            if label == -100:
                continue
            example_index_buckets.setdefault(seq_len, []).append(
                torch.tensor(example_idx, dtype=torch.long)
            )
            token_index_buckets.setdefault(seq_len, []).append(
                torch.tensor(seq_len - 1, dtype=torch.long)
            )
            window_length_buckets.setdefault(seq_len, []).append(
                torch.tensor(seq_len, dtype=torch.long)
            )

        if n_tokens >= window_size:
            full_labels = token_labels[window_size - 1 :]
            keep = full_labels != -100
            if keep.any():
                kept_token_indices = torch.arange(
                    window_size - 1,
                    window_size - 1 + full_labels.shape[0],
                    dtype=torch.long,
                )[keep]
                n_kept = int(kept_token_indices.shape[0])
                example_index_buckets.setdefault(window_size, []).extend(
                    [torch.tensor(example_idx, dtype=torch.long) for _ in range(n_kept)]
                )
                token_index_buckets.setdefault(window_size, []).extend(
                    list(kept_token_indices)
                )
                window_length_buckets.setdefault(window_size, []).extend(
                    [torch.tensor(window_size, dtype=torch.long) for _ in range(n_kept)]
                )

    metadata_batches: List[dict[str, torch.Tensor]] = []
    for seq_len in sorted(example_index_buckets):
        metadata_batches.append(
            {
                "example_indices": torch.stack(example_index_buckets[seq_len], dim=0),
                "token_indices": torch.stack(token_index_buckets[seq_len], dim=0),
                "window_lengths": torch.stack(window_length_buckets[seq_len], dim=0),
            }
        )
    return metadata_batches


def score_dataset(
    dataset: Dataset,
    *,
    hf_extractor: Any,
    probe: Any,
    batch_size: int,
    chat_template_kwargs: Dict[str, Any],
) -> dict[str, np.ndarray]:
    from data import build_probe_feature_batches, collate_fn, prepare_probe_batch

    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    example_indices_all: List[torch.Tensor] = []
    token_indices_all: List[torch.Tensor] = []
    window_lengths_all: List[torch.Tensor] = []
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    n_scored_tokens = 0

    progress = tqdm(
        iter_dataset_batches(dataset, batch_size),
        total=n_batches,
        desc="Scoring dataset",
        unit="batch",
    )

    for batch_num, (batch_indices, rows) in enumerate(progress, start=1):
        collated = collate_fn(rows)
        annotations = collated["annotations_val"].float()
        prepared_batch = prepare_probe_batch(
            hf_extractor.tokenizer,
            collated["prompt"],
            collated["completion"],
            annotations,
            chat_template_kwargs=chat_template_kwargs,
        )
        prompt_token_lengths = [len(ids) for ids in prepared_batch.token_id_lists]
        progress.set_postfix(
            examples=len(rows),
            min_tok=min(prompt_token_lengths),
            max_tok=max(prompt_token_lengths),
            scored_tok=n_scored_tokens,
        )
        progress.set_description(f"Scoring dataset ({batch_num}/{n_batches})")

        hidden_states_dict = hf_extractor.extract(prepared_batch.token_id_lists)
        hidden_states_list = hidden_states_dict[probe.layer_idx]
        feature_batches = build_probe_feature_batches(
            hidden_states_list,
            prepared_batch.completion_token_labels,
            probe.cfg.model,
        )
        metadata_batches = build_feature_batch_metadata(
            hidden_states_list,
            prepared_batch.completion_token_labels,
            probe.cfg.model,
        )

        for bucket_num, (feature_batch, metadata_batch) in enumerate(
            zip(feature_batches, metadata_batches),
            start=1,
        ):
            progress.set_postfix(
                examples=len(rows),
                min_tok=min(prompt_token_lengths),
                max_tok=max(prompt_token_lengths),
                bucket=f"{bucket_num}/{len(feature_batches)}",
                bucket_shape=str(tuple(feature_batch.features.shape)),
                scored_tok=n_scored_tokens,
            )
            with torch.no_grad():
                logits = probe.model(feature_batch.features).squeeze(-1).detach().to("cpu", dtype=torch.float32)
            labels = feature_batch.labels.detach().to("cpu", dtype=torch.float32)
            local_example_indices = metadata_batch["example_indices"].detach().to("cpu", dtype=torch.int64)
            global_example_indices = torch.tensor(
                [batch_indices[int(i)] for i in local_example_indices.tolist()],
                dtype=torch.int64,
            )

            logits_all.append(logits)
            labels_all.append(labels)
            example_indices_all.append(global_example_indices)
            token_indices_all.append(metadata_batch["token_indices"].detach().to("cpu", dtype=torch.int64))
            window_lengths_all.append(metadata_batch["window_lengths"].detach().to("cpu", dtype=torch.int64))
            n_scored_tokens += int(logits.shape[0])

    progress.close()

    if not logits_all:
        raise ValueError("No labelled token scores were produced for the selected dataset")

    logits = torch.cat(logits_all, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {
        "logits": logits.astype(np.float32),
        "labels": labels.astype(np.uint8),
        "probs": probs.astype(np.float32),
        "example_indices": torch.cat(example_indices_all, dim=0).numpy().astype(np.int32),
        "token_indices": torch.cat(token_indices_all, dim=0).numpy().astype(np.int32),
        "window_lengths": torch.cat(window_lengths_all, dim=0).numpy().astype(np.int16),
    }


def compute_roc(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray | float]:
    labels = labels.astype(np.uint8)
    scores = scores.astype(np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be 1D arrays of the same length")

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC requires both positive and negative labels")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(sorted_scores) - 1]

    tpr = tp[threshold_indices] / n_pos
    fpr = fp[threshold_indices] / n_neg
    thresholds = sorted_scores[threshold_indices]

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    auc = float(np.trapezoid(tpr, fpr))
    return {
        "fpr": fpr.astype(np.float64),
        "tpr": tpr.astype(np.float64),
        "thresholds": thresholds.astype(np.float64),
        "auc": auc,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    seq_auc = payload.get("auc_roc_sequence")
    seq_auc_str = f"{seq_auc:.6f}" if seq_auc is not None else "undefined"
    lines = [
        f"n_scored_tokens={payload['n_scored_tokens']}",
        f"n_positive_tokens={payload['n_positive_tokens']}",
        f"n_negative_tokens={payload['n_negative_tokens']}",
        f"n_sequences={payload['n_sequences']}",
        f"n_sequences_violating={payload['n_sequences_violating']}",
        f"n_sequences_clean={payload['n_sequences_clean']}",
        f"auc_roc_token={payload['auc_roc_token']:.6f}",
        f"auc_roc_sequence={seq_auc_str}",
        f"logit_min={payload['logit_min']:.6f}",
        f"logit_max={payload['logit_max']:.6f}",
        f"logit_mean={payload['logit_mean']:.6f}",
        f"prob_mean={payload['prob_mean']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_roc_curve(roc: dict[str, np.ndarray | float], out_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    fpr = np.asarray(roc["fpr"])
    tpr = np.asarray(roc["tpr"])
    auc = float(roc["auc"])

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--probe-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to probe_head.bin (default: <output_dir>/probe_head.bin).",
)
@click.option(
    "--split",
    type=click.Choice(["all", "train", "eval"]),
    default="all",
    help="Which dataset split to score (ignored when --annotations-override is set).",
)
@click.option(
    "--annotations-override",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to an alternative annotated jsonl (e.g. a held-out test set). "
    "When set, ignores --split and scores every row in this file.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Examples per HF forward pass (default: config train_batch_size).",
)
@click.option(
    "--output-prefix",
    type=click.Path(path_type=Path),
    default=None,
    help="Output prefix for .npz/.png/.json/.txt files.",
)
def main(
    config_path: Path,
    probe_path: Path | None,
    split: str,
    annotations_override: Path | None,
    batch_size: int | None,
    output_prefix: Path | None,
) -> None:
    data_module = importlib.import_module("data")
    models_module = importlib.import_module("models")
    utils_module = importlib.import_module("utils")
    build_annotations_dataset = data_module.build_annotations_dataset
    build_annotations_dataset_dict = data_module.build_annotations_dataset_dict
    probe_prefill_token_count = utils_module.probe_prefill_token_count
    ProbeConfig = models_module.ProbeConfig
    ProbeModelConfig = models_module.ProbeModelConfig
    ValueHeadProbe = models_module.ValueHeadProbe
    TrainingConfig = importlib.import_module("config").TrainingConfig
    HFHiddenStateExtractor = importlib.import_module("hf_backend").HFHiddenStateExtractor

    cfg = TrainingConfig.from_yaml(config_path.expanduser().resolve())
    print(f"[setup] loaded config from {config_path.expanduser().resolve()}")
    probe_path = (
        probe_path.expanduser().resolve()
        if probe_path is not None
        else Path(cfg.output_dir).resolve() / "probe_head.bin"
    )
    if not probe_path.is_file():
        raise SystemExit(f"Probe checkpoint not found: {probe_path}")

    dataset: Dataset
    if annotations_override is not None:
        dataset = build_annotations_dataset(annotations_override.expanduser().resolve())
        split = annotations_override.stem  # used for output naming
        print(f"[setup] loaded held-out dataset from {annotations_override} "
              f"with {len(dataset)} examples before truncation")
    elif split == "all":
        dataset = build_annotations_dataset(Path(cfg.annotations_jsonl))
        print(f"[setup] loaded dataset split={split} with {len(dataset)} examples before truncation")
    else:
        dataset_dict = build_annotations_dataset_dict(
            path=Path(cfg.annotations_jsonl),
            test_size=cfg.val_fraction,
            seed=cfg.seed,
        )
        dataset = dataset_dict[split]
        print(f"[setup] loaded dataset split={split} with {len(dataset)} examples before truncation")

    dtype = resolve_dtype(cfg.dtype)
    print(f"[setup] dtype={dtype}")

    lora_path = cfg.lora_path if cfg.lora_present else None
    hf_extractor = HFHiddenStateExtractor(
        model_name=cfg.model_name,
        layers=[cfg.layer_idx],
        dtype=dtype,
        lora_path=lora_path,
    )
    tokenizer = hf_extractor.tokenizer
    print(
        f"[setup] HF extractor ready "
        f"(model={cfg.model_name}, layer={cfg.layer_idx}, hidden_size={hf_extractor.hidden_size}, "
        f"device={hf_extractor.device})"
    )

    max_prefill_tokens = cfg.max_model_len - 1

    def within_limit(example: Dict[str, Any]) -> bool:
        n = probe_prefill_token_count(
            tokenizer,
            example["prompt"],
            example["completion"],
            chat_template_kwargs=cfg.chat_template_kwargs,
        )
        return n <= max_prefill_tokens

    dataset = dataset.filter(within_limit)
    print(f"[setup] dataset size after length filter = {len(dataset)} examples")

    probe_model_cfg = ProbeModelConfig(
        probe_model_type=cfg.probe.probe_model_type,
        hidden_size=hf_extractor.hidden_size,
        hidden_sizes=list(cfg.probe.hidden_sizes),
        output_size=cfg.probe.output_size,
        covseq=cfg.probe.covseq,
    )
    probe = ValueHeadProbe(
        ProbeConfig(
            layer_idx=cfg.layer_idx,
            model=probe_model_cfg,
            underlying_model=cfg.model_name,
            path=probe_path,
            policy=None,
        )
    )
    probe.model.eval()
    print(
        f"[setup] probe loaded "
        f"(model_type={probe.cfg.model.model_type}, hidden_size={probe.cfg.model.hidden_size})"
    )

    batch_size = batch_size or cfg.train_batch_size
    scored = score_dataset(
        dataset,
        hf_extractor=hf_extractor,
        probe=probe,
        batch_size=batch_size,
        chat_template_kwargs=cfg.chat_template_kwargs,
    )

    roc = compute_roc(scored["labels"], scored["logits"])

    # Sequence-level: one score (max prob across scored tokens) and one label
    # (any labelled token is positive) per example. Closer to the deployment
    # use case — "flag this completion for review" rather than per-token.
    ex_indices = scored["example_indices"]
    unique_examples = np.unique(ex_indices)
    seq_scores = np.zeros(len(unique_examples), dtype=np.float64)
    seq_labels = np.zeros(len(unique_examples), dtype=np.uint8)
    for i, ex in enumerate(unique_examples):
        mask = ex_indices == ex
        seq_scores[i] = float(scored["probs"][mask].max())
        seq_labels[i] = int(scored["labels"][mask].any())
    try:
        seq_roc = compute_roc(seq_labels, seq_scores)
        seq_auc: float | None = float(seq_roc["auc"])
    except ValueError:
        seq_roc = None
        seq_auc = None
        print("[warn] sequence-level AUC undefined — need both violating and clean examples")

    output_prefix = (
        output_prefix.expanduser().resolve()
        if output_prefix is not None
        else Path(cfg.output_dir).resolve() / f"probe_scores_{split}"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        split=np.asarray(split),
        logits=scored["logits"],
        probs=scored["probs"],
        labels=scored["labels"],
        example_indices=scored["example_indices"],
        token_indices=scored["token_indices"],
        window_lengths=scored["window_lengths"],
        roc_fpr=np.asarray(roc["fpr"]),
        roc_tpr=np.asarray(roc["tpr"]),
        roc_thresholds=np.asarray(roc["thresholds"]),
        roc_auc=np.asarray([roc["auc"]], dtype=np.float64),
        seq_scores=seq_scores,
        seq_labels=seq_labels,
        seq_roc_auc=np.asarray(
            [seq_auc if seq_auc is not None else np.nan], dtype=np.float64
        ),
    )

    summary_payload = {
        "split": split,
        "n_scored_tokens": int(len(scored["labels"])),
        "n_positive_tokens": int(scored["labels"].sum()),
        "n_negative_tokens": int(len(scored["labels"]) - scored["labels"].sum()),
        "auc_roc_token": float(roc["auc"]),
        "auc_roc_sequence": seq_auc,
        "n_sequences": int(len(unique_examples)),
        "n_sequences_violating": int(seq_labels.sum()),
        "n_sequences_clean": int(len(seq_labels) - seq_labels.sum()),
        "logit_min": float(np.min(scored["logits"])),
        "logit_max": float(np.max(scored["logits"])),
        "logit_mean": float(np.mean(scored["logits"])),
        "prob_mean": float(np.mean(scored["probs"])),
    }
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    txt_path = output_prefix.with_suffix(".txt")
    write_summary(txt_path, summary_payload)
    png_path = output_prefix.with_suffix(".png")
    title_parts = [f"Probe ROC ({split})", f"token AUC={float(roc['auc']):.4f}"]
    if seq_auc is not None:
        title_parts.append(f"seq AUC={seq_auc:.4f}")
    plot_roc_curve(
        roc,
        png_path,
        title=" — ".join(title_parts),
    )

    print(f"Scored split={split} with {summary_payload['n_scored_tokens']} labelled tokens "
          f"across {summary_payload['n_sequences']} completions "
          f"({summary_payload['n_sequences_violating']} violating / "
          f"{summary_payload['n_sequences_clean']} clean)")
    print(f"Token-level ROC AUC:    {summary_payload['auc_roc_token']:.6f}")
    if seq_auc is not None:
        print(f"Sequence-level ROC AUC: {seq_auc:.6f}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
