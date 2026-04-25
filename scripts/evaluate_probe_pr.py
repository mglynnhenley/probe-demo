#!/usr/bin/env python3
"""Score a trained probe and plot a Precision-Recall curve.

Answers the product question: at a given detection threshold, what fraction
of hallucinations am I catching (recall) and what fraction of my flags are
genuine hallucinations (precision)?

Can load scores from a pre-computed .npz (e.g. from evaluate_probe_roc.py)
to skip re-running inference:

  uv run python scripts/evaluate_probe_pr.py --npz outputs/probe_scores_all.npz

Or run the full pipeline from a config:

  uv run python scripts/evaluate_probe_pr.py configs/gemma4_31b_cluster.yaml
  uv run python scripts/evaluate_probe_pr.py configs/gemma4_31b_cluster.yaml --split eval
"""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys
from typing import Any, Dict

import click
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
for path in (ROOT, TRAIN_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def compute_pr_curve(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray | float]:
    labels = labels.astype(np.uint8)
    scores = scores.astype(np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be 1D arrays of the same length")

    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError("PR curve requires at least one positive label")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    # Deduplicate at tied thresholds (take the last index in each tie group
    # so precision/recall are computed after processing all tied examples).
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(sorted_scores) - 1]

    precision = tp[threshold_indices] / (tp[threshold_indices] + fp[threshold_indices])
    recall = tp[threshold_indices] / n_pos
    thresholds = sorted_scores[threshold_indices]

    # Prepend the (recall=0, precision=1) anchor point (threshold → +∞).
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[np.inf, thresholds]

    # AUC-PR via trapezoidal rule (recall is the x-axis).
    auc_pr = float(np.trapezoid(precision, recall))

    # Baseline: fraction of tokens that are positive (random classifier AUC).
    baseline = float(n_pos / len(labels))

    return {
        "precision": precision.astype(np.float64),
        "recall": recall.astype(np.float64),
        "thresholds": thresholds.astype(np.float64),
        "auc_pr": auc_pr,
        "baseline": baseline,
    }


def plot_pr_curve(pr: dict[str, np.ndarray | float], out_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    precision = np.asarray(pr["precision"])
    recall = np.asarray(pr["recall"])
    auc_pr = float(pr["auc_pr"])
    baseline = float(pr["baseline"])

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(recall, precision, label=f"PR curve (AUC={auc_pr:.4f})", linewidth=1.5)
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1.0, alpha=0.7,
               label=f"Random classifier ({baseline:.3f})")
    ax.set_xlabel("Recall (fraction of hallucinations caught)")
    ax.set_ylabel("Precision (fraction of flags that are genuine hallucinations)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"n_scored_tokens={payload['n_scored_tokens']}",
        f"n_positive_tokens={payload['n_positive_tokens']}",
        f"n_negative_tokens={payload['n_negative_tokens']}",
        f"auc_pr={payload['auc_pr']:.6f}",
        f"baseline={payload['baseline']:.6f}",
        f"logit_min={payload['logit_min']:.6f}",
        f"logit_max={payload['logit_max']:.6f}",
        f"logit_mean={payload['logit_mean']:.6f}",
        f"prob_mean={payload['prob_mean']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    data = np.load(npz_path, allow_pickle=False)
    if "logits" not in data or "labels" not in data:
        raise SystemExit(f"npz file missing 'logits' or 'labels': {npz_path}")
    split = str(data["split"]) if "split" in data else "unknown"
    return data["logits"], data["labels"], split


def _run_full_pipeline(
    config_path: Path,
    split: str,
    batch_size: int | None,
    probe_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference + probe scoring; return (logits, labels)."""
    import vllm_probe_plugin
    from vllm.lora.request import LoRARequest
    from evaluate_probe_roc import (
        load_training_config,
        score_dataset,
    )
    from vllm_probe_plugin import configure_llm
    from vllm_probe_plugin.utils import hidden_size_from_hf_config

    data_module = importlib.import_module("data")
    models_module = importlib.import_module("models")
    utils_module = importlib.import_module("utils")
    build_annotations_dataset = data_module.build_annotations_dataset
    build_annotations_dataset_dict = data_module.build_annotations_dataset_dict
    ProbeConfig = models_module.ProbeConfig
    ProbeModelConfig = models_module.ProbeModelConfig
    ValueHeadProbe = models_module.ValueHeadProbe
    effective_probe_max_model_len = utils_module.effective_probe_max_model_len
    get_compute_capability = utils_module.get_compute_capability
    optimal_batch_size = utils_module.optimal_batch_size
    probe_prefill_token_count = utils_module.probe_prefill_token_count
    resolve_gpu_counts = utils_module.resolve_gpu_counts

    cfg = load_training_config(config_path.expanduser().resolve())
    probe_path = (
        probe_path.expanduser().resolve()
        if probe_path is not None
        else Path(cfg.output_dir).resolve() / "probe_head.bin"
    )
    if not probe_path.is_file():
        raise SystemExit(f"Probe checkpoint not found: {probe_path}")

    if split == "all":
        dataset = build_annotations_dataset(Path(cfg.annotations_jsonl))
    else:
        dataset_dict = build_annotations_dataset_dict(
            path=Path(cfg.annotations_jsonl),
            test_size=cfg.val_fraction,
            seed=cfg.seed,
        )
        dataset = dataset_dict[split]
    print(f"[setup] loaded dataset split={split} with {len(dataset)} examples")

    vllm_probe_plugin.register()
    tp, dp = resolve_gpu_counts(cfg.tensor_parallel_size)
    major_cc = get_compute_capability() if torch.cuda.is_available() else 0
    chunked_prefill = cfg.enable_chunked_prefill if cfg.enable_chunked_prefill is not None else major_cc >= 8

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
        cudagraph_capture_sizes=[8, 16, 32, 64, 128],
    )
    tokenizer = llm.get_tokenizer()
    hf_config = llm.llm_engine.model_config.hf_config
    effective_mml = effective_probe_max_model_len(llm, cfg.max_model_len)

    def within_limit(example: Dict[str, Any]) -> bool:
        n = probe_prefill_token_count(
            tokenizer, example["prompt"], example["completion"],
            chat_template_kwargs=cfg.chat_template_kwargs,
        )
        return n <= effective_mml - 1

    dataset = dataset.filter(within_limit)

    lora_request = (
        LoRARequest(lora_name="lora_probe", lora_int_id=1, lora_path=cfg.lora_path)
        if cfg.lora_present else None
    )
    probe_model_cfg = ProbeModelConfig(
        probe_model_type=cfg.probe.probe_model_type,
        hidden_size=hidden_size_from_hf_config(hf_config),
        hidden_sizes=list(cfg.probe.hidden_sizes),
        output_size=cfg.probe.output_size,
        covseq=cfg.probe.covseq,
    )
    probe = ValueHeadProbe(ProbeConfig(
        layer_idx=cfg.layer_idx,
        model=probe_model_cfg,
        underlying_model=cfg.model_name,
        path=probe_path,
        policy=None,
    ))
    probe.model.eval()

    kv_max_batch = optimal_batch_size(llm, effective_mml)
    requested = batch_size or cfg.train_batch_size
    actual_batch_size = min(requested, kv_max_batch)

    scored = score_dataset(
        dataset, llm=llm, tokenizer=tokenizer, probe=probe,
        batch_size=actual_batch_size,
        chat_template_kwargs=cfg.chat_template_kwargs,
        lora_request=lora_request,
    )
    return scored["logits"], scored["labels"]


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("config_path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--npz",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Load pre-computed scores from .npz (skips re-running inference).",
)
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
    help="Dataset split to score (only used when running from config).",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Examples per vLLM call (only used when running from config).",
)
@click.option(
    "--output-prefix",
    type=click.Path(path_type=Path),
    default=None,
    help="Output prefix for .png/.json/.txt files.",
)
def main(
    config_path: Path | None,
    npz: Path | None,
    probe_path: Path | None,
    split: str,
    batch_size: int | None,
    output_prefix: Path | None,
) -> None:
    if npz is not None:
        logits, labels, split = _load_npz(npz)
        print(f"[setup] loaded {len(labels)} scored tokens from {npz} (split={split})")
        if output_prefix is None:
            output_prefix = npz.parent / f"probe_pr_{split}"
    elif config_path is not None:
        logits, labels = _run_full_pipeline(config_path, split, batch_size, probe_path)
        if output_prefix is None:
            cfg_module = importlib.import_module("config")
            cfg = cfg_module.TrainingConfig.from_yaml(config_path.expanduser().resolve())
            output_prefix = Path(cfg.output_dir).resolve() / f"probe_pr_{split}"
    else:
        raise click.UsageError("Provide either CONFIG_PATH or --npz.")

    pr = compute_pr_curve(labels, logits)
    probs = 1.0 / (1.0 + np.exp(-logits))

    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "split": split,
        "n_scored_tokens": int(len(labels)),
        "n_positive_tokens": int(labels.sum()),
        "n_negative_tokens": int(len(labels) - labels.sum()),
        "auc_pr": float(pr["auc_pr"]),
        "baseline": float(pr["baseline"]),
        "logit_min": float(np.min(logits)),
        "logit_max": float(np.max(logits)),
        "logit_mean": float(np.mean(logits)),
        "prob_mean": float(np.mean(probs)),
    }

    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    txt_path = output_prefix.with_suffix(".txt")
    write_summary(txt_path, summary_payload)
    png_path = output_prefix.with_suffix(".png")
    plot_pr_curve(
        pr,
        png_path,
        title=f"Probe Precision-Recall ({split})  AUC={float(pr['auc_pr']):.4f}",
    )

    print(f"Scored split={split} with {summary_payload['n_scored_tokens']} labelled tokens")
    print(f"Positive prevalence: {summary_payload['baseline']:.3%}")
    print(f"AUC-PR: {summary_payload['auc_pr']:.6f}  (baseline: {summary_payload['baseline']:.6f})")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
