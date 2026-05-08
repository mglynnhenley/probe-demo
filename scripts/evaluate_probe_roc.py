#!/usr/bin/env python3
"""Score a trained probe on annotated data and plot ROC from saved token logits.

Example:
  uv run python scripts/evaluate_probe_roc.py configs/gemma4_31b_cluster.yaml
  uv run python scripts/evaluate_probe_roc.py configs/gemma4_31b_cluster.yaml --split eval
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
import vllm
import vllm_probe_plugin
from vllm.lora.request import LoRARequest
from tqdm.auto import tqdm

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
for path in (ROOT, TRAIN_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def load_training_config(path: Path) -> Any:
    TrainingConfig = importlib.import_module("config").TrainingConfig
    get_dtype = importlib.import_module("utils").get_dtype

    cfg = TrainingConfig.from_yaml(path)
    if cfg.dtype in (None, "auto"):
        if torch.cuda.is_available():
            cfg.dtype = get_dtype()
        else:
            cfg.dtype = None
    return cfg


def iter_dataset_batches(dataset: Dataset, batch_size: int) -> Iterable[tuple[List[int], List[Dict[str, Any]]]]:
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(len(dataset), start + batch_size)))
        yield indices, [dataset[i] for i in indices]


def get_hidden_states(llm: Any, token_id_lists: List[List[int]], lora_request: LoRARequest | None) -> Dict[int, List[torch.Tensor]]:
    from vllm_probe_plugin import extract_prefill_hidden_states

    sampling_params = vllm.SamplingParams(n=1, temperature=0.0, max_tokens=1)
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": ids} for ids in token_id_lists],
        sampling_params=sampling_params,
        use_tqdm=False,
        lora_request=lora_request,
    )
    hidden_states_dict: Dict[int, List[torch.Tensor]] = {}
    for output in outputs:
        per_req = extract_prefill_hidden_states(output)
        for layer_id, hs in per_req.items():
            hidden_states_dict.setdefault(layer_id, []).append(hs)
    return hidden_states_dict


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
    llm: Any,
    tokenizer: Any,
    probe: Any,
    batch_size: int,
    chat_template_kwargs: Dict[str, Any],
    lora_request: LoRARequest | None,
) -> dict[str, np.ndarray]:
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
        from data import build_probe_feature_batches, collate_fn, prepare_probe_batch

        collated = collate_fn(rows)
        annotations = collated["annotations_val"].float()
        prepared_batch = prepare_probe_batch(
            tokenizer,
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
        progress.set_description(
            f"Scoring dataset ({batch_num}/{n_batches})"
        )
        try:
            hidden_states_dict = get_hidden_states(llm, prepared_batch.token_id_lists, lora_request)
        except Exception as exc:
            print(
                f"[score] batch {batch_num}/{n_batches}: failure during vLLM hidden-state generation "
                f"(n_prompts={len(prompt_token_lengths)}, "
                f"min_tokens={min(prompt_token_lengths)}, "
                f"max_tokens={max(prompt_token_lengths)})"
            )
            raise RuntimeError("Failed during vLLM hidden-state generation") from exc
        if probe.cfg.model.model_type == "multi_layer_covseq":
            hidden_states_list = [
                hidden_states_dict[li] for li in probe.cfg.model.layer_indices
            ]
        else:
            hidden_states_list = hidden_states_dict[probe.cfg.model.layer_idx]
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
            try:
                with torch.no_grad():
                    probe_dtype = next(probe.model.parameters()).dtype
                    features = feature_batch.features.to(dtype=probe_dtype)
                    logits = probe.model(features).squeeze(-1).detach().to("cpu", dtype=torch.float32)
            except Exception as exc:
                print(
                    f"[score] batch {batch_num}/{n_batches}: failure during probe scoring "
                    f"for feature_shape={tuple(feature_batch.features.shape)}"
                )
                raise RuntimeError("Failed during probe scoring") from exc
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
    lines = [
        f"n_scored_tokens={payload['n_scored_tokens']}",
        f"n_positive_tokens={payload['n_positive_tokens']}",
        f"n_negative_tokens={payload['n_negative_tokens']}",
        f"auc_roc={payload['auc_roc']:.6f}",
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
    help="Which dataset split to score.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Examples per vLLM call (default: config train_batch_size).",
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
    batch_size: int | None,
    output_prefix: Path | None,
) -> None:
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
    from vllm_probe_plugin import configure_llm
    from vllm_probe_plugin.utils import hidden_size_from_hf_config

    cfg = load_training_config(config_path.expanduser().resolve())
    print(f"[setup] loaded config from {config_path.expanduser().resolve()}")
    probe_path = (
        probe_path.expanduser().resolve()
        if probe_path is not None
        else Path(cfg.output_dir).resolve() / "probe_head.bin"
    )
    if not probe_path.is_file():
        raise SystemExit(f"Probe checkpoint not found: {probe_path}")

    dataset: Dataset
    if split == "all":
        dataset = build_annotations_dataset(Path(cfg.annotations_jsonl))
    else:
        dataset_dict = build_annotations_dataset_dict(
            path=Path(cfg.annotations_jsonl),
            test_size=cfg.val_fraction,
            seed=cfg.seed,
        )
        dataset = dataset_dict[split]
    print(f"[setup] loaded dataset split={split} with {len(dataset)} examples before truncation")

    vllm_probe_plugin.register()
    print("[setup] registered vLLM probe plugin")
    tp, dp = resolve_gpu_counts(cfg.tensor_parallel_size)
    major_cc = get_compute_capability() if torch.cuda.is_available() else 0
    if cfg.enable_chunked_prefill is not None:
        chunked_prefill = cfg.enable_chunked_prefill
    else:
        chunked_prefill = major_cc >= 8

    print(
        f"[setup] initializing vLLM "
        f"(model={cfg.model_name}, tp={tp}, dp={dp}, max_model_len={cfg.max_model_len}, "
        f"dtype={cfg.dtype}, chunked_prefill={chunked_prefill})"
    )
    if not cfg.probe.layer_indices:
        raise ValueError(
            "probe.layer_indices must list at least one layer in the YAML config"
        )
    llm = configure_llm(
        model=cfg.model_name,
        layers=list(cfg.probe.layer_indices),
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
    print("[setup] vLLM initialized successfully")
    tokenizer = llm.get_tokenizer()
    hf_config = llm.llm_engine.model_config.hf_config
    effective_mml = effective_probe_max_model_len(llm, cfg.max_model_len)
    print(f"[setup] effective max model len = {effective_mml}")

    def within_limit(example: Dict[str, Any]) -> bool:
        n = probe_prefill_token_count(
            tokenizer,
            example["prompt"],
            example["completion"],
            chat_template_kwargs=cfg.chat_template_kwargs,
        )
        return n <= effective_mml - 1

    dataset = dataset.filter(within_limit)
    print(f"[setup] dataset size after length filter = {len(dataset)} examples")

    if cfg.lora_present:
        lora_request = LoRARequest(
            lora_name="lora_probe",
            lora_int_id=1,
            lora_path=cfg.lora_path,
        )
    else:
        lora_request = None

    probe_model_cfg = ProbeModelConfig(
        probe_model_type=cfg.probe.probe_model_type,
        hidden_size=hidden_size_from_hf_config(hf_config),
        hidden_sizes=list(cfg.probe.hidden_sizes),
        output_size=cfg.probe.output_size,
        covseq=cfg.probe.covseq,
        layer_indices=list(cfg.probe.layer_indices),
    )
    probe = ValueHeadProbe(
        ProbeConfig(
            model=probe_model_cfg,
            underlying_model=cfg.model_name,
            path=probe_path,
            policy=None,
            dtype=cfg.dtype,
        )
    )
    probe.model.eval()
    print(
        f"[setup] probe loaded successfully "
        f"(model_type={probe.cfg.model.model_type}, hidden_size={probe.cfg.model.hidden_size})"
    )

    kv_max_batch = optimal_batch_size(llm, effective_mml)
    requested_batch_size = batch_size or cfg.train_batch_size
    if requested_batch_size > kv_max_batch:
        print(
            f"[batch_size] requested {requested_batch_size} exceeds vLLM KV budget "
            f"({kv_max_batch} seqs at worst-case len={effective_mml}); capping."
        )
        batch_size = kv_max_batch
    else:
        batch_size = requested_batch_size
    scored = score_dataset(
        dataset,
        llm=llm,
        tokenizer=tokenizer,
        probe=probe,
        batch_size=batch_size,
        chat_template_kwargs=cfg.chat_template_kwargs,
        lora_request=lora_request,
    )

    roc = compute_roc(scored["labels"], scored["logits"])
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
    )

    summary_payload = {
        "split": split,
        "n_scored_tokens": int(len(scored["labels"])),
        "n_positive_tokens": int(scored["labels"].sum()),
        "n_negative_tokens": int(len(scored["labels"]) - scored["labels"].sum()),
        "auc_roc": float(roc["auc"]),
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
    plot_roc_curve(
        roc,
        png_path,
        title=f"Probe ROC ({split}) AUC={float(roc['auc']):.4f}",
    )

    print(f"Scored split={split} with {summary_payload['n_scored_tokens']} labelled tokens")
    print(f"ROC AUC: {summary_payload['auc_roc']:.6f}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
