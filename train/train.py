#!/usr/bin/env python
"""
Train the policy-violation probe via Hugging Face Trainer + vLLM prefill hidden states.

Labels are **highly imbalanced**: ``1`` only on annotated violation spans (rare), ``0`` on
almost all completion characters. Use ``pos_weight: auto`` (default) so BCE upweights
the positive class; see ``summarize_token_class_balance`` and ``compute_pos_weight`` in
``data.py``.
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
import vllm_probe_plugin
from transformers import TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from vllm.lora.request import LoRARequest

from config import TrainingConfig
from data import (
    build_annotations_dataloader,
    collate_fn,
    compute_pos_weight,
    summarize_token_class_balance,
    truncate_dataset,
)
from models import ProbeConfig, ValueHeadProbe
from trainer import ProbeTrainer
from utils import get_compute_capability, get_dtype, optimal_batch_size, resolve_gpu_counts, save_training_config_json
from vllm_probe_plugin import configure_llm


def load_training_config(path: Path) -> TrainingConfig:
    """Load YAML and resolve ``dtype`` when unset or ``auto`` (compute-capability based on CUDA)."""
    cfg = TrainingConfig.from_yaml(path)
    if cfg.dtype in (None, "auto"):
        if torch.cuda.is_available():
            cfg.dtype = get_dtype()
            print(f"[dtype] {cfg.dtype!r} (from GPU compute capability)")
        else:
            print("[dtype] None — no CUDA; vLLM will use its default resolution")
    else:
        print(f"[dtype] {cfg.dtype!r} (from config)")
    return cfg


class TqdmMetricsCallback(ProgressCallback):
    """Fold logged metrics into the tqdm bar postfix instead of JSON lines."""

    _FIELDS = (
        "loss",
        "eval_loss",
        "grad_norm",
        "f1",
        "prec_viol",
        "tpr",
        "tnr",
        "frac_pred_viol",
        "eval_f1",
        "eval_precision_viol",
        "eval_tpr",
        "eval_tnr",
        "eval_baseline_nonviol_acc",
        "learning_rate",
        "epoch",
    )
    _FORMAT = {
        "learning_rate": lambda v: ("lr", f"{v:.2e}"),
        "epoch": lambda v: ("ep", f"{v:.2f}"),
    }

    def __init__(self) -> None:
        super().__init__()
        self._postfix: dict = {}

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if not state.is_local_process_zero or self.training_bar is None or not logs:
            return
        for key in self._FIELDS:
            if key not in logs:
                continue
            val = logs[key]
            if key in self._FORMAT:
                display_key, display_val = self._FORMAT[key](val)
            elif isinstance(val, float):
                display_key, display_val = key, f"{val:.4f}"
            else:
                display_key, display_val = key, val
            self._postfix[display_key] = display_val
        self.training_bar.set_postfix(self._postfix)


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def main(config_path: Path) -> None:
    cfg = load_training_config(config_path)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    annotations_dataloader = build_annotations_dataloader(
        path=Path(cfg.annotations_jsonl),
        test_size=cfg.val_fraction,
    )
    print(f"Loaded annotations from {cfg.annotations_jsonl}")

    vllm_probe_plugin.register()

    tp, dp = resolve_gpu_counts(cfg.tensor_parallel_size)
    major_cc = get_compute_capability() if torch.cuda.is_available() else 0
    chunked_prefill = True
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"Loading vLLM: tensor_parallel_size={tp}, data_parallel_size={dp} "
        f"(requested TP={cfg.tensor_parallel_size}, GPUs={n_gpus}, CC major={major_cc})"
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
    )

    if cfg.lora_present:
        print(f"LoRA enabled (adapter path): {cfg.lora_path}")
        lora_request = LoRARequest(
            lora_name="lora_probe",
            lora_int_id=1,
            lora_path=cfg.lora_path,
        )
    else:
        print("No LoRA adapter — continuing without LoRA")
        lora_request = None

    hf_config = llm.llm_engine.model_config.hf_config
    probe_cfg = ProbeConfig(
        layer_idx=cfg.layer_idx,
        hidden_size=hf_config.hidden_size,
        hidden_sizes=cfg.probe_hidden_sizes or None,
        output_size=1,
        underlying_model=cfg.model_name,
        policy=None,
    )
    probe = ValueHeadProbe(probe_cfg)

    kv_max_batch = optimal_batch_size(llm, cfg.max_model_len)
    if cfg.train_batch_size > kv_max_batch:
        print(
            f"[batch_size] {cfg.train_batch_size} exceeds KV capacity ({kv_max_batch} seqs at "
            f"max_model_len={cfg.max_model_len}); capping."
        )
        cfg.train_batch_size = kv_max_batch
    print(f"[batch_size] using {cfg.train_batch_size} sequences per vLLM call")

    tokenizer = llm.get_tokenizer()

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.train_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        learning_rate=cfg.probe_lr,
        num_train_epochs=cfg.epochs,
        logging_steps=cfg.log_interval,
        eval_strategy="steps",
        eval_steps=cfg.val_interval,
        per_device_eval_batch_size=cfg.train_batch_size,
        save_steps=cfg.checkpoint_interval,
        no_cuda=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        max_grad_norm=cfg.max_grad_norm,
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    annotations_dataloader = truncate_dataset(annotations_dataloader, tokenizer, cfg.max_model_len)
    train_dataset = annotations_dataloader["train"]
    eval_dataset = annotations_dataloader["test"]

    for name, ds in ("train", train_dataset), ("eval", eval_dataset):
        stats = summarize_token_class_balance(ds)
        n_v = stats["n_violation_tokens"]
        n_n = stats["n_nonviolation_tokens"]
        frac = stats["fraction_violation_tokens"]
        ratio = stats["neg_tokens_per_violation_token"]
        ratio_s = f"{ratio:.1f}" if ratio != float("inf") else "inf"
        print(
            f"[class balance] {name}: violation_tokens={n_v}, nonviolation_tokens={n_n}, "
            f"fraction_violation={frac:.6f}, neg/pos≈{ratio_s}"
        )

    if cfg.pos_weight == "auto":
        raw_pw = compute_pos_weight(train_dataset, max_pos_weight=None)
        if cfg.pos_weight_max is not None:
            pos_weight = min(raw_pw, cfg.pos_weight_max)
            cap_note = (
                f" (capped from raw {raw_pw:.2f})"
                if raw_pw > cfg.pos_weight_max
                else ""
            )
        else:
            pos_weight = raw_pw
            cap_note = ""
        print(
            f"[class balance] BCE pos_weight={pos_weight:.4f} "
            f"(upweights rare violation=1 vs common=0){cap_note}"
        )
    elif cfg.pos_weight is not None:
        pos_weight = float(cfg.pos_weight)
        print(f"[class balance] pos_weight = {pos_weight:.4f} (manual)")
    else:
        pos_weight = None
        print(
            "[class balance] WARNING: pos_weight disabled. With extreme imbalance (few 1s, many 0s), "
            "the loss is dominated by non-violation tokens and the probe often collapses to predicting 0. "
            "Strongly recommended: pos_weight: auto in YAML."
        )

    trainer = ProbeTrainer(
        vllm_llm=llm,
        tokenizer=tokenizer,
        probe=probe,
        pos_weight=pos_weight,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        lora_request=lora_request,
        args=training_args,
        chat_template_kwargs=cfg.chat_template_kwargs,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(TqdmMetricsCallback())

    print(f"Training dataset size: {len(trainer.train_dataset)}")
    print(f"Evaluation dataset size: {len(trainer.eval_dataset)}")

    trainer.train()

    probe.save(Path(cfg.output_dir) / "probe_head.bin")
    save_training_config_json(cfg, Path(cfg.output_dir) / "config.json")
    print(f"Saved probe to {cfg.output_dir}/probe_head.bin and config.json")


if __name__ == "__main__":
    main()
