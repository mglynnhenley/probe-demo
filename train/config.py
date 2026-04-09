"""YAML-driven training configuration for the policy-violation probe (vLLM + probe head)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class TrainingConfig:
    """Training run settings. Loaded from YAML; see ``configs/default_config.yaml``."""

    # ── Model / vLLM ─────────────────────────────────────────────────────
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # None or "auto": after load, set from compute capability (CUDA) via get_dtype(); else explicit string
    dtype: Optional[str] = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.75
    # Tensor parallel width required for this model; combined with available GPUs → (tp, dp)
    tensor_parallel_size: int = 1
    layer_idx: int = 30  # hidden state layer extracted by vLLM (valid range depends on model depth)
    trust_remote_code: bool = True
    # Passed to tokenizer/processor apply_chat_template (e.g. Gemma 4: enable_thinking: false)
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    lora_present: bool = False
    lora_path: str = "output/lora_adapter/"

    # ── Probe head (set hidden_size to the base model’s d_model) ──────────
    hidden_size: int = 4096
    probe_hidden_sizes: List[int] = field(default_factory=list)  # [] = linear probe

    # ── Data ─────────────────────────────────────────────────────────────
    annotations_jsonl: str = "data/annotated.jsonl"  # path to JSONL
    val_fraction: float = 0.1
    # BCE pos_weight for rare violation tokens (1) vs common non-violation (0); "auto" = n_neg/n_pos on train
    pos_weight: Union[str, float, None] = "auto"
    # If set, caps auto pos_weight (e.g. 500) when violations are extremely rare
    pos_weight_max: Optional[float] = None

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 3
    train_batch_size: int = 1
    grad_accumulation_steps: int = 1
    probe_lr: float = 1e-3
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    val_interval: int = 500
    log_interval: int = 1
    checkpoint_interval: int = 100

    # ── Output & misc ────────────────────────────────────────────────────
    output_dir: str = "output/probe_run"  # directory for checkpoints / logs
    seed: int = 42

    @staticmethod
    def from_yaml(path: Path | str) -> TrainingConfig:
        """Load a flat YAML mapping into :class:`TrainingConfig`. Unknown keys raise."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

        cfg = TrainingConfig()
        valid = {f.name for f in fields(TrainingConfig)}
        for key, value in raw.items():
            if key not in valid:
                raise ValueError(f"Unknown config key: {key!r} (valid: {sorted(valid)})")
            setattr(cfg, key, value)
        return cfg
