"""YAML-driven training configuration for the policy-violation probe (vLLM + probe head)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from models import ProbeModelConfig


@dataclass
class TrainingConfig:
    """Training run settings. Loaded from YAML; see ``configs/default_config.yaml``."""

    # ── Model / vLLM ─────────────────────────────────────────────────────
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # None or "auto": after load, set from compute capability (CUDA) via get_dtype(); else explicit string
    dtype: Optional[str] = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.75
    # None: enable chunked prefill only on Ampere+ (cc>=8); false saves VRAM on V100/Pascal.
    enable_chunked_prefill: Optional[bool] = None
    # Tensor parallel width required for this model; combined with available GPUs → (tp, dp)
    tensor_parallel_size: int = 1
    layer_idx: int = 30  # hidden state layer extracted by vLLM (valid range depends on model depth)
    trust_remote_code: bool = True
    # Passed to tokenizer/processor apply_chat_template (e.g. Gemma 4: enable_thinking: false)
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    lora_present: bool = False
    lora_path: str = "output/lora_adapter/"

    # ── Probe head ────────────────────────────────────────────────────────
    probe: ProbeModelConfig = field(default_factory=ProbeModelConfig)

    # ── Data ─────────────────────────────────────────────────────────────
    annotations_jsonl: str = "data/annotated.jsonl"  # path to JSONL
    val_fraction: float = 0.1
    # BCE pos_weight for rare violation tokens (1) vs common non-violation (0); "auto" = effective n_neg/n_pos given row sampling
    pos_weight: Union[str, float, None] = "auto"
    # If set, caps auto pos_weight (e.g. 500) when violations are extremely rare
    pos_weight_max: Optional[float] = None
    # If True, print weight stats, Σ(w·n), P(batch has zero pos tokens), etc. (see data.debug_print_weighted_sampling_stats)
    debug_sampling: bool = False
    # WeightedRandomSampler: w = w_min + (w_max - w_min) * frac_pos, frac_pos = violation completion tokens / all labelled completion tokens
    sampler_row_weight_min: float = 1.0
    sampler_row_weight_max: float = 10.0

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 3
    # Sequences per vLLM prefill call (capped to KV budget in train.py; default >1 for throughput)
    train_batch_size: int = 8
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
        """Load YAML into :class:`TrainingConfig`. Unknown keys raise."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

        cfg = TrainingConfig()
        valid = {f.name: f for f in fields(TrainingConfig)}
        type_hints = get_type_hints(TrainingConfig)
        for key, value in raw.items():
            if key not in valid:
                raise ValueError(f"Unknown config key: {key!r} (valid: {sorted(valid)})")
            setattr(cfg, key, _coerce_dataclass_field(type_hints.get(key, valid[key].type), value))
        return cfg


def _coerce_dataclass_field(field_type: Any, value: Any) -> Any:
    """Recursively instantiate nested dataclass fields from YAML mappings."""
    dataclass_type = _resolve_dataclass_type(field_type)
    if dataclass_type is None or value is None:
        return value
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected mapping for nested dataclass {dataclass_type.__name__}, "
            f"got {type(value).__name__}"
        )

    kwargs: dict[str, Any] = {}
    valid = {f.name: f for f in fields(dataclass_type)}
    type_hints = get_type_hints(dataclass_type)
    for key, nested_value in value.items():
        if key not in valid:
            raise ValueError(
                f"Unknown config key {key!r} for {dataclass_type.__name__} "
                f"(valid: {sorted(valid)})"
            )
        kwargs[key] = _coerce_dataclass_field(
            type_hints.get(key, valid[key].type),
            nested_value,
        )
    return dataclass_type(**kwargs)


def _resolve_dataclass_type(field_type: Any) -> Any:
    if is_dataclass(field_type):
        return field_type
    origin = get_origin(field_type)
    if origin is Union:
        for candidate in get_args(field_type):
            if candidate is type(None):
                continue
            if is_dataclass(candidate):
                return candidate
    return None
