#!/usr/bin/env python3
"""Download a probe from obalcells/hallucination-probes and translate it into
the probe-demo on-disk schema.

The HF repo ships:
  probe_head.bin       — flat state dict {"weight": [1, hidden], "bias": [1]}
  probe_config.json    — {"target_layer_name", "layer_idx", "hidden_size"}
  training_config.json — full training metadata (incl. base model id)

The probe-demo backend expects:
  model.safetensors    — keys "layers.0.weight" / "layers.0.bias" matching MLP
  probe_config.json    — matches ProbeConfig.to_dict() schema

Currently supports the *_linear probe variants only (single nn.Linear head,
no LoRA adapter). LoRA variants are intentionally out of scope.

Usage:
    uv run python scripts/probe_from_hf.py \\
        --probe-id llama3_1_8b_linear \\
        --output-dir output/hf_probe/llama3_1_8b_linear

After translation, deploy via modal_backend.py with:
    PROBE_PATH=output/hf_probe/llama3_1_8b_linear/model.safetensors
    MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file as safetensors_save

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ID = "obalcells/hallucination-probes"

# Map HF probe_id → base model id served by vLLM. Verified from each variant's
# training_config.json on the HF repo.
_BASE_MODEL = {
    "llama3_1_8b_linear":     "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3_3_70b_linear":    "meta-llama/Llama-3.3-70B-Instruct",
    "gemma2_9b_linear":       "google/gemma-2-9b-it",
    "mistral_small_24b_linear": "mistralai/Mistral-Small-24B-Instruct-2501",
    "qwen2_5_7b_linear":      "Qwen/Qwen2.5-7B-Instruct",
}


def _download(probe_id: str, filename: str, token: str | None) -> Path:
    """Fetch one file from the HF probe repo into the local cache."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{probe_id}/{filename}",
        token=token,
    )
    return Path(path)


def _translate_state_dict(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename HF linear-probe keys to match probe-demo's MLP(hidden_sizes=[])."""
    if set(src.keys()) != {"weight", "bias"}:
        raise ValueError(
            f"Unexpected probe_head.bin keys {set(src.keys())}. "
            "This translator only supports the *_linear variants (single nn.Linear "
            "with weight + bias)."
        )
    return {
        "layers.0.weight": src["weight"].contiguous(),
        "layers.0.bias":   src["bias"].contiguous(),
    }


def _build_probe_demo_config(
    hf_cfg: dict,
    base_model: str,
    dtype: torch.dtype,
) -> dict:
    """Synthesise a probe-demo probe_config.json from the HF probe_config.json."""
    layer_idx = int(hf_cfg["layer_idx"])
    hidden_size = int(hf_cfg["hidden_size"])
    return {
        "model": {
            "probe_model_type": "mlp",
            "hidden_size": hidden_size,
            "hidden_sizes": [],
            "output_size": 1,
            "layer_indices": [layer_idx],
        },
        "underlying_model": base_model,
        "path": None,
        "policy": "hallucination",
        "dtype": str(dtype).replace("torch.", "torch."),
    }


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option(
    "--probe-id",
    default="llama3_1_8b_linear",
    help="Probe directory inside obalcells/hallucination-probes. Must end with '_linear'.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory. Default: <repo-root>/output/hf_probe/<probe_id>/",
)
@click.option(
    "--base-model",
    default=None,
    help="Override the base model id (default: looked up from _BASE_MODEL by probe-id).",
)
@click.option(
    "--token",
    default=None,
    help="HuggingFace token (default: $HF_TOKEN).",
)
def main(
    probe_id: str,
    output_dir: Path | None,
    base_model: str | None,
    token: str | None,
) -> None:
    if not probe_id.endswith("_linear"):
        raise click.UsageError(
            f"probe-id={probe_id!r} is not a *_linear variant. "
            "LoRA probes are not supported by this translator."
        )

    base_model = base_model or _BASE_MODEL.get(probe_id)
    if base_model is None:
        raise click.UsageError(
            f"No default base model known for probe-id={probe_id!r}. "
            f"Pass --base-model. Known: {sorted(_BASE_MODEL)}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (output_dir or repo_root / "output" / "hf_probe" / probe_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = token or os.environ.get("HF_TOKEN")

    print(f"[probe-from-hf] downloading {REPO_ID}/{probe_id}/probe_head.bin")
    bin_path = _download(probe_id, "probe_head.bin", token)
    print(f"[probe-from-hf] downloading {REPO_ID}/{probe_id}/probe_config.json")
    cfg_path = _download(probe_id, "probe_config.json", token)

    src_state = torch.load(bin_path, map_location="cpu", weights_only=False)
    if not isinstance(src_state, dict):
        raise SystemExit(f"Unexpected probe_head.bin payload type: {type(src_state).__name__}")

    hf_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print(f"[probe-from-hf] HF probe_config: {hf_cfg}")

    translated = _translate_state_dict(src_state)
    weight_dtype = translated["layers.0.weight"].dtype
    out_cfg = _build_probe_demo_config(hf_cfg, base_model, weight_dtype)

    safetensors_path = output_dir / "model.safetensors"
    config_path = output_dir / "probe_config.json"

    safetensors_save(translated, str(safetensors_path))
    config_path.write_text(json.dumps(out_cfg, indent=2) + "\n", encoding="utf-8")

    print(f"[probe-from-hf] wrote {safetensors_path}")
    print(f"[probe-from-hf] wrote {config_path}")
    print()
    print("Deploy via:")
    print(f"  PROBE_DEMO_APP_NAME=probe-demo-{probe_id.replace('_', '-')} \\")
    rel = safetensors_path.relative_to(repo_root)
    print(f"  PROBE_PATH={rel} \\")
    print(f"  MODEL_NAME={base_model} \\")
    print("  modal deploy backend/modal_backend.py")


if __name__ == "__main__":
    main()
