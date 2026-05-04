"""Modal deployment for the probe-demo backend.

Deploy the two probe instances from the probe-demo repo root:

    # Hallucination probe
    PROBE_DEMO_APP_NAME=probe-demo-hallucination \
    PROBE_PATH=output/gemma4_31b_hallucination_probe/checkpoint-200/model.safetensors \
    modal deploy backend/modal_backend.py

    # Financial-advice probe
    PROBE_DEMO_APP_NAME=probe-demo-finadvice \
    PROBE_PATH=output/gemma4_31b_probe_gemma4data/checkpoint-196/model.safetensors \
    modal deploy backend/modal_backend.py

Environment variables (read from local env at `modal deploy` time):
  PROBE_DEMO_APP_NAME   Modal app name. Required (default: probe-demo).
  PROBE_PATH            Path relative to probe-demo/ root to the .safetensors file. Required.
  HF_TOKEN              HuggingFace token for gated Gemma model weights. Required.

The probe checkpoint directory (model.safetensors + probe_config.json) is
bundled directly into the image via add_local_dir.

Gemma weights are downloaded from HuggingFace on first cold start and cached
in a named Modal Volume so subsequent restarts skip the download.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Build-time config (resolved when `modal deploy` runs locally)
# ---------------------------------------------------------------------------

APP_NAME = os.environ.get("PROBE_DEMO_APP_NAME", "probe-demo")
PROBE_PATH_REL = os.environ.get(
    "PROBE_PATH",
    "output/gemma4_31b_hallucination_probe/checkpoint-200/model.safetensors",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent  # probe-demo/
_PROBE_ABS = _REPO_ROOT / PROBE_PATH_REL

if not _PROBE_ABS.exists():
    raise FileNotFoundError(
        f"Probe checkpoint not found: {_PROBE_ABS}\n"
        f"Set PROBE_PATH to a path relative to the probe-demo repo root."
    )

_PROBE_DIR_ABS = _PROBE_ABS.parent   # the checkpoint-NNN/ directory
_VLLM_PLUGIN_DIR = _REPO_ROOT.parent / "vllm_probe_plugin"

# ---------------------------------------------------------------------------
# HF token secret
# ---------------------------------------------------------------------------

if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")

_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_HF_SECRET = (
    modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})
    if _HF_TOKEN
    else modal.Secret.from_dict({})
)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

_IGNORE = ["__pycache__", "*.pyc", ".git", ".env", ".ruff_cache", ".venv", "*.egg-info"]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({
        "PYTHONPATH": "/root/repo/backend:/root/repo/train",
        # Baked-in inference settings for google/gemma-4-31B-it on H100.
        "MODEL_NAME": "google/gemma-4-31B-it",
        "PROBE_PATH": "/root/probe_checkpoint/model.safetensors",
        "VLLM_DTYPE": "auto",
        "VLLM_MAX_MODEL_LEN": "8192",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
        "VLLM_TENSOR_PARALLEL_SIZE": "1",
        "VLLM_TRUST_REMOTE_CODE": "1",
    })
    # Install vllm without its dependency resolver so we can override the
    # transformers<5 pin it carries. Gemma 4 31B requires transformers>=5.5.0
    # which vllm==0.18.1 would reject if installed normally.
    .pip_install(
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
        "fastapi[standard]",
        "uvicorn[standard]",
        "uvloop>=0.17.0",
        "python-dotenv>=1.0",
    )
    .run_commands(
        # vllm==0.18.1 brings its own torch==2.10.0, gguf, and other deps.
        "pip install vllm==0.18.1",
        # transformers 5.x requires huggingface_hub>=0.27 (is_offline_mode was
        # removed). Upgrade both together so their APIs stay in sync; --no-deps
        # prevents pip from downgrading anything vllm already installed.
        "pip install 'huggingface_hub>=0.27.0' 'transformers>=5.5.0' 'accelerate>=0.20.0' --upgrade --no-deps",
    )
    .add_local_dir(
        _REPO_ROOT / "backend",
        remote_path="/root/repo/backend",
        copy=True,
        ignore=_IGNORE,
    )
    .add_local_dir(
        _REPO_ROOT / "train",
        remote_path="/root/repo/train",
        copy=True,
        ignore=_IGNORE,
    )
    # Probe checkpoint: model.safetensors + probe_config.json
    .add_local_dir(
        _PROBE_DIR_ABS,
        remote_path="/root/probe_checkpoint",
        copy=True,
        ignore=_IGNORE,
    )
    .add_local_dir(
        _VLLM_PLUGIN_DIR,
        remote_path="/root/vllm_probe_plugin",
        copy=True,
        ignore=_IGNORE,
    )
    .run_commands("pip install -e /root/vllm_probe_plugin --no-deps")
)

# ---------------------------------------------------------------------------
# Modal app + HF weights volume
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME, image=image)

# Each deployment gets its own volume so the two apps don't share weight storage.
weights_volume = modal.Volume.from_name(f"{APP_NAME}-weights", create_if_missing=True)
WEIGHTS_PATH = "/root/hf_cache"

# ---------------------------------------------------------------------------
# ASGI web endpoint — runs main.py's FastAPI app inside the GPU container.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100",
    scaledown_window=300,
    volumes={WEIGHTS_PATH: weights_volume},
    timeout=600,
    secrets=[_HF_SECRET],
    max_containers=1,
    min_containers=0,
)
@modal.asgi_app()
def api() -> "fastapi.FastAPI":
    import os as _os
    # Point HuggingFace at the volume so weights survive cold starts.
    _os.environ["HF_HOME"] = WEIGHTS_PATH
    _os.environ["HF_HUB_CACHE"] = f"{WEIGHTS_PATH}/hub"

    from main import app as fastapi_app  # noqa: PLC0415 (import inside function intentional)
    return fastapi_app
