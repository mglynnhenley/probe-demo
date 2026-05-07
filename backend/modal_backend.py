"""Modal deployment for the probe-demo backend.

Deploy the two Gemma probe instances from the probe-demo repo root:

    # Hallucination probe (Gemma 31B)
    PROBE_DEMO_APP_NAME=probe-demo-hallucination \
    PROBE_PATH=output/gemma4_31b_hallucination_probe/checkpoint-200/model.safetensors \
    modal deploy backend/modal_backend.py

    # Financial-advice probe (Gemma 31B)
    PROBE_DEMO_APP_NAME=probe-demo-finadvice \
    PROBE_PATH=output/gemma4_31b_probe_gemma4data/checkpoint-196/model.safetensors \
    modal deploy backend/modal_backend.py

    # External probe served against a different base model (e.g. Llama 3.1 8B):
    PROBE_DEMO_APP_NAME=probe-demo-obalcells-llama-linear \
    PROBE_PATH=output/hf_probe/llama3_1_8b_linear/model.safetensors \
    MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.85 \
    modal deploy backend/modal_backend.py

Environment variables (read from local env at `modal deploy` time):
  PROBE_DEMO_APP_NAME       Modal app name. Default: probe-demo.
  PROBE_PATH                Path relative to probe-demo/ root to the .safetensors file. Required.
  MODEL_NAME                HuggingFace model id served by vLLM. Default: google/gemma-4-31B-it.
  MAX_MODEL_LEN             vLLM max sequence length. Default: 8192.
  GPU_MEMORY_UTILIZATION    vLLM GPU memory fraction. Default: 0.85.
  TENSOR_PARALLEL_SIZE      vLLM TP. Default: 1.
  HF_TOKEN                  HuggingFace token for gated weights (Gemma, Llama). Required.

The probe checkpoint directory (model.safetensors + probe_config.json) is
bundled directly into the image via add_local_dir.

Base-model weights are downloaded from HuggingFace on first cold start and
cached in a named Modal Volume so subsequent restarts skip the download.
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
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-31B-it")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
TENSOR_PARALLEL_SIZE = os.environ.get("TENSOR_PARALLEL_SIZE", "1")

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

_SECRET = modal.Secret.from_dict({
    k: os.environ[k]
    for k in ("HF_TOKEN", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")
    if os.environ.get(k)
})

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
        # Inference settings — defaults are tuned for google/gemma-4-31B-it on H100;
        # override per deployment via env at `modal deploy` time.
        "MODEL_NAME": MODEL_NAME,
        "PROBE_PATH": "/root/probe_checkpoint/model.safetensors",
        "DTYPE": "auto",
        "MAX_MODEL_LEN": MAX_MODEL_LEN,
        "GPU_MEMORY_UTILIZATION": GPU_MEMORY_UTILIZATION,
        "TENSOR_PARALLEL_SIZE": TENSOR_PARALLEL_SIZE,
        "TRUST_REMOTE_CODE": "1",
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
        # vllm==0.19.0 fixes the NoneType.dtype crash in _patch_config when
        # loading Gemma 4 via the Transformers backend.
        "pip install vllm==0.19.0",
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
    secrets=[_SECRET],
    max_containers=1,
    min_containers=0,
)
@modal.asgi_app()
def api():
    import os as _os
    # Point HuggingFace at the volume so weights survive cold starts.
    _os.environ["HF_HOME"] = WEIGHTS_PATH
    _os.environ["HF_HUB_CACHE"] = f"{WEIGHTS_PATH}/hub"

    from main import app as fastapi_app  # noqa: PLC0415 (import inside function intentional)
    return fastapi_app
