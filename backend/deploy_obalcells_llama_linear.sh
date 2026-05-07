#!/usr/bin/env bash
# Deploy the obalcells/hallucination-probes Llama-3.1-8B linear probe to Modal.
#
# Prerequisites: run scripts/probe_from_hf.py first to download and translate
# the probe into output/hf_probe/llama3_1_8b_linear/.
#
# Run from the probe-demo/ directory:
#   uv run python scripts/probe_from_hf.py --probe-id llama3_1_8b_linear
#   bash backend/deploy_obalcells_llama_linear.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROBE_DEMO_APP_NAME=probe-demo-obalcells-llama-linear \
PROBE_PATH=output/hf_probe/llama3_1_8b_linear/model.safetensors \
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.85 \
TENSOR_PARALLEL_SIZE=1 \
modal deploy backend/modal_backend.py
