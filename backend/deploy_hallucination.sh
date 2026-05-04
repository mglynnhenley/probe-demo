#!/usr/bin/env bash
# Deploy the hallucination probe backend to Modal.
# Run from the probe-demo/ directory:
#   bash backend/deploy_hallucination.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROBE_DEMO_APP_NAME=probe-demo-hallucination \
PROBE_PATH=output/gemma4_31b_hallucination_probe/checkpoint-200/model.safetensors \
modal deploy backend/modal_backend.py
