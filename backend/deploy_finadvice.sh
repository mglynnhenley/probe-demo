#!/usr/bin/env bash
# Deploy the financial-advice probe backend to Modal.
# Run from the probe-demo/ directory:
#   bash backend/deploy_finadvice.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROBE_DEMO_APP_NAME=probe-demo-finadvice \
PROBE_PATH=output/gemma4_31b_probe_gemma4data/checkpoint-196/model.safetensors \
modal deploy backend/modal_backend.py
