#!/usr/bin/env bash
# Start the backend with every trained probe under output/qwen_mac_*/probe_head.bin
# loaded at once. Meant to pair with backend/examples/demo_all_probes.py.
#
# Usage:
#   scripts/serve_all_probes.sh            # auto-discover all probes
#   scripts/serve_all_probes.sh --dry-run  # print the PROBE_PATH and exit
#   OUTPUT_GLOB="output/custom_*/probe_head.bin" scripts/serve_all_probes.sh
#   PYTHON=/path/to/other/.venv/bin/python scripts/serve_all_probes.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_GLOB="${OUTPUT_GLOB:-output/qwen_mac_*/probe_head.bin}"
PYTHON="${PYTHON:-.venv/bin/python}"

# shellcheck disable=SC2206  # intentional word-splitting for glob expansion
probes=( $OUTPUT_GLOB )
if [[ ! -f "${probes[0]}" ]]; then
  echo "No probes match $OUTPUT_GLOB — run scripts/train_all_new.sh first." >&2
  exit 1
fi

probe_path=$(IFS=,; echo "${probes[*]}")
echo "Loading ${#probes[@]} probes:"
for p in "${probes[@]}"; do
  echo "  - $p"
done
echo
echo "PROBE_PATH=$probe_path"

if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found at $PYTHON" >&2
  echo "Set PYTHON=... to point at an interpreter with transformers/torch installed." >&2
  exit 1
fi

export PROBE_PATH="$probe_path"
exec "$PYTHON" backend/main.py
