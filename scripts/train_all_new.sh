#!/usr/bin/env bash
# Train all 10 new probes sequentially on Mac MPS (HFHiddenStateExtractor).
# Skips a probe if its probe_head.bin already exists.
set -euo pipefail

cd "$(dirname "$0")/.."

POLICIES=(
  religious_truth_claim
  partisan_endorsement
  definitive_diagnosis
  personal_investment_rec
  moralising
  sycophancy
  unsolicited_disclaimer
  fabricated_quote
  structuring
  guaranteed_returns
)

for name in "${POLICIES[@]}"; do
  out_dir="output/qwen_mac_${name}"
  probe_bin="${out_dir}/probe_head.bin"
  config="configs/probe/qwen_mac_${name}.yaml"
  log_file="data/logs/train_${name}.log"

  if [[ -f "$probe_bin" ]]; then
    echo "[skip] $name: $probe_bin already exists"
    continue
  fi

  if [[ ! -f "data/${name}_annotated.jsonl" ]]; then
    echo "[skip] $name: no annotated jsonl — run annotate_all_new.sh first"
    continue
  fi

  echo "[run ] train $name"
  .venv/bin/python train/train.py "$config" > "$log_file" 2>&1
  echo "       → $probe_bin"
done

echo "Training complete."
