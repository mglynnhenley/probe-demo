#!/usr/bin/env bash
# Evaluate all 10 new probes on train+val and held-out splits.
# Skips a run if its npz already exists.
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
  config="configs/probe/qwen_mac_${name}.yaml"
  out_dir="output/qwen_mac_${name}"
  probe_bin="${out_dir}/probe_head.bin"

  if [[ ! -f "$probe_bin" ]]; then
    echo "[skip] $name: probe not trained yet"
    continue
  fi

  # train+val split
  tv_npz="${out_dir}/probe_scores_all.npz"
  if [[ -f "$tv_npz" ]]; then
    echo "[skip] $name train+val: $tv_npz exists"
  else
    echo "[run ] eval $name on train+val"
    .venv/bin/python scripts/evaluate_probe_roc.py "$config" \
      > "data/logs/eval_${name}_tv.log" 2>&1
  fi

  # held-out split
  ho_file="data/${name}_heldout_annotated.jsonl"
  ho_npz="${out_dir}/probe_scores_${name}_heldout_annotated.npz"
  if [[ ! -f "$ho_file" ]]; then
    echo "[skip] $name held-out: no heldout annotated file"
    continue
  fi
  if [[ -f "$ho_npz" ]]; then
    echo "[skip] $name held-out: $ho_npz exists"
  else
    echo "[run ] eval $name on held-out"
    .venv/bin/python scripts/evaluate_probe_roc.py "$config" \
      --annotations-override "$ho_file" \
      > "data/logs/eval_${name}_ho.log" 2>&1
  fi
done

echo "Evaluation complete."
