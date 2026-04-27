#!/usr/bin/env bash
# Annotate all 10 new policies (pilot + held-out) via OpenRouter.
# Resume-safe: annotate.py skips records already present in the output file.
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
  financial_advice
  tax_advice
)

# fabricated_quote benefits from web search to verify real quotations;
# others use the default gpt-4o-mini annotator.
WEB_SEARCH_POLICIES=(fabricated_quote)

is_web_search() {
  for p in "${WEB_SEARCH_POLICIES[@]}"; do
    [[ "$p" == "$1" ]] && return 0
  done
  return 1
}

for name in "${POLICIES[@]}"; do
  extra=()
  if is_web_search "$name"; then
    extra=(--web-search --model openai/gpt-4o)
  fi

  for suffix in "" "_heldout"; do
    in_file="data/${name}${suffix}_generations.jsonl"
    out_file="data/${name}${suffix}_annotated.jsonl"
    log_file="data/logs/annotate_${name}${suffix}.log"

    if [[ ! -f "$in_file" ]]; then
      echo "[skip] $name${suffix}: no generations file"
      continue
    fi

    echo "[run ] $name${suffix}  ${extra[*]:-}"
    .venv/bin/python annotation_pipeline/annotate.py \
      --input "$in_file" \
      --output "$out_file" \
      ${extra[@]+"${extra[@]}"} \
      > "$log_file" 2>&1
    n=$(wc -l < "$out_file" | tr -d ' ')
    echo "       → $out_file  ($n rows)"
  done
done

echo "Annotation complete."
