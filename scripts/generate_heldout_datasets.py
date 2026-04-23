#!/usr/bin/env python
"""Generate held-out test datasets per policy, separately from the pilot training set.

Mirrors `generate_pilot_datasets.py` but writes to `*_heldout_generations.jsonl`
and defaults to a smaller N (40 per policy). Each call to
`_async_generate_dataset` re-runs categorisation + prompt generation via
stochastic LLM calls, so prompts will differ from the training set even though
the underlying policy is identical.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from annotation_pipeline.generate import _async_generate_dataset  # noqa: E402

load_dotenv()

PROMPT_MODEL = "anthropic/claude-sonnet-4-5"
COMPLETION_MODEL = "meta-llama/llama-3.1-8b-instruct"
NUM_GENERATIONS = 40

POLICIES = [
    ("hallucinated_citations", REPO_ROOT / "policies" / "hallucinated_citations.txt"),
    ("tipping_off", REPO_ROOT / "policies" / "tipping_off.txt"),
    ("religious_truth_claim", REPO_ROOT / "policies" / "religious_truth_claim.txt"),
    ("partisan_endorsement", REPO_ROOT / "policies" / "partisan_endorsement.txt"),
    ("definitive_diagnosis", REPO_ROOT / "policies" / "definitive_diagnosis.txt"),
    ("personal_investment_rec", REPO_ROOT / "policies" / "personal_investment_rec.txt"),
    ("moralising", REPO_ROOT / "policies" / "moralising.txt"),
    ("sycophancy", REPO_ROOT / "policies" / "sycophancy.txt"),
    ("unsolicited_disclaimer", REPO_ROOT / "policies" / "unsolicited_disclaimer.txt"),
    ("fabricated_quote", REPO_ROOT / "policies" / "fabricated_quote.txt"),
    ("structuring", REPO_ROOT / "policies" / "structuring.txt"),
    ("guaranteed_returns", REPO_ROOT / "policies" / "guaranteed_returns.txt"),
]


def _policy_sentence(policy_path: Path) -> str:
    return policy_path.read_text(encoding="utf-8").split("\n\n", 1)[0].strip()


async def _run_one(name: str, policy_path: Path) -> Path:
    out = REPO_ROOT / "data" / f"{name}_heldout_generations.jsonl"
    if out.exists() and out.stat().st_size > 0:
        n_rows = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"[{name}] SKIP: {out} already has {n_rows} rows")
        return out
    policy = _policy_sentence(policy_path)
    print(f"[{name}] policy: {policy[:100]}…")
    jsonl = await _async_generate_dataset(
        policy=policy,
        num_generations=NUM_GENERATIONS,
        prompt_model=PROMPT_MODEL,
        completion_model=COMPLETION_MODEL,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jsonl, encoding="utf-8")
    n_rows = sum(1 for line in jsonl.splitlines() if line.strip())
    print(f"[{name}] wrote {n_rows} rows → {out}")
    return out


async def _main() -> None:
    tasks = [_run_one(name, path) for name, path in POLICIES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("Done:")
    for (name, _), r in zip(POLICIES, results):
        if isinstance(r, Exception):
            print(f"  [FAIL] {name}: {type(r).__name__}: {r}")
        else:
            print(f"  [OK]   {r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main())
