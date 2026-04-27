#!/usr/bin/env python
"""Generate pilot policy-violation datasets for two policies concurrently via OpenRouter.

Reads the policy sentence (first paragraph) from each policy text file, runs
`generate_dataset`'s inner coroutine for both in the same event loop so the
OpenRouter calls interleave, and writes per-policy JSONL files under `data/`.

Pass policy names as positional args to narrow to a subset:
    python scripts/generate_pilot_datasets.py tipping_off financial_advice tax_advice
"""

from __future__ import annotations

import argparse
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
NUM_GENERATIONS = 100

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
    ("financial_advice", REPO_ROOT / "policies" / "financial_advice.txt"),
    ("tax_advice", REPO_ROOT / "policies" / "tax_advice.txt"),
]


def _policy_sentence(policy_path: Path) -> str:
    """First paragraph only — the rest is human-facing annotation guidance."""
    return policy_path.read_text(encoding="utf-8").split("\n\n", 1)[0].strip()


async def _run_one(name: str, policy_path: Path) -> Path:
    out = REPO_ROOT / "data" / f"{name}_generations.jsonl"
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


async def _main(selected: list[tuple[str, Path]]) -> None:
    tasks = [_run_one(name, path) for name, path in selected]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("Done:")
    for (name, _), r in zip(selected, results):
        if isinstance(r, Exception):
            print(f"  [FAIL] {name}: {type(r).__name__}: {r}")
        else:
            print(f"  [OK]   {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "only", nargs="*",
        help="Policy names to generate for. Default: all policies in POLICIES.",
    )
    args = parser.parse_args()

    if args.only:
        known = {name for name, _ in POLICIES}
        unknown = [n for n in args.only if n not in known]
        if unknown:
            parser.error(f"unknown policy name(s): {unknown}; known: {sorted(known)}")
        selected = [p for p in POLICIES if p[0] in set(args.only)]
    else:
        selected = POLICIES

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main(selected))
