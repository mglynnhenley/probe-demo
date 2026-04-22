#!/usr/bin/env python
"""Generate pilot policy-violation datasets for two policies concurrently via OpenRouter.

Reads the policy sentence (first paragraph) from each policy text file, runs
`generate_dataset`'s inner coroutine for both in the same event loop so the
OpenRouter calls interleave, and writes per-policy JSONL files under `data/`.
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
NUM_GENERATIONS = 100

POLICIES = [
    ("hallucinated_citations", REPO_ROOT / "policies" / "hallucinated_citations.txt"),
    ("tipping_off", REPO_ROOT / "policies" / "tipping_off.txt"),
]


def _policy_sentence(policy_path: Path) -> str:
    """First paragraph only — the rest is human-facing annotation guidance."""
    return policy_path.read_text(encoding="utf-8").split("\n\n", 1)[0].strip()


async def _run_one(name: str, policy_path: Path) -> Path:
    policy = _policy_sentence(policy_path)
    print(f"[{name}] policy: {policy[:100]}…")
    jsonl = await _async_generate_dataset(
        policy=policy,
        num_generations=NUM_GENERATIONS,
        prompt_model=PROMPT_MODEL,
        completion_model=COMPLETION_MODEL,
    )
    out = REPO_ROOT / "data" / f"{name}_generations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jsonl, encoding="utf-8")
    n_rows = sum(1 for line in jsonl.splitlines() if line.strip())
    print(f"[{name}] wrote {n_rows} rows → {out}")
    return out


async def _main() -> None:
    results = await asyncio.gather(
        *(_run_one(name, path) for name, path in POLICIES)
    )
    print("Done:")
    for p in results:
        print(f"  {p}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main())
