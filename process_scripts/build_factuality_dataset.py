#!/usr/bin/env python3
"""Build a mixed factuality question set for probe training.

75% SimpleQA (hard factual questions, expected high hallucination rate from
the target model) + 25% TriviaQA rc.nocontext (common-knowledge questions,
expected low hallucination rate). The TriviaQA slice gives the probe clean
fully-correct sequences as negatives.

Output is a JSONL of {question, answer} pairs.

Usage:
    uv run python process_scripts/build_factuality_dataset.py
    uv run python process_scripts/build_factuality_dataset.py --easy-frac 0.25 --seed 42
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import click
from datasets import load_dataset


def _qid(source: str, question: str) -> str:
    return hashlib.sha256(f"{source}::{question}".encode()).hexdigest()[:16]


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("data/generations.jsonl"),
    help="JSONL file to write {question, answer} rows into.",
)
@click.option(
    "--easy-frac",
    type=float,
    default=0.25,
    help="Fraction of the final dataset that should come from the easy (TriviaQA) source.",
)
@click.option(
    "--simpleqa-limit",
    type=int,
    default=None,
    help="Cap SimpleQA rows (useful for smoke tests). Default: all rows.",
)
@click.option("--seed", default=42, type=int)
def main(
    output_path: Path,
    easy_frac: float,
    simpleqa_limit: int | None,
    seed: int,
) -> None:
    if not 0.0 < easy_frac < 1.0:
        raise click.BadParameter("--easy-frac must be in (0, 1)")

    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading basicv8vc/SimpleQA...")
    simpleqa = load_dataset("basicv8vc/SimpleQA", split="test")
    if simpleqa_limit is not None:
        simpleqa = simpleqa.shuffle(seed=seed).select(range(min(simpleqa_limit, len(simpleqa))))

    rows: list[dict] = [
        {
            "id": _qid("simpleqa", row["problem"]),
            "question": row["problem"],
            "answer": row["answer"],
            "source_dataset": "simpleqa",
        }
        for row in simpleqa
    ]
    n_hard = len(rows)

    # n_easy / (n_hard + n_easy) = easy_frac  →  n_easy = n_hard * easy_frac / (1 - easy_frac)
    n_easy_target = round(n_hard * easy_frac / (1.0 - easy_frac))
    print(f"SimpleQA: {n_hard} rows. TriviaQA target: {n_easy_target} rows ({easy_frac:.0%}).")

    print("Loading mandarjoshi/trivia_qa (rc.nocontext)...")
    trivia = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    n_easy_target = min(n_easy_target, len(trivia))
    trivia = trivia.shuffle(seed=seed).select(range(n_easy_target))

    rows.extend(
        {
            "id": _qid("triviaqa", row["question"]),
            "question": row["question"],
            "answer": row["answer"]["value"],
            "source_dataset": "triviaqa",
        }
        for row in trivia
    )

    rng.shuffle(rows)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
