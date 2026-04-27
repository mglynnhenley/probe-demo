#!/usr/bin/env python3
"""Download obalcells/longfact-augmented-prompts and write a questions JSONL
ready for annotation_pipeline/local_generate.py.

Usage:
    uv run python process_scripts/download_longfact.py
    uv run python process_scripts/download_longfact.py --output data/longfact_questions.jsonl
    uv run python process_scripts/download_longfact.py --limit 100
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import click
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

_HF_DATASET = "obalcells/longfact-augmented-prompts"
_SOURCE_DATASET = "longfact-augmented-prompts"


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option(
    "--output", "output_path",
    type=click.Path(path_type=Path),
    default=ROOT / "data" / "longfact_questions.jsonl",
    help="Output questions JSONL (input for local_generate.py).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after writing this many records (useful for testing).",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(output_path: Path, limit: int | None, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    from datasets import load_dataset

    print(f"Downloading {_HF_DATASET}…")
    ds = load_dataset(_HF_DATASET, split="train")
    print(f"  {len(ds)} rows")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in tqdm(ds, unit="row"):
            question = row["question"]
            rec = {
                "id": hashlib.sha256(question.encode()).hexdigest()[:16],
                "question": question,
                "source_dataset": _SOURCE_DATASET,
                "policy": "",
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(ds)} records written to {output_path}")


if __name__ == "__main__":
    main()
