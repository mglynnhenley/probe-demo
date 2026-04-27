#!/usr/bin/env python3
"""Generate completions for a questions JSONL using OpenRouter.

Reads a JSONL of records with at least {id, question} fields, generates
completions via OpenRouter, and writes GenerationRecord-format output JSONL.

Resume-safe: records whose id already appears in the output file are skipped.
Writes each record to disk immediately after it completes.

Usage:
    uv run python process_scripts/generate_completions.py \
        --input data/longfact_questions.jsonl \
        --output data/generations.jsonl \
        --model google/gemma-4-31b-it
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from annotation_pipeline.generate import (  # noqa: E402
    _DEFAULT_COMPLETION_MODEL,
    _MAX_CONCURRENT_COMPLETIONS,
    _MAX_RETRIES,
    _RETRY_BACKOFF,
)

log = logging.getLogger(__name__)


async def _generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    record: dict,
    model: str,
    output_path: Path,
    pbar: tqdm,
) -> None:
    from openai import RateLimitError
    attempt = 0
    while True:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": record["question"]}],
                )
            choice = response.choices[0]
            if choice.finish_reason == "stop":
                completed = {**record, "completion": choice.message.content, "model": model}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(completed, ensure_ascii=False) + "\n")
            break
        except RateLimitError:
            if attempt >= _MAX_RETRIES * 3:
                log.error("Rate limit retry limit reached for id=%s, dropping.", record["id"])
                break
            wait = min(2 ** attempt, 60)
            log.warning("Rate limited, retrying in %ss (attempt %d)…", wait, attempt + 1)
            await asyncio.sleep(wait)
            attempt += 1
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF * (2 ** attempt)
                log.warning("Retry %d/%d after %ss: %s", attempt + 1, _MAX_RETRIES, wait, e)
                await asyncio.sleep(wait)
                attempt += 1
            else:
                log.error("Failed after %d retries: %s", _MAX_RETRIES, e)
                break
    pbar.update(1)


async def _run(
    pending: list[dict],
    model: str,
    output_path: Path,
    concurrency: int,
) -> None:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    semaphore = asyncio.Semaphore(concurrency)

    with tqdm(total=len(pending), unit="rec", desc="Generating") as pbar:
        tasks = [
            _generate_one(client, semaphore, r, model, output_path, pbar)
            for r in pending
        ]
        await asyncio.gather(*tasks)



@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input questions JSONL.",
)
@click.option(
    "--output", "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output generations JSONL (appended to if it exists).",
)
@click.option(
    "--model",
    default=_DEFAULT_COMPLETION_MODEL,
    show_default=True,
    help="OpenRouter model slug.",
)
@click.option(
    "--concurrency",
    type=int,
    default=_MAX_CONCURRENT_COMPLETIONS,
    show_default=True,
    help="Number of concurrent requests to OpenRouter.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(input_path: Path, output_path: Path, model: str, concurrency: int, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    records = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d: %s", i, e)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"Resuming: skipping {len(done_ids)} already-generated record(s)")

    pending = [r for r in records if r.get("id") not in done_ids]
    print(f"Records to generate: {len(pending)} / {len(records)}")

    if not pending:
        print("Nothing to do.")
        return

    asyncio.run(_run(pending, model, output_path, concurrency))
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
