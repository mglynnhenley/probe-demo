#!/usr/bin/env python3
"""Generate completions for prompts using a locally hosted LLM.

Reads questions and policies from an input JSONL (e.g. generations.jsonl or a
prompts-only JSONL), sends each question to the local backend, and writes
records in the same GenerationRecord format.

Resume-safe: any IDs already present in the output file are skipped.

Usage:
    uv run python annotation_pipeline/local_generate.py
    uv run python annotation_pipeline/local_generate.py --input data/generations.jsonl --output data/local_generations.jsonl
    uv run python annotation_pipeline/local_generate.py --limit 100 --concurrency 8
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
import httpx
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT / "annotation_pipeline"))
from data_models import GenerationRecord  # noqa: E402

log = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    input_path: Path = field(default_factory=lambda: ROOT / "data" / "generations.jsonl")
    output_path: Path = field(default_factory=lambda: ROOT / "data" / "local_generations.jsonl")
    backend_url: str = "http://localhost:8000"
    concurrency: int = 1
    max_tokens: int = 4096
    temperature: float = 0.7
    limit: int | None = None
    max_retries: int = 3
    retry_backoff: float = 2.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_input_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed line {i} in {path}: {e}")
    return records


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def _append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def _generate_one(
    client: AsyncOpenAI,
    *,
    question: str,
    model: str,
    max_tokens: int,
    temperature: float,
    token_bar: tqdm,
    max_retries: int,
    retry_backoff: float,
) -> str | None:
    """Stream a single completion, updating token_bar as tokens arrive.

    Semaphore must be held by the caller before this is invoked.
    """
    for attempt in range(max_retries):
        try:
            chunks: list[str] = []
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                timeout=None,  # generation can take arbitrarily long
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    chunks.append(delta)
                    token_bar.update(1)
                finish = chunk.choices[0].finish_reason if chunk.choices else None
                if finish and finish not in ("stop", "length"):
                    log.warning(f"Unexpected finish_reason={finish!r}")
            return "".join(chunks) or None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_backoff * (2 ** attempt)
                log.warning(f"Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                await asyncio.sleep(wait)
            else:
                log.error(f"Failed after {max_retries} retries: {e}")
                return None


async def _wait_for_backend(backend_url: str, poll_interval: float = 5.0) -> str:
    """Poll /v1/models until the backend is ready, then return the model name.

    Uses direct httpx calls (not the OpenAI SDK) so SDK retry logic cannot
    turn each probe into a multi-minute hang.
    """
    url = f"{backend_url.rstrip('/')}/v1/models"
    bar = tqdm(desc="Waiting for backend", unit=" poll", dynamic_ncols=True)
    try:
        while True:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=5.0)) as probe:
                    r = await probe.get(url)
                    r.raise_for_status()
                    models = r.json().get("data", [])
                    bar.close()
                    return models[0]["id"] if models else "local"
            except Exception:
                bar.update(1)
                await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        bar.close()
        raise


async def _run(cfg: GenerateConfig) -> None:
    model_name = await _wait_for_backend(cfg.backend_url)
    log.info(f"Backend model: {model_name}")

    client = AsyncOpenAI(
        base_url=f"{cfg.backend_url.rstrip('/')}/v1",
        api_key=os.environ.get("PROBE_BACKEND_API_KEY", "not-required"),
        timeout=httpx.Timeout(30.0, connect=10.0),
    )

    all_records = _load_input_records(cfg.input_path)
    existing_ids = _load_existing_ids(cfg.output_path)

    pending = [
        r for r in all_records
        if r.get("question") and r.get("id") not in existing_ids
    ]
    if cfg.limit is not None:
        pending = pending[:cfg.limit]

    log.info(
        f"Input records: {len(all_records)}  |  "
        f"Already generated: {len(existing_ids)}  |  "
        f"To generate: {len(pending)}"
    )

    if not pending:
        log.info("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(cfg.concurrency)
    done = 0
    failed = 0
    active: set[str] = set()

    overall = tqdm(total=len(pending), desc="Completions", unit="req", position=0, dynamic_ncols=True)
    token_bar = tqdm(desc="Tokens", unit=" tok", position=1, leave=False, dynamic_ncols=True)

    def _refresh_postfix() -> None:
        sample = next(iter(active), "")[:40]
        overall.set_postfix_str(
            f"active={len(active)} failed={failed}" + (f"  {sample!r}" if sample else ""),
            refresh=True,
        )

    async def _process(record: dict) -> None:
        nonlocal done, failed
        question = record["question"]
        policy = record.get("policy", "")
        record_id = record["id"]

        async with semaphore:
            active.add(question)
            _refresh_postfix()
            token_bar.set_description_str(f"  {question[:40]!r}")

            completion = await _generate_one(
                client,
                question=question,
                model=model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                token_bar=token_bar,
                max_retries=cfg.max_retries,
                retry_backoff=cfg.retry_backoff,
            )

        active.discard(question)
        _refresh_postfix()

        if completion is None:
            failed += 1
            _refresh_postfix()
            overall.update(1)
            return

        out_record = GenerationRecord(
            id=record_id,
            question=question,
            source_dataset=record.get("source_dataset", "local"),
            model=model_name,
            completion=completion,
            policy=policy,
            answer=record.get("answer"),
            annotations=None,
            annotator_model=None,
        ).model_dump()

        _append_record(cfg.output_path, out_record)
        done += 1
        overall.update(1)

    tasks = [_process(r) for r in pending]
    await asyncio.gather(*tasks)

    overall.close()
    token_bar.close()
    log.info(
        f"Done. Written: {done}  |  Failed/skipped: {failed}  |  "
        f"Output: {cfg.output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULTS = GenerateConfig()


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option(
    "--input", "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=_DEFAULTS.input_path,
    help="Input JSONL with questions and policies.",
)
@click.option(
    "--output", "output_path",
    type=click.Path(path_type=Path),
    default=_DEFAULTS.output_path,
    help="Output JSONL for local completions (appended to if it exists).",
)
@click.option(
    "--backend-url",
    default=lambda: os.environ.get("PROBE_BACKEND_URL", _DEFAULTS.backend_url),
    help="Base URL of the local inference backend.",
)
@click.option(
    "--concurrency",
    type=int,
    default=_DEFAULTS.concurrency,
    help="Number of concurrent requests to the backend.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=_DEFAULTS.max_tokens,
    help="Maximum tokens per completion.",
)
@click.option(
    "--temperature",
    type=float,
    default=_DEFAULTS.temperature,
    help="Sampling temperature.",
)
@click.option(
    "--limit",
    type=int,
    default=_DEFAULTS.limit,
    help="Stop after generating this many new completions.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(
    input_path: Path,
    output_path: Path,
    backend_url: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    limit: int | None,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    cfg = GenerateConfig(
        input_path=input_path,
        output_path=output_path,
        backend_url=backend_url,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        limit=limit,
    )
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
