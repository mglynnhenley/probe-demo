#!/usr/bin/env python3
"""Annotate completions with policy-violation spans.

Usage:
    export ANTHROPIC_API_KEY=...
    uv run python annotation_pipeline/annotate_policy.py \
        --input data/generations.jsonl \
        --output data/annotated.jsonl \
        --backend anthropic --batch --model claude-sonnet-4-6

    # Real-time via OpenRouter
    export OPENROUTER_API_KEY=...
    uv run python annotation_pipeline/annotate_policy.py \
        --input data/generations.jsonl \
        --output data/annotated.jsonl \
        --model openai/gpt-4o-mini
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from annotation_pipeline.annotate import (  # noqa: E402
    POLICY_SYSTEM_PROMPT,
    format_policy_user_message,
    load_done_ids,
    load_input_records,
    parse_policy_spans,
    resolve_model,
    run_anthropic_batch,
    run_realtime,
)

log = logging.getLogger(__name__)


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option("--input", "input_path", default=None, type=click.Path(exists=True, path_type=Path),
              help="Generations JSONL.")
@click.option("--supabase", is_flag=True, default=False)
@click.option("--table", "supabase_table", default=None, type=str)
@click.option("--supabase-limit", default=None, type=int)
@click.option("--output", "output_path", default="data/annotated.jsonl", type=click.Path(path_type=Path))
@click.option("--backend", default="openrouter", type=click.Choice(["openrouter", "anthropic"]))
@click.option("--model", default=None, help="Annotator model id.")
@click.option("--max-tokens", default=8192)
@click.option("--temperature", default=0.0)
@click.option("--max-concurrent", default=3)
@click.option("--num-items", default=None, type=int)
@click.option("--batch", "use_batch", is_flag=True, default=False,
              help="Use Anthropic Message Batch API (async, lower cost). Only with --backend anthropic.")
@click.option("--batch-poll-interval", default=60.0, type=float)
@click.option("--cancel-batch", "cancel_batch", is_flag=True, default=False)
@click.option("--verbose", "-v", is_flag=True)
def main(
    input_path: Optional[Path],
    supabase: bool,
    supabase_table: Optional[str],
    supabase_limit: Optional[int],
    output_path: Path,
    backend: str,
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    num_items: Optional[int],
    use_batch: bool,
    batch_poll_interval: float,
    cancel_batch: bool,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if cancel_batch:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set.")
        batch_state_path = Path(output_path).with_suffix(".batch_state.json")
        if not batch_state_path.exists():
            raise SystemExit(f"No batch state file found at {batch_state_path}.")
        state = json.loads(batch_state_path.read_text())
        batch_id = state.get("batch_id")
        if not batch_id:
            raise SystemExit("batch_state.json has no batch_id.")
        import anthropic
        async def _do_cancel():
            client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            await cancel_batch(client, batch_id)
            batch_state_path.unlink()
        asyncio.run(_do_cancel())
        return

    if use_batch and backend != "anthropic":
        raise SystemExit("--batch is only supported with --backend anthropic.")
    if backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is not set.")
    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    model = resolve_model(backend, model)
    records = load_input_records(input_path, supabase, supabase_table, supabase_limit, num_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)
    if done_ids:
        log.info("Resuming: %d record(s) already annotated", len(done_ids))

    to_do = [r for r in records if r.id not in done_ids and r.completion]
    print(f"Records to annotate: {len(to_do)}")
    if not to_do:
        print("Nothing to do.")
        return

    def _build_user_message(rec):
        return format_policy_user_message(rec.policy, rec.question, rec.completion)

    if use_batch:
        try:
            n = asyncio.run(run_anthropic_batch(
                to_do=to_do,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                output_path=output_path,
                system_prompt=POLICY_SYSTEM_PROMPT,
                build_user_message=_build_user_message,
                parse_spans=parse_policy_spans,
                poll_interval=batch_poll_interval,
            ))
        except KeyboardInterrupt:
            return
    else:
        n = run_realtime(
            to_do=to_do,
            output_path=output_path,
            backend=backend,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
            system_prompt=POLICY_SYSTEM_PROMPT,
            build_user_message=_build_user_message,
            parse_spans=parse_policy_spans,
        )

    print(f"Done. {n} records annotated, written to {output_path}")


if __name__ == "__main__":
    main()
