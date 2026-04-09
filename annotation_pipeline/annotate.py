#!/usr/bin/env python3
"""
Annotate completions with policy-violation spans (OpenRouter only).

Reads generations JSONL (policy, question, completion, …), calls an annotator
model via OpenRouter, and writes JSONL with `annotations` and
`annotator_model` set. Skips already-annotated IDs (resume-safe).

Usage:
    export OPENROUTER_API_KEY=...
    python annotation_pipeline/annotate.py \\
        --input data/generations.jsonl \\
        --output data/annotated.jsonl \\
        --model anthropic/claude-3.5-sonnet
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import click
import httpx
from dotenv import load_dotenv
from pydantic import TypeAdapter
from pydantic_core import from_json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from annotation_pipeline.data_models import GenerationRecord, PolicyViolationSpan

load_dotenv()
log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (_PROMPTS_DIR / "policy_violation.txt").read_text().strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _format_user_message(policy: str, question: str, completion: str) -> str:
    return (
        f"Policy:\n<policy>{policy}</policy>\n\n"
        f"User question / instruction:\n<question>{question}</question>\n\n"
        f"Model completion to audit:\n<completion>{completion}</completion>"
    )


# ---------------------------------------------------------------------------
# Fuzzy span matching (aligned with glm5 hallucination annotate.py)
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _rfuzz = None  # type: ignore[assignment]
    _HAS_RAPIDFUZZ = False

try:
    from rouge_score import rouge_scorer as _rs_mod
    _ROUGE = _rs_mod.RougeScorer(["rougeL"], use_stemmer=False)
except ImportError:
    _ROUGE = None


def _norm(text: str) -> str:
    text = re.sub(r"(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", text)
    text = re.sub(r"[\'\"``\u2018\u2019\u201c\u201d\u201e\u201f\u2039\u203a:;()\[\]\-\u2013\u2014]"
                  r"|[.,](?!(?<=\d[.,])\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _trim_edges(query: str, match: str, use_norm: bool = False) -> str:
    if not match or match == query or _ROUGE is None:
        return match

    def _score(q: str, m: str) -> float:
        if use_norm:
            return _ROUGE.score(_norm(q), _norm(m))["rougeL"].recall
        return _ROUGE.score(q, m)["rougeL"].recall

    best, best_r = match, _score(query, match)
    for i in range(1, min(len(match) // 2, 20)):
        t = match[i:]
        if _score(query, t) >= best_r:
            best, best_r = t, _score(query, t)
        else:
            break
    match = best
    for i in range(1, min(len(match) // 2, 20)):
        t = match[:-i]
        if _score(query, t) >= best_r:
            best, best_r = t, _score(query, t)
        else:
            break
    return best


def _find_closest(
    query: str,
    text: str,
    window: int = 10,
    min_sim: float = 0.9,
    use_norm: bool = False,
) -> Optional[str]:
    if query in text:
        return query

    if not _HAS_RAPIDFUZZ and _ROUGE is None:
        return None

    qlen = len(query)
    nq = _norm(query) if use_norm else query
    best_s, best_sub = -math.inf, ""

    for start in range(len(text)):
        for length in range(max(1, qlen - window), min(len(text), qlen + window) + 1):
            end = start + length
            if end > len(text):
                break
            cand = text[start:end].strip()
            nc = _norm(cand) if use_norm else cand
            if _HAS_RAPIDFUZZ:
                score = _rfuzz.ratio(nq, nc) / 100.0
            else:
                score = _ROUGE.score(nq, nc)["rougeL"].fmeasure  # type: ignore[union-attr]
            if score > best_s:
                best_s, best_sub = score, cand
            if best_s > 0.95:
                best_sub = _trim_edges(query, best_sub, use_norm)
                assert best_sub in text
                return best_sub

    if best_s >= min_sim:
        best_sub = _trim_edges(query, best_sub, use_norm)
        assert best_sub in text
        return best_sub
    if not use_norm:
        return _find_closest(query, text, window, min_sim, use_norm=True)
    return None


def _match_span(span: str, text: str, cur_idx: int = 0, min_sim: float = 0.8):
    match = _find_closest(span, text[cur_idx:])
    if match is not None:
        idx = text[cur_idx:].index(match)
        return match, cur_idx + idx
    if cur_idx > 0:
        return _match_span(span, text, cur_idx=0, min_sim=min_sim)
    return None, None


_SPAN_ADAPTER = TypeAdapter(list[PolicyViolationSpan])


def _parse_spans(llm_response: str) -> list[PolicyViolationSpan]:
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", llm_response)
    text = re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", text)
    text = re.sub(r"[\u201C\u201D\u2018\u2019\u201E\u201F\u2039\u203A\u00AB\u00BB]", '"', text)
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in annotator response: {llm_response[:200]!r}")
    parsed = from_json(m.group(0).strip(), allow_partial=True)
    if isinstance(parsed, dict):
        parsed = [parsed]
    items = _SPAN_ADAPTER.validate_python(parsed)
    return [s for s in items if s.span.strip()]


def _assign_positions(
    spans: list[PolicyViolationSpan], text: str, min_sim: float = 0.8
) -> list[PolicyViolationSpan]:
    results: list[PolicyViolationSpan] = []
    cur_idx = 0
    used: set[int] = set()

    for span in spans:
        closest, idx = _match_span(span.span, text, cur_idx=cur_idx, min_sim=min_sim)

        if closest is None:
            log.warning("Could not locate span %r in completion; dropping", span.span)
            continue

        if idx is not None and all(p in used for p in range(idx, idx + len(closest))):
            log.warning("Span %r matched at already-used position; skipping", span.span)
            continue

        if closest != span.span:
            log.info("Span %r → %r", span.span, closest)

        span.span = closest
        span.index = idx
        results.append(span)
        used.update(range(idx, idx + len(closest)))
        cur_idx = max(cur_idx, idx + len(closest))

    return results


async def _annotate_openrouter(
    policy: str,
    question: str,
    completion: str,
    model: str,
    max_tokens: int,
    temperature: float,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> list[PolicyViolationSpan]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_message(policy, question, completion)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.post(
                    OPENROUTER_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                        "Content-Type": "application/json",
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
            break
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 2**attempt
                code = getattr(e, "response", None) and getattr(e.response, "status_code", None)
                if code in (429, 502, 503, 529):
                    log.warning("Retryable error (%s), sleeping %ss", code, wait)
                else:
                    log.warning("Request error: %s, sleeping %ss", e, wait)
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise last_err  # pragma: no cover

    response_text: str = resp.json()["choices"][0]["message"]["content"]
    spans = _parse_spans(response_text)
    return _assign_positions(spans, completion)


def run(
    input_path: Path,
    output_path: Path,
    model: str,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    max_concurrent: int = 3,
    num_items: Optional[int] = None,
) -> int:
    """Annotate completions. Returns number of records written."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[GenerationRecord] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                raw.pop("logprobs", None)
                records.append(GenerationRecord.model_validate(raw))
    if num_items is not None:
        records = records[:num_items]

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("id") and rec.get("annotations") is not None:
                        done_ids.add(rec["id"])
                except json.JSONDecodeError:
                    pass
        if done_ids:
            log.info("Resuming: %d records already annotated", len(done_ids))

    to_do = [r for r in records if r.id not in done_ids and r.completion]
    print(f"Records to annotate: {len(to_do)}")
    if not to_do:
        print("Nothing to do.")
        return 0

    async def _run_all() -> int:
        semaphore = asyncio.Semaphore(max_concurrent)
        n_written = 0
        t0 = time.monotonic()

        async def _process(rec: GenerationRecord) -> Optional[GenerationRecord]:
            try:
                spans = await _annotate_openrouter(
                    rec.policy,
                    rec.question,
                    rec.completion,
                    model,
                    max_tokens,
                    temperature,
                    http_client,
                    semaphore,
                )
                rec.annotations = spans
                rec.annotator_model = model
                return rec
            except Exception as e:
                log.error("Annotation failed for id=%s: %s", rec.id, e)
                return None

        async with httpx.AsyncClient() as http_client:
            tasks = [asyncio.create_task(_process(r)) for r in to_do]
            with open(output_path, "a") as out_f:
                with tqdm(total=len(tasks), desc="Annotating", unit="rec") as pbar:
                    for fut in asyncio.as_completed(tasks):
                        result = await fut
                        if result is not None:
                            out_f.write(result.model_dump_json(exclude_none=False) + "\n")
                            out_f.flush()
                            n_written += 1
                        elapsed = time.monotonic() - t0
                        pbar.set_postfix(
                            written=n_written,
                            r_s=f"{n_written/elapsed:.1f}" if elapsed else "?",
                        )
                        pbar.update(1)
        return n_written

    try:
        n = asyncio.run(_run_all())
    except KeyboardInterrupt:
        return 0
    print(f"Done. {n} records annotated, written to {output_path}")
    return n


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", default="data/annotated.jsonl", type=click.Path(path_type=Path))
@click.option(
    "--model",
    default="anthropic/claude-3.5-sonnet",
    help="OpenRouter model id (e.g. anthropic/claude-3.5-sonnet, openai/gpt-4o).",
)
@click.option("--max-tokens", default=8192)
@click.option("--temperature", default=0.0)
@click.option("--max-concurrent", default=3)
@click.option("--num-items", default=None, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(
    input_path: Path,
    output_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    num_items: Optional[int],
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        input_path=input_path,
        output_path=output_path,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent=max_concurrent,
        num_items=num_items,
    )


if __name__ == "__main__":
    main()
