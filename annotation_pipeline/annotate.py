#!/usr/bin/env python3
"""
Core annotation utilities shared by annotate_policy.py and annotate_hallucinations.py.

Provides:
  - Fuzzy span matching / position assignment
  - JSON parsing for both PolicyViolationSpan and AnnotatedSpan
  - Anthropic real-time and Message Batch backends
  - OpenRouter backend
  - Record I/O helpers

CLI entry points live in:
  annotation_pipeline/annotate_policy.py       — policy-violation mode
  annotation_pipeline/annotate_hallucinations.py — hallucination / fact-check mode
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Union

import httpx
from dotenv import load_dotenv
from pydantic import TypeAdapter
from pydantic_core import from_json
from rapidfuzz import fuzz as _rfuzz
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from annotation_pipeline.data_models import AnnotatedSpan, GenerationRecord, PolicyViolationSpan
from annotation_pipeline.downloader import default_generations_table, fetch_generation_records

load_dotenv()
log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

POLICY_SYSTEM_PROMPT = (_PROMPTS_DIR / "policy_violation.txt").read_text(encoding="utf-8").strip()
HALLUCINATION_SYSTEM_PROMPT = (_PROMPTS_DIR / "entity_annotation.txt").read_text(encoding="utf-8").strip()
HALLUCINATION_SYSTEM_PROMPT_NO_SEARCH = (_PROMPTS_DIR / "entity_annotation_no_search.txt").read_text(encoding="utf-8").strip()

_DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

AnySpan = Union[PolicyViolationSpan, AnnotatedSpan]


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def resolve_model(backend: str, model: Optional[str]) -> str:
    if model and str(model).strip():
        resolved = str(model).strip()
    elif backend == "openrouter":
        resolved = os.environ.get("OPENROUTER_ANNOTATOR_MODEL", "").strip() or _DEFAULT_OPENROUTER_MODEL
    elif backend == "anthropic":
        resolved = os.environ.get("ANTHROPIC_ANNOTATOR_MODEL", "").strip() or _DEFAULT_ANTHROPIC_MODEL
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if backend == "anthropic" and resolved.startswith("anthropic/"):
        resolved = resolved.removeprefix("anthropic/")

    return resolved


# ---------------------------------------------------------------------------
# User message formatting
# ---------------------------------------------------------------------------


def format_policy_user_message(policy: str, question: str, completion: str) -> str:
    return (
        f"Policy:\n<policy>{policy}</policy>\n\n"
        f"User question / instruction:\n<question>{question}</question>\n\n"
        f"Model completion to audit:\n<completion>{completion}</completion>"
    )


def format_hallucination_user_message(question: str, completion: str) -> str:
    return (
        f"Here's the instruction:\n<instruction>{question}</instruction>\n\n"
        f"Here's the completion you'll have to fact-check:\n<completion>{completion}</completion>"
    )


# ---------------------------------------------------------------------------
# Fuzzy span matching
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", text)
    text = re.sub(r"[\'\"``:;()\[\]\-]|[.,](?!(?<=\d[.,])\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _trim_edges(query: str, match: str, use_norm: bool = False) -> str:
    if not match or match == query:
        return match

    def _score(q: str, m: str) -> float:
        if use_norm:
            return _rfuzz.ratio(_norm(q), _norm(m)) / 100.0
        return _rfuzz.ratio(q, m) / 100.0

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
    window: int = 20,
    min_sim: float = 0.85,
    use_norm: bool = False,
) -> Optional[str]:
    if query in text:
        return query

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
            score = _rfuzz.ratio(nq, nc) / 100.0
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


def _assign_positions(
    spans: list[AnySpan], text: str, min_sim: float = 0.8
) -> list[AnySpan]:
    results: list[AnySpan] = []
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


def _should_skip_write_after_assignment(
    rec_id: str,
    parsed_spans: list[AnySpan],
    assigned_spans: list[AnySpan],
) -> bool:
    if parsed_spans and not assigned_spans:
        log.error(
            "All %d parsed span(s) were dropped during position assignment for id=%s; not writing record",
            len(parsed_spans),
            rec_id,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_POLICY_SPAN_ADAPTER = TypeAdapter(list[PolicyViolationSpan])
_HALLUCINATION_SPAN_ADAPTER = TypeAdapter(list[AnnotatedSpan])


def _normalise_llm_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[   -​  　﻿]", " ", text)
    text = re.sub(r"[“”‘’„‟‹›«»]", '"', text)
    text = re.sub(r"[–—―]", "-", text)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    return text



def _fix_json_escapes(text: str) -> str:
    """Double lone backslashes inside JSON string values that aren't valid JSON escape sequences."""
    _VALID_ESCAPES = set('"' + chr(92) + '/bfnrtu')
    result = []
    i = 0
    in_string = False

    while i < len(text):
        ch = text[i]

        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if ch == '"':
            in_string = False
            result.append(ch)
            i += 1
        elif ch == chr(92):
            if i + 1 >= len(text):
                result.append(chr(92) + chr(92))
                i += 1
            else:
                nxt = text[i + 1]
                if nxt in _VALID_ESCAPES:
                    result.append(ch)
                    result.append(nxt)
                    i += 2
                else:
                    # Invalid escape — double the backslash so the char is preserved literally
                    result.append(chr(92) + chr(92))
                    result.append(nxt)
                    i += 2
        else:
            result.append(ch)
            i += 1

    return "".join(result)

def parse_policy_spans(llm_response: str) -> list[PolicyViolationSpan]:
    text = _normalise_llm_text(llm_response)
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in annotator response: {llm_response[:200]!r}")
    parsed = from_json(m.group(0).strip(), allow_partial=True)
    if isinstance(parsed, dict):
        parsed = [parsed]
    items = _POLICY_SPAN_ADAPTER.validate_python(parsed)
    return [s for s in items if s.span.strip()]


def parse_hallucination_spans(llm_response: str) -> list[AnnotatedSpan]:
    text = _normalise_llm_text(llm_response)
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in annotator response: {llm_response[:200]!r}")
    json_text = _fix_json_escapes(m.group(0).strip())
    parsed = from_json(json_text, allow_partial=True)
    if isinstance(parsed, dict):
        parsed = [parsed]
    return _HALLUCINATION_SPAN_ADAPTER.validate_python(parsed)


# ---------------------------------------------------------------------------
# OpenRouter URL helper
# ---------------------------------------------------------------------------


def _openrouter_chat_url() -> str:
    explicit = os.environ.get("OPENROUTER_API_URL", "").strip()
    base = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    default_full = "https://openrouter.ai/api/v1/chat/completions"
    raw = explicit or base or default_full
    raw = raw.rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    if raw.endswith("/v1") or raw.endswith("/api/v1"):
        return f"{raw}/chat/completions"
    return raw


# ---------------------------------------------------------------------------
# Anthropic real-time backend
# ---------------------------------------------------------------------------


async def annotate_anthropic(
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_message: str,
    web_search: bool = False,
    max_searches: int = 5,
    max_retries: int = 5,
) -> list[AnySpan]:
    import anthropic  # lazy import

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
    )
    if web_search:
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.messages.create(**kwargs)
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Claude rate limited/overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except anthropic.APIConnectionError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise

    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    if not text_parts:
        raise ValueError("Anthropic response contained no text block")
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# OpenRouter real-time backend
# ---------------------------------------------------------------------------


async def annotate_openrouter(
    model: str,
    max_tokens: int,
    temperature: float,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_message: str,
    max_retries: int = 5,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    url = _openrouter_chat_url()
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    timeout=120.0,
                )
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            last_err = e
            code = e.response.status_code
            body_snip = (e.response.text or "")[:400]
            if code in (401, 403):
                raise RuntimeError(f"OpenRouter HTTP {code}. Check OPENROUTER_API_KEY. Body: {body_snip!r}") from e
            if code == 404:
                raise RuntimeError(f"OpenRouter HTTP 404 for {url!r}. Body: {body_snip!r}") from e
            if 400 <= code < 500 and code != 429:
                raise RuntimeError(f"OpenRouter HTTP {code}. Body: {body_snip!r}") from e
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                log.warning("Retryable HTTP %s, sleeping %ss", code, wait)
                await asyncio.sleep(wait)
            else:
                raise
        except httpx.TransportError as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                log.warning("Transport error: %s, sleeping %ss", e, wait)
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise last_err  # pragma: no cover

    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Anthropic Message Batch backend
# ---------------------------------------------------------------------------


async def cancel_batch(client, batch_id: str) -> None:
    try:
        await client.messages.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancellation requested.")
    except Exception as e:
        print(f"Warning: could not cancel batch {batch_id}: {e}")


async def run_anthropic_batch(
    to_do: list[GenerationRecord],
    model: str,
    max_tokens: int,
    temperature: float,
    output_path: Path,
    system_prompt: str,
    build_user_message,   # callable: (rec) -> str
    parse_spans,          # callable: (str) -> list[AnySpan]
    web_search: bool = False,
    max_searches: int = 5,
    poll_interval: float = 60.0,
    max_retries: int = 8,
) -> int:
    """Submit all records as a single Anthropic Message Batch, poll until done, write results."""
    import anthropic  # lazy import

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    batch_state_path = output_path.with_suffix(".batch_state.json")

    batch_id: Optional[str] = None
    submitted_ids: list[str] = []

    if batch_state_path.exists():
        try:
            state = json.loads(batch_state_path.read_text())
            batch_id = state.get("batch_id")
            submitted_ids = state.get("submitted_ids", [])
            print(f"Resuming existing batch {batch_id} ({len(submitted_ids)} records submitted)")
        except Exception as e:
            log.warning("Could not read batch state file (%s); will resubmit.", e)

    _id_counts: dict[str, int] = {}
    custom_id_to_rec: dict[str, GenerationRecord] = {}
    for rec in to_do:
        count = _id_counts.get(rec.id, 0)
        _id_counts[rec.id] = count + 1
        custom_id = rec.id if count == 0 else f"{rec.id}-{count}"
        custom_id_to_rec[custom_id] = rec

    if batch_id is None:
        n_dupes = sum(v - 1 for v in _id_counts.values() if v > 1)
        if n_dupes:
            print(f"NOTE: {n_dupes} duplicate question IDs disambiguated with suffix for batch submission.")

        requests = []
        for custom_id, rec in custom_id_to_rec.items():
            params: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": build_user_message(rec)}],
            }
            if web_search:
                params["tools"] = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]
            requests.append({"custom_id": custom_id, "params": params})

        try:
            for attempt in range(max_retries):
                try:
                    batch = await client.messages.batches.create(requests=requests)
                    batch_id = batch.id
                    submitted_ids = list(custom_id_to_rec.keys())
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code in (429, 529) and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(f"Rate limited submitting batch (attempt {attempt + 1}/{max_retries}), retrying in {wait}s…")
                        await asyncio.sleep(wait)
                    else:
                        raise
                except anthropic.APIConnectionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            if batch_id is not None:
                print("\nInterrupted during submission — cancelling batch…")
                await cancel_batch(client, batch_id)
                if batch_state_path.exists():
                    batch_state_path.unlink()
            raise KeyboardInterrupt

        batch_state_path.write_text(json.dumps({"batch_id": batch_id, "submitted_ids": submitted_ids}))
        print(f"Batch submitted: {batch_id} ({len(submitted_ids)} records)")

    print(f"Polling batch {batch_id} every {poll_interval:.0f}s…")
    print("(Ctrl+C will stop polling and exit — the batch keeps running. Re-run to resume.)")
    print("(To cancel the batch entirely, use: --cancel-batch)")

    try:
        while True:
            for attempt in range(max_retries):
                try:
                    batch = await client.messages.batches.retrieve(batch_id)
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code in (429, 529) and attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
                except anthropic.APIConnectionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

            counts = batch.request_counts
            print(
                f"  [{batch.processing_status}] "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}"
            )

            if batch.processing_status == "ended":
                break

            await asyncio.sleep(poll_interval)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            f"\nPolling stopped. Batch {batch_id} is still processing on Anthropic's servers.\n"
            f"Re-run the same command to resume polling.\n"
            f"State saved to: {batch_state_path}\n"
            f"To cancel the batch, run with: --cancel-batch"
        )
        raise KeyboardInterrupt

    n_written = 0
    n_errors = 0

    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            with open(output_path, "a") as out_f:
                for attempt in range(max_retries):
                    try:
                        pending: list[tuple] = []

                        with tqdm(total=len(custom_id_to_rec), desc="Streaming results", unit="rec") as pbar:
                            async for result in await client.messages.batches.results(batch_id):
                                rec = custom_id_to_rec.get(result.custom_id)
                                if rec is None:
                                    log.warning("Unknown custom_id in batch results: %s", result.custom_id)
                                    pbar.update(1)
                                    continue

                                result_type = result.result.type
                                if result_type == "succeeded":
                                    response = result.result.message
                                    text_parts = [b.text for b in response.content if b.type == "text"]
                                    if not text_parts:
                                        # No text block — write record with empty annotations so it won't be retried
                                        log.warning("No text block in batch result for id=%s; writing with empty annotations", rec.id)
                                        rec.annotations = []
                                        rec.annotator_model = model
                                        out_f.write(rec.model_dump_json(exclude_none=False) + "\n")
                                        out_f.flush()
                                        n_written += 1
                                    else:
                                        try:
                                            spans = parse_spans("\n".join(text_parts))
                                            fut = loop.run_in_executor(executor, _assign_positions, spans, rec.completion)
                                            pending.append((fut, rec, spans))
                                        except Exception as e:
                                            log.error("Parse failed for id=%s: %s", rec.id, e)
                                            n_errors += 1
                                elif result_type == "errored":
                                    log.error("Batch request errored for id=%s: %s", rec.id, result.result.error)
                                    n_errors += 1
                                elif result_type == "expired":
                                    log.error("Batch request expired for id=%s", rec.id)
                                    n_errors += 1
                                pbar.update(1)

                        with tqdm(total=len(pending), desc="Assigning positions", unit="rec") as pbar:
                            for fut, rec, parsed_spans in pending:
                                try:
                                    assigned_spans = await fut
                                    if _should_skip_write_after_assignment(rec.id, parsed_spans, assigned_spans):
                                        n_errors += 1
                                        pbar.update(1)
                                        continue
                                    rec.annotations = assigned_spans
                                    rec.annotator_model = model
                                    out_f.write(rec.model_dump_json(exclude_none=False) + "\n")
                                    out_f.flush()
                                    n_written += 1
                                except Exception as e:
                                    log.error("Position assignment failed for id=%s: %s", rec.id, e)
                                    n_errors += 1
                                pbar.update(1)
                                pbar.set_postfix(written=n_written, errors=n_errors)

                        break
                    except anthropic.APIStatusError as e:
                        if e.status_code in (429, 529) and attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            raise
                    except anthropic.APIConnectionError:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            raise
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted while collecting results. Re-run to collect remaining results.")
        raise KeyboardInterrupt

    print(f"Batch complete: {n_written} written, {n_errors} errors/expired.")
    if batch_state_path.exists():
        batch_state_path.unlink()
    return n_written


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_records_from_jsonl(path: Path) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                raw.pop("logprobs", None)
                records.append(GenerationRecord.model_validate(raw))
    return records


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec and rec.get("annotations") is not None:
                    done.add(str(rec["id"]))
            except json.JSONDecodeError:
                pass
    return done


def load_input_records(
    input_path: Optional[Path],
    from_supabase: bool,
    supabase_table: Optional[str],
    supabase_limit: Optional[int],
    num_items: Optional[int],
) -> list[GenerationRecord]:
    if from_supabase:
        table = supabase_table if supabase_table is not None else default_generations_table()
        records = fetch_generation_records(table=table, limit=supabase_limit)
        print(f"Loaded {len(records)} row(s) from Supabase table {table!r}")
    else:
        if input_path is None:
            raise SystemExit("Provide --input PATH.jsonl or use --supabase.")
        records = load_records_from_jsonl(input_path)
    if num_items is not None:
        records = records[:num_items]
    return records


# ---------------------------------------------------------------------------
# Real-time annotation loop (shared by both CLI scripts)
# ---------------------------------------------------------------------------


def run_realtime(
    to_do: list[GenerationRecord],
    output_path: Path,
    backend: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    system_prompt: str,
    build_user_message,  # callable: (rec) -> str
    parse_spans,         # callable: (str) -> list[AnySpan]
    web_search: bool = False,
    max_searches: int = 5,
) -> int:
    async def _run_all() -> int:
        semaphore = asyncio.Semaphore(max_concurrent)
        n_written = 0
        t0 = time.monotonic()
        assign_workers = min(12, max(1, (os.cpu_count() or 4)))
        loop = asyncio.get_running_loop()

        with concurrent.futures.ProcessPoolExecutor(max_workers=assign_workers) as assign_executor:

            async def _process(rec: GenerationRecord) -> Optional[GenerationRecord]:
                try:
                    user_message = build_user_message(rec)
                    if backend == "anthropic":
                        response_text = await annotate_anthropic(
                            model, max_tokens, temperature,
                            semaphore, system_prompt, user_message, web_search, max_searches,
                        )
                    else:
                        response_text = await annotate_openrouter(
                            model, max_tokens, temperature, http_client, semaphore,
                            system_prompt, user_message,
                        )
                    spans = parse_spans(response_text)
                    assigned_spans = await loop.run_in_executor(assign_executor, _assign_positions, spans, rec.completion)
                    if _should_skip_write_after_assignment(rec.id, spans, assigned_spans):
                        return None
                    rec.annotations = assigned_spans
                    rec.annotator_model = model
                    return rec
                except Exception as e:
                    log.error("Annotation failed for id=%s: %s", rec.id, e)
                    return None

            async def _write_results(tasks):
                nonlocal n_written
                with open(output_path, "a") as out_f:
                    with tqdm(total=len(tasks), desc="Annotating", unit="rec") as pbar:
                        for fut in asyncio.as_completed(tasks):
                            result = await fut
                            if result is not None:
                                out_f.write(result.model_dump_json(exclude_none=False) + "\n")
                                out_f.flush()
                                n_written += 1
                            elapsed = time.monotonic() - t0
                            pbar.set_postfix(written=n_written, r_s=f"{n_written/elapsed:.1f}" if elapsed else "?")
                            pbar.update(1)

            if backend == "openrouter":
                async with httpx.AsyncClient() as http_client:
                    await _write_results([asyncio.create_task(_process(r)) for r in to_do])
            else:
                http_client = None
                await _write_results([asyncio.create_task(_process(r)) for r in to_do])

        return n_written

    try:
        return asyncio.run(_run_all())
    except KeyboardInterrupt:
        return 0
