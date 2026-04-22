#!/usr/bin/env python3
"""
Annotate completions with policy-violation spans.

Input is either a generations JSONL file or rows from Supabase (table matching
README columns: id, question, completion, model, source_dataset, policy).

Backends:
  - openrouter (default): chat completions via OpenRouter (concurrent requests).
  - anthropic: Claude via Anthropic API; use ``--batch`` for the Message Batch API
    (lower cost, async; same semantics as glm5_hallucination_probes annotate.py).

Writes JSONL with ``annotations`` and ``annotator_model``. Resume-safe: skips any
input row whose ``id`` already has ``annotations`` in the output file (not merely
present with a failed/partial line). Only un-annotated rows are sent to the API.

Usage:
    export OPENROUTER_API_KEY=...
    export OPENROUTER_ANNOTATOR_MODEL=openai/gpt-4o-mini   # or pass --model (required)
    python annotation_pipeline/annotate.py \\
        --input data/generations.jsonl \\
        --output data/annotated.jsonl

    # Anthropic Message Batch (lower cost vs real-time; no concurrent cap)
    export ANTHROPIC_API_KEY=...
    python annotation_pipeline/annotate.py \\
        --input data/generations.jsonl \\
        --output data/annotated.jsonl \\
        --backend anthropic --batch --model claude-sonnet-4-20250514

    # From Supabase (see annotation_pipeline/downloader.py)
    export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...  # or SUPABASE_KEY / anon key
    python annotation_pipeline/annotate.py --supabase --output data/annotated.jsonl --model openai/gpt-4o-mini
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
from typing import Optional

import click
import httpx
from dotenv import load_dotenv
from pydantic import TypeAdapter
from pydantic_core import from_json
from rapidfuzz import fuzz as _rfuzz
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from annotation_pipeline.data_models import GenerationRecord, PolicyViolationSpan
from annotation_pipeline.downloader import default_generations_table, fetch_generation_records

load_dotenv()
log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# Single system prompt for policy-span auditing (must match PolicyViolationSpan in data_models.py).
SYSTEM_PROMPT = (_PROMPTS_DIR / "policy_violation.txt").read_text(encoding="utf-8").strip()

_DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

def _openrouter_chat_url() -> str:
    """URL for POST chat completions.

    Same convention as the OpenAI SDK / hallucination-probes-backend ``OPENROUTER_BASE_URL``:
    ``https://openrouter.ai/api/v1`` is valid; we append ``/chat/completions`` when missing.

    ``OPENROUTER_API_URL`` (full URL) wins over ``OPENROUTER_BASE_URL`` (base only).
    """
    explicit = os.environ.get("OPENROUTER_API_URL", "").strip()
    base = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    default_full = "https://openrouter.ai/api/v1/chat/completions"
    raw = explicit or base or default_full
    raw = raw.rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    # OpenAI-style base_url (what AsyncOpenAI(base_url=...) uses for OpenRouter)
    if raw.endswith("/v1") or raw.endswith("/api/v1"):
        return f"{raw}/chat/completions"
    return raw


def _resolve_model(backend: str, model: Optional[str]) -> str:
    """Resolve backend-appropriate model defaults and normalize simple aliases."""
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


def _format_user_message(policy: str, question: str, completion: str) -> str:
    return (
        f"Policy:\n<policy>{policy}</policy>\n\n"
        f"User question / instruction:\n<question>{question}</question>\n\n"
        f"Model completion to audit:\n<completion>{completion}</completion>"
    )


# ---------------------------------------------------------------------------
# Fuzzy span matching (RapidFuzz only; used heavily — keep hot path in workers)
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    text = re.sub(r"(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", text)
    text = re.sub(r"[\'\"``\u2018\u2019\u201c\u201d\u201e\u201f\u2039\u203a:;()\[\]\-\u2013\u2014]"
                  r"|[.,](?!(?<=\d[.,])\d)", " ", text)
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
    window: int = 10,
    min_sim: float = 0.9,
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


def _should_skip_write_after_assignment(
    rec_id: str,
    parsed_spans: list[PolicyViolationSpan],
    assigned_spans: list[PolicyViolationSpan],
) -> bool:
    """Do not write records where the model found spans but none could be aligned."""
    if parsed_spans and not assigned_spans:
        log.error(
            "All %d parsed span(s) were dropped during position assignment for id=%s; not writing record",
            len(parsed_spans),
            rec_id,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Anthropic batch backend (aligned with glm5_hallucination_probes/annotate.py)
# ---------------------------------------------------------------------------


async def _cancel_batch(client, batch_id: str) -> None:
    """Best-effort batch cancellation — errors are logged, not raised."""
    try:
        await client.messages.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancellation requested.")
    except Exception as e:
        print(f"Warning: could not cancel batch {batch_id}: {e}")


async def _run_anthropic_batch(
    to_do: list[GenerationRecord],
    model: str,
    max_tokens: int,
    temperature: float,
    output_path: Path,
    poll_interval: float = 60.0,
    max_retries: int = 8,
) -> int:
    """Submit all records as a single Anthropic Message Batch, poll until done, write results.

    Saves a sidecar ``<output>.batch_state.json`` so that a killed run can resume
    against the already-submitted batch rather than re-submitting.

    Ctrl+C behaviour matches glm5: during polling, batch keeps running; re-run to resume.
    """
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
            print(f"NOTE: {n_dupes} duplicate question IDs disambiguated with ':N' suffix for batch submission.")

        requests = []
        for custom_id, rec in custom_id_to_rec.items():
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": [{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    "messages": [{
                        "role": "user",
                        "content": _format_user_message(rec.policy, rec.question, rec.completion),
                    }],
                },
            })

        try:
            for attempt in range(max_retries):
                try:
                    batch = await client.messages.batches.create(requests=requests)
                    batch_id = batch.id
                    submitted_ids = list(custom_id_to_rec.keys())
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code in (429, 529) and attempt < max_retries - 1:
                        wait = 2**attempt
                        print(
                            f"Rate limited submitting batch (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait}s…"
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
                except anthropic.APIConnectionError:
                    if attempt < max_retries - 1:
                        wait = 2**attempt
                        print(
                            f"Connection error submitting batch (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait}s…"
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            if batch_id is not None:
                print("\nInterrupted during submission — cancelling batch…")
                await _cancel_batch(client, batch_id)
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
                        wait = 2**attempt
                        await asyncio.sleep(wait)
                    else:
                        raise
                except anthropic.APIConnectionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        raise

            counts = batch.request_counts
            print(
                f"  [{batch.processing_status}] "
                f"processing={counts.processing} "
                f"succeeded={counts.succeeded} "
                f"errored={counts.errored} "
                f"canceled={counts.canceled} "
                f"expired={counts.expired}"
            )

            if batch.processing_status == "ended":
                break

            await asyncio.sleep(poll_interval)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            f"\nPolling stopped. Batch {batch_id} is still processing on Anthropic's servers.\n"
            f"Re-run the same command to resume polling and collect results.\n"
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
                                        log.error("No text block in batch result for id=%s", rec.id)
                                        n_errors += 1
                                    else:
                                        try:
                                            spans = _parse_spans("\n".join(text_parts))
                                            fut = loop.run_in_executor(
                                                executor, _assign_positions, spans, rec.completion
                                            )
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
                                        pbar.set_postfix(written=n_written, errors=n_errors)
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
                            wait = 2**attempt
                            print(
                                f"Rate limited fetching results (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {wait}s…"
                            )
                            await asyncio.sleep(wait)
                        else:
                            raise
                    except anthropic.APIConnectionError:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                        else:
                            raise
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            f"\nInterrupted while collecting results. Batch {batch_id} is complete on Anthropic's servers.\n"
            f"Re-run the same command to collect the remaining results."
        )
        raise KeyboardInterrupt

    print(f"Batch complete: {n_written} written, {n_errors} errors/expired.")

    if batch_state_path.exists():
        batch_state_path.unlink()

    return n_written


async def _annotate_anthropic(
    policy: str,
    question: str,
    completion: str,
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> list[PolicyViolationSpan]:
    """Single-request Anthropic Messages API (non-batch)."""
    import anthropic  # lazy import

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_content = _format_user_message(policy, question, completion)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=[{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": user_content}],
                )
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                wait = 2**attempt
                print(f"Claude rate limited/overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
        except anthropic.APIConnectionError:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(f"Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise

    text_parts = [b.text for b in response.content if b.type == "text"]
    if not text_parts:
        raise ValueError("Anthropic response contained no text block")
    response_text = "\n".join(text_parts)
    return _parse_spans(response_text)


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
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    url = _openrouter_chat_url()
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            last_err = e
            code = e.response.status_code
            body_snip = (e.response.text or "")[:400]
            if code in (401, 403):
                raise RuntimeError(
                    f"OpenRouter HTTP {code} for {url!r}. "
                    "Check OPENROUTER_API_KEY (non-empty, valid). "
                    f"Body (truncated): {body_snip!r}"
                ) from e
            if code == 404:
                try:
                    j = json.loads(e.response.text or "{}")
                    em = (j.get("error") or {}).get("message")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    em = None
                if isinstance(em, str) and "No endpoints found" in em:
                    raise RuntimeError(
                        f"OpenRouter HTTP 404: {em} "
                        "That model id is not available on OpenRouter (retired or wrong id). "
                        "Set --model or OPENROUTER_ANNOTATOR_MODEL to a model from https://openrouter.ai/models"
                    ) from e
                raise RuntimeError(
                    f"OpenRouter HTTP 404 for {url!r}. "
                    "Check OPENROUTER_BASE_URL / OPENROUTER_API_URL if you override them; "
                    "a wrong path or a blocking proxy can 404. "
                    f"Body (truncated): {body_snip!r}"
                ) from e
            if 400 <= code < 500 and code != 429:
                raise RuntimeError(
                    f"OpenRouter HTTP {code} for {url!r}. Body (truncated): {body_snip!r}"
                ) from e
            if code not in (429, 500, 502, 503, 504, 529):
                raise RuntimeError(
                    f"OpenRouter HTTP {code} for {url!r}. Body (truncated): {body_snip!r}"
                ) from e
            if attempt < max_retries - 1:
                wait = min(2**attempt, 30)
                log.warning("Retryable HTTP %s, sleeping %ss", code, wait)
                await asyncio.sleep(wait)
            else:
                raise
        except httpx.TransportError as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = min(2**attempt, 30)
                log.warning("Transport error: %s, sleeping %ss", e, wait)
                await asyncio.sleep(wait)
            else:
                raise
    else:
        raise last_err  # pragma: no cover

    response_text: str = resp.json()["choices"][0]["message"]["content"]
    return _parse_spans(response_text)


def _load_records_from_jsonl(path: Path) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                raw.pop("logprobs", None)
                records.append(GenerationRecord.model_validate(raw))
    return records


def run(
    output_path: Path,
    model: Optional[str] = None,
    input_path: Optional[Path] = None,
    from_supabase: bool = False,
    supabase_table: Optional[str] = None,
    supabase_limit: Optional[int] = None,
    backend: str = "openrouter",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    max_concurrent: int = 3,
    num_items: Optional[int] = None,
    use_batch: bool = False,
    batch_poll_interval: float = 60.0,
    web_search: bool = False,
) -> int:
    """Annotate completions. Returns number of records written."""
    if use_batch and backend != "anthropic":
        raise ValueError("--batch is only supported with --backend anthropic.")
    if web_search and backend != "openrouter":
        raise ValueError("--web-search is only supported with --backend openrouter.")

    model = _resolve_model(backend, model)

    if web_search and not model.endswith(":online"):
        # OpenRouter routes `:online` models through Exa web search; results are injected
        # into the prompt, so the model can verify URLs/DOIs/cases before labelling.
        model = f"{model}:online"
        print(f"Web search enabled — annotator model set to {model!r}")

    if backend == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise SystemExit("OPENROUTER_API_KEY is not set or is empty.")
    elif backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise SystemExit("ANTHROPIC_API_KEY is not set or is empty.")
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if from_supabase:
        table = supabase_table if supabase_table is not None else default_generations_table()
        records = fetch_generation_records(table=table, limit=supabase_limit)
        print(f"Loaded {len(records)} row(s) from Supabase table {table!r}")
    else:
        if input_path is None:
            raise SystemExit("Provide --input PATH.jsonl or use --supabase.")
        records = _load_records_from_jsonl(input_path)

    if num_items is not None:
        records = records[:num_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip rows that already have annotations in the output (same rule as glm5 batch annotate).
    # Lines without ``annotations`` (failed/partial runs) are not treated as done.
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec and rec.get("annotations") is not None:
                        done_ids.add(str(rec["id"]))
                except json.JSONDecodeError:
                    pass
        if done_ids:
            log.info("Resuming: %d record(s) already annotated in %s", len(done_ids), output_path)

    to_do = [r for r in records if r.id not in done_ids and r.completion]
    print(f"Records to annotate: {len(to_do)}")
    if not to_do:
        print("Nothing to do.")
        return 0

    async def _run_all() -> int:
        if use_batch:
            return await _run_anthropic_batch(
                to_do=to_do,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                output_path=output_path,
                poll_interval=batch_poll_interval,
            )

        semaphore = asyncio.Semaphore(max_concurrent)
        n_written = 0
        t0 = time.monotonic()
        assign_workers = min(12, max(1, (os.cpu_count() or 4)))
        loop = asyncio.get_running_loop()

        # CPU-heavy span → index alignment runs in worker processes (same idea as batch retrieval).
        with concurrent.futures.ProcessPoolExecutor(max_workers=assign_workers) as assign_executor:

            async def _process(rec: GenerationRecord) -> Optional[GenerationRecord]:
                try:
                    if backend == "anthropic":
                        spans = await _annotate_anthropic(
                            rec.policy,
                            rec.question,
                            rec.completion,
                            model,
                            max_tokens,
                            temperature,
                            semaphore,
                        )
                    else:
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
                    assigned_spans = await loop.run_in_executor(
                        assign_executor, _assign_positions, spans, rec.completion
                    )
                    if _should_skip_write_after_assignment(rec.id, spans, assigned_spans):
                        return None
                    rec.annotations = assigned_spans
                    rec.annotator_model = model
                    return rec
                except Exception as e:
                    log.error("Annotation failed for id=%s: %s", rec.id, e)
                    return None

            if backend == "openrouter":
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
            else:
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
@click.option(
    "--input",
    "input_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Generations JSONL (README schema). Not used with --supabase.",
)
@click.option(
    "--supabase",
    is_flag=True,
    default=False,
    help="Load generations from Supabase instead of --input (see downloader.py).",
)
@click.option(
    "--table",
    "supabase_table",
    default=None,
    type=str,
    help="Supabase table name (default: env SUPABASE_GENERATIONS_TABLE, else 'generations').",
)
@click.option(
    "--supabase-limit",
    default=None,
    type=int,
    help="Maximum rows to fetch from Supabase (default: no limit; API may still cap).",
)
@click.option("--output", "output_path", default="data/annotated.jsonl", type=click.Path(path_type=Path))
@click.option(
    "--backend",
    default="openrouter",
    type=click.Choice(["openrouter", "anthropic"]),
    help="openrouter: chat completions API. anthropic: Claude Messages API or Message Batch (--batch).",
)
@click.option(
    "--model",
    default=None,
    help="Annotator model id. Defaults to OPENROUTER_ANNOTATOR_MODEL or openai/gpt-4o-mini "
    "for openrouter, and ANTHROPIC_ANNOTATOR_MODEL or claude-sonnet-4-20250514 for anthropic.",
)
@click.option("--max-tokens", default=8192)
@click.option("--temperature", default=0.0)
@click.option("--max-concurrent", default=3)
@click.option("--num-items", default=None, type=int)
@click.option(
    "--batch",
    "use_batch",
    is_flag=True,
    default=False,
    help="Use Anthropic Message Batch API (lower cost, async). Only with --backend anthropic.",
)
@click.option(
    "--batch-poll-interval",
    default=60.0,
    type=float,
    help="Seconds between batch status polls (batch mode only).",
)
@click.option(
    "--cancel-batch",
    is_flag=True,
    default=False,
    help="Cancel the in-progress batch in <output>.batch_state.json, then exit.",
)
@click.option(
    "--web-search",
    is_flag=True,
    default=False,
    help="Route annotator through OpenRouter's web-search plugin (appends ':online' to the model id). "
    "Useful for policies that require verifying real-world facts (e.g. hallucinated citations).",
)
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
    web_search: bool,
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
            raise SystemExit(f"No batch state file found at {batch_state_path}. Nothing to cancel.")
        state = json.loads(batch_state_path.read_text())
        batch_id = state.get("batch_id")
        if not batch_id:
            raise SystemExit("batch_state.json has no batch_id. Nothing to cancel.")

        import anthropic

        async def _do_cancel() -> None:
            client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            await _cancel_batch(client, batch_id)
            batch_state_path.unlink()
            print(f"State file {batch_state_path} removed.")

        asyncio.run(_do_cancel())
        return

    if not supabase and input_path is None:
        raise SystemExit("Provide --input PATH.jsonl or use --supabase.")
    run(
        output_path=output_path,
        model=model,
        input_path=input_path,
        from_supabase=supabase,
        supabase_table=supabase_table,
        supabase_limit=supabase_limit,
        backend=backend,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent=max_concurrent,
        num_items=num_items,
        use_batch=use_batch,
        batch_poll_interval=batch_poll_interval,
        web_search=web_search,
    )


if __name__ == "__main__":
    main()
