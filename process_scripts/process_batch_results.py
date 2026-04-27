#!/usr/bin/env python3
"""Convert downloaded Anthropic Message Batch results into annotated JSONL.

The Anthropic batch API returns a JSONL where each line is:
    {"custom_id": "<record_id>", "result": {"type": "succeeded", "message": {...}}}

This script combines those results with the original generations JSONL (which
holds question, completion, model, source_dataset, policy) to produce the
annotated JSONL format defined in STANDARDS.md.

Resume-safe: records whose id already appears with non-null annotations in the
output file are skipped.

Usage:
    uv run python process_scripts/process_batch_results.py \\
        --batch  data/batches/msgbatch_XXXX_results.jsonl \\
        --source data/generations.jsonl \\
        --output data/annotated.jsonl

    # Process multiple batch files against the same source
    uv run python process_scripts/process_batch_results.py \\
        --batch  data/batches/*.jsonl \\
        --source data/generations.jsonl \\
        --output data/annotated.jsonl
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path

import click
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_pipeline.annotate import _assign_positions, parse_policy_spans as _parse_spans  # noqa: E402
from annotation_pipeline.data_models import GenerationRecord  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_generations(path: Path) -> dict[str, GenerationRecord]:
    """Load a generations JSONL keyed by id."""
    records: dict[str, GenerationRecord] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                raw.pop("logprobs", None)
                rec = GenerationRecord.model_validate(raw)
                records[rec.id] = rec
            except Exception as e:
                log.warning("Skipping malformed line %d in %s: %s", i, path, e)
    return records


def _load_done_ids(path: Path) -> set[str]:
    """Return ids already written to the output file with non-null annotations."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
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


def _iter_batch_results(path: Path):
    """Yield (custom_id, annotator_model, response_text | None) for each batch line."""
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d in %s: %s", i, path, e)
                continue

            custom_id: str = entry.get("custom_id", "")
            result = entry.get("result", {})
            result_type = result.get("type")

            if result_type != "succeeded":
                log.warning("id=%s result type=%r, skipping", custom_id, result_type)
                continue

            message = result.get("message", {})
            stop_reason = message.get("stop_reason")
            annotator_model: str = message.get("model", "unknown")

            if stop_reason == "refusal":
                log.info("id=%s was refused by the model, skipping", custom_id)
                continue

            content_blocks = message.get("content", [])
            text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
            if not text_parts:
                log.warning("id=%s has no text content, skipping", custom_id)
                continue

            yield custom_id, annotator_model, "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def _process_batch_file(
    batch_path: Path,
    generations: dict[str, GenerationRecord],
    done_ids: set[str],
    output_path: Path,
    assign_workers: int,
) -> tuple[int, int, int]:
    """Process a single batch results file. Returns (written, skipped, errors)."""
    written = 0
    skipped = 0
    errors = 0

    # Collect work items first so we can show a progress bar with a total
    work: list[tuple[GenerationRecord, str, str]] = []  # (rec, annotator_model, response_text)
    missing_ids: list[str] = []

    for custom_id, annotator_model, response_text in _iter_batch_results(batch_path):
        if custom_id in done_ids:
            skipped += 1
            continue
        rec = generations.get(custom_id)
        if rec is None:
            log.warning("id=%s not found in generations file, skipping", custom_id)
            missing_ids.append(custom_id)
            errors += 1
            continue
        work.append((rec, annotator_model, response_text))

    if missing_ids:
        log.warning(
            "%d id(s) from batch not found in source generations. "
            "Make sure --source points to the correct generations file.",
            len(missing_ids),
        )

    if not work:
        return written, skipped, errors

    with concurrent.futures.ProcessPoolExecutor(max_workers=assign_workers) as executor:
        # Submit span parsing + position assignment as CPU-bound work
        futures: list[tuple[concurrent.futures.Future, GenerationRecord, str, list]] = []
        for rec, annotator_model, response_text in work:
            try:
                spans = _parse_spans(response_text)
            except Exception as e:
                log.error("Parse failed for id=%s: %s", rec.id, e)
                errors += 1
                continue
            fut = executor.submit(_assign_positions, spans, rec.completion)
            futures.append((fut, rec, annotator_model, spans))

        with open(output_path, "a", encoding="utf-8") as out_f:
            with tqdm(total=len(futures), desc=f"Processing {batch_path.name}", unit="rec") as pbar:
                for fut, rec, annotator_model, parsed_spans in futures:
                    try:
                        assigned_spans = fut.result()
                    except Exception as e:
                        log.error("Position assignment failed for id=%s: %s", rec.id, e)
                        errors += 1
                        pbar.update(1)
                        continue

                    if parsed_spans and not assigned_spans:
                        log.error(
                            "All %d parsed span(s) dropped during position assignment for id=%s; skipping",
                            len(parsed_spans),
                            rec.id,
                        )
                        errors += 1
                        pbar.update(1)
                        continue

                    rec.annotations = assigned_spans
                    rec.annotator_model = annotator_model
                    out_f.write(rec.model_dump_json(exclude_none=False) + "\n")
                    out_f.flush()
                    written += 1
                    pbar.update(1)
                    pbar.set_postfix(written=written, errors=errors)

    return written, skipped, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option(
    "--batch",
    "batch_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Batch results JSONL file(s) downloaded from Anthropic. May be specified multiple times.",
)
@click.option(
    "--source",
    "source_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Generations JSONL used when the batch was submitted (provides question, completion, etc.).",
)
@click.option(
    "--output",
    "output_path",
    default="data/annotated.jsonl",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output annotated JSONL file (appended to if it already exists).",
)
@click.option(
    "--workers",
    default=min(12, max(1, (os.cpu_count() or 4))),
    show_default=True,
    type=int,
    help="Number of parallel workers for span position assignment.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
def main(
    batch_paths: tuple[Path, ...],
    source_path: Path,
    output_path: Path,
    workers: int,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"Loading generations from {source_path}…")
    generations = _load_generations(source_path)
    print(f"  Loaded {len(generations)} generation record(s).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} record(s) already annotated in {output_path}")

    total_written = 0
    total_skipped = 0
    total_errors = 0

    for batch_path in batch_paths:
        print(f"\nProcessing {batch_path}…")
        written, skipped, errors = _process_batch_file(
            batch_path=batch_path,
            generations=generations,
            done_ids=done_ids,
            output_path=output_path,
            assign_workers=workers,
        )
        # Update done_ids so subsequent batch files don't re-write the same records
        if written:
            done_ids |= _load_done_ids(output_path)
        total_written += written
        total_skipped += skipped
        total_errors += errors
        print(f"  written={written}  skipped={skipped}  errors={errors}")

    print(
        f"\nDone. Total written={total_written}  skipped={total_skipped}  errors={total_errors}"
        f"\nOutput: {output_path}"
    )


if __name__ == "__main__":
    main()
