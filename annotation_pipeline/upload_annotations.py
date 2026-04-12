#!/usr/bin/env python3
"""
Upload ``annotated.jsonl`` rows into Supabase ``annotations`` table.

Expects each JSONL line to match the annotation README shape (``id``, ``annotations``,
``annotator_model``, plus generation fields). The DB row needs ``generation_uuid``;
this script resolves it by looking up ``generations`` where the match column
(``id`` by default) equals the annotated row's ``id`` (prompt hash).

Environment (same Supabase setup as ``downloader.py``):

- ``SUPABASE_URL``, one API key (e.g. ``SUPABASE_SERVICE_ROLE_KEY``).
- ``SUPABASE_GENERATIONS_TABLE`` — table that holds generation rows (default: ``generations``).
- ``SUPABASE_GENERATION_MATCH_COLUMN`` — column on ``generations`` equal to annotated JSONL ``id``
  (prompt hash; default: ``id``).
- ``SUPABASE_GENERATION_UUID_COLUMN`` — UUID primary key column on ``generations`` (default: ``uuid``).
- ``SUPABASE_ANNOTATIONS_TABLE`` — insert target (default: ``annotations``).

Usage::

    uv run python annotation_pipeline/upload_annotations.py data/annotated.jsonl
    uv run python annotation_pipeline/upload_annotations.py data/annotated.jsonl --dry-run --limit 5
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import click
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from annotation_pipeline.downloader import (  # noqa: E402
    _raise_if_html_404,
    get_supabase_client,
)


load_dotenv()


def _table_env(name: str, default: str) -> str:
    v = (os.environ.get(name) or default).strip()
    return v or default


def _fetch_generation_uuid(
    client: Any,
    *,
    generations_table: str,
    match_column: str,
    uuid_column: str,
    record_id: str,
) -> Optional[str]:
    try:
        r = (
            client.table(generations_table)
            .select(uuid_column)
            .eq(match_column, record_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        _raise_if_html_404(e, table=generations_table)
        raise
    if not r.data:
        return None
    u = r.data[0].get(uuid_column)
    return str(u) if u is not None else None


def _annotation_exists(
    client: Any,
    *,
    annotations_table: str,
    record_id: str,
) -> bool:
    try:
        r = (
            client.table(annotations_table)
            .select("uuid")
            .eq("id", record_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        _raise_if_html_404(e, table=annotations_table)
        raise
    return bool(r.data)


def run_upload(
    input_path: Path,
    *,
    dry_run: bool,
    skip_existing: bool,
    limit: Optional[int],
) -> int:
    generations_table = _table_env("SUPABASE_GENERATIONS_TABLE", "generations")
    annotations_table = _table_env("SUPABASE_ANNOTATIONS_TABLE", "annotations")
    match_column = _table_env("SUPABASE_GENERATION_MATCH_COLUMN", "id")
    uuid_column = _table_env("SUPABASE_GENERATION_UUID_COLUMN", "uuid")

    lines: list[str] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    if limit is not None:
        lines = lines[:limit]

    if dry_run:
        print(
            f"Would process {len(lines)} line(s) → {annotations_table!r} "
            f"(lookups: {generations_table!r}.{match_column!r} → {uuid_column!r})"
        )
        return 0

    client = get_supabase_client()
    inserted = 0
    skipped_lookup = 0
    skipped_dup = 0
    errors = 0

    for line in tqdm(lines, desc="Uploading", unit="row"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skip bad JSON: {e}", file=sys.stderr)
            errors += 1
            continue

        rid = row.get("id")
        if not rid:
            print("Skip row without id", file=sys.stderr)
            errors += 1
            continue

        if skip_existing and _annotation_exists(client, annotations_table=annotations_table, record_id=rid):
            skipped_dup += 1
            continue

        cu = _fetch_generation_uuid(
            client,
            generations_table=generations_table,
            match_column=match_column,
            uuid_column=uuid_column,
            record_id=rid,
        )
        if not cu:
            tqdm.write(f"No generation row for id={rid!r} in {generations_table!r}.{match_column!r}")
            skipped_lookup += 1
            continue

        annotations = row.get("annotations")
        if annotations is None:
            tqdm.write(f"Skip id={rid!r}: missing annotations field")
            errors += 1
            continue

        model = row.get("annotator_model")
        if not model:
            tqdm.write(f"Skip id={rid!r}: missing annotator_model")
            errors += 1
            continue

        payload = {
            "generation_uuid": cu,
            "id": rid,
            "annotations": annotations,
            "annotator_model": model,
        }

        try:
            client.table(annotations_table).insert(payload).execute()
            inserted += 1
        except Exception as e:
            _raise_if_html_404(e, table=annotations_table)
            tqdm.write(f"Insert failed id={rid!r}: {e}")
            errors += 1

    print(
        f"Done. inserted={inserted} skipped_no_generation={skipped_lookup} "
        f"skipped_existing={skipped_dup} errors={errors}"
    )
    return 0 if errors == 0 else 1


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--dry-run", is_flag=True, help="Only print how many rows would be processed.")
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip rows whose id already exists in the annotations table.",
)
@click.option("--limit", default=None, type=int, help="Process at most this many lines (for testing).")
def main(input_path: Path, dry_run: bool, skip_existing: bool, limit: Optional[int]) -> None:
    raise SystemExit(run_upload(input_path, dry_run=dry_run, skip_existing=skip_existing, limit=limit))


if __name__ == "__main__":
    main()
