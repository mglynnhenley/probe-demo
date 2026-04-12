"""
Supabase helpers: load generation rows for the annotation pipeline.

Expects a table whose columns match ``annotation_pipeline/README.md`` (generations format):
``id``, ``question``, ``completion``, ``model``, ``source_dataset``, ``policy``.

Environment:

- ``SUPABASE_URL`` — project URL (required).
- **Exactly one** API key (first non-empty wins, in this order):

  1. ``SUPABASE_KEY`` — optional convenience alias.
  2. ``SUPABASE_SERVICE_ROLE_KEY`` — preferred for trusted scripts (bypasses RLS; full table read).
  3. ``SUPABASE_ANON_KEY``
  4. ``SUPABASE_SECRET_KEY``
  5. ``SUPABASE_PUBLISHABLE_KEY`` — Supabase’s “anon/public” style key; use if RLS allows ``SELECT``.

  You do **not** need ``SUPABASE_KEY`` if any of the others is set.

- ``SUPABASE_GENERATIONS_TABLE`` — PostgREST table name for generation rows (default: ``generations``).

**Important:** ``SUPABASE_URL`` must be the **REST API base** from Dashboard → **Settings → API →
Project URL**, shaped like ``https://xxxxxxxx.supabase.co``. If you use the dashboard URL
(``supabase.com/dashboard/...``) or any wrong host, requests return **HTML 404 pages**; the
client then fails while parsing that as JSON (“Invalid JSON”, “validating JSON”).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Same package root resolution as ``annotate.py`` (probe-demo project root on sys.path).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_GENERATIONS_TABLE = "generations"


def default_generations_table() -> str:
    """Table name for README-shaped generation rows: env ``SUPABASE_GENERATIONS_TABLE`` or ``generations``."""
    name = (os.environ.get("SUPABASE_GENERATIONS_TABLE") or _DEFAULT_GENERATIONS_TABLE).strip()
    return name or _DEFAULT_GENERATIONS_TABLE


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing environment variable {name} (required for Supabase).")
    return v


def _supabase_api_key() -> str:
    for name in (
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
    ):
        v = os.environ.get(name)
        if v:
            return v
    raise RuntimeError(
        "Set one of SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY, "
        "SUPABASE_SECRET_KEY, or SUPABASE_PUBLISHABLE_KEY for Supabase access."
    )


def _normalize_supabase_url(url: str) -> str:
    u = url.strip().rstrip("/")
    if "supabase.com/dashboard" in u or "/dashboard/" in u:
        raise RuntimeError(
            "SUPABASE_URL looks like the Supabase dashboard URL. Use the Project URL instead: "
            "Dashboard → Settings → API → Project URL (https://<ref>.supabase.co)."
        )
    if not u.startswith("https://"):
        raise RuntimeError("SUPABASE_URL should start with https://")
    return u


def get_supabase_client():
    """Return a configured Supabase client."""
    from supabase import create_client

    url = _normalize_supabase_url(_require_env("SUPABASE_URL"))
    key = _supabase_api_key()
    return create_client(url, key)


def _raise_if_html_404(exc: BaseException, *, table: str) -> None:
    """PostgREST returns HTML on wrong base URL; client then fails JSON parse — give a clear hint."""
    text = repr(exc) + str(exc)
    low = text.lower()
    if (
        "404" in text
        or "<!doctype html" in low
        or "json_invalid" in low
        or "json could not be generated" in low
    ):
        raise RuntimeError(
            "Supabase/PostgREST did not return JSON (often HTTP 404 HTML from a wrong API URL).\n"
            "  • Set SUPABASE_URL to Project URL: https://<project-ref>.supabase.co\n"
            "    (Dashboard → Settings → API, not the supabase.com dashboard link).\n"
            f"  • Table requested: {table!r}\n"
            "  • If the URL is correct, check the table exists and your key can SELECT it (RLS)."
        ) from exc
    raise exc


def fetch_generations_raw(
    table: str | None = None,
    *,
    limit: Optional[int] = None,
    columns: str = "*",
) -> list[dict[str, Any]]:
    """Fetch rows from ``table`` as plain dicts (``table`` defaults via :func:`default_generations_table`)."""
    if table is None:
        table = default_generations_table()
    client = get_supabase_client()
    q = client.table(table).select(columns)
    if limit is not None:
        q = q.limit(limit)
    try:
        resp = q.execute()
    except Exception as e:
        _raise_if_html_404(e, table=table)
        raise
    data = resp.data
    if data is None:
        return []
    return list(data)


def fetch_generation_records(
    table: str | None = None,
    *,
    limit: Optional[int] = None,
) -> list[Any]:
    """Fetch rows from Supabase and validate as :class:`~annotation_pipeline.data_models.GenerationRecord`."""
    from annotation_pipeline.data_models import GenerationRecord

    rows = fetch_generations_raw(table=table, limit=limit)  # None → default_generations_table()
    out = []
    for row in rows:
        row = dict(row)
        row.pop("logprobs", None)
        out.append(GenerationRecord.model_validate(row))
    return out
