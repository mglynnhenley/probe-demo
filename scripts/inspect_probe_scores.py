#!/usr/bin/env python3
"""Qualitative inspection of a trained probe's per-token scores.

Renders three views from a previously saved probe_scores_<split>.npz so we can see
whether the probe fires on actual policy-violating text:

  1. Top-K highest-scoring tokens with ±context, flagged by whether they fall
     inside a labelled violation span.
  2. Per-sequence summary — max / mean-top-k probe score split by whether the
     completion contains any labelled violation span.
  3. Rendered completions with `[tok]` marking probe-flagged tokens (prob ≥
     threshold) and `{tok}` marking ground-truth span tokens, grouped into
     TP / FP / FN / TN buckets.

Example:
  .venv/bin/python scripts/inspect_probe_scores.py \
      configs/probe/qwen_mac_tipping_off.yaml --split all
"""
from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
for path in (ROOT, TRAIN_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def _decode_completion_tokens(tokenizer, prompt: str, completion: str, chat_template_kwargs):
    """Return (token_id, token_text) for each completion token, in order."""
    utils_module = importlib.import_module("utils")
    encode_probe_chat = utils_module.encode_probe_chat
    probe_completion_char_start = utils_module.probe_completion_char_start

    enc = encode_probe_chat(
        tokenizer,
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        chat_template_kwargs=chat_template_kwargs,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    comp_char_start = probe_completion_char_start(
        tokenizer, prompt, completion, chat_template_kwargs=chat_template_kwargs
    )
    comp_idx = next(
        (idx for idx, (_, end) in enumerate(offsets) if end > comp_char_start),
        len(offsets),
    )
    comp_ids = input_ids[comp_idx:]
    texts = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in comp_ids]
    return texts


def _has_any_span(example: Dict[str, Any]) -> bool:
    vals = example.get("annotations_val") or []
    return any(v == 1.0 for v in vals)


def _scan_top_tokens(
    per_example: Dict[int, Dict[str, Any]],
    top_k: int = 30,
    context_window: int = 8,
) -> str:
    """Top-K highest-scoring tokens globally, with context."""
    flat: List[Tuple[float, int, int, int]] = []  # (prob, example_idx, token_idx, label)
    for ex_idx, rec in per_example.items():
        for tok_idx, prob, label in zip(rec["token_indices"], rec["probs"], rec["labels"]):
            flat.append((float(prob), int(ex_idx), int(tok_idx), int(label)))
    flat.sort(key=lambda x: -x[0])

    lines = [f"# Top-{top_k} highest-scoring tokens (probe prob, ↓)\n"]
    lines.append(f"{'rank':>4}  {'prob':>6}  {'label':>5}  ex   context")
    for rank, (prob, ex_idx, tok_idx, label) in enumerate(flat[:top_k], start=1):
        texts = per_example[ex_idx]["token_texts"]
        labels_all = per_example[ex_idx]["label_by_pos"]  # dict: token_idx -> label
        lo = max(0, tok_idx - context_window)
        hi = min(len(texts), tok_idx + context_window + 1)
        parts = []
        for i in range(lo, hi):
            tok_text = texts[i].replace("\n", "⏎")
            is_target = i == tok_idx
            is_span = labels_all.get(i, 0) == 1
            marker = "◀" if is_target else ""
            if is_target:
                parts.append(f"⟦{tok_text}⟧{marker}")
            elif is_span:
                parts.append(f"{{{tok_text}}}")
            else:
                parts.append(tok_text)
        ctx = "".join(parts).strip()
        lines.append(f"{rank:>4}  {prob:>6.3f}  {label:>5}  {ex_idx:>3}  {ctx[:180]}")
    return "\n".join(lines) + "\n"


def _per_sequence_summary(per_example: Dict[int, Dict[str, Any]]) -> str:
    rows: List[Tuple[int, bool, int, int, float, float]] = []
    for ex_idx, rec in per_example.items():
        probs = np.asarray(rec["probs"], dtype=np.float32)
        labels = np.asarray(rec["labels"], dtype=np.uint8)
        if probs.size == 0:
            continue
        has_pos = bool(labels.any())
        max_p = float(probs.max())
        top3 = float(np.sort(probs)[-min(3, len(probs)):].mean())
        rows.append((ex_idx, has_pos, int(labels.sum()), int(probs.size), max_p, top3))

    def _fmt_block(header: str, rs):
        if not rs:
            return f"# {header}\n(empty)\n"
        rs_sorted = sorted(rs, key=lambda r: -r[4])
        max_ps = np.array([r[4] for r in rs_sorted], dtype=np.float32)
        lines = [
            f"# {header} — n={len(rs_sorted)}  "
            f"max_prob: mean={max_ps.mean():.3f} median={np.median(max_ps):.3f} "
            f"min={max_ps.min():.3f} max={max_ps.max():.3f}",
            f"{'ex':>3}  {'max_p':>6}  {'top3':>6}  {'n_pos':>5}  {'n_tok':>5}",
        ]
        for ex_idx, _has, n_pos, n_tok, max_p, top3 in rs_sorted:
            lines.append(f"{ex_idx:>3}  {max_p:>6.3f}  {top3:>6.3f}  {n_pos:>5}  {n_tok:>5}")
        return "\n".join(lines) + "\n"

    violating = [r for r in rows if r[1]]
    clean = [r for r in rows if not r[1]]
    return (
        _fmt_block("Completions WITH labelled violation spans", violating)
        + "\n"
        + _fmt_block("Completions WITHOUT labelled violation spans (clean)", clean)
    )


def _render_example(rec: Dict[str, Any], threshold: float) -> str:
    texts = rec["token_texts"]
    label_by = rec["label_by_pos"]
    prob_by = rec["prob_by_pos"]
    parts = []
    for i, text in enumerate(texts):
        text_disp = text
        is_span = label_by.get(i, 0) == 1
        is_flagged = prob_by.get(i, 0.0) >= threshold
        if is_span and is_flagged:
            parts.append(f"[{{{text_disp}}}]")
        elif is_flagged:
            parts.append(f"[{text_disp}]")
        elif is_span:
            parts.append(f"{{{text_disp}}}")
        else:
            parts.append(text_disp)
    return "".join(parts)


def _render_failures(
    per_example: Dict[int, Dict[str, Any]],
    dataset_rows: List[Dict[str, Any]],
    threshold: float,
    n_worst: int = 5,
) -> str:
    rows = []
    for ex_idx, rec in per_example.items():
        probs = np.asarray(rec["probs"], dtype=np.float32)
        labels = np.asarray(rec["labels"], dtype=np.uint8)
        if probs.size == 0:
            continue
        rows.append({
            "ex_idx": ex_idx,
            "has_pos": bool(labels.any()),
            "max_p": float(probs.max()),
            "n_flagged": int((probs >= threshold).sum()),
        })

    # FP: no span but high max_p. FN: has span but low max_p.
    fps = sorted([r for r in rows if not r["has_pos"]], key=lambda r: -r["max_p"])[:n_worst]
    fns = sorted([r for r in rows if r["has_pos"]], key=lambda r: r["max_p"])[:n_worst]
    tps = sorted([r for r in rows if r["has_pos"]], key=lambda r: -r["max_p"])[:n_worst]
    tns = sorted([r for r in rows if not r["has_pos"]], key=lambda r: r["max_p"])[:n_worst]

    def _section(title: str, bucket):
        out = [f"\n========== {title} (threshold prob ≥ {threshold}) =========="]
        out.append("  legend: [flagged-by-probe]  {ground-truth-span}  [{both}]")
        for r in bucket:
            rec = per_example[r["ex_idx"]]
            prompt = dataset_rows[r["ex_idx"]]["prompt"]
            rendered = _render_example(rec, threshold)
            out.append(
                f"\n--- ex={r['ex_idx']}  max_p={r['max_p']:.3f}  "
                f"n_flagged={r['n_flagged']}  has_span={r['has_pos']} ---"
            )
            out.append(f"PROMPT: {prompt[:300]}")
            out.append(f"COMPLETION:\n{rendered}")
        return "\n".join(out) + "\n"

    return (
        _section("TRUE POSITIVES (has span, high probe score)", tps)
        + _section("FALSE POSITIVES (no span, high probe score)", fps)
        + _section("FALSE NEGATIVES (has span, low probe score)", fns)
        + _section("TRUE NEGATIVES (no span, low probe score)", tns)
    )


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--split", type=click.Choice(["all", "train", "eval"]), default="all",
              help="Ignored when --annotations-override is set.")
@click.option(
    "--annotations-override",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Alternative annotated jsonl (e.g. a held-out test set). "
    "When set, the inspection script loads rows from this file and expects scores at "
    "<output_dir>/probe_scores_<stem-of-override>.npz unless --scores-path is given.",
)
@click.option(
    "--scores-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to probe_scores_<split>.npz (default: <output_dir>/probe_scores_<split>.npz).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write inspection_*.txt (default: alongside the npz).",
)
@click.option("--threshold", type=float, default=0.5, help="Prob threshold for 'flagged'.")
@click.option("--top-k", type=int, default=30, help="How many top-scoring tokens to list.")
@click.option("--n-worst", type=int, default=5, help="Examples per TP/FP/FN/TN bucket.")
def main(
    config_path: Path,
    split: str,
    annotations_override: Path | None,
    scores_path: Path | None,
    output_dir: Path | None,
    threshold: float,
    top_k: int,
    n_worst: int,
) -> None:
    TrainingConfig = importlib.import_module("config").TrainingConfig
    data_module = importlib.import_module("data")
    utils_module = importlib.import_module("utils")
    build_annotations_dataset = data_module.build_annotations_dataset
    build_annotations_dataset_dict = data_module.build_annotations_dataset_dict
    probe_prefill_token_count = utils_module.probe_prefill_token_count

    cfg = TrainingConfig.from_yaml(config_path.expanduser().resolve())

    if annotations_override is not None:
        dataset = build_annotations_dataset(annotations_override.expanduser().resolve())
        split = annotations_override.stem
    elif split == "all":
        dataset = build_annotations_dataset(Path(cfg.annotations_jsonl))
    else:
        dd = build_annotations_dataset_dict(
            path=Path(cfg.annotations_jsonl),
            test_size=cfg.val_fraction,
            seed=cfg.seed,
        )
        dataset = dd[split]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    max_prefill_tokens = cfg.max_model_len - 1

    def within_limit(example):
        n = probe_prefill_token_count(
            tokenizer,
            example["prompt"],
            example["completion"],
            chat_template_kwargs=cfg.chat_template_kwargs,
        )
        return n <= max_prefill_tokens

    dataset = dataset.filter(within_limit)
    rows = [dataset[i] for i in range(len(dataset))]
    print(f"[setup] dataset size after length filter = {len(rows)} examples")

    scores_path = (
        scores_path.expanduser().resolve()
        if scores_path is not None
        else Path(cfg.output_dir).resolve() / f"probe_scores_{split}.npz"
    )
    if not scores_path.is_file():
        raise SystemExit(f"Scores file not found: {scores_path}")
    d = np.load(scores_path, allow_pickle=True)
    print(f"[setup] loaded scores from {scores_path}")

    example_indices = d["example_indices"]
    token_indices = d["token_indices"]
    probs = d["probs"]
    labels = d["labels"]

    per_example: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"token_indices": [], "probs": [], "labels": []}
    )
    for ex_i, tok_i, p, lab in zip(example_indices, token_indices, probs, labels):
        rec = per_example[int(ex_i)]
        rec["token_indices"].append(int(tok_i))
        rec["probs"].append(float(p))
        rec["labels"].append(int(lab))

    # Attach decoded completion tokens + per-position lookups.
    for ex_idx, rec in per_example.items():
        row = rows[ex_idx]
        token_texts = _decode_completion_tokens(
            tokenizer, row["prompt"], row["completion"], cfg.chat_template_kwargs
        )
        rec["token_texts"] = token_texts
        rec["label_by_pos"] = {tok_i: lab for tok_i, lab in zip(rec["token_indices"], rec["labels"])}
        rec["prob_by_pos"] = {tok_i: p for tok_i, p in zip(rec["token_indices"], rec["probs"])}

    output_dir = (
        output_dir.expanduser().resolve() if output_dir is not None else scores_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_path = output_dir / f"inspection_top_tokens_{split}.txt"
    seq_path = output_dir / f"inspection_sequence_summary_{split}.txt"
    ren_path = output_dir / f"inspection_rendered_{split}.txt"

    top_path.write_text(
        _scan_top_tokens(per_example, top_k=top_k, context_window=8),
        encoding="utf-8",
    )
    seq_path.write_text(_per_sequence_summary(per_example), encoding="utf-8")
    ren_path.write_text(
        _render_failures(per_example, rows, threshold=threshold, n_worst=n_worst),
        encoding="utf-8",
    )

    print(f"Wrote {top_path}")
    print(f"Wrote {seq_path}")
    print(f"Wrote {ren_path}")


if __name__ == "__main__":
    main()
