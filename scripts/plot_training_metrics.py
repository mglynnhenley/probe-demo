#!/usr/bin/env python3
"""Plot curves from ``training_metrics.json`` (from ``save_training_metrics_json``).

Example:
  uv run python scripts/plot_training_metrics.py output/gemma4_31b_probe/training_metrics.json
  uv run python scripts/plot_training_metrics.py output/gemma4_31b_probe/training_metrics.json -o run.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _load_history(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary", {}), payload.get("log_history", [])


def _split_log_history(
    log_history: List[Dict[str, Any]],
) -> tuple[
    List[tuple],
    List[tuple],
    List[tuple],
]:
    """Return (loss_rows, train_probe_rows, eval_rows) as lists of tuples aligned by step."""
    loss_data: List[tuple] = []
    probe_data: List[tuple] = []
    eval_data: List[tuple] = []

    for e in log_history:
        if not isinstance(e, dict):
            continue
        step = e.get("step")
        if step is None:
            continue
        istep = int(step)

        # Independent branches: merged v2 rows may contain loss + probe + eval in one dict.
        if "eval_f1" in e:
            eval_data.append(
                (
                    istep,
                    float(e.get("eval_loss", float("nan"))),
                    float(e["eval_f1"]),
                    float(e.get("eval_token_accuracy", float("nan"))),
                    float(e.get("eval_precision_viol", float("nan"))),
                    float(e.get("eval_tpr", float("nan"))),
                    float(e.get("eval_tnr", float("nan"))),
                    float(e.get("eval_baseline_nonviol_acc", float("nan"))),
                )
            )
        if "loss" in e:
            loss_data.append(
                (
                    istep,
                    float(e["loss"]),
                    float(e.get("learning_rate", float("nan"))),
                    float(e.get("grad_norm", float("nan"))),
                )
            )
        if "f1" in e and "n_violation_tokens_in_batch" in e:
            probe_data.append(
                (
                    istep,
                    float(e["f1"]),
                    float(e.get("token_accuracy", float("nan"))),
                    float(e.get("prec_viol", float("nan"))),
                    float(e.get("tpr", float("nan"))),
                    float(e.get("frac_pred_viol", float("nan"))),
                    float(e.get("n_violation_tokens_in_batch", float("nan"))),
                    float(e.get("n_nonviolation_tokens_in_batch", float("nan"))),
                )
            )

    return loss_data, probe_data, eval_data


def plot_metrics(
    summary: Dict[str, Any],
    loss_data: List[tuple],
    probe_data: List[tuple],
    eval_data: List[tuple],
    title: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title or "Training metrics", fontsize=12)

    # --- Loss & LR ---
    ax = axes[0, 0]
    if loss_data:
        steps = [x[0] for x in loss_data]
        losses = [x[1] for x in loss_data]
        ax.plot(steps, losses, color="C0", alpha=0.85, linewidth=0.8, label="train loss")
        ax.set_ylabel("loss")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        ax2 = ax.twinx()
        lrs = [x[2] for x in loss_data]
        ax2.plot(steps, lrs, color="C2", alpha=0.6, linewidth=0.7, label="lr")
        ax2.set_ylabel("learning rate", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")
    else:
        ax.text(0.5, 0.5, "no loss entries", ha="center", va="center", transform=ax.transAxes)

    # --- Train probe (per forward) ---
    ax = axes[0, 1]
    if probe_data:
        steps = [x[0] for x in probe_data]
        ax.plot(steps, [x[1] for x in probe_data], color="C0", linewidth=0.6, alpha=0.8, label="f1")
        ax.plot(steps, [x[3] for x in probe_data], color="C1", linewidth=0.6, alpha=0.7, label="prec_viol")
        ax.plot(steps, [x[4] for x in probe_data], color="C3", linewidth=0.6, alpha=0.7, label="tpr")
        ax.set_ylabel("score")
        ax.set_xlabel("step")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Train (microbatch) probe metrics")
    else:
        ax.text(0.5, 0.5, "no probe metric entries", ha="center", va="center", transform=ax.transAxes)

    # --- Eval ---
    ax = axes[1, 0]
    if eval_data:
        steps = [x[0] for x in eval_data]
        ax.plot(steps, [x[2] for x in eval_data], "o-", color="C0", markersize=3, label="eval_f1")
        ax.plot(steps, [x[4] for x in eval_data], "s-", color="C1", markersize=2, alpha=0.85, label="eval_prec")
        ax.plot(steps, [x[5] for x in eval_data], "^-", color="C3", markersize=2, alpha=0.85, label="eval_tpr")
        ax.set_ylabel("score")
        ax.set_xlabel("step")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Evaluation")
        ax2 = ax.twinx()
        ax2.plot(steps, [x[1] for x in eval_data], color="gray", alpha=0.5, linestyle="--", label="eval_loss")
        ax2.set_ylabel("eval_loss", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
    else:
        ax.text(0.5, 0.5, "no eval entries", ha="center", va="center", transform=ax.transAxes)

    # --- Train frac_pred_viol ---
    ax = axes[1, 1]
    if probe_data:
        steps = [x[0] for x in probe_data]
        ax.plot(steps, [x[5] for x in probe_data], color="purple", linewidth=0.5, alpha=0.75)
        ax.set_xlabel("step")
        ax.set_ylabel("frac_pred_viol")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title("Predicted violation rate (train microbatch)")
    else:
        ax.text(0.5, 0.5, "no probe data", ha="center", va="center", transform=ax.transAxes)

    footer = (
        f"summary: global_step={summary.get('global_step')}  epoch={summary.get('epoch')}  "
        f"best_metric={summary.get('best_metric')}  |  "
        f"points: loss={len(loss_data)}  probe={len(probe_data)}  eval={len(eval_data)}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="dimgray")

    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training_metrics.json curves.")
    p.add_argument(
        "metrics_json",
        type=Path,
        help="Path to training_metrics.json",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: alongside JSON as training_metrics_plot.png)",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    path = args.metrics_json.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"not found: {path}")

    summary, log_history = _load_history(path)
    loss_data, probe_data, eval_data = _split_log_history(log_history)

    out = args.output
    if out is None:
        out = path.parent / "training_metrics_plot.png"
    else:
        out = out.expanduser().resolve()

    title = f"{path.parent.name} / {path.name}"
    fig = plot_metrics(summary, loss_data, probe_data, eval_data, title=title)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
