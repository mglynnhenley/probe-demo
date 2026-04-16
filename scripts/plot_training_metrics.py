#!/usr/bin/env python3
"""Plot curves from ``training_metrics.json`` (from ``save_training_metrics_json``).

Example:
  uv run python scripts/plot_training_metrics.py output/gemma4_31b_probe/training_metrics.json
  uv run python scripts/plot_training_metrics.py output/gemma4_31b_probe/training_metrics.json -o run.png
"""

from __future__ import annotations

import argparse
import json
import math
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
                    float(e.get("tnr", float("nan"))),
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
        ax.plot(steps, [x[5] for x in probe_data], color="C4", linewidth=0.6, alpha=0.7, label="tnr")
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
        ax.plot(steps, [x[6] for x in eval_data], "d-", color="C4", markersize=2, alpha=0.85, label="eval_tnr")
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
        ax.plot(steps, [x[6] for x in probe_data], color="purple", linewidth=0.5, alpha=0.75)
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


def _last_finite(values: List[float]) -> float | None:
    for value in reversed(values):
        if math.isfinite(value):
            return value
    return None


def _best_by_max(rows: List[tuple], value_idx: int) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    for row in rows:
        value = float(row[value_idx])
        if not math.isfinite(value):
            continue
        step = int(row[0])
        if best is None or value > best[1]:
            best = (step, value)
    return best


def _best_by_min(rows: List[tuple], value_idx: int) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    for row in rows:
        value = float(row[value_idx])
        if not math.isfinite(value):
            continue
        step = int(row[0])
        if best is None or value < best[1]:
            best = (step, value)
    return best


def build_summary_text(
    summary: Dict[str, Any],
    loss_data: List[tuple],
    probe_data: List[tuple],
    eval_data: List[tuple],
    *,
    source_path: Path,
) -> str:
    lines: List[str] = []
    lines.append(f"Run: {source_path.parent.name}/{source_path.name}")
    lines.append(
        "Summary: "
        f"global_step={summary.get('global_step')}  "
        f"epoch={summary.get('epoch')}  "
        f"best_metric={summary.get('best_metric')}  "
        f"best_model_checkpoint={summary.get('best_model_checkpoint')}"
    )
    lines.append(
        f"Points: loss={len(loss_data)}  train_probe={len(probe_data)}  eval={len(eval_data)}"
    )

    if loss_data:
        final_step = int(loss_data[-1][0])
        final_loss = float(loss_data[-1][1])
        final_lr = _last_finite([float(x[2]) for x in loss_data])
        final_grad_norm = _last_finite([float(x[3]) for x in loss_data])
        best_loss = _best_by_min(loss_data, 1)
        final_lr_s = f"{final_lr:.6g}" if final_lr is not None else "nan"
        lines.append("")
        lines.append("Train loss:")
        lines.append(
            f"final step={final_step}  loss={final_loss:.6f}  "
            f"lr={final_lr_s}"
        )
        if final_grad_norm is not None:
            lines.append(f"final grad_norm={final_grad_norm:.6f}")
        if best_loss is not None:
            lines.append(f"best loss={best_loss[1]:.6f} at step={best_loss[0]}")

    if probe_data:
        final = probe_data[-1]
        best_f1 = _best_by_max(probe_data, 1)
        best_tpr = _best_by_max(probe_data, 4)
        lines.append("")
        lines.append("Train probe metrics:")
        lines.append(
            f"final step={int(final[0])}  f1={float(final[1]):.6f}  "
            f"token_accuracy={float(final[2]):.6f}  "
            f"prec_viol={float(final[3]):.6f}  "
            f"tpr={float(final[4]):.6f}  "
            f"tnr={float(final[5]):.6f}  "
            f"frac_pred_viol={float(final[6]):.6f}"
        )
        lines.append(
            f"final batch token counts: violation={int(final[7])}  nonviolation={int(final[8])}"
        )
        if best_f1 is not None:
            lines.append(f"best train f1={best_f1[1]:.6f} at step={best_f1[0]}")
        if best_tpr is not None:
            lines.append(f"best train tpr={best_tpr[1]:.6f} at step={best_tpr[0]}")

    if eval_data:
        final = eval_data[-1]
        best_eval_f1 = _best_by_max(eval_data, 2)
        best_eval_loss = _best_by_min(eval_data, 1)
        lines.append("")
        lines.append("Evaluation:")
        lines.append(
            f"final step={int(final[0])}  eval_loss={float(final[1]):.6f}  "
            f"eval_f1={float(final[2]):.6f}  "
            f"eval_token_accuracy={float(final[3]):.6f}  "
            f"eval_precision_viol={float(final[4]):.6f}  "
            f"eval_tpr={float(final[5]):.6f}  "
            f"eval_tnr={float(final[6]):.6f}  "
            f"eval_baseline_nonviol_acc={float(final[7]):.6f}"
        )
        if best_eval_f1 is not None:
            lines.append(f"best eval_f1={best_eval_f1[1]:.6f} at step={best_eval_f1[0]}")
        if best_eval_loss is not None:
            lines.append(f"best eval_loss={best_eval_loss[1]:.6f} at step={best_eval_loss[0]}")
    else:
        lines.append("")
        lines.append("Evaluation:")
        lines.append("no eval entries")

    return "\n".join(lines)


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
    summary_text = build_summary_text(
        summary,
        loss_data,
        probe_data,
        eval_data,
        source_path=path,
    )
    summary_out = out.with_name(f"{out.stem}_summary.txt")
    summary_out.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    print("")
    print(f"Wrote {out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
