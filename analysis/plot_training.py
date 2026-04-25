#!/usr/bin/env python
"""
Plot training curves and final metrics from fit_model.json / fit_kernel.json.

Produces up to three figures:
  1. Training loss (per batch, smoothed) for all neural architectures
  2. Grid of per-epoch validation metrics for all neural architectures
  3. Bar chart of final test metrics for kernel methods (fit_kernel.json),
     optionally overlaid with the final-epoch scores of the neural probes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import matplotlib.pyplot as plt
import numpy as np

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

VAL_METRICS = ["eval_loss", "f1", "auprc", "auroc", "precision", "recall", "tnr", "accuracy"]


def _smooth(values: List[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _load_models(data: Dict[str, Any]) -> Dict[str, Dict]:
    """Flatten mlp and covseq entries into a single {label: data} dict."""
    models: Dict[str, Dict] = {}
    for label, v in data.get("mlp", {}).items():
        models[label] = v
    for key, v in data.get("covseq", {}).items():
        ws = key.removeprefix("window_")
        models[f"CovSeq w={ws}"] = v
    return models


def plot_train_loss(
    models: Dict[str, Dict],
    smooth_window: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    for label, v in models.items():
        tl = v.get("train_loss", [])
        if not tl:
            continue
        xs = [e["epoch_frac"] for e in tl]
        ys = [e["loss"] for e in tl]
        ys_s = _smooth(ys, smooth_window)
        xs_s = xs[: len(ys_s)] if smooth_window > 1 else xs
        ax.plot(xs_s, ys_s, label=label, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title(f"Training loss  (smoothed window={smooth_window})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved training loss → {output_path}")
    plt.close(fig)


def plot_val_metrics(
    models: Dict[str, Dict],
    metrics: List[str],
    output_path: Path,
) -> None:
    n = len(metrics)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = axes.flat if nrows > 1 else ([axes] if ncols == 1 else axes)

    for ax, metric in zip(axes_flat, metrics):
        for label, v in models.items():
            vpe = v.get("val_per_epoch", [])
            if not vpe or metric not in vpe[0]:
                continue
            xs = [e["epoch"] for e in vpe]
            ys = [e[metric] for e in vpe]
            ax.plot(xs, ys, marker="o", markersize=4, label=label, linewidth=1.5)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7)

    for ax in list(axes_flat)[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle("Validation metrics per epoch", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved val metrics    → {output_path}")
    plt.close(fig)


BAR_METRICS = ["f1", "auprc", "auroc", "precision", "recall", "tnr", "accuracy"]
TABLE_METRICS = ["f1", "auprc", "auroc", "precision", "recall", "tnr", "accuracy"]


def print_results_table(
    models: Dict[str, Dict],
    metrics: List[str],
    fmt: str = "markdown",
) -> None:
    """Print a final-metrics table to stdout.

    fmt: "markdown" (GFM pipe table) or "text" (plain aligned columns).
    """
    model_names = list(models.keys())
    finals = {
        name: v.get("final_metrics", {})
        for name, v in models.items()
    }

    if fmt == "markdown":
        header = "| metric | " + " | ".join(model_names) + " |"
        sep    = "| --- | " + " | ".join(["---"] * len(model_names)) + " |"
        print(header)
        print(sep)
        for metric in metrics:
            row = f"| {metric} | "
            row += " | ".join(
                f"{finals[n].get(metric, float('nan')):.4f}" for n in model_names
            )
            row += " |"
            print(row)
    else:
        col_w = max(14, max(len(n) for n in model_names) + 2)
        header = f"{'metric':<12}" + "".join(f"{n:>{col_w}}" for n in model_names)
        print(header)
        print("-" * len(header))
        for metric in metrics:
            row = f"{metric:<12}" + "".join(
                f"{finals[n].get(metric, float('nan')):>{col_w}.4f}" for n in model_names
            )
            print(row)


def plot_kernel_bars(
    kernel_results: Dict[str, Dict[str, float]],
    metrics: List[str],
    output_path: Path,
    neural_finals: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Bar chart of final test metrics for kernel methods.

    neural_finals, if provided, overlays the final-epoch scores of neural
    probes as individual markers on each bar group so everything is comparable
    in one view.
    """
    kernel_names = list(kernel_results.keys())
    n_metrics = len(metrics)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = list(axes.flat) if nrows > 1 else list(axes) if ncols > 1 else [axes]

    x = np.arange(len(kernel_names))
    bar_width = 0.6

    for ax, metric in zip(axes_flat, metrics):
        vals = [kernel_results[k].get(metric, float("nan")) for k in kernel_names]
        bars = ax.bar(x, vals, width=bar_width, color="steelblue", alpha=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

        # Overlay neural probe finals as horizontal dashes
        if neural_finals:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, (nlabel, nmetrics) in enumerate(neural_finals.items()):
                nval = nmetrics.get(metric)
                if nval is None:
                    continue
                ax.axhline(nval, linestyle="--", linewidth=1.2,
                           color=colors[i % len(colors)], label=nlabel, alpha=0.85)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(kernel_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(bottom=0)
        if neural_finals:
            ax.legend(fontsize=6, loc="lower right")

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Kernel probe — final test metrics\n(dashed lines = neural probe finals)", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved kernel bars    → {output_path}")
    plt.close(fig)


@click.command(context_settings=dict(help_option_names=["-h", "--help"], show_default=True))
@click.option("--input-json", type=click.Path(exists=True, path_type=Path),
              default=_RESULTS_DIR / "fit_model.json",
              help="JSON file produced by fit_model.py")
@click.option("--kernel-json", type=click.Path(path_type=Path),
              default=_RESULTS_DIR / "fit_kernel.json",
              help="JSON file produced by fit_kernel.py (optional)")
@click.option("--train-loss-path", type=click.Path(path_type=Path),
              default=_RESULTS_DIR / "training_loss.png",
              help="Output path for the training-loss plot")
@click.option("--val-metrics-path", type=click.Path(path_type=Path),
              default=_RESULTS_DIR / "val_metrics.png",
              help="Output path for the validation-metrics grid")
@click.option("--kernel-bars-path", type=click.Path(path_type=Path),
              default=_RESULTS_DIR / "kernel_bars.png",
              help="Output path for the kernel final-metrics bar chart")
@click.option("--smooth-window", type=int, default=20,
              help="Smoothing window for per-batch training loss (1 = no smoothing)")
@click.option("--val-metrics", default=",".join(VAL_METRICS),
              help="Comma-separated list of val metrics to include in the grid")
@click.option("--bar-metrics", default=",".join(BAR_METRICS),
              help="Comma-separated list of metrics to include in the kernel bar chart")
@click.option("--table-metrics", default=",".join(TABLE_METRICS),
              help="Comma-separated list of metrics to include in the results table")
@click.option("--table-format", type=click.Choice(["markdown", "text"]), default="markdown",
              help="Format for the printed results table")
@click.option("--no-plots", is_flag=True, default=False,
              help="Skip generating plot files (table only)")
def main(
    input_json: Path,
    kernel_json: Optional[Path],
    train_loss_path: Path,
    val_metrics_path: Path,
    kernel_bars_path: Path,
    smooth_window: int,
    val_metrics: str,
    bar_metrics: str,
    table_metrics: str,
    table_format: str,
    no_plots: bool,
) -> None:
    """Plot training curves and final metrics from fit_model.json / fit_kernel.json."""
    data = json.loads(input_json.read_text())
    models = _load_models(data)

    if not models:
        print("No model data found in JSON — exiting.")
        return

    val_metrics_list  = [m.strip() for m in val_metrics.split(",")  if m.strip()]
    bar_metrics_list  = [m.strip() for m in bar_metrics.split(",")  if m.strip()]
    table_metrics_list = [m.strip() for m in table_metrics.split(",") if m.strip()]

    cfg = data.get("config", {})
    print(f"Config: epochs={cfg.get('epochs')}  lr={cfg.get('lr')}  "
          f"train_records={cfg.get('n_train_records')}  test_records={cfg.get('n_test_records')}")
    print(f"Neural models: {list(models)}\n")

    print_results_table(models, table_metrics_list, fmt=table_format)

    if no_plots:
        return

    plot_train_loss(models, smooth_window, train_loss_path)
    plot_val_metrics(models, val_metrics_list, val_metrics_path)

    kernel_path = kernel_json if kernel_json and Path(kernel_json).exists() else None
    if kernel_path is None and kernel_json is not None:
        print(f"kernel-json not found at {kernel_json}, skipping kernel plot")
    if kernel_path is not None:
        kdata = json.loads(kernel_path.read_text())
        kernel_results = kdata.get("results", {})
        print(f"Kernel models: {list(kernel_results)}")

        # Collect final-epoch scores from neural probes for overlay
        neural_finals = {
            label: v["val_per_epoch"][-1]
            for label, v in models.items()
            if v.get("val_per_epoch")
        }
        plot_kernel_bars(kernel_results, bar_metrics_list, kernel_bars_path, neural_finals)


if __name__ == "__main__":
    main()
